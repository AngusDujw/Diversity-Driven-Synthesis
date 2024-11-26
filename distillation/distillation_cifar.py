import os
import random
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data.distributed
from PIL import Image
from perterb import Perterb
from utils import BNFeatureHook, lr_cosine_policy, save_images, clip_image, denormalize_image
import wandb


def get_images(args, model_teacher, hook_for_display, ipc_id):
    save_every = 100
    batch_size = args.batch_size
    best_cost = 1e4

    loss_r_feature_layers = []
    for module in model_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BNFeatureHook(module, args))

    # setup target labels
    if args.dataset=='cifar10':
        targets_all = torch.LongTensor(np.arange(10))
    else:
        targets_all = torch.LongTensor(np.arange(100))
    for kk in range(0, 100, batch_size):
        targets = targets_all[kk : min(kk + batch_size, 100)].to("cuda")

        # init with original image
        loaded_tensor = torch.load(args.init_path + '/tensor_' + str(ipc_id) + '.pt').to('cuda')
        inputs = loaded_tensor.detach().clone()
        inputs.requires_grad_(True)
        iterations_per_layer = args.iteration
        lim_0, lim_1 = args.jitter, args.jitter

        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps=1e-8)
        paras = model_teacher.parameters()
        # perterb weights
        adjusted_optimizer = Perterb(paras, optimizer, rho=args.rho/args.steps)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)  
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)

            # apply random jitter offsets
            off1 = random.randint(0, lim_0)
            off2 = random.randint(0, lim_1)
            inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))

            # forward pass
            optimizer.zero_grad()
            if iteration ==0:
                for ii in range(args.steps):
                    outputs = model_teacher(inputs_jit)
                    loss_ce = criterion(outputs, targets)
                    loss_ce.backward()
                    adjusted_optimizer.first_step(True)

            outputs = model_teacher(inputs_jit)

            # classification loss
            loss_ce = criterion(outputs, targets)

            # feature loss
            rescale = [args.first_bn_multiplier] + [1.0 for _ in range(len(loss_r_feature_layers) - 1)]
            loss_r_bn_feature = [
                mod.r_feature.to(loss_ce.device) * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)
            ]
            loss_r_bn_feature = torch.stack(loss_r_bn_feature).sum()

            loss_aux = args.r_bn * loss_r_bn_feature

            loss = loss_ce + loss_aux

            if iteration % save_every == 0 and args.verifier:
                print("------------iteration {}----------".format(iteration))
                print("loss_ce", loss_ce.item())
                print("loss_aux", loss_aux.item())
                print("loss_total", loss.item())
                # comment below line can speed up the training (no validation process)
                if hook_for_display is not None:
                    acc_jit, _ = hook_for_display(inputs_jit, targets)
                    acc_image, loss_image = hook_for_display(inputs, targets)

                    metrics = {
                        'crop/acc_crop': acc_jit,
                        'image/acc_image': acc_image,
                        'image/loss_image': loss_image,
                    }
                    wandb_metrics.update(metrics)

                metrics = {
                    'crop/loss_ce': loss_ce.item(),
                    'crop/loss_r_bn_feature': loss_r_bn_feature.item(),
                    'crop/loss_total': loss.item(),
                }
                wandb_metrics.update(metrics)
                wandb.log(wandb_metrics)

            # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            inputs.data = clip_image(inputs.data, args.dataset)

            if best_cost > loss.item() or iteration == 1:
                best_inputs = inputs.data.clone()

        if args.store_best_images:
            best_inputs = inputs.data.clone()  # using multicrop, save the last one
            best_inputs = denormalize_image(best_inputs, args.dataset)
            save_images(args, best_inputs, targets, ipc_id)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        optimizer.state = collections.defaultdict(dict)

    torch.cuda.empty_cache()

def main_syn(args, ipc_id):
    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)
    # load teacher model
    import torchvision
    if args.dataset == 'cifar10':
        num_classes = 10
    else:
        num_classes = 100
    model_teacher = torchvision.models.resnet18(num_classes=num_classes)
    model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model_teacher.maxpool = nn.Identity()

    model_teacher = nn.DataParallel(model_teacher).cuda()
    if args.dataset == "cifar10":
        pretrained_path = './pretrain/save/cifar10/resnet18_E200/ckpt.pth'
    else:
        pretrained_path = './pretrain/save/cifar100/resnet18_E200/ckpt.pth'

    checkpoint = torch.load(pretrained_path)
    model_teacher.load_state_dict(checkpoint["state_dict"])

    model_teacher.eval()

    hook_for_display = None
    get_images(args, model_teacher, hook_for_display, ipc_id)


def parse_args():
    parser = argparse.ArgumentParser("DWA: enhancing DD through directed weight adjustment")
    """Data save flags"""
    parser.add_argument("--exp-name", type=str, default="distillation-c100-ipc50", help="name of the experiment, subfolder under syn_data_path")
    parser.add_argument("--dataset", type=str, default='cifar100', help="choose from cifar10 and cifar100")
    parser.add_argument("--syn-data-path", type=str, default="./syn_data", help="where to store synthetic data")
    parser.add_argument("--store-best-images", action="store_true", help="whether to store best images")
    """Optimization related flags"""
    parser.add_argument("--batch-size", type=int, default=100, help="number of images to optimize at the same time")
    parser.add_argument("--iteration", type=int, default=1000, help="num of iterations to optimize the synthetic data")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate for optimization")
    parser.add_argument("--jitter", default=4, type=int, help="random shift on the synthetic data")
    parser.add_argument("--r-bn", type=float, default=0.05, help="coefficient for BN feature distribution regularization")
    parser.add_argument("--first-bn-multiplier", type=float, default=10.0, help="additional multiplier on first bn layer of R_bn")
    """Model related flags"""
    parser.add_argument("--arch-name", type=str, default="resnet18", help="arch name from pretrained torchvision models")
    parser.add_argument("--verifier", action="store_true", help="whether to evaluate synthetic data with another model")
    parser.add_argument("--verifier-arch",type=str, help="arch name from torchvision models to act as a verifier")
    parser.add_argument("--ipc-start", default=0, type=int)
    parser.add_argument("--ipc-end", default=50, type=int)
    """Directed weight adjustment related flags"""
    parser.add_argument('--init-path', default='./distillation/init_images/cifar100', type=str)
    parser.add_argument('--rho', default=0.05, type=float)
    parser.add_argument('--steps', default=1, type=int)
    parser.add_argument("--r-var", default=1, type=float)


    args = parser.parse_args()

    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    return args


if __name__ == "__main__":

    args = parse_args()

    if not wandb.api.api_key:
        wandb.login(key='')
    wandb.init(project='dwa-cifar', name=args.exp_name)
    global wandb_metrics
    wandb_metrics = {}
   
    for ipc_id in range(args.ipc_start, args.ipc_end):
        print("ipc = ", ipc_id)
        wandb.log({'ipc_id': ipc_id})
        import time
        start_time = time.time()
        main_syn(args, ipc_id)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("using ", elapsed_time)

    wandb.finish()
