import argparse
import os
import random
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms

from PIL import Image
from utils import BNFeatureHook, lr_cosine_policy, save_images, validate, denormalize_image, clip_image
from perterb import Perterb
import wandb




def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    normalize = transforms.Normalize(
        mean=[0.4802, 0.4481, 0.3975],
        std=[0.2302, 0.2265, 0.2262]
    )
    preprocess = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        normalize,
    ])
    return preprocess(image)


def get_images(args, model_teacher, hook_for_display, ipc_id):
    print("generating one IPC images (200)")

    batch_size = args.batch_size
    dataset = args.dataset

    loss_r_feature_layers = []
    for module in model_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(BNFeatureHook(module, args))

    # setup target labels
    targets_all = torch.LongTensor(np.arange(200))

    for kk in range(0, 200, batch_size):
                
        start_index = kk
        end_index = min(kk + batch_size, 200)
        targets = targets_all[start_index:end_index].to('cuda')

        inputs = []
        for folder_index in range(start_index, end_index):
            folder_path = os.path.join(args.init_path, args.sorted_path[folder_index])
            image_path = os.path.join(folder_path, os.listdir(folder_path)[ipc_id])
            image = load_image(image_path)
            inputs.append(image)

        inputs = torch.stack(inputs).to('cuda')

        inputs.requires_grad = True
        iterations_per_layer = args.iteration
        lim_0, lim_1 = args.jitter , args.jitter
        optimizer = optim.Adam([inputs], lr=args.lr, betas=[0.5, 0.9], eps = 1e-8)
        paras = model_teacher.parameters()

        adjusted_optimizer = Perterb(paras, optimizer, rho = args.rho / args.steps)
        lr_scheduler = lr_cosine_policy(args.lr, 0, iterations_per_layer)  # 0 - do not use warmup
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()

        for iteration in range(iterations_per_layer):
            # learning rate scheduling
            lr_scheduler(optimizer, iteration, iteration)

            aug_func = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
            ])
            inputs_jit = aug_func(inputs)

            # apply random jitter offsets
            off1 = random.randint(0, lim_0)
            off2 = random.randint(0, lim_1)
            inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

            # forward pass
            optimizer.zero_grad()

            if iteration ==0:
                for _ in range(args.steps):
                    outputs = model_teacher(inputs_jit)
                    loss_ce = criterion(outputs, targets)
                    loss_ce.backward()
                    adjusted_optimizer.first_step(True)

            outputs = model_teacher(inputs_jit)

            # R_cross classification loss
            loss_ce = criterion(outputs, targets)

            # R_feature loss
            rescale = [args.first_bn_multiplier] + [1.]* (len(loss_r_feature_layers) - 1)
            loss_r_bn_feature = [
                mod.r_feature.to(loss_ce.device) * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)
            ]
            loss_r_bn_feature = torch.stack(loss_r_bn_feature).sum()
            
            # final loss
            loss_aux = args.r_bn * loss_r_bn_feature
            loss = loss_ce + loss_aux

            metrics = {
                'syn/loss_ce': loss_ce.item(),
                'syn/loss_aux': loss_aux.item(),
                'syn/loss_total': loss.item(),
                # 'syn/aux_weight': aux_weight,
                # 'image/acc_image': acc_image,
                # 'image/loss_image': loss_image,
                'syn/ipc_id': ipc_id,
            }

            wandb_metrics.update(metrics)
            wandb.log(wandb_metrics)

            save_every = args.batch_size
            if iteration % save_every == 0 and args.verifier:
                print("------------iteration {}----------".format(iteration))
                print("total loss", loss.item())
                print("loss_r_bn_feature", loss_r_bn_feature.item())
                print("main criterion", criterion(outputs, targets).item())
                # comment below line can speed up the training (no validation process)
                if hook_for_display is not None:
                    hook_for_display(inputs, targets)

            # do image update
            loss.backward()
            optimizer.step()

            # clip color outlayers
            inputs.data = clip_image(inputs.data, args.dataset)

        if args.store_best_images:
            best_inputs = inputs.data.clone()  # using multicrop, save the last one
            best_inputs = denormalize_image(best_inputs, args.dataset)
            save_images(args, best_inputs, targets, ipc_id)


def main_syn(ipc_id, args):

    if not os.path.exists(args.syn_data_path):
        os.makedirs(args.syn_data_path)

    model_teacher = torchvision.models.resnet18(num_classes=200)
    model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model_teacher.maxpool = nn.Identity()

    model_teacher = model_teacher.cuda()

    pretrained_path = './pretrain/save/tiny/resnet18_E100/ckpt.pth'
    checkpoint = torch.load(pretrained_path)
    model_teacher.load_state_dict(checkpoint["model"])

    model_teacher.eval()

    hook_for_display = None

    get_images(args, model_teacher, hook_for_display, ipc_id)



def parse_args():
    parser = argparse.ArgumentParser("DWA: enhancing DD through directed weight adjustment")
    """Data save flags"""
    parser.add_argument("--exp-name", type=str, default="distillation-tiny-ipc50", help="name of the experiment, subfolder under syn_data_path")
    parser.add_argument("--dataset", type=str, default='tiny', help="")
    parser.add_argument('--syn-data-path', type=str, default='./syn_data', help='where to store synthetic data')
    parser.add_argument('--store-best-images', action='store_true', help='whether to store best images')
    """Optimization related flags"""
    parser.add_argument('--batch-size', type=int, default=100, help='number of images to optimize at the same time')
    parser.add_argument('--iteration', type=int, default=1000, help='num of iterations to optimize the synthetic data')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate for optimization')
    parser.add_argument('--jitter', default=4, type=int, help='random shift on the synthetic data')
    parser.add_argument('--r-bn', type=float, default=0.05, help='coefficient for BN feature distribution regularization')
    parser.add_argument('--first-bn-multiplier', type=float, default=10., help='additional multiplier on first bn layer of R_bn')
    """Model related flags"""
    parser.add_argument('--arch-name', type=str, default='resnet18',  help='arch name from pretrained torchvision models')
    parser.add_argument('--verifier', action='store_true', help='whether to evaluate synthetic data with another model')
    parser.add_argument('--verifier-arch', type=str, default='', help="arch name from torchvision models to act as a verifier")
    parser.add_argument('--ipc-start', default=0, type=int)
    parser.add_argument('--ipc-end', default=50, type=int)
    """Directed weight adjustment related flags"""
    parser.add_argument('--init-path', default='./distillation/init_images/tiny', type=str)
    parser.add_argument('--rho', type=float, default=0.015)
    parser.add_argument('--steps', default=1, type=int)
    parser.add_argument("--r-var", default=1, type=float)
  
    args = parser.parse_args()

    args.syn_data_path = os.path.join(args.syn_data_path, args.exp_name)
    args.sorted_path = sorted(os.listdir(args.init_path))
    return args


if __name__ == '__main__':
    args = parse_args()

    if not wandb.api.api_key:
        wandb.login(key='')
    wandb.init(project='dwa-tiny', name=args.exp_name)
    global wandb_metrics
    wandb_metrics = {}

    for ipc_id in range(args.ipc_start, args.ipc_end):
        print("ipc = ", ipc_id)
        wandb.log({'ipc_id': ipc_id})
        import time
        start_time = time.time()
        main_syn(ipc_id, args)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("using ", elapsed_time)
        
    wandb.finish()
