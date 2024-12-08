# 🌟 Diversity-Driven Synthesis: Enhancing Dataset Distillation through Directed Weight Adjustment 
## 🔥 NeurIPS 2024 Spotlight 🔥

>[Diversity-Driven Synthesis: Enhancing Dataset Distillation through Directed Weight Adjustment](https://arxiv.org/abs/2409.17612).<br>
> [Jiawei Du](https://scholar.google.com/citations?user=WrJKEzEAAAAJ&hl=zh-CN), [Xin Zhang](https://zhangxin-xd.github.io/), [Juncheng Hu](https://scholar.google.com/citations?user=F8_T6XMAAAAJ&hl=en&oi=sra), [Wenxin Huang](https://scholar.google.com/citations?user=TY2PC4CgIwkC&hl=en), [Joey Tianyi Zhou](https://joeyzhouty.github.io/) <br>
> A*Star, XiDian University,  National University of Singapore, Hubei University
> 
## 📖 Introduction
<p align="justify">
The sharp increase in data-related expenses has motivated research into condensing datasets while retaining the most informative features. Dataset distillation has thus recently come to the fore. This paradigm generates synthetic datasets that are representative enough to replace the original dataset in training a neural network. To avoid redundancy in these synthetic datasets, it is crucial that each element contains unique features and remains diverse from others during the synthesis stage. In this paper, we provide a thorough theoretical and empirical analysis of diversity within synthesized datasets. We argue that enhancing diversity can improve the parallelizable yet isolated synthesizing approach. Specifically, we introduce a novel method that employs dynamic and directed weight adjustment techniques to modulate the synthesis process, thereby maximizing the representativeness and diversity of each synthetic instance. Our method ensures that each batch of synthetic data mirrors the characteristics of a large, varying subset of the original dataset. Extensive experiments across multiple datasets, including CIFAR, Tiny-ImageNet, and ImageNet-1K, demonstrate the superior performance of our method, highlighting its effectiveness in producing diverse and representative synthetic datasets with minimal computational expense.</p>

---

## ⚙️ Installation

To get started, follow these instructions to set up the environment and install dependencies.

1. **Clone this repository**:
    ```bash
    git clone https://github.com/AngusDujw/Diversity-Driven-Synthesis.git
    cd Diversity-Driven-Synthesis
    ```

2. **Install required packages**:
   You don’t need to create a new environment; simply ensure that you have compatible versions of CUDA and PyTorch installed.
---

## 🚀 Usage

Here’s how to use this code for distillation and evaluation:
- **Preparation**
For ImageNet-1K, we utilize the pre-trained weights available in torchvision. As for CIFAR and Tiny-ImageNet datasets, we offer the trained weights on this [link](https://drive.google.com/drive/folders/1dH96COYa4kCquQ4c6wEnt7QobGMl6M3N?usp=sharing). Alternatively, you can generate the pre-trained weights yourself using the following code.
    ```bash
    bash squeeze.sh
    ```
- **Distillation**:
    Before performing distillation, please first prepare the images by randomly sampling from the original dataset and saving them as tensors. We provide the tensor-formatted initialization images at this [link](https://drive.google.com/drive/folders/1ueAnTXOUGiQ_E9iIssNYmEBX4vlVQEDZ?usp=sharing) .

    Cifar:
    ```bash
    python distillation/distillation_cifar.py 
        --iteration 1000 --r-bn 0.01 --batch-size 100 --lr 0.25 
        --exp-name distillation-c100-ipc50 
        --store-best-images 
        --syn-data-path ./syn_data/ 
        --init_path ./distillation/init_images/cifar100 
        --steps 12 --rho 15e-3 --ipc-start 0 --ipc-end 50 --r-var 11 
        --dataset cifar100 
    ```
    Tiny-ImageNet:
    ```bash
    python distillation/distillation_tiny.py 
         --iteration 2000 --r-bn 0.01 --batch-size 200 --lr 0.1 
         --exp-name distillation-tiny-ipc50 
         --store-best-images 
         --syn-data-path ./syn_data/ 
         --init-path ./distillation/init_images/tiny 
         --steps 12 --rho 15e-3 --ipc-start 0 --ipc-end 50 --r-var 11 
         --dataset tiny 
    ```
    ImageNet-1K:
    ```bash
    python distillation/distillation_imgnet.py 
        --exp-name distillation-imgnet-ipc50  
        --syn-data-path ./syn_data/ 
        --init-path ./distillation/init_images/imgnet/ 
        --arch-name resnet18 
        --batch-size 100 --lr 0.25 --iteration 2000 --r-bn 0.01 
        --r-var 2 --steps 15 --rho 15e-3 
        --store-best-images 
        --ipc-start 0 --ipc-end 50 
    ```
- **Evaluation**:
  
    Cifar:
    ```bash
    python validation/validation_cifar.py 
          --epochs 400 --batch-size 128 --ipc 10 
          --syn-data-path ./syn_data/distillation-c100-ipc50 
          --output-dir ./syn_data/validation-c100-ipc50 
          --networks resnet18 --dataset cifar100 
    ```
    Tiny-ImageNet:
    ```bash
    python validation/validation_tiny.py 
            --epochs 200 --batch-size 64 --ipc 50 
            --lr 0.2 --momentum 0.9 --weight-decay 1e-4 
            --lr-scheduler cosineannealinglr 
            --lr-warmup-epochs 5 
            --lr-warmup-method linear 
            --lr-warmup-decay 0.01
            --data-path ./data/tiny/
            --syn-data-path ./syn_data/distillation-tiny-ipc50/ 
            --model resnet18   
    ```
    ImageNet-1K:
    ```bash
    python validation/validation_imgnet.py 
        --epochs 300 --batch-size 128 --ipc 50 
        --mix-type cutmix 
        --cos -T 20 -j 4 
        --train-dir ./syn_data/distillation-imgnet-ipc50 
        --output-dir ./syn_data/validation-imgnet-ipc50 
        --val-dir ./data/Imagenet-1k/val 
        --teacher-model resnet18 
        --model resnet18 
    ```
we also provide the `.sh` script in the `scripts` directory.

---

## 📊 Results

Our experiments demonstrate the effectiveness of the proposed approach across various benchmarks. 

![Results](./imgs/results.png)

For detailed experimental results and further analysis, please refer to the full paper.

---

## 📑 Citation

If you find this code useful in your research, please consider citing our work:

```bibtex
@inproceedings{dwa2024neurips,
    title={Diversity-Driven Synthesis: Enhancing Dataset Distillation through Directed Weight Adjustment},
    author={Du, Jiawei and Zhang, Xin and Hu, Juncheng and Huang, Wenxin and Zhou, Joey Tianyi},
    booktitle={Adv. Neural Inf. Process. Syst. (NeurIPS)},
    year={2024}
}
```
---
## 🎉 Reference
Our code has referred to previous work:

[Squeeze, Recover and Relabel: Dataset Condensation at ImageNet Scale From A New Perspective](https://github.com/VILA-Lab/SRe2L)

