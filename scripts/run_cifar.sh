python distillation/distillation_cifar.py \
     --iteration 1000 --r-bn 0.01 --batch-size 100 --lr 0.25 \
     --exp-name distillation-c100-ipc50 \
     --store-best-images \
     --syn-data-path ./syn_data/ \
     --init-path ./distillation/init_images/cifar100 \
     --steps 12 --rho 15e-3 --ipc-start 0 --ipc-end 50 --r-var 11 \
     --dataset cifar100 



python validation/validation_cifar.py \
    --epochs 400 --batch-size 128 --ipc 10 \
    --syn-data-path ./syn_data/distillation-c100-ipc50 \
    --output-dir ./syn_data/validation-c100-ipc50 \
    --networks resnet18 --dataset cifar100 



