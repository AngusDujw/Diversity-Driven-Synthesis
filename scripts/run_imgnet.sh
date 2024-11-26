# python distillation/distillation_imgnet.py \
#     --exp-name distillation-imgnet-ipc50  \
#     --syn-data-path ./syn_data/ \
#     --init-path ./distillation/init_images/imgnet/ \
#     --arch-name resnet18 \
#     --batch-size 100 --lr 0.25 --iteration 2000 --r-bn 0.01 \
#     --r-var 2 --steps 15 --rho 15e-3 \
#     --store-best-images \
#     --ipc-start 0 --ipc-end 50 


python validation/validation_imgnet.py \
    --epochs 300 --batch-size 128 --ipc 3 \
    --mix-type cutmix \
    --cos -T 20 -j 4 \
    --train-dir ./syn_data/distillation-imgnet-ipc50 \
    --output-dir ./syn_data/validation-imgnet-ipc50 \
    --val-dir /data/zhangxin/data/Imagenet-1k/val \
    --teacher-model resnet18 \
    --model resnet18 