python distillation/distillation_tiny.py \
     --iteration 2000 --r-bn 0.01 --batch-size 200 --lr 0.1 \
     --exp-name distillation-tiny-ipc50 \
     --store-best-images \
     --syn-data-path ./syn_data/ \
     --init-path ./distillation/init_images/tiny \
     --steps 12 --rho 15e-3 --ipc-start 0 --ipc-end 50 --r-var 11 \
     --dataset tiny 



python validation/validation_tiny.py \
    --epochs 200 --batch-size 64 --ipc 50 \
    --lr 0.2 --momentum 0.9 --weight-decay 1e-4 \
    --lr-scheduler cosineannealinglr \
    --lr-warmup-epochs 5 \
    --lr-warmup-method linear \
    --lr-warmup-decay 0.01 \
    --syn-data-path ./syn_data/distillation-tiny-ipc50/ \
    --model resnet18   



