python main_mixskd_imagenet.py \
    --data [your imagenet dataset path] \
    --arch MixSKD_ImageNet_ResNet50 \
    --dist-url 'tcp://127.0.0.1:2222' \
    --dist-backend 'nccl' \
    --batch-size 128 \
    --init-lr 0.1 \
    --warmup-epoch 5 \
    --lr-type cosine \
    --epochs 305 \
    --mixup-alpha 0.2 \
    --checkpoint-dir [your checkpoint saving path] \
    --multiprocessing-distributed \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --world-size 1 --rank 0 --manual_seed 0