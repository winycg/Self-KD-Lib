CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 train_baseline.py \
    --model deeplabv3 \
    --backbone resnet50_original \
    --dataset ade20k \
    --data [your dataset path]/ade20k/ \
    --batch-size 16 \
    --workers 16 \
    --lr 0.02 \
    --crop-size 512 512 \
    --max-iterations 40000 \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --val-per-iters 40000 \
    --pretrained-base [your pretrained-backbone path]/resnet50_mixskd.pth


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 eval.py \
    --model deeplabv3 \
    --backbone resnet50_original \
    --dataset ade20k \
    --data [your dataset path]/ade20k/ \
    --save-dir [your directory path to store checkpoint files] \
    --pretrained [your pretrained model path]

