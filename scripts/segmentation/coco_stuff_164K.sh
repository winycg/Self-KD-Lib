CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --master_addr 127.5.0.9 --master_port 2601 --nproc_per_node=8 \
    train_baseline.py \
    --model deeplabv3 \
    --backbone resnet50_original \
    --dataset coco_stuff_164k \
    --data [your dataset path]/coco_stuff_164k/ \
    --batch-size 16 \
    --workers 8 \
    --crop-size 512 512 \
    --lr 0.02 \
    --max-iterations 40000 \
    --save-dir [your directory path to store checkpoint files] \
    --log-dir [your directory path to store log files] \
    --val-per-iters 40000 \
    --pretrained-base [your pretrained-backbone path]/resnet50_mixskd.pth


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --master_addr 127.5.0.9 --master_port 2601 --nproc_per_node=8 \
    eval.py \
    --model deeplabv3 \
    --backbone resnet50_original \
    --dataset coco_stuff_164k \
    --data [your dataset path]/coco_stuff_164k/ \
    --save-dir [your directory path to store checkpoint files] \
    --pretrained [your pretrained model path]
