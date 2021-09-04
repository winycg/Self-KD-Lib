python mainx.py --gpu 0 --data /home/ycg/hhd/dataset --method mixup
python mainx.py --gpu 0 --data /home/ycg/hhd/dataset --method cutmix
python mainx.py --gpu 0 --data /home/ycg/hhd/dataset --method label_smooth
python mainx.py --gpu 0 --data /home/ycg/hhd/dataset --method FocalLoss
python mainx.py --gpu 0 --data /home/ycg/hhd/dataset --method TF_KD_self_reg
python mainx.py --gpu 0 --data /home/ycg/hhd/dataset --method virtual_softmax
python mainx.py --gpu 0 --data /home/ycg/hhd/dataset --method Maximum_entropy
python main.py --gpu 0 --data /home/ycg/hhd/dataset --arch CIFAR_ResNet18_dks --method DKS --warmup-epoch 5
python main.py --gpu 0 --data /home/ycg/hhd/dataset --arch CIFAR_ResNet18_byot --method BYOT --warmup-epoch 5
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method DDGSD
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method CS-KD
python main.py --gpu 0 --data /home/ycg/hhd/dataset --arch CIFAR_ResNet_BiFPN --method FRSKD --warmup-epoch 5


          