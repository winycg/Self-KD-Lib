'''Self-KD methods'''
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method TF_KD_self_reg
python main.py --gpu 0 --data /home/ycg/hhd/dataset --arch CIFAR_ResNet50_dks --method DKS --warmup-epoch 5
python main.py --gpu 0 --data /home/ycg/hhd/dataset --arch CIFAR_ResNet50_byot --method BYOT --warmup-epoch 5
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method DDGSD
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method SAD
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method CS-KD
python main.py --gpu 0 --data /home/ycg/hhd/dataset --arch CIFAR_ResNet50_BiFPN --method FRSKD --warmup-epoch 5

'''Data augmentation and regularization methods'''
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method mixup
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method cutmix
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method label_smooth
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method FocalLoss
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method virtual_softmax
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method Maximum_entropy
python main.py --gpu 0 --data /home/ycg/hhd/dataset --data-aug cutout --warmup-epoch 5
python main.py --gpu 0 --data /home/ycg/hhd/dataset --data-aug random_erase --warmup-epoch 5
python main.py --gpu 0 --data /home/ycg/hhd/dataset --data-aug auto_aug --warmup-epoch 5
          