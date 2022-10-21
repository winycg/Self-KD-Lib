'''Self-KD methods'''
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method TF_KD_self_reg
python main.py --gpu 0 --data /home/ycg/hhd/dataset --arch CIFAR_ResNet18_dks --method DKS
python main.py --gpu 0 --data /home/ycg/hhd/dataset --arch CIFAR_ResNet18_byot --method BYOT
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method DDGSD
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method SAD
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method CS-KD
python main.py --gpu 0 --data /home/ycg/hhd/dataset --arch CIFAR_ResNet18_BiFPN --method FRSKD
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method PSKD
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method BAKE
python main_mixskd_cifar.py --gpu 0 --data /home/ycg/hhd/dataset --arch MixSKD_CIFAR_ResNet18

'''Data augmentation and regularization methods'''
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method mixup
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method cutmix
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method label_smooth
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method FocalLoss
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method virtual_softmax
python main.py --gpu 0 --data /home/ycg/hhd/dataset --method Maximum_entropy
python main.py --gpu 0 --data /home/ycg/hhd/dataset --data-aug cutout
python main.py --gpu 0 --data /home/ycg/hhd/dataset --data-aug random_erase
python main.py --gpu 0 --data /home/ycg/hhd/dataset --data-aug auto_aug
python main.py --gpu 0 --data /home/ycg/hhd/dataset --data-aug randaug
python main.py --gpu 0 --data /home/ycg/hhd/dataset --data-aug augmix
python main.py --gpu 0 --data /home/ycg/hhd/dataset --data-aug trivalaug
python main.py --arch manifold_mixup_CIFAR_ResNet18 --gpu 0 --data /home/ycg/hhd/dataset --method manifold_mixup


          