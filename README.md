This project provides the implementations of some Self-Knowledge distillation, data augmentation methods and  regularization methods.

## Installation

### Requirements

Ubuntu 18.04 LTS

Python 3.8 ([Anaconda](https://www.anaconda.com/) is recommended)

CUDA 11.1

PyTorch 1.6.0

NCCL for CUDA 11.1

## Perform experiments on CIFAR-100 dataset
### Dataset
CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

unzip to the `./data` folder

The commands for running various methods can be found in `main.sh` 

#### Comparison of Self-KD methods on ResNet-50
| Method | Venue | Accuracy(%) |
|:---------------:|:-----------------:|:-----------------:|
| Cross-entropy | - | 77.79 |
| DDGSD [1] |  AAAI-2019 | 81.73 |
| DKS [2]|  CVPR-2019 | 80.75 |
| SAD [3] |  ICCV-2019 | 78.33 |
| BYOT [4] |  ICCV-2019 | 79.76 |
| Tf-KD-reg [5] | CVPR-2020 | 79.84 | 
| CS-KD [6]|  CVPR-2020 | 79.99 |
|  FRSKD [7]|  CVPR-2021 |  80.51  |

##### Reference
[1] DDGSD: Data-Distortion Guided Self-Distillation for Deep Neural Networks. AAAI-2019

[2] DKS: Deeply-supervised Knowledge Synergy. CVPR-2019.

[3] SAD: Learning Lightweight Lane Detection CNNs by Self Attention Distillation. ICCV-2019.

[4] BYOT: Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation. ICCV-2019

[5] Tf-KD-reg: Revisiting Knowledge Distillation via Label Smoothing Regularization. CVPR-2020.

[6] CS-KD: Regularizing Class-wise Predictions via Self-knowledge Distillation. CVPR-2020.

[7] FRSKD: Refine Myself by Teaching Myself: Feature Refinement via Self-Knowledge Distillation. CVPR-2021.

#### Comparison of advanced regularization methods Self-KD methods on ResNet-50

| Method | Venue | Accuracy |
|:---------------:|:-----------------:|:-----------------:|
| Cross-entropy | - | 77.79 | 
| Label Smoothing [1] | CVPR-2016 | 80.33 | 
| Virtual Softmax [2] | NeurIPS-2018 | 79.68 | 
| Focal Loss [3]| ICCV-2017 | 79.31 | 
| Maximum Entropy [4] | ICLR Workshops 2017 | 78.11 | 
| Cutout [5]| ArXiv.2017 | 80.42 |
| Random Erase [6]| AAAI-2020 | 80.64 |
| Mixup [7]| ICLR-2018 | 81.39 |
| CutMix [8]| ICCV-2019 | 82.47 |
| AutoAugment [9]| CVPR-2019 | 81.41 |

[1] Label Smoothing: Rethinking the inception architecture for computer vision. CVPR-2016.

[2] Virtual Softmax:  Virtual class enhanced discriminative embedding learning. NeurIPS-2018.

[3] Focal Loss: Focal loss for dense object detection. ICCV-2017. 

[4] Maximum Entropy: Regularizing neural networks by penalizing confident output distributions. ICLR Workshops 2017.

[5] Cutout: Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552

[6] Random Erase: Random erasing data augmentation. AAAI-2020.

[7] Mixup: mixup: Beyond empirical risk minimization. ICLR-2018.

[8] CutMix: Cutmix: Regularization strategy to train strong classifiers with localizable features. ICCV-2019.

[9] Autoaugment: Autoaugment: Learning augmentation strategies from data. CVPR-2019.


