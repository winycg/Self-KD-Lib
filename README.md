This project provides the implementations of some data augmentation methods, regularization methods, online Knowledge distillation and Self-Knowledge distillation methods.

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

#### Comparison of Self-KD methods
| Method | Venue | Accuracy(%) |
|:---------------:|:-----------------:|:-----------------:|
| Cross-entropy | - | 77.79 |
| DDGSD |  AAAI-2019 | 81.73 |
| DKS |  CVPR-2019 | 80.75 |
| SAD |  ICCV-2019 |  |
| BYOT |  ICCV-2019 |  |
| Tf-KD-reg | CVPR-2020 | 79.84 | 
| CS-KD |  CVPR-2020 | 79.99 |
|  FRSKD |  CVPR-2021 |    |

##### Reference
DDGSD: Data-Distortion Guided Self-Distillation for Deep Neural Networks. AAAI-2019

DKS: Deeply-supervised Knowledge Synergy. CVPR-2019.

SAD: Learning Lightweight Lane Detection CNNs by Self Attention Distillation. ICCV-2019.

BYOT: Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation. ICCV-2019

Tf-KD-reg: Revisiting Knowledge Distillation via Label Smoothing Regularization. CVPR-2020.

CS-KD: Regularizing Class-wise Predictions via Self-knowledge Distillation. CVPR-2020.

FRSKD: Refine Myself by Teaching Myself: Feature Refinement via Self-Knowledge Distillation. CVPR-2021.

#### Comparison of advanced regularization methods Self-KD methods

| Method | Accuracy | Venue |Paper Link |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|
| Cross-entropy | - | 77.79 | 
| Label Smoothing | - | 80.33 | 
| Virtual Softmax | - | 79.68 | 
| Focal Loss | - | 79.31 | 
| Maximum Entropy | - | 78.11 | 
| Cutout | - |  |
| Random Erase | - |  |
| Mixup | - | 81.39 |
| CutMix | - | 82.47 |

Label Smoothing: Rethinking the inception architecture for computer vision. CVPR-2016.

Virtual Softmax:  Virtual class enhanced discriminative embedding learning. NeurIPS-2018.

Focal Loss: Focal loss for dense object detection. iccv-2017. 

Maximum Entropy: Regularizing neural networks by penalizing confident output distributions. ICLR Workshops 2017.

Cutout: Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552

Random Erase: Random erasing data augmentation. AAAI-2020.

Mixup: mixup: Beyond empirical risk minimization. ICLR-2018.

CutMix: Cutmix: Regularization strategy to train strong classifiers with localizable features. ICCV-2019.


