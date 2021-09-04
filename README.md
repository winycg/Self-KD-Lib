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
| Method | Accuracy | Venue |Paper Link |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|
| Cross-entropy | - |  | 
| DDGSD | - | AAAI-2019 |  | 
| DKS | - | CVPR-2019 |  |
| BYOT | - | ICCV-2019 | [Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Be_Your_Own_Teacher_Improve_the_Performance_of_Convolutional_Neural_ICCV_2019_paper.pdf) |
| Tf-KD-reg |- | CVPR-2020 |  | 
| CS-KD | - | CVPR-2020 | 
|  FRSKD | - | CVPR-2021 |


#### Comparison of advanced regularization methods Self-KD methods

| Method | Accuracy | Venue |Paper Link |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|
| Cross-entropy | - |  | 
| Label Smoothing | - |  | 
| Virtual Softmax | - |  | 
| Focal Loss | - |  | 
| Maximum Entropy | - |  | 
| Cutout | - |  |
| Random Erase | - |  |
| Mixup | - |  |
| CutMix | - |  |

