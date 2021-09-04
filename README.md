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
| DDGSD | - |  | 
| DKS | - |  | 
| BYOT | - |  | 
| Tf-KD-reg | - |  | 
| CS-KD | - |  | 
|  FRSKD | - |  |


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

