This project provides the implementations of some data augmentation methods, regularization methods, online Knowledge distillation and Self-Knowledge distillation methods.

## Installation

### Requirements

Ubuntu 18.04 LTS

Python 3.8 ([Anaconda](https://www.anaconda.com/) is recommended)

CUDA 11.1

PyTorch 1.12 + torchvision 0.13

## Perform experiments on CIFAR-100 dataset
### Dataset
CIFAR-100 : [download](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

unzip to the `./data` folder

The commands for running various methods can be found in [main_cifar.sh](https://github.com/winycg/Self-KD-Lib/blob/main/scripts/main_cifar.sh)

<table>
	<tr>
	    <th colspan="4">Top-1 accuracy(%) of Self-KD and Data Augmentation (DA) methods on ResNet-18</th>
	</tr >
	<tr>
	    <td >Type</td>
	    <td>Method</td>
	    <td>Venue</td>  
      <td>Accuracy(%)</td>  
	</tr >
	<tr >
    <td>Baseline</td>
	    <td>Cross-entropy</td>
	    <td>-</td>
	    <td>76.24</td>
	</tr>
  <tr >
  <td rowspan="10">Self-KD</td>
	    <td>DDGSD [1]</td>
	    <td>AAAI-2019</td>
	    <td>76.61</td>
	</tr>
  <tr >
	    <td>DKS [2]</td>
	    <td>CVPR-2019</td>
	    <td>78.64</td>
	</tr>
    <tr >
	    <td>SAD [3]</td>
	    <td>ICCV-2019</td>
	    <td>76.40</td>
	</tr>
  	</tr>
    <tr >
	    <td>BYOT [4]</td>
	    <td>ICCV-2019</td>
	    <td>77.88</td>
	</tr>
      <tr >
	    <td>Tf-KD-reg [5]</td>
	    <td>CVPR-2020</td>
	    <td>76.61</td>
	</tr>
  </tr>
      <tr >
	    <td>CS-KD [6]</td>
	    <td>CVPR-2020</td>
	    <td>78.66</td>
	</tr>
    </tr>
    <tr >
	    <td>FRSKD [7]</td>
	    <td>CVPR-2021</td>
	    <td>77.71</td>
	</tr>
	<tr >
	    <td>PS-KD [8]</td>
	    <td>ICCV-2021</td>
	    <td>79.31</td>
	</tr>
	<tr >
	    <td>BAKE [9]</td>
	    <td>arXiv:2104.13298</td>
	    <td>76.93</td>
	</tr>
    <tr >
	    <td>MixSKD [10]</td>
	    <td>ECCV-2022</td>
	    <td>80.32</td>
	</tr>
    <tr >
  <td rowspan="12">DA</td>
	    <td>Label Smoothing [1]</td>
	    <td>CVPR-2016</td>
	    <td>78.72</td>
	</tr>
  <tr >
	    <td>Virtual Softmax [2]</td>
	    <td>NeurIPS-2018</td>
	    <td>78.54</td>
	</tr>
    <tr >
	    <td>Focal Loss [3]</td>
	    <td>ICCV-2017</td>
	    <td>76.19</td>
	</tr>
  <tr >
	    <td>Maximum Entropy [4]</td>
	    <td>ICLR Workshops 2017</td>
	    <td>76.50</td>
	</tr>
    <tr >
	    <td>Cutout [5]</td>
	    <td>arXiv:1708.04552</td>
	    <td>76.66</td>
	</tr>
	<tr >
	    <td>Random Erase [6]</td>
	    <td>AAAI-2020</td>
	    <td>76.75</td>
	</tr>
	<tr >
	    <td>Mixup [7]</td>
	    <td>ICLR-2018</td>
	    <td>78.68</td>
	</tr>
	<tr >
	    <td>CutMix [8]</td>
	    <td>ICCV-2019</td>
	    <td>80.17</td>
	</tr>
	<tr >
	    <td>AutoAugment [9]</td>
	    <td>CVPR-2019</td>
	    <td>77.97</td>
	</tr>
	<tr >
	    <td>RandAugment [10]</td>
	    <td>CVPR Workshops-2020</td>
	    <td>76.86</td>
	</tr>
	<tr >
	    <td>AugMix [11]</td>
	    <td>arXiv:1912.02781</td>
	    <td>76.22</td>
	</tr>
	<tr >
	    <td>TrivalAugment [12]</td>
	    <td>ICCV-2021</td>
	    <td>76.03</td>
	</tr>
	
</table>

Some implementations are referred by the official code. Thanks the papers' authors for their released code.


## Perform experiments on ImageNet dataset
| MixSKD | Top-1 Accuracy | Script | Pretrained Model|
| -- | -- | -- |-- |
|  ResNet-50 | 78.76|[sh](https://github.com/winycg/Self-KD-Lib/blob/main/scripts/main_mixskd_imagenet.sh)|[Baidu Cloud](https://pan.baidu.com/s/1X_jJoP7KedjcO2BV3s5FZQ?pwd=1rdr ) |

## Perform experiments of downstream object detection on COCO
Our implementation of object detection is based on MMDetection. Please refer the detailed guideline at  [https://github.com/winycg/detection](https://github.com/winycg/detection ).
| Framework | mAP | Log | Pretrained Model|
| -- | -- | -- |-- |
|  Cascade-Res50 | 41.6 |[Baidu Cloud](https://pan.baidu.com/s/1BhlI_uQZSO6KiBo7k-fWEA?pwd=kkxe ) |[Baidu Cloud](https://pan.baidu.com/s/1eT_rFkxztuTi8iNMGmvW2g?pwd=oc22 ) |

## Perform experiments on downstream semantic segmentation
The training script are based on our previous released segmentation codebase: [https://github.com/winycg/CIRKD](https://github.com/winycg/CIRKD )
| Dataset | mIoU | Script | Log | Pretrained Model|
| -- | -- | -- |-- |-- |
|  ADE20K | 42.37|[sh](https://github.com/winycg/Self-KD-Lib/blob/main/scripts/segmentation/ade20k.sh)|[Baidu Cloud](https://pan.baidu.com/s/1AisXtqqcNXjsv5K0yqr5bQ?pwd=yffl ) |[Baidu Cloud](https://pan.baidu.com/s/15wu2PP8AoOBJAblPvEIMHw?pwd=sbzp ) |
|  COCO-Stuff-164K | 37.12|[sh](https://github.com/winycg/Self-KD-Lib/blob/main/scripts/segmentation/coco_stuff_164K.sh)|[Baidu Cloud](https://pan.baidu.com/s/1QYHcd8rHwTRJn3b_4Mdgkw?pwd=vgh0 ) |[Baidu Cloud](https://pan.baidu.com/s/1ViHjasWkZ1dtrN3hDW6psw?pwd=rwbe ) |
|  Pascal VOC | 78.78|[sh](https://github.com/winycg/Self-KD-Lib/blob/main/scripts/segmentation/pascal_voc.sh)|[Baidu Cloud](https://pan.baidu.com/s/1yr6tZMImtBys4ASpJH4-mQ?pwd=l77q ) |[Baidu Cloud](https://pan.baidu.com/s/1cIEcvXQOGugfOWUU3RmZnQ?pwd=2479 ) |

If you find this repository useful, please consider citing the following paper:

```
@inproceedings{yang2022mixskd,
  title={MixSKD: Self-Knowledge Distillation from Mixup for Image Recognition},
  author={Yang, Chuanguang and An, Zhulin and Zhou, Helong and  Cai, Linhang and Zhi, Xiang and Wu, Jiwen and Xu, Yongjun and Zhang, Qian},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```


