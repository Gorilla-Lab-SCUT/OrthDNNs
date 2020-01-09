# OrthDNNs-CIFAR
By Kui Jia, Shuai Li, Yuxin Wen, Tongliang Liu, and Dacheng Tao.

## Introdution
This repository contains the implementation used for the results in our paper (https://arxiv.org/abs/1905.05929). And this is a Pytorch version of [Singlur Value Bounding method](https://github.com/kui-jia/svb).

## Requirements
- A computer running on Linux
- NVIDIA GPU and NCCL
- Python version 2.7
- Pytorch 1.1

## Usage
Use `python main.py` to train a new model. Here is some example settings:
### For vanilla training:
```
CUDA_VISIBLE_DEVICES=0 python main.py  --dataset CIFAR10 --dataset_dir Dataset/CIFAR10 --nEpoch 160 -nGPU 1 -a ConvNet_WOBN -b 128 --lr_decay_method exp -lr 0.1 --save Exps/ConvNet20_BNSameVar_CIFAR10_Batch128_160E_lr01_Ori
```
### For Stiefel manifold optimization:
```
CUDA_VISIBLE_DEVICES=0 python main.py  --dataset CIFAR10 --dataset_dir Dataset/CIFAR10 --nEpoch 160 -nGPU 1 -a ConvNet_WOBN -b 128 --lr_decay_method exp -lr 0.1  --save Exps/ConvNet20_BNSameVar_CIFAR10_Batch128_160E_lr01_Stiefel -stiefel
```
### For Frobenius norm Restricted optimization:
```
CUDA_VISIBLE_DEVICES=0 python main.py  --dataset CIFAR10 --dataset_dir Dataset/CIFAR10 --nEpoch 160 -nGPU 1 -a ConvNet_WOBN -b 128 --lr_decay_method exp -lr 0.1 --save Exps/ConvNet20_BNSameVar_CIFAR10_Batch128_160E_lr01_Soft --is_soft_regu --soft_lambda 0.1
```
### For Spectral Restricted Isometry Property optimization:
```
CUDA_VISIBLE_DEVICES=0 python main.py  --dataset CIFAR10 --dataset_dir Dataset/CIFAR10 --nEpoch 160 -nGPU 1 -a ConvNet_WOBN -b 128 --lr_decay_method exp -lr 0.1 --save Exps/ConvNet20_BNSameVar_CIFAR10_Batch128_160E_lr01_SRIP --is_SRIP --soft_lambda 0.1
```
### For Singular Value Bounding optimization:
```
CUDA_VISIBLE_DEVICES=0 python main.py  --dataset CIFAR10 --dataset_dir Dataset/CIFAR10 --nEpoch 160 -nGPU 1 -a ConvNet_WOBN -b 128 --lr_decay_method exp -lr 0.1 --save Exps/ConvNet20_BNSameVar_CIFAR10_Batch128_160E_lr01_SVB -svb --svb_factor 0.05
```

## Citation
If you use this method or this code in your paper, then please cite it:
```
@article{Jia_2019,
   title={Orthogonal Deep Neural Networks},
   ISSN={1939-3539},
   url={http://dx.doi.org/10.1109/TPAMI.2019.2948352},
   DOI={10.1109/tpami.2019.2948352},
   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Jia, Kui and Li, Shuai and Wen, Yuxin and Liu, Tongliang and Tao, Dacheng},
   year={2019},
   pages={1–1}
}
```
