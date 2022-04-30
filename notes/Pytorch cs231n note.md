---
title: Pytorch cn231n note (dev)
tags:
  - Pytorch
  - CS231N
categories:
  - 编程
  - Python
  - Pytorch
abbrlink: 88f138aa
date: 2021-10-08 22:20:38
---
# Pytorch cs231n note

跟着作业过一遍，看看有哪些需要熟悉的 pytorch 知识点，笔记里的英文是从 notebook 中的原文

现在看来2017年的 cs231n 课程视频已经有很多内容需要补充了，尤其是现在 transformer 很火的情况下

原来这个老师 **Justin Johnson** 已经不在 Stanford 了，现在在UMich [个人主页](https://web.eecs.umich.edu/~justincj/) 开了一个计算机视觉的课程 [Deep Learning for Computer Vision](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/)，这个课程比较新，有2020年的教学资源

## How do I learn PyTorch?

One of our former instructors, Justin Johnson, made an excellent [tutorial](https://github.com/jcjohnson/pytorch-examples) for PyTorch. (这个 github 仓库感觉好老了，可以看看 pytorch [官方 tutorial](https://pytorch.org/tutorials/))

You can also find the detailed [API doc](http://pytorch.org/docs/stable/index.html) here. If you have other questions that are not addressed by the API docs, the [PyTorch forum](https://discuss.pytorch.org/) is a much better place to ask than StackOverflow.

## GPU

检查 cuda

```python
# basic check
torch.cuda.is_available()
torch.version.cuda
# cuda device
torch.device('cuda', device_oridinal)
```

## Part 1: Preparation

### torchvision.transform

**torchvision 独立于 torch。torchvision 包由流行的数据集（torchvision.datasets）、模型架构(torchvision.models)和用于计算机视觉的常见图像转换组成 t(torchvision.transforms)**

The torchvision.transforms package provides tools for preprocessing data and for performing data augmentation

```python
import torchvison.transfoms as T
# T.Compose: Composes several transforms together
# T.ToTensor: Converts a PIL Image or numpy.ndarray (H x W x C) in the range
# T.Normalize: Normalize a tensor image with mean and standard deviation
[0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
```

### torchvision.dataset

使用 CIFAR10 数据集

```python
import torchvision.datasets as dset
# args
# root: root directory of dataset where it might exist or will be saved to if download
# train: if true create dataset from traning set
# transform: A function/transform that takes in an PIL image and returns a transformed version
cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True, transform=transform)
cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True, transform=transform)
```

这里需要学习处理数据集常用的几个方法

#### DataLoader & Sampler

> DataLoader: Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.

创建 train, val, test 三个 DataLoader

```python
NUM_TRAIN = 49000

from torch.utils.data import DataLoader
fromt torch.utils.data import sampler

loader_train = DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
loader_val = DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
loader_test = DataLoader(cifar10_test, batch_size=64)
```

如何使用？

### train a model



## Part 2: Barebones by PyTorch

### Autograd engine

> When we create a PyTorch Tensor with `requires_grad=True`, then operations involving that Tensor will not just compute values; they will also build up a computational graph in the background

### torch.nn.functional

各种函数都在这个文件当中，作业里用了 relu, conv2d

#### Bug

Run into errors when using  F.conv2d

```python
import torch.nn.funtional as F
filters = torch.randn(8, 4, 3, 3)
inputs = torch.randn(1, 4, 5, 5)
# the next line produce error: Illegal instruction (core dumped)
x = F.conv2d(inputs, filters, padding=1)
```

通过更换 pytorch 版本得到解决，卸载 `conda uninstall pytorch`

重新下载 `conda install pytorch==1.2.0 torchvision==0.4.0`

nn.Conv2d 和 nn.function.conv2d 有什么区别？尤其是传入的参数看

### Kaiming normalization

这个初始化是什么？待整理

### bug

使用 cuda 调用 GPU 失败 

```shell
# run: torch.randn(shape, device=device, dtype=dtype)
RuntimeError: CUDA error: all CUDA-capable devices are busy or unavailable
```

检查了 `torch.cuda.is_available()` 还有 `torch.version.cuda` 都是好的，网上也没有什么好的方法，在 StackOverflow 上找到了方法：**重启!!!** （可恶，重启确实管用）

### to_device

### with no_grad(), with torch.no_grad() 什么意思？

### 格式化输出 %.2f%% 什么意思

 

## Part 3: Pytorch Module API

上面的 Barebone 要求我们手动跟踪所有参数张量，非常不方便，Pytorch 提供了 `nn.Module` API 能够让我们定义任意的网络结构，同时追踪所有网络中的参数

还有优化策略的选择 `torch.optim` 

使用步骤：

1. `nn.Module` 作为子类继承

   Q: 这个类有什么作用呢？如果不定义这个类，直接组合这个类

2. 在 constructor `__init__()` 中定义好需要的网络层作为类的属性，**注意，一定要运行 **`super().__init__()`

   还可以用 `nn.init` 对网络参数进行不同的初始化

3. 在 `forward()` 方法中定义网络的连接

优化的时候要先清零梯度 `optimizer.zero_grad()`



## Part 4: Pytorch Sequential API

Pytorch 定义了一个更方便的 Module: `nn.Sequential`，来实现顺序连接网络，不需要像 `nn.Module` 一样做三个步骤

`nn.Sequential` 自身也是可以嵌套的



##  Part 5: CIFAR-10 open-ended challenge

### Things you might try:

- **Filter size**: Above we used 5x5; would smaller filters be more efficient?
- **Number of filters**: Above we used 32 filters. Do more or fewer do better?
- **Pooling vs Strided Convolution**: Do you use max pooling or just stride convolutions?
- **Batch normalization**: Try adding spatial batch normalization after convolution layers and vanilla batch normalization after affine layers. Do your networks train faster?
- **Network architecture**: The network above has two layers of trainable parameters. Can you do better with a deep network? Good architectures to try include:
  - [conv-relu-pool]xN -> [affine]xM -> [softmax or SVM]
  - [conv-relu-conv-relu-pool]xN -> [affine]xM -> [softmax or SVM]
  - [batchnorm-relu-conv]xN -> [affine]xM -> [softmax or SVM]
- **Global Average Pooling**: Instead of flattening and then having multiple affine layers, perform convolutions until your image gets small (7x7 or so) and then perform an average pooling operation to get to a 1x1 image picture (1, 1 , Filter#), which is then reshaped into a (Filter#) vector. This is used in [Google's Inception Network](https://arxiv.org/abs/1512.00567) (See Table 1 for their architecture).
- **Regularization**: Add l2 weight regularization, or perhaps use Dropout.

上面的这些选择都非常有意义，我需要能够逐个回答他们