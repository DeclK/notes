---
title: 安装 MMDetection 笔记
tags:
  - MMDetection
  - 安装教程
categories:
  - 编程
  - OpenMMLab
abbrlink: '54605061'
date: 2021-08-17 11:30:23
---

# 安装 MMDetection 笔记

想法：通过这种标准化的算法框架，来进行组合实验

但是目前 SE-SSD 没有支持，可以尝试组合一下

## OpenMMLab

官网：https://openmmlab.com/

github: https://github.com/open-mmlab

子项目官方文档：[MMDetection](https://mmdetection.readthedocs.io/zh_CN/latest/) [MMDetection3D](https://mmdetection3d.readthedocs.io/zh_CN/latest/index.html)

什么是 OpenMMLab？什么是 MMDdetction？

> OpenMMLab 是一个计算机视觉开源算法体系和框架，涉及超过10种研究方向，开放超过100种算法、800种预训练模型
>
> MMDetection is an open source object detection toolbox based on PyTorch. It is a part of the OpenMMLab project.

## 安装环境

我打算先熟悉 MMDetection，因为这个项目在 github 上 star 是相对较多的，然后再进一步学习 mmdetection 3D，下面来安装 MMDetection 环境

首先看看依赖的要求

- Linux 和 macOS （Windows 理论上支持）
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ （如果基于 PyTorch 源码安装，也能够支持 CUDA 9.0）
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

之后下载的依赖版本参考要求来

1. 创建 conda 环境 `conda create -n mmlab python=3.7`

2. 安装 pytorch

   在 [pytorch previous versions](https://pytorch.org/get-started/previous-versions/) 中查找你想要的版本，我选择如下

   ```shell
   conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch
   ```

   此时查看我的电脑上安装的 CUDA 版本，使用命令 `nvcc -V`

   ```shell
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2018 NVIDIA Corporation
   Built on Sat_Aug_25_21:08:01_CDT_2018
   Cuda compilation tools, release 10.0, V10.0.130
   ```
   
   该版本为 10.0 和 cudatoolkit 的 10.1 版本并不一样，那么 pytorch 究竟是使用哪个版本呢？答案是使用 10.0 的版本，需要更改一下环境 CUDA_HOME 环境变量。更多的信息，请阅读补充笔记


## 安装 MMDetection

官方推荐使用 [MIM](https://github.com/open-mmlab/mim) 安装 MMDetection

> MIM provides a unified interface for launching and installing OpenMMLab projects and their extensions, and managing the OpenMMLab model zoo.

```shell
pip install openmim
mim install mmdet
```

MIM 自动下载了 mmdet 及其对应依赖

```shell
# mim list
Package    Version    Source
---------  ---------  -----------------------------------------
mmcv-full  1.3.10     https://github.com/open-mmlab/mmcv
mmdet      2.15.0     https://github.com/open-mmlab/mmdetection
```

## 验证

将 MMDetection 仓库克隆到本地，运行如下代码

```python
from mmdet.apis import inference_detector, init_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
img = 'demo/demo.jpg'
result = inference_detector(model, img)
# 绘制结果图
model.show_result(img, result, wait_time=1)
```

## 补充：Pytorch & CUDA

### CUDA & CUDA toolkit & pytorch install with conda

conda 在安装 Pytorch 等会使用到 CUDA 的框架时，会自动为用户安装 cudatoolkit，其主要包含应用程序在使用 CUDA 相关的功能时所依赖的动态链接库。在安装了 cudatoolkit 后，只要系统上存在与当前的 cudatoolkit 所兼容的 Nvidia 驱动，则已经编译好的 CUDA 相关的程序就可以直接运行，而不需要安装完整的 Nvidia 官方提供的 CUDA Toolkit 

经过自己的实验，环境中的 CUDA_HOME 变量是清空的，而且也没有设置 /usr/local/cuda 软链接，但是 pytorch 依然能够调用 GPU，说明只需要你的 Nvidia Driver 和 conda install cudatoolkit 兼容，那么就不需要从 Nvidia 官网中下载完整的 CUDA Toolkit

综上 Pytorch 查找 cuda 路径顺序为：

1. 先查找环境变量中是否存在 CUDA_PATH / CUDA_HOME 如有则使用该路径指向的 CUDA 版本。其中 CUDA_PATH 为 windows 环境变量，CUDA_HOME 为 Linux 环境变量

2. 若找不到 CUDA_HOME，则查看 usr/local/cuda 是否存在，若存在则使用该路径指向的 CUDA 版本

3. 若两个路径都不存在，则使用 conda 下载的 cudatoolkit，也即下载 pytorch 时自动下载的 cudatoolkit

推荐 [参考链接](https://www.cnblogs.com/yhjoker/p/10972795.html) 进行深入了解
