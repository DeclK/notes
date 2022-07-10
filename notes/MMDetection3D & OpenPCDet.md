---
title: MMDetection3D & OpenPCDet Installation
date: 2021-11-04 11:08:00
tag:
  - MMDetection3D
  - OpenPCDet
categories:
  - 编程
  - OpenMMLab
abbrlink: 93e5b117
---

# MMDetection3D & OpenPCDet

2022.06.24 重新整理如何安装 mmdet3d & OpenPCDet

## Install mmdet3d

### Docker

可以参考 [官方 doc](https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#) 进行下载。其中提供了如何使用 conda 从零下载，但是环境不仅仅包含 conda 环境，还有 GCC, CUDA 等编译环境。使用 docker 就能解决这些环境问题

```shell
# build an image with PyTorch 1.6, CUDA 10.1
docker build -t mmdetection3d docker/

docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection3d/data mmdetection3d
```

由于每次启动 docker 都要传入很多参数，所以在这里记录启动容器的命令，以后直接复制粘贴

```shell
docker run --gpus all --shm-size=8g -it -v /home/chk/data:/shared -v /home/chk/.Xauthority:/root/.Xauthority -e DISPLAY --net=host --name [name] [image_id]
# 其中 -e 和 --net 是为了设置图形化操作，在之后详细介绍
```

clone MMDetection3D 时网络出了问题，clone 失败了，网络真的很重要🤣。再次尝试 `docker build` 后成功

#### Nvidia-Docker 

为了让容器能够使用 GPU，需要安装 Nvidia-docker，过程也比较简单，具体可以参考这篇 [知乎](https://zhuanlan.zhihu.com/p/361934132)

### Conda

如果没有 docker 通过 conda 也挺方便的，前提是对 cuda 配置比较熟悉。下面简单记录安装脚本

1. 安装 pytorch，老生常谈了这个

   ```shell
   pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. 安装 `mmcv-full && mmdet && mmseg`，如果 clone repo 遇到问题的话，可以选择源码安装，源码从 gitee 上下载

   ```shell
   pip install openmim && pip install mmcv-full
   # 也可以下载指定版本
   pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
   
   # 安装时最好看一下项目对于两个版本的要求，mmdet 更新很快，尽量使用和项目相同的版本
   pip install mmdet
   pip install mmsegmentation
   ```

   如果对于 mmcv-full 的版本有要求的话，参考 [mmcv installation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 即可

4. 安装 `mmdet3d` or 其他具体项目

   ```python
   git clone https://github.com/open-mmlab/mmdetection3d.git # or 其他项目 git clone xxx
   cd mmdetection3d	# or cd xxx
   pip install -e .
   ```


## Install OpenPCDet

相比 mmdet3d，OpenPCDet 的安装就更加简单了。也可以使用 docker 安装，这里就不过多赘述，下面整理使用 conda 安装

1. 安装 pytorch

   ```shell
   pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. 安装 [spconv v2.x](https://github.com/traveller59/spconv)，感谢 Yan Yan 大佬，现在安装变得更加简单了

   ```shell
   pip install spconv-cu111
   ```

3. 安装 OpenPCDet

   ```shell
   git clone https://github.com/open-mmlab/OpenPCDet.git
   cd OpenPCDet
   
   pip install -r requirements.txt
   python setup.py develop
   ```

## KITTI 

使用 KITTI 数据集进行实验，下载通过 [GRAVITI](https://gas.graviti.cn/dataset/data-decorators/KITTIObject)

<img src="MMDetection3D & OpenPCDet/image-20211028134043307.png" style="zoom:80%;" />

将数据集放在一个文件夹下，全部解压
