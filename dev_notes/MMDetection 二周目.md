# MMDetection 二周目

现在想要更加熟练细致地掌握 mmdetection 的一些使用方法，所以作此整理

## Installation

现在的安装也没有什么坑了，可以参考 [bilibili](https://www.bilibili.com/video/BV1NL4y1c7ki?p=2)

1. 安装 pytorch。这一步只要查找好自己的 cuda 版本，然后使用镜像就可以轻松安装了

2. 安装 mmcv-full，官方推荐使用 mim 安装

   ```shell
   pip install -U openmim
   mim install mmcv-full
   ```

3. 安装 mmdetection

   ```shell
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   pip install -v -e .
   ```

   国内也可以 clone gitee，速度飞快。官方有 repo

   ```shell
   git clone https://gitee.com/open-mmlab/mmdetection.git
   git clone https://github.com/open-mmlab/mmdetection.git
   ```

一波安装，简直丝滑

## 数据集

