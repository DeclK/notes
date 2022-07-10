# TODO BOX

## Current

1. **秋招岗位投递，Leetcode 算法整理，NMS 重写**
2. **语义分割实践学习**
3. **GFL/Faster R-CNN 代码，总结 mmdetection 运行逻辑**
4. **论文：Loss & Experiment writing**
5. 重构 SSL 代码，通过更改 `raw` 的 info 为 labeled info 就可以简单使用。实现 Soft Teacher cls branch，利用 GFL 实现更好的 location consistency loss，bring background loss into attention
6. CG-SSD 试验：
   1. CGAM max pooling corner predictions
   1. IoU + CGAM + GFL
7. 整理工具库 tensorboard logging yaml

## Deep Learning Theory

1. 论文整理：
   1. RPVNet, **Soft teacher(end to end semi-supervise), Dense Teacher**, DKD, **LargeKernel3D & FocalConv**
   2. lidar + rgb: **BEVDet** & BEVFusion
2. 图像目标检测前沿：VitDet, ConvNext（local and long range modeling, focus is not on transformer!）CAE, DETR, GFL, KLD, Localization Distillation for Dense Object Detection
3. 胡思乱想
   1. 能不能将自监督和半监督结合起来？（MAE + MoCo）
   2. 工程上的实现：IA-SSD 中的采样策略可能可以用于加速。将 下采样 & 点生成结合起来
4. 机器学习理论：决策树，逻辑回归，PCA，朴素贝叶斯，凸优化基础，kernel method，图论算法（裸的网络流），卡尔曼滤波，四元数与旋转，信息论基础整理，强化学习基础
5. 优化器 AdamW, cosine learning schedule 整理
6. 百面机器学习
7. [完整矩阵求导体系](https://zhuanlan.zhihu.com/p/24709748)

## Basic theory

1. 线性代数 & 概率论与数理统计 & 凸优化 & 运筹学
1. 计算机网络
2. 操作系统
3. 数据库

## Interest

1. 网球 tracking 优化，姿态识别
2. OpenCV projects，opencv-基础整理
3. 经济学原理、固定收益证券、量化交易
4. Vscode 搜索原理，建立方便的可搜索数据库
4. 升级 hexo mattery mermaid 功能
