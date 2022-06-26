# TODO BOX

## Experiment

1. 整理工具库 **tensorboard logging yaml**
2. CG-SSD 试验：

   1. **CGAM + IoU** rectify
   1. **CGAM speed up**
3. Mean Teacher 试验：

   1. 利用 GFL 实现更好的 location consistency loss，bring background loss into attention
4. **PillarNet iou regression rebuild**
5. 阅读 **GFL/FCOS/Faster R-CNN 代码，总结 mmdetection 运行逻辑**
6. **如何把握学习方向？？理论和实现需要相互指导**
8. **开始写论文：anchor-based + res + mean teacher**

## Deep learning

1. 论文整理：
   1. RangeDet, Pyramid R-CNN, VoTr, PillarNet, RPVNet, CFGNet, **Soft teacher(end to end semi-supervise)**, **DKD**
   2. lidar + rgb: **BEVFusion**
2. 图像目标检测前沿：VitDet, ConvNext（local and long range modeling, focus is not on transformer!），CAE, DETR, GFL, **Localization Distillation for Dense Object Detection**
3. **胡思乱想**
   1. mean teacher 半监督以及 MoCo 自监督的本质是什么（似乎可以从防止过拟合的角度看）？为什么有效？似乎知识蒸馏只是一个华丽的表象
   2. 能不能将自监督和半监督结合起来？（MAE + MoCo）
   3. **工程上的实现：IA-SSD 中的采样策略可能可以用于加速。将 下采样 & 点生成结合起来，也可以将下采样和 SST 结合起来！**
4. **机器学习理论：决策树，逻辑回归，PCA，朴素贝叶斯**，凸优化基础，kernel method，图论算法（最大流之类），卡尔曼滤波，四元数与旋转，信息论基础整理，强化学习基础
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
3. 经济学原理、固定收益证券
4. Vscode 搜索原理，建立方便的可搜索数据库
4. 升级 hexo mattery mermaid 功能
