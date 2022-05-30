# TODO BOX

## Experiment

1. 整理工具库 **tensorboard logging yaml**

2. 整理 OpenPCDet 工具库

3. > >CG-SSD 试验：
   > >
   > >1. 经过不断排查，database 还是除了问题：由于 data augmentation 改变了 gt boxes 的位置，所以 corner index 会随之改变。修复了这个 bug 过后，再加入 multi channel 依然没有提升，确实用单个头去做这么多的任务还是很困难的，最后尝试一波 multi head 看看结果
   > >3. 使用 centerpoint + CGAM 有无 SOTA
   > >3. **考虑使用 CGAM 的结果，进一步提升检测表现，就像 GFLv2.0 一样思考**
   
4. > > Mean Teacher 试验：
   > >
   > > 1. 使用 centerpoint 作为 student 无法提升行人点，但是 second 可以提升（是否说明输出已经很一致了？）
   > > 2. 尝试了两种匹配策略（iou & dist based），之后可以哪种好用哪个，从定性来看都能提升
   > > 5. **给 teacher 加入 augmentation！！**这个 idea 目前来看是有效的，在更长 epoch，更大 lr 下能提升了 0.8 个点。我再尝试对 regression 加入 consistency loss，不知道会不会有提升。同时我也认为 ONCE 论文里提到的 regression consistency loss 没用是不太准确的，因为论文里完全没有对 teacher 使用任何的 augmentation，这样 noise 的效果就变差了
   
7. **阅读 SST & Lidar-RCNN 代码**

8. 尝试增加 IoU prediction head，据说效果不错

8. 阅读 faster rcnn 代码，便于之后实现自己的兴趣小项目

## Deep learning

1. 论文整理：
   1. **SST，Lidar R-CNN** （已阅读，待整理）
   2. FCOS 也是一篇经典的目标检测，很多地方都在用
   3. RangeDet，Pyramid R-CNN，VoTr，PillarNet
   4. lidar + rgb: SFD，代码未开源
2. 图像目标检测前沿：ConvNext，CAE，DETR，**GFL**，**Localization Distillation for Dense Object Detection**
3. **胡思乱想**
   1. mean teacher 半监督以及 MoCo 自监督的本质是什么（似乎可以从防止过拟合的角度看）？为什么有效？似乎知识蒸馏只是一个华丽的表面现象
   2. 能不能将自监督和半监督结合起来？（MAE + MoCo）
   3. **工程上的实现：将 mask & 点生成结合起来**
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
