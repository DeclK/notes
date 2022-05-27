# TODO BOX

## Experiment

1. 整理工具库 **tensorboard logging yaml**

2. 整理 OpenPCDet 工具库

3. > >CG-SSD 目前已经跑起来了，最后看看效果，**现在基本超越了 baseline，行人还是差：**
   > >
   > >1. 使用了多个 channel 对每个 corner offset 单独预测，行人的结果更差了...但其他结果还是好的，我怀疑是特征融合出了问题，明天再问问作者 plug-in 的具体细节，CNN 用了几个？
   > >3. 最后使用 centerpoint + CGAM 有无 SOTA
   
4. > > 已经基本跑通 consistency loss，记录如下：
   > >
   > > 1. 使用 centerpoint 作为 student 无法提升行人点，但是 second 可以提升（是否说明输出已经很一致了？）
   > > 2. 尝试了两种匹配策略（iou & dist based），之后可以哪种好用哪个，从定性来看都能提升
   > > 5. **new idea**：给 teacher 加入 augmentation！！
   
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

