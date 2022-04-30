# TODO BOX

## Experiment

1. 整理工具库 ~~pathlib~~ tensorboard tqdm pdb logging yaml
2. 整理 OpenPCDet ~~loss~~ & 工具库
3. **Pytorch 分布式训练**
4. 辅助任务：~~点云对 bbox 的偏移量~~，~~使用多余的 channel 进行预测~~。使用两阶段提供 soft target 给单阶段。~~PointPillars 是不是能够也使用辅助网络，并且达到非常快的速度~~
5. **一些可以了解的研究方向**
   1. **stable variance 与知识蒸馏**
   2. 稀疏信息与完整信息的转换（MAE 自监督学习/对比学习）
   3. **工程上的实现：将 mask & 点生成结合起来**
6. 完整实现 SE-SSD 的训练
7. 阅读 CenterPoint 的代码，需要其中处理 center heatmap & regression 的代码理解。将完全放弃 SA-SSD 中的辅助网络结构，无法复现里面的效果，但是 CIA-SSD 里面的 Attention 结构是确实有效果的，我能够在 ONCE 数据集上体现出来。接下来尝试复现一个类似 CG-SSD 中的 corner guided 辅助模块，应该会有不错的结果
8. **Transformer!!!!!!!!!**

## Deep learning

1. 百面机器学习， ~~矩阵求导整理案例~~，~~transformer & bert~~ ~~pytorch & numpy 常用方法 & NMS 实现方法~~，[完整矩阵求导体系](https://zhuanlan.zhihu.com/p/24709748)，信息论基础整理
2. 论文整理：
   1. ~~**SPG BtC Lidar R-CNN**~~
   2. **CG-SSD** （已阅读，待整理）
   3. **SST, RangeDet**
   4. ~~**Dynamic voxel**~~
   5. ICCV 2021 Pyramid CNN, VoTr
   3. ~~Not all points are equal (ultra speed)~~
3. 图像目标检测前沿：~~ViT Moco MAE~~ ~~Swin~~
4. **白板推导机器学习：~~SVM~~，决策树，逻辑回归，PCA，朴素贝叶斯**，凸优化基础，kernel method，图论算法（最大流之类的），卡尔曼滤波，四元数与旋转
5. 强化学习基础

## Autowise

构建物体的分类网络

1. 构建 target & target assinger
2. 获得场景下的 object 表示
3. 使用简单的线性网络进行分类
4. 训练网络，获得结果

## Basic theory

1. 计算机网络
2. 操作系统
3. 数据库

## Interest

1. 网球 tracking 优化，姿态识别
2. OpenCV projects，opencv-基础整理
3. 经济学原理、固定收益证券
3. Vscode 搜索原理，建立自己的数据库
3. misc

