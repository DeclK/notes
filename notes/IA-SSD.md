---
title: IA-SSD
tags:
  - Point Cloud
categories:
  - papers
mathjax: true
date: 2022-4-10 22:20:38
---

# IA-SSD

---

Zhang, Yifan, Qingyong Hu, Guoquan Xu, Yanxin Ma, Jianwei Wan, and Yulan Guo. “Not All Points Are Equal: Learning Highly Efficient Point-Based Detectors for 3D LiDAR Point Clouds.” *ArXiv:2203.11139 [Cs]*, March 21, 2022. http://arxiv.org/abs/2203.11139.

---

好久没有看新的文章了...来看看新鲜出炉的 CVPR 2022 的论文吧！

## Introduction

依然是解决老问题：体素化的需要解决信息损失 ，直接对点进行特征提取速度又不太行。这篇论文是 point-based SSD，想要使用 point 的精细特征，又不想要大量的计算，如果能够有不错的方法对点云进行筛选就好了

之前的 point-based 的采样方法都是 furthest points sampling (FPS) 基于距离的最远采样法，这样甚至会丢失一些重要的前景点。论文直接使用了一个网络来预测某个点是否为前景点，并基于这些预测结果进行下采样，这样就保证了重要的点不被丢弃

## IA-SSD

### Class-aware sampling

简单来说论文使用了两个 MLP，将每个点的特征用于预测每个点的类别。损失函数为交叉熵损失
$$
L_{c l s-a w a r e}=-\sum_{c=1}^{C}\left(s_{i} \log \left(\hat{s_{i}}\right)+\left(1-s_{i}\right) \log \left(1-{\hat{s_{i}}}\right)\right)
$$
C 代表的是数据集的所有标签类别，$s_i$ 为标签的 one-hot 向量。在推理的时候，取前 k 个最高得分的点

### Centroid-aware sampling

这是论文进一步提出的采样方法，让损失函数更聚焦于接近物体中心的点
$$
L_{c t r-a w a r e}=-\sum_{c=1}^{C}\left(M a s k_{i} \cdot s_{i} \log \left(\hat{s_{i}}\right)+\left(1-s_{i}\right) \log \left(1-\hat{s_{i}}\right)\right)
\\
M a s k_{i}=\sqrt[3]{\frac{\min \left(f^{*}, b^{*}\right)}{\max \left(f^{*}, b^{*}\right)} \times \frac{\min \left(l^{*}, r^{*}\right)}{\max \left(l^{*}, r^{*}\right)} \times \frac{\min \left(u^{*}, d^{*}\right)}{\max \left(u^{*}, d^{*}\right)}}
$$
其中 $f^*,b^*,l^*,r^*,u^*,d^*$ 代表的是这个点到 ground truth bbox 前后左右上下表面的距离

在推理/训练?时，直接下采样得分最高的 top k 个点以及它们的特征继续进行选框预测

### Centroid Prediction

继续给网络加入一些先验知识：让网络去做一些上下文的预测，能够让检测的结果更好，论文选择去做中心回归预测

$$
\begin{aligned}
L_{c e n t}=& \frac{1}{\left|\mathcal{F}_{+}\right|} \frac{1}{\left|\mathcal{S}_{+}\right|} \sum_{i} \sum_{j}\left(\left|\Delta \hat c_{i j}-\Delta c_{i j}\right|+\left| \hat{c_{i j}}-\overline{c_{i}}\right|\right) \cdot \mathbf{I}_{\mathcal{S}}\left(p_{i j}\right) \\
& \text { where } \quad \overline{c_{i}}=\frac{1}{\left|\mathcal{S}_{+}\right|} \sum_{j} c_{i j}, \quad \mathbf{I}_{\mathcal{S}}: \mathcal{P} \rightarrow\{0,1\}
\end{aligned}
$$
先介绍一下公式中标记的含义：

1. F 代表 ground truth boxes 个数
2. S   代表（某 gt box 内）进行预测的点
3. c 代表 center offset，i 代表第 i 个 gt box，j 代表在对应 gt box 中第 j 个点
4. I 为示性函数，代表这个点有没有在某个 gt 中，也就是是否为前景点

可能上述的公式也不是特别严谨，但是整体还是很好理解的：

1. 每一个（前景）点都有一个对应的 gt bbox，损失的计算都是点与各自的 gt 之间计算的。损失进对前景点进行，并使用前景点的数量进行归一化
2. 增加了一个类似于方差的损失项，希望预测的中心尽量聚集在一个地方，也就是方差尽量的小

### Centroid-based Instance Aggregation

有个预测的中心点过后，就可以基于这些中心点做特征提取。论文使用 PointNet++ 对点集进行 set abstraction。论文这里讲得特别的模糊，直接一句带过了，说是用了 local canonical coordinate system，但是我去看代码的时候好像没有看到对应的 canonical transformation。直接使用了预测得到的中心作为 new_xyz（需要做特征聚集的点）然后使用 PointNet++ 的方式抽取特征。特征来源是之前 SA 的输出，即经过 centroid-aware sampling 后的输出

### Proposal

论文也是一句带过。应该就是使用点的特征预测 bbox，然后使用 NMS 做过滤。论文还使用了 corner loss，即是八个角点的损失函数。这个损失似乎越来越多在用了
$$
L_{\text {corner }}=\sum_{m=1}^{8}\left\|P_{m}-G_{m}\right\|
$$
