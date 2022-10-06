# DETR

[zhihu](https://zhuanlan.zhihu.com/p/348060767)	[bilibili](https://www.bilibili.com/video/BV1GB4y1X72R)

DETR 的特点：简单！端到端！no anchor！在性能（表现/速度）上和 Faster RCNN 相似。虽然和当时最好的方法相差10个点，但是这个框架太好了，是一个挖坑性质的文章，所以这也是件好事儿

## Intro

现阶段目标检测器很大程度上都受限于后处理 **NMS** 的方法，不管是 anchor-based or anchor-free，RCNN or SSD or YOLO，都有这个问题，这也让目标检测在目前的深度学习方法里都是比较复杂的，在优化和调参的难度上都比较大

DETR 的网络流程如下图所示

<img src="DETR/image-20220919001758670.png" alt="image-20220919001758670" style="zoom:67%;" />

用语言总结如下：

1. 使用 CNN 抽取图像特征
2. 使用 Transformer encoder 获得全局特征
3. 使用 Transformer decoder 获得预测框
4. 将预测框和 ground truth boxes 做匹配并计算 loss

更具体的细节显然没办法从图里获得了，需要的是代码，在之后的 代码 章节里介绍

受益于 transformer 的全局建模能力，DETR 对于大物体的检测能力非常强，但是对小物体的比较差，并且 DETR 收敛的速度非常慢。改进方法在 Deformable DETR 中提出，依然是使用多尺度的特征图谱 + Deformable attention

前人也有使用二分图匹配的方法，或者使用 RNN 做 encoder-decoder 来进行目标检测，但是都没有用 transformer 所以性能上不去。所以说 DETR 的成功，也是 transformer 的成功

##  Model

### Bipartite Matching Loss

论文认为结构都是比较简单好理解的，所以先讲了损失函数这一块：如何使用二分图匹配来计算损失

DETR 预测输出是一个固定值，即预测固定的 N(=100) 个预测

关于二分图匹配算法（匈牙利算法），我在之前的博客 **图论算法** 里有一些总结可以参考，在 DETR 的场景下该匹配算法的作用为：将 N 个 prediction 与 N 个 gt 进行配对（没有 N 个 gt 则需要 padding）。预测有了 gt 过后就可以计算损失函数了

配对使用的 cost matrix 计算公式如下
$$
\hat{\sigma}=\underset{\sigma \in \mathfrak{S}_{N}}{\arg \min } \sum_{i}^{N} \mathcal{L}_{\operatorname{match}}\left(y_{i}, \hat{y}_{\sigma(i)}\right) \\
\mathcal{L}_{\operatorname{match}}\left(y_{i}, \hat{y}_{\sigma(i)}\right)=
-\mathbb{1}_{\left\{c_{i} \neq \varnothing\right\}} \hat{p}_{\sigma(i)}\left(c_{i}\right)+\mathbb{1}_{\left\{c_{i} \neq \varnothing\right\}} \mathcal{L}_{\mathrm{box}}\left(b_{i}, \hat{b}_{\sigma(i)}\right)\\

\mathcal{L}_{\mathrm{box}}=\lambda_{\text {iou }} \mathcal{L}_{\text {iou }}\left(b_{i}, \hat{b}_{\sigma(i)}\right)+\lambda_{\mathrm{L} 1}\left\|b_{i}-\hat{b}_{\sigma(i)}\right\|_{1}
$$
其中 $\sigma$ 可以看作一个排列或者映射，$\sigma(i)$ 代表第 i 个 gt 所匹配的预测的 index，box 损失使用的是 GIoU 损失和 L1 损失的加权。注意到空 gt 和任何 prediction 的 cost 都是 0，所以本质上就是 N 个 prediction 和 M 个 gt 之间的匹配，用于确定 M 个正样本 prediction 和 N - M 个负样本 prediction

匹配完成后，就可以计算损失函数
$$
\mathcal{L}_{\text {Hungarian }}(y, \hat{y})=\sum_{i=1}^{N}\left[-\log \hat{p}_{\hat{\sigma}(i)}\left(c_{i}\right)+\mathbb{1}_{\left\{c_{i} \neq \varnothing\right\}} \mathcal{L}_{\text {box }}\left(b_{i}, \hat{b}_{\hat{\sigma}}(i)\right)\right]
$$
论文提到在计算分类损失时，对于空类 gt $\varnothing$ 的分类损失要除以 10 用于平衡正负样本  

### Codes

#### Structure

TODO

#### Loss

TODO

1. 多个中间 loss 监督
2. cross attention
3. positional embedding

如果不使用预训练权重，会怎样？

DC5 是什么意思

大物体和小物体效果相差大

encoder 和 decoder 的作用，和 CNN 类似，UNet，FCN

object query 的预测结果可视化
