# Swin and Better ViTs

现在是 2023/06，所以 swin transformer 已经是两年前的工作了，之前看了论文觉得很复杂，没看懂，不喜欢🤣现在代码能力更强了过后再来整理一番！

这一篇笔记的目的是想要对 vision transformer 做一个代表性的整理，因为 ViT 有很多变体，咱们当然只看效果最好的那几个，并掌握其精髓，入选的模型如下：

1. Swin Transformer
2. Deformable Attention Transformer, DAT
3. MaxViT
4. DaViT

解释一下 DAT 入选的原因，因为我个人不太喜欢 detr 中对 deformable 的实现，一直想要找到其替代品，了解到了 DAT，故打算整理一下

实现参考来自于两个项目：[vit-pytorch](https://github.com/huggingface/pytorch-image-models) & [timm](https://github.com/huggingface/pytorch-image-models)

## Positional Embeddings

### relative position bias

这里不再是 positional **embeddings**，而是一个 bias！并且则个 bias 是可学习的

einsum 用于替代矩阵乘法以更好地获得阅读性

## DAT

lucidrian 的代码读起来太赏心悦目辣！

非常简洁的 deformable attention 的实现，完全可以替代 deformable detr 中的实现
