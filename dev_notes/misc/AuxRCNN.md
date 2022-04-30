# AuxRCNN

## 搭建自己的 AuxRCNN

### auxrcnn_backbone

改进 VoxelBackBone8x，加入辅助网络，并且需要计算辅助网络损失函数，文件位于 `/OpenPCDet/pcdet/models/backbones_3d`

有不少可复用代码

由于 spconv 的使用习惯，voxel 相关的好多都是 zyx 的维度顺序，要注意一下

### aux_rcnn.py

搭建网络，文件位于 `/OpenPCDet/pcdet/models/detectors`

### registry

把 AuxRCNN 注册到框架中，修改 models 中的 `__init__.py` 



### 区分训练与测试

更改 `nn.Module` 中的内置属性 `self.train` 以区分是训练还是测试。也可以通过 `model.train()` 进行区分 

### build aux & compute aux

weights for aux task no longer matters

[issue](https://github.com/skyhehe123/SA-SSD/issues/66)

## Train

!!!!!可以开始训练啦！！！！

遇到问题

```python
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss. You can enable unused parameter detection by (1) passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`; (2) making sure all `forward` function outputs participate in calculating loss. If you already have done the above two steps, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
```

设置 find_unused_parameters=True

因为我想要看看分类的准确率如何，所以想要计算一下准确率 acc (thres = 0.7)，仅计算 in_box 的准确率。但这个 acc 和损失函数是没有关系的 [CSDN](https://blog.csdn.net/weixin_44966641/article/details/120385212)，所以有了上面的报错

## Experiment

现在获得的最好结果是85.21，比之前的最好结果要高0.3%，提升是有效果的，现在就看能不能把这个超参数给调出来

首先尝试了 epoch，显然即使是足够长的 epoch 也不能够获得良好的效果。尝试了300个 epoch，效果甚至没有120个 epoch 的85.05效果好，只能达到84.88

这里是没有搜索到一个更好的最优解，我需要尝试更小的 batch size 和学习率。既然更多的 epoch 不能够调整出更好的模型，这里就对 epoch 有了一个指导：最高为200个 epoch，更高的 epoch 将不会有其他的指导意义

现在我每隔2个 epoch 保存一个模型，能够看到更精细的变化了