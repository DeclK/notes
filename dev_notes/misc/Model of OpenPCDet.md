# Model of OpenPCDet

## Model function

from collections import namedtuple

1. load data to gpu

   ```python
   batch_dict[key] = torch.from_numpy(val).float().cuda()
   ```

2. 向 model 传入 batch_dict

3. 更新 global step，相当于在模型内记录 Iteration 次数

## Model

### MeanVFE

给 voxel 特征做平均

```python
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
```

torch.clamp 截断

### VoxelBackBone8x

grid_size[::-1] 表示序列取反 grid_size.shape = (D, H, W)

暂时无法理解为什么使用这个形状 [1600, 1408, 41]，[github issue](https://github.com/open-mmlab/OpenPCDet/issues/502)

#### Sparse convolution 和 submanifold convolution 的混合结构

> submanifold convolution [27] restricts an output location to be active if and only if the corresponding input location is active. This avoids the generation of too many active locations

<img src="Model of OpenPCDet/image-20211111152232190.png" style="zoom:67%;" />

使用普通稀疏卷积用于下采样，再使用子流形卷积进行进一步特征提取

```python
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                - encoded_spconv_tensor: sparse tensor
                    To be exact: in KITTI, spatial shape is [200, 176, 2], channel num is 128
                    But haven't converted to dense format yet
                - encoded_spconv_tensor_stride
                - multi_scale_3d_features
                - multi_scale_3d_strides
```

### Height compression

作用：map to bev

### BaseBEVBackbone

```python
def __init__()        
    Args:
        NAME: BaseBEVBackbone

        LAYER_NUMS: [5, 5]
        LAYER_STRIDES: [1, 2]
        NUM_FILTERS: [128, 256]
        UPSAMPLE_STRIDES: [1, 2]
        NUM_UPSAMPLE_FILTERS: [256, 256]

def forward()
        Args:
            data_dict:
                spatial_features: shape is (N, C * D, H, W)
        Returns:
            data_dict: spatial_features_2d
```

对 BEV 特征做卷积操作

### AnchorHead

这应该是最难的一个部分，anchor 生成，target 分配，损失函数的计算，预测结果，都将在这个 Dense head 中完成

#### anchor generate & target assign

先看 Init

#### pdb

通过 pdb 模块对 python 代码进行调试 

https://zhuanlan.zhihu.com/p/37294138

https://www.bilibili.com/video/BV1bg4y1B7vK

torch.repeat

torch.expand

Tensor[...]

#### 关于 anchor 生成的问题

[github issue](https://github.com/open-mmlab/OpenPCDet/issues/272)

[github issue](https://github.com/open-mmlab/mmdetection3d/issues/648#issuecomment-917540581)

作者说是为了兼容之前的实验，自己可以将 center alignment 调整为 true 尝试一下，估计影响不大，因为 anchor 本来就是密集摆放的

anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]

pdb 断点不停止

https://stackoverflow.com/questions/7617066/pdb-wont-stop-on-breakpoint

#### feature map

这部分比较简单，直接拿之前的 BEV map 过两个卷积层就可以

#### get loss

```python
    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict
```



Tensor.scatter 经常在创建 one hot 标签的时候使用，主要根据 index 矩阵，修改目标矩阵中的值

tensor.to 也可以转换属性，也可以将张量送入 cpu/gpu

logits, sigmoid, softmax 的关系，细节！sigmoid_cross_entropy_with_logits，链接在下面

https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits



给类的功能定义一定要明确，这样才能在处理不同情况的时候，有明确的使用途径

比如在生成 anchors 的时候，是对一个特征图谱生成每个类别的 anchors，而不是生成整个 batch 的 anchors，一定要明确输入，输出！

clamp 经常使用，因为可能会除以0

使用 weights 作为 mask 去除 negative anchors



glob 库可用于查询文件

# Evaluation

下一步，弄清如何 evaluate，并对推理速度值进行测试



# Experiment schedule

只实验车子的类型尝试一下效果，这样时间会很快

1. second-fast
2. second-fast-iou
3. centerpoint with second fast iou
4. sa-ssd with second-fast-iou
5. se-ssd with second-fast-iou
6. se-ssd with an-second-fast-iou 
7. se-ssd with an-second-fast-iou-centerpoint

## 可视化

只需要弄懂几个 mayavi 的关键函数就可以了

https://github.com/hailanyi/3D-Detection-Tracking-Viewer

torch.max(input, dim=) 会返回一个 namedtuple(values, indices)

vtk camera 参数

https://blog.csdn.net/minmindianzi/article/details/84279290

![Azimuth, elevation and roll angles for heads | Download Scientific Diagram](Model of OpenPCDet/Azimuth-elevation-and-roll-angles-for-heads.png)

# 2021-11-18

## tensorboard on vscode 

https://code.visualstudio.com/docs/datascience/pytorch-support#_tensorboard-integration

## visualization with kitti

视场角？

r40 应该是指 precision-recall 曲线中，recall 横坐标应该是取 40 个标记点，这样更密集一些

https://github.com/hailanyi/3D-Detection-Tracking-Viewer

目前已经完成了视角的转变，点云和 bbox 的添加，并成功完成了图片保存（也是需要将 show offscreen 设置为 true）

# 2021-11-19

需要将 gt 和标签放在同一个场景当中进行区别，如果不是预测的类型则使用其他颜色，把它们的 score 打印出来

推理的时间可以加入到 log 信息当中

进行多层感知机的整理

论文阅读并制作PPT

昨天一直在思考 kitti evaluation 的问题，昨天一直没有明白，今天算是能够说服自己了，虽然也没有一步一个坑地完全明白。起因是为什么 score_thres = 0.0 的时候，kitti evaluation 结果并没有改变，AP 依然还是那么多？关键在于对于预测选框的排序，先匹配得分高的预测框，当质量不错的选框匹配得差不多了过后，其实召回率已经差不多了接近一了，最后的一点预测，其实对于 AP 的影响是几乎没有的

## 2021-11-29

回归问题是个大麻烦，因为 center based 没有锚框作为基本的参考，所以通过特征确实比较难回归，但两阶段将会很好的弥补这个问题，因为预选框提供了更好的参考

是否能够再次将回归问题转换为分类问题？

深入思考一阶段和二阶段的区别：

直观来说二阶段利用更细粒度的空间信息，但从回归的角度来说

辅助网络+二阶段，给细粒度的空间信息提供更多语义信息

fpn可以用于检测小物体
