---
title: OpenPCDet VoxelRCNN
tags:
  - OpenPCDet
categories:
  - 编程
  - OpenMMLab
mathjax: true
abbrlink: c3e4f95d
date: 2021-12-20 22:20:38
---

# Voxel R-CNN

## VoxelRCNN

来看看 Voxel R-CNN 的实现，由于高度的抽象化和良好的封装，其模型代码和 SECOND 相比，仅多一行 `loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)` 与 roi 相关

```python
class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
```

## MeanVFE (VFE)

将体素中每个点的特征进行平均，batch_dict 添加了 vfe_features 关键字

```python
def forward(self, batch_dict, **kwargs):
    """
    Args:
        batch_dict:
            voxels: (num_voxels, max_points_per_voxel, C)
            voxel_num_points: optional (num_voxels)
        **kwargs:

    Returns:
        vfe_features: (num_voxels, C)
```

## VoxelBackBone8x (BACKBONE_3D)

### Spconv

首先总结一下 spconv1.2 的操作逻辑

1. 生成 `SparseConvTensor` 

   ```python
   import spconv
   
   features = # [N, num_channels]
   indices = # your indices/coordinates with shape [N, ndim + 1] (in 3D ndim=3), batch index must be put in indices[:, 0]
   spatial_shape = # spatial shape of your sparse tensor, (in 3D its shape=[3]) .
   batch_size = # batch size of your sparse tensor.
   x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
   ```

2. 像 Pytorch 一样使用卷积，常用两种卷积 `SparseConv3d & SubMconv3d`，常用一个容器模块 `spconv.SparseSequential`，`indice_key` 可以用于节省相同形状/相同索引的稀疏卷积层建立时间

   ```python
   self.net = spconv.SparseSequential(
               spconv.SparseConv3d(32, 64, 3), # just like nn.Conv3d but don't support group
               nn.BatchNorm1d(64), # non-spatial layers can be used directly in SparseSequential.
               nn.ReLU(),
               spconv.SubMConv3d(64, 64, 3, indice_key="subm0"),
               nn.BatchNorm1d(64),
               nn.ReLU(),
               # when use submanifold convolutions, their indices can be shared to save indices generation time.
               spconv.SubMConv3d(64, 64, 3, indice_key="subm0"),
               nn.BatchNorm1d(64),
               nn.ReLU(),
               spconv.SparseConvTranspose3d(64, 64, 3, 2),
               nn.BatchNorm1d(64),
               nn.ReLU(),
               spconv.ToDense(), # convert spconv tensor to dense and convert it to NCHW format.
               nn.Conv3d(64, 64, 3),
               nn.BatchNorm1d(64),
               nn.ReLU(),
           )
   ```

3. 将 `SparseConvTensor` 转变为正常的 dense tensor

   ```python
   x_dense_NCHW = x.dense() # convert sparse tensor to dense (N,C,D,H,W) tensor.
   ```

### VoxelBackBone8x

这部分直接看前向方程会有更直观的理解，最终返回了一个字典，不仅包含了输出的特征图谱，还有在卷积过程中每一个分辨率的特征图谱也保存下来了 `encoded_spconv_tensor & multi_scale_3d_features`，卷积层的具体设置请直接看源码

```python
def forward(self, batch_dict):
    """
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
    """
    voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
    batch_size = batch_dict['batch_size']
    input_sp_tensor = spconv.SparseConvTensor(
        features=voxel_features,
        indices=voxel_coords.int(),
        spatial_shape=self.sparse_shape,
        batch_size=batch_size
    )

    x = self.conv_input(input_sp_tensor)

    x_conv1 = self.conv1(x)
    x_conv2 = self.conv2(x_conv1)
    x_conv3 = self.conv3(x_conv2)
    x_conv4 = self.conv4(x_conv3)

    # for detection head
    # [200, 176, 5] -> [200, 176, 2]
    out = self.conv_out(x_conv4)

    batch_dict.update({
        'encoded_spconv_tensor': out,
        'encoded_spconv_tensor_stride': 8
    })
    batch_dict.update({
        'multi_scale_3d_features': {
            'x_conv1': x_conv1,
            'x_conv2': x_conv2,
            'x_conv3': x_conv3,
            'x_conv4': x_conv4,
        }
    })
    batch_dict.update({
        'multi_scale_3d_strides': {
            'x_conv1': 1,
            'x_conv2': 2,
            'x_conv3': 4,
            'x_conv4': 8,
        }
    })

    return batch_dict

```

发现代码库里实现了残差模块 `VoxelResBackBone8x`，但是在论文当中并没有使用残差网络

## HeightCompression (MAP_TO_BEV)

这部分就是将 `SparseConvTensor` 转为 dense tensor 并将高度的特征堆叠

```python
def forward(self, batch_dict):
    """
    Args:
        batch_dict:
            encoded_spconv_tensor: sparse tensor
    Returns:
        batch_dict:
            - spatial_features: shape is (N, C * D, H, W)
            - spatial_feature_stride: encoded_spconv_tensor_stride

    """
    encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
    spatial_features = encoded_spconv_tensor.dense()
    N, C, D, H, W = spatial_features.shape
    spatial_features = spatial_features.view(N, C * D, H, W)
    batch_dict['spatial_features'] = spatial_features
    batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
    return batch_dict
```

## BaseBEVBackbone (BACKBONE_2D)

接下来进入 2D 卷积网络，对 `encoded_spconv_tensor` 进行特征提取，这里用配置文件进行说明比较方便

```yaml
BACKBONE_2D:
    NAME: BaseBEVBackbone

    # 5 convolution layers, input channel == output channel == num_filters[idx], stride = 1
    LAYER_NUMS: [5, 5]
    LAYER_STRIDES: [1, 2]
    NUM_FILTERS: [64, 128]
    UPSAMPLE_STRIDES: [1, 2]
    NUM_UPSAMPLE_FILTERS: [128, 128]
```

使用了两个卷积块，每个卷积块由 (5 + 1) 个卷积层组成，+1 代表的卷积层用于通道数的转换，两个卷积块有不同的 stride 以获得不同分辨率。之后使用上采样将两个不同分辨率的特征图谱转换成相同的特征图谱，然后将二者进行通道连接

```python
def forward(self, data_dict):
    """
    Args:
        data_dict:
            spatial_features: shape is (N, C * D, H, W)
    Returns:
        data_dict: spatial_features_2d (N, channels, H, W)
    """
```

## AnchorHeadSingle (DENSE_HEAD)

这将会比较复杂的部分。anchor 生成，target 分配，损失函数的计算，预测结果，都将在这个 `DENSE_HEAD` 中完成。该类的实现也是有基类的 `AnchorHeadTemplate`

### AnchorHeadTemplate

这个基类功能也非常多，我暂且把它的功能分为两大类：anchor 相关和 loss 相关

####  Anchor 相关

```python
class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        """
        Attributes:
        	- self.anchors
            - self.box_coder
            - self.target_assigner
            - self.forward_ret_dict
        """
```

1. `anchors`：一个列表，每个成员对一个类别，一般成员的张量形状都是一样的，`anchors[0].shape = (z, y, x, num_anchor_size, num_anchor_rotation, 7)`，anchors 的生成就不过多介绍了

2. `box_encoder`：可以看作生成回归目标的类，有两个主要功能：输入 anchors 和 gt_boxes，将返回二者的残差；输入残差和 anchors 返回真实的 boxes

3. `target_assigner`：其 `assign_targets` 方法返回一个字典

   ```python
   all_targets_dict = {
               'box_cls_labels': cls_labels,   	# shape is (4, 211200) in KITTI, bg box is 0, fg box is int like (1, 2, 3)
       											# those don't care is -1
               'box_reg_targets': bbox_targets,    # (4, 211200, 7) to be exact, (4, 200*176*2*3, 7)
               'reg_weights': reg_weights			# 1 or 1 / positive_anchors (if normalize), negative anchors are 0
       											# regression weights 在之后似乎并没有用到，而是直接从 cls_label 里进行判断
           }
   ```

   发现这里没有使用 $sin(\Delta \theta)$ 对 target 进行编码，而是直接使用 $\Delta \theta$ 表示方向残差，之后单独用 `add_sin_difference` 处理。assign targets 是**分批分类**进行处理的，这里贴一下其中的核心代码，了解处理一个 sample 一个类该怎么做，因为制作 targets 的过程比较细，不好好看一下真的不清晰

   ```python
   labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
   gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
   
   # 返回与 gt 重叠最大的 anchor index 一般形状为 (num_gt,)
   anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
   # 返回被选中的 gt index, anchor_to_gt_argmax.shape = (num_anchors,)
   gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
   # 标记 max_overlap anchor 并记录其 gt index
   labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
   gt_ids[anchors_with_max_overlap] = gt_inds_force.int()
   
   # 需要注意的是 positive anchor 与 anchors_with_max_overlap 是两个不同的集合
   pos_inds = anchor_to_gt_max >= matched_threshold
   gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
   # 标记 positive anchor 并记录其 gt index
   labels[pos_inds] = gt_classes[gt_inds_over_thresh]
   gt_ids[pos_inds] = gt_inds_over_thresh.int()
   
   # 需要注意的是 negative thres 和 positive thres 之间是有间隙的，二者不相等
   bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
   # 标记 background anchor
   labels[bg_inds] = 0
   # 综上 fg_inds 是 positive_inds | anchors_with_max_overlap - bg_inds
   ```

   **重要的总结再说一遍：综上 `fg_inds` 是 `positive_inds | anchors_with_max_overlap - bg_inds`**

4. `forward_ret_dict`: 虽然这是与 loss 相关的部分，暂且先放在这里。该字典存储向前传播中的预测结果及其标签，用于之后计算 loss

   ```python
   cls_preds
   box_preds
   dir_cls_preds
   box_cls_labels
   box_reg_targets
   reg_weights
   ```

#### Loss 相关

有了 anchor 和对应的 target 还需要预测结果 prediction 才能够计算，需要注意的是：预测的结果是选框残差，还需要 `generate_predicted_boxes` 产生实际的选框结果。

`get_loss` 将获得每个 batch 的损失函数，需要注意的是，loss 相关的方法一般不会在 AnchorHeadSingle 的前向方程中使用，**而是在总模型的前向方程中调用（即：在 VoxelRCNN 类中）**。下面具体看看其组成内容，也不做细节了解

1. `get_loss` 将返回分类损失和回归损失，其中方向分类损失是在回归损失中计算的。`get_loss` 的调用一般是在模型的 `get_training_loss` 中，请查看该笔记之前记录的 `SECOND` 部分中的 `forward` 代码 

   ```python
   def get_loss(self):
       # 获得分类损失 (batch, num_anchors_all)
       cls_loss, tb_dict = self.get_cls_layer_loss()
       # 获得回归损失
       box_loss, tb_dict_box = self.get_box_reg_layer_loss()
       tb_dict.update(tb_dict_box)
       rpn_loss = cls_loss + box_loss
   
       tb_dict['rpn_loss'] = rpn_loss.item()
       return rpn_loss, tb_dict
   ```

   可学习的技巧：使用 weight 来进行筛选，因为在并行运算的情况下，乘法比索引筛选更快

1. `generate_predicted_boxes`

   ```python
    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
       """
       Args:
           batch_size:
           cls_preds: (N, H, W, C1)
           box_preds: (N, H, W, C2)
           dir_cls_preds: (N, H, W, C3)
   
       Returns:
           batch_cls_preds: (B, num_boxes, num_classes)
           batch_box_preds: (B, num_boxes, 7+C)
       """
   ```

   因为 target 是残差，所以在预测了结果过后需要 `box_coder` 进行解码，变为真实的 box

### AnchorHeadSingle

有了基类的功能，`AnchorHeadSingle` 就可以把中心放在网络的搭建之上了，模块在 `__init__` 中定义了三个卷积层：`conv_cls & conv_box & conv_dir_cls` 分别对类别，bbox，朝向进行预测，注意这里并没有全连接层的存在，直接把各个 channel 中的结果作为预测结果。获得预测结果过后：

1. 如果是单阶段检测器基本前向方程就结束了，接下来回到 `Detector` 模块中计算损失函数。如果是测试阶段，需要将预测结果进一步生成最终选框（因为预测的结果是残差），然后进行 NMS 后处理
2. 如果是两阶段检测器，也是将预测结果进一步生成选框，然后继续向前计算

#### forward

下面来看看前向方程，有一个具体感受

```python
def forward(self, data_dict):
    # 获得 BEV 特征图谱
    spatial_features_2d = data_dict['spatial_features_2d']

    # 分类和回归图谱
    cls_preds = self.conv_cls(spatial_features_2d)
    box_preds = self.conv_box(spatial_features_2d)

    # 把 channel 移动到最后一个维度，便于之后计算损失函数
    cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
    box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

    self.forward_ret_dict['cls_preds'] = cls_preds
    self.forward_ret_dict['box_preds'] = box_preds

    # 方向分类图谱
    if self.conv_dir_cls is not None:
        dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
        self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
    else:
        dir_cls_preds = None

    # targets_dict 返回值在 anchor 相关部分里
    if self.training:
        targets_dict = self.assign_targets(
            gt_boxes=data_dict['gt_boxes']
        )
        self.forward_ret_dict.update(targets_dict)

    # 如果是测试或者是两阶段检测器，则需要生成选框预测
    if not self.training or self.predict_boxes_when_training:
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=data_dict['batch_size'],
            cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
        )
        data_dict['batch_cls_preds'] = batch_cls_preds
        data_dict['batch_box_preds'] = batch_box_preds
        data_dict['cls_preds_normalized'] = False

    return data_dict
```

## VoxelRCNNHead (ROI_HEAD)

从配置文件看，这一部分也是最复杂的。不过不要担心，有了足够的理论知识和对之前代码的解读，咱也能大致掌握其中的重要的流程🤨先来看基类 `RoIHeadTemplate` 实现

### RoIHeadTemplate

还是按照之前 AnchorHeadTemplate 的总结方式，分为两类：proposal 相关和 loss 相关

#### Proposal 相关

```python
class RoIHeadTemplate(nn.Module):
    def __init__(self, num_class, model_cfg, **kwargs):
        """
        Attribute:
            - self.box_coder: ResidualCoder
            - self.proposal_target_layer
            - self.forward_ret_dict
        """
```

以上是初始化函数，`box_coder` 和之前的是一样的 ` ResidualCoder`；比较重要的是 `proposal_target_layer`，这部分对应的配置是 `TARGET_CONFIG`，其功能是对 NMS 筛选过后的 proposal（此时应该叫 rois）进行采样并制作其标签；`forward_ret_dict` 用于存储向前传播中的预测结果及其标签，用于之后计算 loss

下面进行进一步介绍 `RoIHeadTemplate` 的相关方法：

1. `proposal_layer` 方法。使用 NMS，（通常）返回 512 个 rois（不足的用0填上）

```python
def forward(self, batch_dict):
    """
    Args:
        batch_dict:
            batch_size:
            rois: (B, num_rois, 7 + C)
            roi_scores: (B, num_rois)
            gt_boxes: (B, N, 7 + C + 1)
            roi_labels: (B, num_rois)
    Returns:
        batch_dict:
            rois: (B, M, 7 + C)
            gt_of_rois: (B, M, 7 + C)
            gt_iou_of_rois: (B, M)
            roi_scores: (B, M)
            roi_labels: (B, M) cls-based label
            reg_valid_mask: (B, M) positive bbox
            rcnn_cls_labels: (B, M) iou-based label
    """
```

2. `assign_targets` 方法。通常在调用 `proposal_layer` 方法后使用，该方法完成了两个事情：

   1. 再在 rois 采样 M 个 roi，并获得其对应的 gt 标签、iou 标签、正样本 mask等。通常采样 128 个 roi，以 1:1 正负比例进行采样，正负判定条件依然为 iou 相关阈值，负采样仅参与置信度损失的计算，不参与回归损失的计算。Target 的分配是通过 `ProposalTargetLayer` 类的前向方程完成，也即 `self.proposal_target_layer`

      ```python
      targets_dict = self.proposal_target_layer.forward(batch_dict)
      """
      Args:
          batch_dict:
              batch_size:
              rois: (B, num_rois, 7 + C)
              roi_scores: (B, num_rois)
              gt_boxes: (B, N, 7 + C + 1)
              roi_labels: (B, num_rois)
      Returns:
          batch_dict:
              rois: (B, M, 7 + C)
              gt_of_rois: (B, M, 7 + C)
              gt_iou_of_rois: (B, M)
              roi_scores: (B, M) cls-based score
              roi_labels: (B, M) cls-based label
              reg_valid_mask: (B, M) positive bbox
              rcnn_cls_labels: (B, M) iou-based label
      """
      ```

   2. 将 gt 转换到对应的 roi 坐标系当中（平移+旋转），需要注意的是还对方向相反的 gt 进行了 flip orientation 处理，以减少错误预测的损失。虽然这不是真实的标签，但损失太大可能不利于维护 R-CNN 训练的稳定
   
   2. 这里提一下，所有获得的 target 都使用 detach 从计算图中分离，也就是不希望更新用于预测 roi 部分的网络参数，仅关注用于预测残差以及 backbone 中的网络参数。建议画一下计算图

#### Loss 相关

与 AnchorHeadTemplate 一样，这里也是两个主要方法：

1. `get_loss` 计算分类损失和回归损失，有趣的是在回归损失中还使用了一个 `corner_loss`，这是论文中没有提到的，而且在代码的注释中也写了 `TODO: NEED TO BE CHECK`
2. `generate_predicted_boxes`，将预测的残差结果，还原为真实选框

这么一看是不是结构就清晰很多了呢？下面就看看 VoxelRCNNHead 干了些什么吧！

### VoxelRCNNHead

当有了基类 RoiHeadTemplate 过后就可以专注实现 R-CNN 的核心功能，即 roi pooling 提取特征

#### init

在搭建该部分的网络之前，先来看看需要哪些子模块需要定义

1. `pool_layer`，这是通过另一个类实现 `NeighborVoxelSAModuleMSG`。这个层用于 voxel query 寻找附近的非空体素，并对 grouping 特征进行特征提取
2. `shared_fc_layer`，过渡全连接层，`nn.Linear + nn.BatchNorm1d + nn.ReLU`
3. `cls_fc_layer`，分类全连接层，`nn.Linear + nn.BatchNorm1d + nn.ReLU`
4. `reg_fc_layer`，预测全连接层，`nn.Linear + nn.BatchNorm1d + nn.ReLU`

#### roi_grid_pooling

这个功能函数基本上就是 roi grid pooling 的核心，其作用简单叙述为：在不同分辨率 feature source 下，对每个 grid point 进行 roi pooling，并将不同分辨率的结果连接起来。下面具体分析其中的步骤：

1. 获得 roi 中的 grid point 在当前 feature source 特征图谱的 voxel 坐标 (B, x, y, z) 以及 lidar 坐标 (BxN, 6x6x6, 3)，用于之后的 grouping。我一直有一个疑问：如果 roi 不是一个合法的选框应该怎么办？比如长宽高为负数。实际上这样的问题不会发生，因为得到的选框是基于 anchor 进行变换的，保证了预测选框的合法性

2. 获得 `pooled_feature_list`，即对每个分辨率的 feature source，使用 `pool_layer` 对每个 grid point 进行 roi pooling。然后将所有分辨率的特征连接起来 `torch.concat` 得到每个 grid point 最终的特征。`pool_layer` 是比较复杂的一个层，是类 `NeighborVoxelSAModuleMSG` 的对象，其有四个子模块

   1. `mlp_in`，对所有的 grid point 进行统一特征提取，由 kernel size = 1 的 `Conv1d` 完成

      ```python
      ##################### Note ########################
      # Ni 是第 i 个样本的非空体素的个数
      # 并且本笔记中没有对 channel 数量进行区分，都用 C 表示
      # k 表示 k 个不同分辨率的特征图谱
      ##################### Note ########################
      # features_in: (1, C, N1+N2+...)
      features_in = self.mlps_in[k](features_in)
      features_in = features_in.permute(0, 2, 1).contiguous()	# features_in: (1, M1+M2+..., C)
      features_in = features_in.view(-1, features_in.shape[-1])	# features_in: (M1+M2+..., C)
      ```

      获得了 `features_in` 之后将会输入到 `self.groupers` 当中

   2. `self.groupers`，执行 grouping 操作，由类 `VoxelQueryAndGrouping` 实现。当邻居数量没有 nsample 这么多时使用第一个 sample grid 进行补位。再通过标注 `empty_ball_mask` 得知该 grid point 是否有邻居，在之后使用 MLP 提取特征时把空 grid point 的特征设置为 0 即可。该类的前向方程获得的结果如下

      ```python
          def forward(self, new_coords: torch.Tensor, xyz: torch.Tensor, xyz_batch_cnt: torch.Tensor,
                      new_xyz: torch.Tensor, new_xyz_batch_cnt: torch.Tensor,
                      features: torch.Tensor, voxel2point_indices: torch.Tensor):
              """
              Args:
                  # voxel data
                  xyz: (N1 + N2 ..., 3) xyz coordinates of the features
                  xyz_batch_cnt: (batch_size), [N1, N2, ...]
                  features: (N1 + N2 ..., C) tensor of features to group
                  voxel2point_indices: (B, Z, Y, X) tensor of points indices of feature source voxels
      
                  # grid point data
                  new_coords: (M1 + M2 ..., 3) centers voxel indices of the ball query
                  new_xyz: (M1 + M2 ..., 3) centers of the ball query
                  new_xyz_batch_cnt: (batch_size), [M1, M2, ...] Mi = 128x6x6x6
      
              Returns:
                  grouped_xyz: (M1 + M2 ..., 3, nsample)
                  empty_ball_mask: (M1 + M2 ...,)
                  grouped_features: (M1 + M2 ..., C, nsample)
              """
      ```

   3. `mlp_pos`，对 group 得到的 nsample 个 voxel positions 进行特征提取，由 kernel size = 1 的  `Conv2d` 完成 ` grouped_xyz: (1, 3, M1+M2+..., nsample)`

      ```python
      # grouped_xyz: (1, 3, M1+M2+..., nsample)
      position_features = self.mlps_pos[k](grouped_xyz)	# position_features: (1, C, M1+M2+..., nsample)
      ```

   4. `mlp_out`，在使用该 MLP 之前，需要将前两个 MLP 提取的特征加起来 `mlp_in + mpl_pos`，然后使用 max pooling 消除 nsample 维度，得到汇聚特征 (1, C, M1+M2+...)。然后再进行特征提取，由 `Conv1d` 完成

   **下面整体来看看 `pool_layer` 层的输入和输出**

   ```python
    pooled_features = pool_layer(
                   xyz=cur_voxel_xyz.contiguous(),
                   xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                   new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                   new_xyz_batch_cnt=roi_grid_batch_cnt,
                   new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                   features=cur_sp_tensors.features.contiguous(),
                   voxel2point_indices=v2p_ind_tensor
               )
   # return:
   # new_xyz: (M1 + M2 ..., 3) tensor of the new features' xyz
   # new_features: (M1 + M2 ..., C) tensor of the new_features descriptors
   ```

3. 连接不同的 feature source 下得到的 roi pooling 特征并改变其维度得到 (BxMi, 6x6x6, C)

#### forward

看完了之前的复杂模块，感觉头都要晕了...好消息是到了这一步，基本上就没有其他复杂模块了，事情变成了简单的组合。有了基类处理 proposal & target & loss，有了 roi grid pooling 获得 grid point features，直接使用定义好的 MLP 进行预测

```python
    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        # 根据 NMS 获得 512 个得分最高的选框
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            # 采样 128 个 rois 并获得 gt，并将 gt 移动到对应 rois 的坐标系当中
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        # Box Refinement
        pooled_features = pooled_features.view(pooled_features.size(0), -1) # (BxN, -1)
        shared_features = self.shared_fc_layer(pooled_features)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict

```

## 感言

至此，Voxel R-CNN 的框架就总结完了😭一路上真的有太多的困难了，但是整个代码看下来感觉自己还是收获不少！虽然路还很长，但是至少迈出了第一步

过程中遇到了代码之外的问题，例如 localhost:10.0 问题，一般是因为没能找到本机的显示器，把所有的东西都重启一遍，包括你自己的电脑！

## TODO

1. 整理损失函数
2. 整理常用功能函数 utils 以及三方库
3. pytorch 技巧总结
4. Summary SPG, Lidar R-CNN
5. SA-SSD 代码阅读
