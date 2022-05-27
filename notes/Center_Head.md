---
title: OpenPCDet Center_Head
tags:
  - OpenPCDet
categories:
  - 编程
  - OpenMMLab
mathjax: true
date: 2022-05-11 11:20:00
---

# Center_Head

我又重新思考了一下该如何去学习一个复杂的项目，难点在于复杂项目的体量会很大，怎样将其中的细节和整体很好的把握住？我打算用 Center_Head 来实验一下这个策略：

1. **关注整体的模块**，对输入 & 输出 & 运行逻辑需要特别清楚
2. **细节部分过一下就好**，看看有没有特别的 trick 或者奇怪的地方需要注意的，主要目的为学习代码技巧

具体来说，对于类，按照 init 和 methods 进行简单划分，对于每一个 methods，按照 logic + input + output/attributes + used functions + 学习点 进行整理 

以上是学习模块的思路，但除了模块除此之外可能还要整理一下项目的大框架逻辑，从而递归地完成对整个项目的掌握

## SeperateHead

这个类就是用于实现子任务的

### init

1. input params

```python
class SeparateHead(nn.Module):
    def __init__(self, input_channels, sep_head_dict, init_bias=-2.19, use_bias=False):
```

2. attributes

   1. `self.sep_head_dict` 存放参数信息，大概长这样

      ```yaml
                  HEAD_DICT: {
                      'center': {'out_channels': 2, 'num_conv': 2},
                      'center_z': {'out_channels': 1, 'num_conv': 2},
                      'dim': {'out_channels': 3, 'num_conv': 2},
                      'rot': {'out_channels': 2, 'num_conv': 2},
                  }
      ```

      其中 hm head 的 out_channels 是根据 classes each head 决定，故没有写在 dict 中，但最终的字典里是有的
   
   2. 不同的 fc，用于不同的头，其名称就是上面 dict 里的 key。从 dict 来看都是由两层 Conv2d 组成

### forward

直接上函数，非常简单的前向

```python
    def forward(self, x):
        ret_dict = {}
        for cur_name in self.sep_head_dict:
            ret_dict[cur_name] = self.__getattr__(cur_name)(x)

        return ret_dict
```

学习点：

1. 使用 `__setattr__` 方法设置 fc 的名称，然后使用 `__getattr__` 调用。当然使用 `getattr & setattr` 方法也是可以实现一样的功能
2. 使用 `fc.modules` 循环模块，并对其进行初始化

## CenterHead

### init

1. input params

   ```python
   class CenterHead(nn.Module):
       def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training=True):
   ```

2. attribute

   1. `self.shared_conv`，一层 kernel_size=3 的 Conv2d

   2. `self.heads_list`，其实就是将多个 `SeqarateHead` 放到一个 `nn.ModuleList` 中。在 nuScenes 中有多个 head（可能因为类别太多，把相似的类别放到一个 head 处理），但是在 waymo 里面只有一个 head

   3. `self.build_losses()` 创建损失函数模块，具体的损失函数分析在之后的笔记中写（挖坑）

      ```python
          def build_losses(self):
              self.add_module('hm_loss_func', loss_utils.FocalLossCenterNet())
              self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())
      ```
      
   4. `self.class_names_each_head` 在 nuScenes 中好像把相似的物体用一个头来处理
   
      ```yaml
              CLASS_NAMES_EACH_HEAD: [
                  ['car'], 
                  ['truck', 'construction_vehicle'],
                  ['bus', 'trailer'],
                  ['barrier'],
                  ['motorcycle', 'bicycle'],
                  ['pedestrian', 'traffic_cone'],
              ]
      
      ```
   
      但是在 waymo 没有这么做，车人都是用一个 head 处理
      
   5. `self.class_id_mapping_each_head`，在之后需要把 pred label 还原到原始 index。因为分了多头，每个类的数字标签是重新开始算的
   
      ```python
              CLASS_NAMES_EACH_HEAD: [
                  ['car'], 	# label [1]
                  ['truck', 'construction_vehicle'],	# label [1, 2]
                  ...
              ]
      ```
   
      但在最后生成 boxes 时，要还原为原始数字标签 [1, 2, 3, ...]

### assign_target_of_single_head

该函数就是给单个 sample 里的指定 classes 分配目标

1. input params

   ```python
       def assign_target_of_single_head(
               self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
               gaussian_overlap=0.1, min_radius=2
       ):
           """
           Args:
               gt_boxes: (N, 8)
               feature_map_size: (2), [x, y] == [W, H]
   ```

2. output

   ```python
           return heatmap, ret_boxes, inds, mask
   ```

   1. `heatmap`: (num_classes, y, x) 代表为各个类的可能性
   2. `ret_boxes`: (num_max_objs, 8+C) 是经过 encode 过后的 boxes $(o_x, o_y, z, log(dx), log(dy), log(dz), sin(\theta), cos(\theta),...)$：
      1. 中心预测 offset
      2. 长宽高为原始的 log
      3. 去除了 class 维度，但把角度维度变为两个 (cos, sin)
   3. `inds`: (num_max_objs) 每一个目标的序号，是把二维表格排序为一维过后的序号
   4. `mask` (num_max_objs) 表示是否为空目标

3. used functions
   1. `centernet_utils.gaussian_radius` 通过 box 的长宽以及最小重叠比例，可以算出一个半径值，[zhihu](https://zhuanlan.zhihu.com/p/96856635)
   2. `centernet_utils.draw_gaussian_to_heatmap` 将扩张过后的 gaussian 放到 heatmap 上

学习点：

1. gaussian2D 方法使用了 `np.ogrid` 产生我们希望的格点序列，然后使用广播的方式得到我们需要的格点值
2. tensor 的切片是和原 tensor 共享存储的，而不像 list 的切片生成新的对象

### assign_targets

利用前面定义的 `assign_target_of_single_head`，对每个 head，每一个 batch 进行单独计算，然后对 batch 的结果 concat，对每个 head 的结果 append。个人感觉这里的 assign_targets 不像 anchor-based 方法，**assign 这个动作不是很明显，更像是在 build targets** 

1. input params

   ```python
       def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
           """
           Args:
               gt_boxes: (B, M, 8+C)
               feature_map_size: (2) [H, W]
           Returns:
   ```
   
   对于 gt_boxes，最后一个 dim 是类别，所以有 7 + 1 个 dim。对于 nuScenes 还有速度标签 (vx, vy)，所以 dim 为 10

2. output

   ```python
           ret_dict = {
               'heatmaps': [],
               'target_boxes': [],
               'inds': [],
               'masks': [],
               'heatmap_masks': []	# not used
           }
   ```

   返回一个字典，每一个字典是一个列表，再次强调，对于 waymo 来说每个列表只有一个元素（只有一个头），只有 nuScenes 把类别分给了多个头去处理，所以列表有多个元素，每个元素的形状如下

   1. `heatmaps`，(B, num_classes, y, x) $\leftrightarrow$ (B, num_classes, H, W)
   2. `target_boxes`，(B, num_max_objs, 8 + C)
   3. `inds`，(B, num_max_objs)
   4. `masks`，(B, num_max_objs)

### sigmoid

写了一个简单的 sigmoid 函数，对上下界进行 clamp

```python
    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y
```

### get_loss

对于每一个 head 分别计算 loss，包括 focal loss & reg loss，并给这些 loss 乘以相应的权重，然后全部加起来

1. input params: none，没有 input，使用的是 `self.forward_ret_dict` 里面的预测数据
2. output: loss, tb_dict
3. used functions: 这部分其实相对复杂，OpenPCDet 直接搬运了 CenterPoint 里面的原始 loss functions，有两个类型：
   1. FocalLossCenterNet
   2. RegLossCenterNet

#### FocalLossCenterNet

这个类实际上是调用了 `neg_loss_cornernet` 函数，所以直接看这个函数的就好。这个函数是 focal loss 的一个变体，进一步抑制负样本的 loss
$$
L_{\text {det }}=\frac{-1}{N} \sum_{c=1}^{C} \sum_{i=1}^{H} \sum_{j=1}^{W}\left\{\begin{array}{cc}
\left(1-p_{c i j}\right)^{\alpha} \log \left(p_{c i j}\right) & \text { if } y_{c i j}=1 \\
\left(1-y_{c i j}\right)^{\beta}\left(p_{c i j}\right)^{\alpha} \log \left(1-p_{c i j}\right) & \text { otherwise }
\end{array}\right.
$$

1. input params

   ```python
   def neg_loss_cornernet(pred, gt, mask=None):
       """
       Refer to https://github.com/tianweiy/CenterPoint.
       Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
       Args:
           pred: (batch x c x h x w)
           gt: (batch x c x h x w)
           mask: (batch x h x w)
       Returns:
       """
   ```

   c 代表类别数量

2. output: loss with shape []，是一个标量

#### RegLossCenterNet

对 positive point 计算回归损失

1. input params

   ```python
       def forward(self, output, mask, ind=None, target=None):
           """
           Args:
               output: (batch x dim x h x w) or (batch x max_objects)
               mask: (batch x max_objects)
               ind: (batch x max_objects)
               target: (batch x max_objects x dim)
           Returns:
           """
   ```

2. output: loss with shape [dim]，也是 normalized by mask (true gt)

3. used functions

   1. `_transpose_and_gather_feat(output, ind)` 把 feature map 拉直并变换一下维度顺序 (B, M, dim)，然后使用 gather 通过 inds 选出需要的 feature
   2. `_reg_loss` 计算各个维度的 L1 损失函数。除此之外还有两点注意：
      1. 处理了 nan 值，把其置0
      2. 使用了 mask 将空 gt 掩去

学习点：

1. `tensor.expand` 和 `tensor.repeat` 的区别有两点，一个是 expand 仅仅改变 view，与原 tensor 共享内存，但 repeat 是生成新对象；另一个是输入参数不一样，expand 是 desired shape，repeat 是重复次数。也可以使用 `tensor.expand_as(tensor)` 简化
2. 可以使用 unsqueeze 方法，这样在 broadcast 时（例如 expand）不容易出错
3. `torch.gather` 和 `torch.scatter` 是相反的操作，[gather](https://pytorch.org/docs/stable/generated/torch.gather.html#torch.gather) 可以看作在 tensor 里面取一些元素，生成新的 tensor。在这里是使用 inds 在 output 中取一些元素，这些元素与 gt 的位置对应
3. 使用 `(~ torch.isnan(gt_regr)).float()` 生成 nan 的 mask，把一些异常点筛除

### generate_predicted_boxes

对 encode 的 boxes 进行 decode，并且根据 post processing 配置进行过滤

1. input params

   ```python
       def generate_predicted_boxes(self, batch_size, pred_dicts):
   ```

   依然是对每一个 head，每一个 batch 进行循环处理

2. output

   ```python
           ret_dict = [{
               'pred_boxes': boxes_all_heads,		# (num_boxes_all_heads, 7+C),
               'pred_scores': scores_all_heads,	# (num_boxes_all_heads,)
               'pred_labels': labels_all_heads,	# (num_boxes_all_heads,)
           } for k in range(batch_size)]
   ```

   这和 assign_targets 的输出有点相似，但后者是 list of dict of list，比前者还多一个 batch 套娃

3. used functions：

   1. `centernet_utils.decode_bbox_from_heatmap`，功能就是函数名，其 input params 如下

      ```python
                  final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                      heatmap=batch_hm,	# pred heatmap, note is batch data
                      rot_cos=batch_rot_cos, rot_sin=batch_rot_sin,	# pred rotation
                      center=batch_center, center_z=batch_center_z,	# pred center offset & z
                      dim=batch_dim,		# pred h w l 
                      vel=batch_vel,		# pred velocity
                      point_cloud_range=self.point_cloud_range,		# point cloud range
                      voxel_size=self.voxel_size,						# voxel size
                      feature_map_stride=self.feature_map_stride,		# feature map stride
                      K=post_process_cfg.MAX_OBJ_PER_SAMPLE,			# max objects per sample
                      circle_nms=False,	# not implemented yet
                      score_thresh=post_process_cfg.SCORE_THRESH,		# score thresh
                      post_center_limit_range=post_center_limit_range	# center limit range
                  )
      ```

      前面的 input 就是为了生成 box，**后面四个参数是为了过滤得分低或者在视角范围外的选框**。output 是单个 head 的 batch 结果

      ```python
              ret_dict = [{
                  'pred_boxes': boxes_single_head,	# (num_boxes, 7 + C)
                  'pred_scores': scores_single_head,	# (num_boxes,)
                  'pred_labels': labels_single_head,	# (num_boxes,)
              } for k in range(batch_size)]
      ```

      pred_labels 会在之后经过 class_id_mapping_each_head 恢复为原始 id

   2. `model_nms_utils.class_agnostic_nms`，是对 batch 中的每个 sample 进行 nms，获得最终的final boxes。Input params 如下

      ```python
      def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
      ```

      NMS_CONFIG 大概长这样

      ```yaml
                  NMS_CONFIG:
                      NMS_TYPE: nms_gpu
                      NMS_THRESH: 0.2
                      NMS_PRE_MAXSIZE: 1000	# not useful
                      NMS_POST_MAXSIZE: 83
      ```

      output 有两个：1. selected index; 2. selected scores

学习点：

1. `torch.topk(tensor, k, dim)` 可以在某个维度选择 top k 个数据，返回 (topk_tensor, topk_index)
2. `torch.LongTensor` or `torch.Tensor `仅输入一个数字 n 创建一个 shape 为 (n,) 的 tensor

### reorder_rois_for_refining

由于每个 batch 获得最终的 boxes 个数是不一样的，而为了让这些 boxes 送到 roi head 中，需要将这些 boxes 合并为一个 tensor

1. input params

   ```python
       @staticmethod
       def reorder_rois_for_refining(batch_size, pred_dicts):
   ```

   pred_dicts，就是 generate_predicted_boxes 的输出

2. output

   ```python
           return rois, roi_scores, roi_labels
   # rois: (B, num_max_boxes, 7 + C)
   # roi_scores: (B, num_max_boxes)
   # roi_labels: (B, num_max_boxes)
   ```

   空 roi，使用0代替，并且 num_max_boxes 的最小值 clamp 为1，以避免程序错误

### forward

forward 把之前的组件串起来：

0. 获得 spatial features 2d
1. 使用 seperate heads 预测
2. 为每个 head，每个 batch，构建 targets
3. 更新 forward_ret_dict
4. 如果需要的话，生成预测选框，并构建 rois，然后更新 data_dict

