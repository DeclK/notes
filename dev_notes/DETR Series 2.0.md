# DETR Series 2.0

这一次必定拿下 DETR 系列以及 Deformable DETR 以及 Vision Transformer 的构建过程🤔

一开始整理都是以一种线性的思维在做，看一点，整理一点。这样可能整理完过后没有一个逻辑的总结，回看时也比较空洞，但这也是没办法的事情，毕竟是第一次整理

于是在第二遍进行总结时就需要注意总结的逻辑通顺问题

这篇笔记所有的代码都参考自 detrex 以保证统一性

## Blocks

### MultiHead Attention

Attention 的功能就是让 query 去询问 (key, value) 然后加权输出
$$
\operatorname{softmax}\left(\frac{\mathbf{Q K}^{\top}}{\sqrt{d}}\right) \mathbf{V} \in \mathbb{R}^{n \times v}
$$
在 detrex 当中直接调用了 `nn.MultiheadAttentnion` 并加入 identity connection & posotional embedding，完成了整体包装 `MultiheadAttention`

```python
    def forward(
        self,
        query: torch.Tensor,
        key = None,
        value = None,
        identity = None,
        query_pos = None,
        key_pos = None,
        attn_mask = None,
        key_padding_mask = None,
    ) -> torch.Tensor:

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query

        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]
		# self.proj_drop = nn.Dropout(r)
        return identity + self.proj_drop(out)
```

在使用 `nn.MultiheadAttentnion` 时有两个技巧和两个注意点：

1. `attn_mask`：其形状为 `(num_query, num_key)`，是 bool 矩阵，用于去除不希望计算的部分元素
2. `key_padding_mask`：其形状为 `(bs, num_query)`，用于忽略不需要计算的 key，本质上可由 `attn_mask` 替代，但是该接口更方便
3. 可以进行两次 dropout，一次在 attentnion，一次在 linear projection
4. Positional embedding 仅对 query 和 key 存在，对 value 不存在

这里放一个 `MultiHead-SelfAttention` 的无掩码实现版本，是面试常考，参考 timm 的实现

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # qkv: (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

Self-Attention 的 qkv 都是一样的，如果要做 cross-attentnion 只要把那个大的线性层 `self.proj` 换成3个 qkv 各自的线性层即可 

### Deformable Attention

Deformable Attention 可以说跟 Attention 相差很大，基本上没什么联系🤣其本质就是可变卷积！

用语言来简要形容：

1. 给定 query，通过 query 计算 sampling offsets & attention weights
2. 给定 reference points for query，通过 reference points + sampling offsets 获得采样点位置
3. 给定 value，通过采样点和 attention weights 获得最终输出

其中需要考虑 multi-scale value 的情况，规定 reference points 在各个 scale 的采样点相同（有没有可能直接增加采样点是一样的效果），可简答理解为如下图，其特征形状为 (B, N1+N2+N3+N4, C)

<img src="DETR Series 2.0/image-20230403214527125.png" alt="image-20230403214527125" style="zoom:50%;" />

要完成 multi-scale 的功能需要知道每一个 scale 的 pixel 数量，然后使用循环完成对每一个 scale 的询问

```python

class MultiScaleDeformableAttention(nn.Module):
    """ Modified from Deformable-DETR for 1D
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_levels: int = 1,
        num_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(
        self,
        query,	                       # (B, N, C)
        spatial_shapes = None,         # (levels, 2), sum is sequence len N
        reference_points = None,       # (B, N, level, 2)
        value = None,
        identity = None,
        query_pos = None,
        **kwargs):
        
        if value is None:	# True, when self-attetion
            value = query
        if identity is None:	# True
            identity = query
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)

        value = value.view(bs, num_value, self.num_heads, -1)
        
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )
        
        # bs, num_query, num_heads, num_levels, num_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        
        output = multi_scale_deformable_attn_pytorch(
                 value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return self.dropout(output) + identity
```

在使用 Deformable Attention 时有两个特点和两个疑问：

1. 整个计算过程没有 `key` 的存在！仅由 `query` 和 `value` 计算
2. 通过约束采样点的数量，deformable attention 可以显著减少计算量
3. 为什么要对 reference points 进行这样的归一化？可能是希望对尺寸有一定感知
4. sampling offsets 仅由 query 产生，没有与 key 进行交流，这会不会损失一定性能

输入中仍有两个未说明的变量：

1. 如何生成 reference points
   1. 在 self-attention 中，reference points 即为每个 pixel 本身的坐标
   2. 在 cross-attention 中，reference points 有两种方法：
      - 使用一个 `nn.Linear(dim, 2)` 处理 query，然后再 sigmoid 生成
      - 在两阶段中，来源于第一阶段的 proposal
2. 如何生成 query positional embedding
   1. 在 self-attention 中，使用常规的 sin positional embedding + learnable level embedding
   2. 在 cross-attention 中，可不使用 positional embedding，因为有了 reference points。但加入之后效果更佳，即使用一个 `nn.Linear(dim, dim)` 对 reference points 进行编码，这就是 Dynamic Anchor 的主要升级（两行代码就行）

### 构建Transformer

构建 transformer 还要有一个环节，就是 FFN，即两个线性层

```python
class FFN(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        feedforward_dim=1024,
        output_dim=None,
        num_fcs=2,
        activation=nn.ReLU(inplace=True),
        ffn_drop=0.0,
        fc_bias=True,
        add_identity=True,
    ):
        super(FFN, self).__init__()
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.num_fcs = num_fcs
        self.activation = activation

        output_dim = embed_dim if output_dim is None else output_dim

        layers = []
        in_channels = embed_dim
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_dim, bias=fc_bias),
                    self.activation,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_dim
        layers.append(nn.Linear(feedforward_dim, output_dim, bias=fc_bias))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity

    def forward(self, x, identity=None) -> torch.Tensor:

        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out
```

注意，这里的 `FFN` 依然保留了 identity connection，和上述 attention 一样。下面就可以通过组合各个网络层来搭建 transformer block 啦😎

一般来讲 transformer block 就两种形式

```python
operation_order=("self_attn", "norm", "ffn", "norm")
operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm")
```

前者为 encoder，后者为 decoder，norm 层为 `nn.LayerNorm`

## Deformable DETR 流程

### init

需要构架的模块如下：

1. `self.backbone & self.neck` 用于提取图像特征，通常由 resnet or swin 提取，然后用卷积将各个分辨率的输出统一到相同 channel 数量
2. `self.position_embedding` 用于产生图像位置嵌入
3. `self.transformer` 即 Deformabel DETR 的核心模块，负责编码和解码
4. `self.class_embed & self.bbox_embed` 用于预测类别和位置残差，其中 `class_embed` 为一层 Linear，`bbox_embed` 为3层 MLP。这里有两个区别：
   1. 如果有 box refine trick，则每一个 decoder 的中间输出使用独立的 `class_embed & bbox_embed`，用 `copy.deepcopy` 完成复制；
   2. 如果有 two stage，需要多余一个 `class_embed & bbox_embed` 通过 encoder 输出获得 proposal
5. `self.criterion` 用于计算损失函数

### forward

1. 图像预处理，获得 $(B, 3, H, W)$ 的 `ImageList`，并记录了每一张图缩放前后的 image size

2. 创建 `image_masks` 用于后面进行 `query_key_padding_mask`

3. 创建 `multi_level_positional_embeddings`

4. 初始化 `query_embeds`：

   1. 如果为两阶段，query 是由第一阶段的 proposal 产生，所以初始化为 None
   2. 如果为单阶段，query 则由 `nn.Embedding(num_query, dim)` 产生，这里也说明 `nn.Embedding.weight` 能够简单替代 `nn.Parameter`

5. 将图像输入到 transformer 当中获得 logits

   ```python
           (
               inter_states,
               init_reference,
               inter_references,
               enc_state,
               enc_reference,
           ) = self.transformer(
               multi_level_feats, multi_level_masks, multi_level_position_embeddings, query_embeds
           )
   ```

   各个输出分别代表：

   1. `inter_states` decoder 各个层输出的 logits
   2. `init_reference` 为第一阶段产生的 proposal/reference points (+ denoising ground truth 如果为 DINO
   3. `inter_references` decoder 各个层输出的 proposal/reference points
   4. `enc_state` 第一阶段产生的 logits
   5. `enc_reference` 为第一阶段产生的 proposal/reference points，与 `init_reference` 等价！

6. 再把 decoder 的中间输出又预测一遍用于计算损失。这是因为 decoder 中间输出的 reference points 是 detached tensor，所以不能直接计算梯度

7. 计算每一层的损失

## DINO Improves

### Box Refinement

Box refinement算法是一种迭代微调策略，它类似于 Cascade R-CNN 方法，可以在每个解码器层上对边界框进行微调，所以在创建的 `self.box_embed & self.class_embed` 是各自独立的，不共享参数

### Look Forward Twice

实际上就是把 box refinement 中的 reference points 没有从中 detach 出来（一行代码就行），实际上就是增加了梯度的计算复杂度以提升效果

### Two Stage

两阶段方法就是利用 encoder 提出 proposal 作为 query 的 reference points

由于 two stage 的出现，所有的 reference points 由2维的点，变成了4维的选框，这在 deformable attention 里有做简要处理，但重点仍然还是选框的中心

代码中的 valid ratios 应该可以移除，甚至可能是有害的，并且代码里有一些 bug 没有修复，先除以后乘以，基本上抵消了

### Mixed Query Selection

也很简单，就是只要 proposal 的位置作为 reference points，不用 proposal 作为 query。真正的 content query 依然是 learnable parameters

### Contrastive Denoising

Denoising 思想非常简单：将经过噪声处理的 gt 作为 query，输入到 decoder 当中去重建 gt。其准备过程如下：

1. 确定 `dn_groups`，就是每一个 gt 需要多少个噪声选框。一个 group 由 positive 和 negative 两个部分组成

2. 创建 `known_labels & known_bbox` 其形状为 `labels & bbox` 在第一个维度重复 $(dn\_groups\times 2\times num\_gt)$。2代表 pos & neg，实际上 negative 区别于 positive 噪声就是 box 的缩放更大一些

3. 对 labels 进行随机噪声，即将部分类别随机替换为其他类别

4. 对 boxes 进行随机噪声，即对选框进行随机位移和缩放

5. 创建 `input_query_label & input_query_box`：

   1. `input_query_label` 形状为 $(B, dn\_groups \times2\times pad\_size, C)$，其中 `pad_size` 是一个 batch 所有样本中 gt 数量的最大值，`C` 为 embed dim (= 128)，使用一个 `nn.Embed` 进行转换
   2. `input_query_batch` 形状为 $(B, dn\_groups \times2\times pad\_size, 4)$，作为可变注意力的 reference points

6. 创建 attention mask，因为 gt 噪声不能被真正的 query 所看见，但是 gt 噪声可以看见真正的 query，各个 gt 噪声 groups 之间不能相互看见，最后的 mask 形状可如图所示

   <img src="DETR Series 2.0/image-20230407151520353.png" alt="image-20230407151520353" style="zoom:50%;" />

   灰色部分 `attention_mask=True`

并且由于 `pad_size` 的存在，在之后计算损失时，会有零填充的 query 做出预测结果，我认为需要把这些预测从损失计算中去除，但是源代码中没有做这一步，可能是因为影响不大？

## Loss

这一部分我要详细整理一下代码，是非常通用的结构

### Matcher

整体思路为：利用匈牙利匹配法获得最小损失匹配。关键在于计算损失矩阵 Cost Matrix

1. 使用 cross entropy style 或者 focal loss style 来计算分类损失，着重理解 cross entropy style 就好

   ```python
        if self.cost_class_type == "ce_cost":
            cost_class = -out_prob[:, tgt_ids]
            
        elif self.cost_class_type == "focal_loss_cost":
            alpha = self.alpha
            gamma = self.gamma
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
   ```

   这里 `out_prob` 即为预测的可能性，其形状为 $(B\times num\_queries, num\_classes)$，通过取得 `tgt_ids` 来获得对应类别的损失

   并且这里的 `focal_loss` 似乎是计算错了，[issue](https://github.com/IDEA-Research/detrex/issues/196) 也没有很好的回复，说是借用的 deformabel detr 的源代码

2. 计算 L1 距离和 `generalized_box_iou` 

   ```python
           # Compute the L1 cost between boxes
           cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
   
           # Compute the giou cost betwen boxes
           cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
   ```

   GIoU  `return iou - (area - union) / (area + 1e-6)`

   **需要注意的是，这里将所有 batch sample 都合到一块了！在后面用切片解决** 

3. 利用加权计算损失矩阵

   ```python
           # Final cost matrix
           C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
           C = C.view(bs, num_queries, -1).cpu()
   ```

4. 匈牙利匹配

   ```python
           sizes = [len(v["boxes"]) for v in targets]
           indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
   ```

   然后将 `indices` 转为 tensor，`indices` 本身为一个 list of tuple (index_i, index_j)

   ```python
           return [
               (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
               for i, j in indices
           ]
   ```

### Last Layer Loss

#### bboxes

为了计算损失，首先应该通过 matcher 算出的 index 获得匹配的 boxes，相当于再 concat 起来，并加上 batch idx

```python
        ...
    	idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
```

有了匹配结果过后就直接计算 L1 和 GIoU

```python

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
```

#### classification

分类也是一样的，先计算匹配的标签，然后构造 one hot 向量，最后计算 focal loss

```python
    def loss_labels(self, outputs, targets, indices, num_boxes):

        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],	# shape (B, num_queries)
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        # Computation classification loss
        if self.loss_class_type == "ce_loss":
            loss_class = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes, self.empty_weight
            )
        elif self.loss_class_type == "focal_loss":
            # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
            # target_classes_one_hot = (2, 300, 80)
            target_classes_onehot = torch.zeros(
                [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                dtype=src_logits.dtype,
                layout=src_logits.layout,
                device=src_logits.device,
            )
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            loss_class = (
                sigmoid_focal_loss(
                    src_logits,
                    target_classes_onehot,
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                * src_logits.shape[1]
            )

        losses = {"loss_class": loss_class}
```

构造 One hot 向量可以用 scatter 也可以用 `F.one_hot`

#### full loss

完整的 api 调用如下

```python
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
            
            
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "class": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
```

### Aux Loss

中间层损失输出就是 Last Layer Loss 的循环，完全一致！

```python
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
```

### Two Stage Loss

依然也是同样的损失计算

```python
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            if self.two_stage_binary_cls:
                for bt in targets:
                    bt["labels"] = torch.zeros_like(bt["labels"])
            indices = self.matcher(enc_outputs, targets)
            for loss in self.losses:
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)
```

不过可以设置 `binary_cls`，这就完全靠齐了传统的两阶段方法：第一个阶段只预测前景，不预测类别。**这里直接将标签全部设置为 0 就可完成该目标！**实际上第一阶段的最重要作用还是提供 reference points 所以这个类别不重要

### DN Loss

DN 唯一不同的是不需要 Matcher 进行匹配，其正负样本都已经分配好了

```python
target_idx = [[0, 1, ..., n],
              [0, 1, ..., n]
              ...(repeat dn_num times)
              [0, 1, ..., n]]	# n ground truth, (dn_num, gt_num_all)
output_idx = (torch.tensor(range(dn_num)) * single_padding	# each starting point
                    ).long().cuda().unsqueeze(1) + t	# (dn_num, gt_num_all)

dn_idx = (output_idx, target_idx)	# match indices
              
            for loss in self.losses:
                losses.update(
                    self.get_loss(loss, output_known_lbs_bboxes, targets, dn_idx, num_boxes * dn_num))
              
              # aux loss
              for i in range(aux_num):
              	...
```

并且该分配结果对中间层的预测结果依然如此，从始至终不改变！
