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

Transformer 大致结构如下

<img src="C:\Data\Projects\notes\dev_notes\DETR\image-20221215140623383.png" alt="image-20221215140623383" style="zoom:50%;" />

图中 Decoder 这边的结构是不够完整的，或者说不够正确的。实际上 Decoder 关于 query 的输入有两个：1

1. Query itself
2. Query positional encoding, **i.e. Object Query**

图中没有把 Query 本身给画出来，论文里是**初始化为0**，所以图中直接省略不画了，导致 Decoder 下侧的 `+` 号意义不清晰

### 一些结论

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

### Detection Loss

匹配完成后，就可以计算损失函数
$$
\mathcal{L}_{\text {Hungarian }}(y, \hat{y})=\sum_{i=1}^{N}\left[-\log \hat{p}_{\hat{\sigma}(i)}\left(c_{i}\right)+\mathbb{1}_{\left\{c_{i} \neq \varnothing\right\}} \mathcal{L}_{\text {box }}\left(b_{i}, \hat{b}_{\hat{\sigma}}(i)\right)\right]
$$
论文提到在计算分类损失时，对于空类 gt $\varnothing$ 的分类损失要除以 10 用于平衡正负样本，在实现上是直接指定 `F.cross_entropy(..., weight=)` 完成

另外还是使用了中间监督，或者称为辅助损失。即把 decoder 的中间 block 的输出也作为预测结果，并计算检测损失

swin v2

DC5，是 dilated convolution at resnet stage 5 的一个简称，在 DETR 中使用的具体描述为

> Following [21, FCN], we also increase the feature resolution by adding a dilation to the last stage of the backbone and removing a stride from the first convolution of this stage.

但是在后面的 DETR-based 目标检测器中没有常用

### DETR 问题

大物体和小物体效果相差大

object query 的预测结果可视化

## Deformable DETR & Deformable Attention

### Multi-Scaele Deformable Attention

一开始看 Deformable DETR 的时候，注意力很容易集中到 Deformable 上，但我觉得 Multi-Scale 同样非常重要。为了彻底弄清 Multi-Scale Deformable Attention，我想从如下几个问题入手

1. 如何表示 multi-scale feature
2. 如何表示 multi-scale feature's positional embedding 
3. 如何表示 reference points
4. 如何完成 multi-scale deformable attention

#### Multi-scale Feature

这非常好理解，就是从 backbone **ResNet 50** 中输出的中间特征层，越深的特征层，分辨率越小

```python
# output feature dict
features = self.backbone(images.tensor)

# project backbone features to the reuired dimension of transformer
multi_level_feats = self.neck(features)
```

最终 neck 把这些特征图谱映射到同一个维度，以输入到 transformer 中做点积

<img src="C:\Data\Projects\notes\dev_notes\DETR\image-20221216190202156.png" alt="image-20221216190202156" style="zoom:50%;" />

#### Multi-scale Positional Embedding

**Encoder 阶段**

直接处理 multi-scale positional embeddings，就是逐个 level 获得

```python
for feat in multi_level_feats:
    multi_level_masks.append(
        F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
    )
    multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))
```

解释：positional embedding 是根据一个 2D mask 直接生成的，对于不同 scale 的 mask 是直接根据 img mask 插值获得（img mask 中的非零值即表示该像素点被忽略） 

为了对不同 scale 的 positional embedding 进行区分，再加入 scale embedding

```python
self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))
# for each scale, pos_embed (B, H*W, C)
for lvl, pos_embed in enumerate(multi_level_pos_embeds):
	lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
```

**Decoder 阶段**

上述的 multi-scale positional embedding 是给 encoder query 使用，在 decoder 中 query 不再是复杂的多尺度特征图谱，而就是一般的 embedding，所以使用的 query positional embedding 就是 DETR 中的 object query，也是一般的 embedding（所谓一般，指的是非预设，如 sine embed）

#### Reference Points

**Encoder 阶段**

reference points 就是每个像素点中心的**归一化坐标**。每一个 scale 的 reference points 为一个张量，形状为 (H, W, 2)，那么多个 scale 的 reference points 合起来应该是 `(B, h1w1 + h2w2 + ..., 2)` 才对，但实际上的代码并不是这么做的

```python
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Args:
            spatial_shapes (Tensor): The shape of all feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid points on the feature map, has shape (bs, num_levels, 2)
        Returns:
            Tensor: reference points used in decoder, has shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)   # (1, HW) / (B, 1) -> (B, HW)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)   # (B, HW, 2)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # (B, N1+N2+..., 2)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
```

可以看到最终的输出形状是 `(bs, num_keys, num_levels, 2)`，**实际上这是为了各个 scale 之间的交互**，即某个 scale 的 reference point 可以去另一个 scale 进行采样

**Decoder 阶段**

由于 Decoder 中的 query 是一般的 embedding，所以不可能像上述一样生成与位置强相关的 reference points，就只能用一个简单的线性层对 embedding 进行转换

```python
self.reference_points = nn.Linear(self.embed_dim, 2)
```

然后再 sigmoid 进行归一化。但之后为了解决这个问题就使用了两个技巧：query selection & iterative box refinement

#### MSDeformAttn

下面正式介绍 multi-scale deformable attention 模块，先简单描述下代码干了什么事情

1. 判断是否为自注意力，如果没有传入 value 则使用 query 本身

2. 给 query 加入 positional embedding，**注意，这里没有给 key 加入 positional embedding，更确切地说，在 deformable attetion 里没有 key 的概念，key 极其 attention 是通过其他方式获得**

3. value 经过一个线性层，维度不变，并且 mask 掉不需要进行注意力的点。然后再 view 为多头的形式

4. 将 query 送到 `self.sampling_points` 线性层，进行偏移量预测，也对偏移量进行归一化

   ```python
   self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
   ```

5. 将 query 送到 `self.attention_weight` 线性层，并用 softmax 计算注意力分数

   ```python
   self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
   ```

6. 有了前面的准备工作，就能够愉快计算可变注意力了，pure pytorch 实现为 `multi_scale_deformable_attn_pytorch`，注意这里的输出已经合并了多头

7. 获得的结果再过一个线性层，维度不变

由于代码过于 mmlab 的风格🤣，我删除了一些，以方便理解

```python
    def forward(
        self,
        query: torch.Tensor,	# (B, N, C)
        key = None,
        value = None,
        identity = None,
        query_pos = None,
        key_padding_mask = None,
        reference_points = None,
        spatial_shapes = None,
        level_start_index = None,
        **kwargs) -> torch.Tensor:
        
        if value is None:	# True, when self-attetion
            value = query
        if identity is None:	# True
            identity = query
        if query_pos is not None:	# True
            query = query + query_pos
        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        
        if key_padding_mask is not None:	# True
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(bs, num_value, self.num_heads, -1)
        
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        # bs, num_query, num_heads, num_levels, num_points, 2
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = (
            reference_points[:, :, None, :, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
        
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)
        
        output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)

        return self.dropout(output) + identity
```

pure pytorch 代码块如下，比较难看的是张量的形状，我都以注释给出，应该比较好理解。用简洁的语言概括为：

1. 对每一个 scale/level，计算所有 sampling points 在该 level 插值得到的特征向量
2. 对插值得到的 multi scale + multi points 的特征向量进行注意力加权整合
3. 合并多个 head 的特征向量

```python
def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims -> bs*num_heads, embed_dims, H_, W_
        value_l_ = (
            value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        )
        # bs, num_queries, num_heads, num_points, 2 -> bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) -> (bs*num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = ( # (bs*num_heads, embed_dims, num_quries, num_levels*num_points)
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights) 
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()
```

### Decoder

看完了 Deformable Attention，实际上在 Decoder 上的东西也很不一样。从此开始 Decoder 不再保持原始 DETR 的简洁性，为了让收敛更快，我们需要使用更强的先验假设

1. query selection
2. encode new query

#### Query Selection

感觉有 beam search 的意思：不需要每次对全部的选择进行暴力搜索，而基于排名靠前的选择继续进行搜索。这个方法也被称为 DETR two stage，两阶段方法。**Query 现在不是来自于随机初始化的 embedding，而是来自于 Encoder Output + Preset Anchor（没错，anchor is EVERYWHERE!）**

Query Selection 显然需要完成两件事：

1. 生成 query / proposal。方法是基于预设 anchor 的 proposal 预测。每一个预设 anchor 对应一个 encoder output pixel，然后生成一个选框极其对应分类。所以说这里的 **query 就是 proposal**
2. 排序 query / proposal。得分高 proposal 的排序靠前，并注意分类任务为二分类，即只分前景和背景，但这个二分类任务完成得很妙，和原来的 80 类直接进行了一个统一：直接把所有的 label 标签指定为0，即可完成二分类

完成上述任务需要调用下方函数 `gen_encoder_output_proposals`，这里仅给出简单注释

```python
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        """
        Args:
            - memory: output of encoder, (B, S, C)
            - memory_padding_mask: (B, S)
            - spatial_shapes: (B, 2)
        Return:
            - output_memory: **MASKED** and projected new memory (B, S, C)
            - output_proposals: reference point **UNsigmoid** anchors (B, S, 4)
        """
```

此时还得到一个好处，既然我们的 query 已经和位置相关了，那么其 reference points 也能够直接使用该 proposal 的位置和大小

另外一点，对于初始化的 anchor，网络对其值并不敏感

#### Iterative Box Refinement

这里感觉是残差，或者 step by step 的思想，整个 trick 非常对我的口味👍我不了解 Diffusion Model，不知道这种 step by step 是不是类似的

在 DETR 中 decoder 每一层都会去预测最终的选框，每一层的预测可能差距非常大，这样的训练过程显然不够稳定。问题提出来了，改进方法也是跃然纸上：要求每一层的预测是基于上一层的预测，这样就只预测一个变化量

同时，我们还可以根据这个预测去更新我们的 reference points，因为预测的框会朝着 gt 去探索，reference points 与其一起更新会收敛更快，而不是从头到尾使用同一套 reference points，并且此时 referece points 为 reference box，最后一个维度是 4

注意，每个 scale/level 的 `bbox_embed or class_embed` 都是独立的

```python
for layer_idx, layer in enumerate(self.layers):
    output = layer(q, k, v, ...)
    
    tmp = self.bbox_embed[layer_idx](output)
    new_reference_points = tmp + inverse_sigmoid(reference_points)
    new_reference_points = new_reference_points.sigmoid()
    
	reference_points = new_reference_points.detach()	# no detach is better!
    
    intermediate.append(output)
    intermediate_reference_points.append(reference_points)
```

我觉得可以直接对 reference points 计算损失，省略计算，但原代码重新计算了

```python
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            tmp += reference
            
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
```

## DeNoising Training

TODO 可以理解为辅助任务

## Q

### 为什么不需要 NMS

难道说 DETR 的误检率到底如何？

### Encoder 中也加入监督信号

不知道有没有类似的工作，感觉像中学的阅读理解，会先看一下问题，然后再去看文章，带着问题去阅读

Forward-Forward
