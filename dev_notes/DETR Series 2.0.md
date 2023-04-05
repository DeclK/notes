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

## DINO Improves

### Box Refinement

Box refinement算法是一种迭代微调策略，它类似于Cascade R-CNN方法，可以在每个解码器层上对边界框进行微调，所以在创建的 `self.box_embed & self.class_embed` 是各自独立的，不共享参数

### Look Forward Twice

实际上就是把 box refinement 中的 reference points 没有从中 detach 出来（一行代码就行），实际上就是增加了梯度的计算复杂度以提升效果

### Two Stage

两阶段方法就是利用 encoder 提出 proposal 作为 query 的 reference points

由于 two stage 的出现，所有的 reference points 由2维的点，变成了4维的选框，这在 deformable attention 里有做简要处理，但重点仍然还是选框的中心

代码中的 valid ratios 应该可以移除，甚至可能是有害的，并且代码里有一些 bug 没有修复，先除以后乘以，基本上抵消了

### Mixed Query Selection

也很简单，就是只要 proposal 的位置作为 reference points，不用 proposal 作为 query。真正的 content query 依然是 learnable parameters

### Contrastive Denoising

代码逻辑理顺，把每一个小块的目的描述出来

用伪代码的形式整理DINO

## More

1. 中间损失函数的计算
2. 第一阶段损失函数的计算
3. COCO dataset