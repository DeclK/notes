# PillarNet code

对 PillarNet 感兴趣的代码进行整理

## SpMiddlePillarEncoder

有两个重要模块：

1. Sparse2DBasicBlockV
2. Sparse2DBasicBlock

如果了解 OpenPCDet 中的 `SparseBasicBlock` 理解这两个就会非常顺畅。其中 `replace_feature` 是为了操作 sparse tensor 内的 torch tensor，直接将 sparse tensor 经过 nn.ReLU 会报错

这两个模块就是残差模块的实现，`Sparse2DBasicBlockV` 要比 `Sparse2DBasicBlock` 多一个 conv 用于转换维度，便于残差相加。下面就放后者的核心代码

```python
class Sparse2DBasicBlock(spconv.SparseModule):
    def __init__(
        self, inplanes, planes, stride=1, dilation=1, norm_cfg=None, indice_key=None):
        # planes are channels, dilation is padding
        super(Sparse2DBasicBlock, self).__init__()
        bias = norm_cfg is not None
        self.conv1 = spconv.SparseSequential(
            conv2D3x3(inplanes, planes, stride, dilation=dilation, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1])

        self.conv2 = spconv.SparseSequential(
            conv2D3x3(planes, planes, indice_key=indice_key, bias=bias),
            build_norm_layer(norm_cfg, planes)[1])

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.relu(out.features))
        out = self.conv2(out)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out
```

