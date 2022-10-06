---
title: CPGNet
tags:
  - Point Cloud
categories:
  - papers
mathjax: true
abbrlink: 3338fc23
---

# CPGNet Code

## Useful Tools

### torchrun

[torchrun doc](https://pytorch.org/docs/stable/elastic/run.html) 根据文档，torchrun 变得更加优雅简单，并且能够很好地兼容 pytorch.distributed.launch

```shell
Single-node multi-worker

>>> torchrun
    --standalone
    --nnodes=1
    --nproc_per_node=$NUM_TRAINERS
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

在获取 local_rank 的时候直接从环境变量获取，而不从 args --local_rank 获取

```python
import os
local_rank = int(os.environ["LOCAL_RANK"])
```

其余可以直接参考之前的整理

### yaml

[菜鸟教程](https://www.runoob.com/w3cnote/yaml-intro.html) YAML 的语法和其他高级语言类似，并且可以简单表达清单、散列表，标量等数据形态。基本语法：

1. 使用缩进表示层级关系
2. '#'表示注释
3. 对象键值对使用冒号结构表示 **key: value**，冒号后面要加一个空格。也可以使用 **key:{key1: value1, key2: value2, ...}**
4. 以 **-** 开头的行表示构成一个数组。也可以使用 **key:{key1: value1, key2: value2, ...}**

### hydra & omegaconf

建议阅读 [hydra doc](https://hydra.cc/docs/intro/) 来了解如何使用 hydra，其底层是 [OmegaConf](https://mybinder.org/v2/gh/omry/omegaconf/master?filepath=docs%2Fnotebook%2FTutorial.ipynb)

1. [Basic Example](https://hydra.cc/docs/intro/#basic-example)

   ```yaml
   # conf/config.yaml
   db:
     driver: mysql
     user: omry
     pass: secret
   ```

   python codes

   ```python
   import hydra
   from omegaconf import DictConfig, OmegaConf
   
   @hydra.main(version_base=None, config_path="conf", config_name="config")
   def my_app(cfg : DictConfig) -> None:
       print(OmegaConf.to_yaml(cfg))
   
   if __name__ == "__main__":
       my_app()
   ```

   指定好 `config_path & config_name` 就能够把配置传入到程序当中了

2. [default list](https://hydra.cc/docs/tutorials/basic/your_first_app/defaults/#config-group-defaults) 可以使用 `defaults` 关键字添加子配置，此时子配置会融合到主配置当中，遇到相同关键字时取最后一个

   ```yaml
   # Directory layout
   ├─ conf
   │  └─ db
   │      ├─ mysql.yaml
   │      └─ postgresql.yaml
   |  └─ config.yaml
   └── my_app.py
   
   # db/mysql.yaml
   driver: mysql
   user: omry
   password: secret
   
   # config.yaml
   defaults:
     - db: mysql
   
   # multiple dir
   defaults:
    - db: mysql
    - db/mysql/engine: innodb
   ```

   如果希望能够改变子配置的内容，在 `defaults` 的最后加入 `_self_` 关键字

   ```yaml
   # config.yaml
   defaults:
     - db: mysql
     - _self_
   
   db:
     user: root
   ```

   并且能够通过 `@` 对 default list 中的 key 进行等效替代

   ```yaml
   defaults:
    - db@backup: mysql
   ```

   最后生成的配置文件中，关键字就从 `db` 变成了 `backup`

3. 利用 hydra 可以进行 multi run，便于调参，只需要调整 hydra 关键字

   ```yaml
   hydra:
     sweeper:
       params:
         db: mysql,postgresql
         schema: warehouse,support,school
   ```

4. [resolver](https://hydra.cc/docs/configure_hydra/intro/#resolvers-provided-by-hydra) provided by hydra 可以在 yaml 中生成一些常用的变量，最常用的是 **now** resolver `${now:%H-%M-%S}` 能够提供 `strftime` 格式时间，也可以直接调用 yaml 文件里的变量 `${CONFIG_KEY}` 进行引用，引用成员也是非常方便的，不论是字典还是序列

5. 一般 hydra 会自动创建一个 output dir 用于存放输出，可以自定义输出路径

   ```yaml
   hydra:
     run:
       dir: ./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
   ```

6. 可以使用 hydra 进行更加方便的 logging，会直接输出到 output dir

   ```python
   import logging
   
   # A logger for this file
   log = logging.getLogger(__name__)
   ```

7. 可以对 cfg 文件使用 `to_container` 将其转化为一个字典或者列表

## CPGNet

自底向上地整理 CPGNet 的各个模块

### CBN2d

```python
class CBN2d(nn.Module):
    """ Conv2d + BacthNorm2d + NoLinear """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True,
                 no_linear=nn.ReLU())
```

NoLinear 默认为 ReLU，但为了更加灵活还可以指定为其他函数，如 MaxPool2d

### MLP

简单的多层感知机

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, end_nolinear=False):
```

可以最后层是否使用非线性层 ReLU

### DualDownSamplingBlock & AttentionFeaturePyramidFusion

<img src="CPGNet Code/image-20220715114033695.png" alt="image-20220715114033695" style="zoom: 80%;" />

上面是示意图，简单来说 dual down-sampling block 就是更好的下采样层，而 attention feature pyramid fusion 是更好的上采样融合机制。完全可以替代常规的 backbone2d neck

这里实现 attention 的时候有点变化。预测多个 channel 而不是一个 channel，进一步增加权重的灵活性

```python
class DualDownSamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(DualDownSamplingBlock, self).__init__()

        self.cbn1 = CBN2d(in_channels, out_channels, stride=stride, no_linear=None)
        self.cbn2 = CBN2d(in_channels, out_channels, kernel_size=1, padding=0,
                          no_linear=nn.MaxPool2d(kernel_size=3, stride=stride, padding=1))

    def forward(self, inputs):
        outputs1 = self.cbn1(inputs)
        outputs2 = self.cbn2(inputs)
        
        return F.relu(outputs1 + outputs2)
    
class AttentionFeaturePyramidFusion(nn.Module):

    def __init__(self, lower_channels, higher_channels, out_channels, stride, output_padding):
        super(AttentionFeaturePyramidFusion, self).__init__()

        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(higher_channels, out_channels, 3, stride, 1, output_padding),
            CBN2d(out_channels, out_channels)
        )
        self.cbn = CBN2d(lower_channels, out_channels)
        self.attn = nn.Conv2d(2*out_channels, 2, 3, 1, 1)

    def forward(self, inputs_lower, inputs_higher):

        outputs_higher = self.up_sample(inputs_higher)
        outputs_lower = self.cbn(inputs_lower)

        attn_weight = torch.softmax(self.attn(torch.cat([outputs_higher, outputs_lower], dim=1)), dim=1)
        outputs = outputs_higher * attn_weight[:, 0:1, :, :] + outputs_lower * attn_weight[:, 1:, :, :]
        
        return outputs
```

### CPGFCN

```python
class CPGFCN(nn.Module):

    def __init__(self, 
                 in_channels=64, 
                 encoder_channels=[32, 64, 128, 128], 
                 stride=[2, 2, 2, 1],
                 decoder_channels=[96, 64, 64], 
                 output_padding=[1, 1, 1]):
        super(CPGFCN, self).__init__()
```

类似于 U-Net 的风格，获得全分辨率特征图谱

<img src="CPGNet Code/image-20220715154108384.png" alt="image-20220715154108384" style="zoom: 80%;" />

先用 DualDownSamplingBlock 进行下采样 然后再使用 AttentionFeaturePyramidFusion 进行上采样融合。实现依然和上图有出入，skip connection 是错位的，因为 attention fusion 是需要 low resolution & hight resolution 特征图谱作为输入的

### Projection

这是一个功能类，用于计算投影过后的点云坐标，并且把范围之外的点云给过滤掉。投影分为两个类：BEV 投影和 range view 投影

1. BEV 投影

   ```python
       def init_bev_coord(self, coord, eps=0.1):
           """Convert the coordinates to bev coordinates and save them as member variables
           
           Param:
               coord: points coordinates with shape :math:`(P, 4)` that contain (batch_index, x, y, z)
               eps: Prevent coordinate out of bounds of `self.bev_shape`
   
           Return:
               bev_coord: bev coordinate in valid range with shape :math:`(P_reduce, 3)`, that contain (batch_index, u, v)
               keep: mask of points in valid range with shape :math: `(P, )`
           """
           
       def p2g_bev(self, points, batch_size):
           """
           Param:
               points: with shape :math:`(P_reduce, D)`, where P_reduce is valid points number, D is feature dimention.
               batch_size: batch size. 
           
           Return:
               bev_map: with shape :math:`(N, C, H, W)` where N is batch size, H and W is bev feature map. 
           """
       def g2p_bev(self, bev_map):
           """ reverse of p2g"""
   ```

   主要有3个计算函数，一个用于计算坐标，一个用于形成 (N,C,H,W) 形状张量，最后一个用于从特征图谱中挑选需要的特征。`P_reduce` 意思是过滤了范围之外的点。后者再形成张量时使用了 `scatter_max` 方法，可以参考 [pytorch scatter](https://pytorch-scatter.readthedocs.io/en/1.3.0/functions/max.html)，index 和 source 在 dim 轴上的长度必须保持一致。在之后还需要在特征图中 `gather` 需要的 feature，使用的是 bilinear 插值

2. range view 投影也是类似的，计算过程和论文里的公式是类似的

   <img src="CPGNet Code/image-20220715155721631.png" alt="image-20220715155721631" style="zoom: 80%;" />

   注意这里的 arctan 函数应该使用 `torch.atan2` 方法，其值域为 $[-\pi, \pi]$，而这里的 $f_{up} \& f_{down}$ 是不好理解的，但代码更好理解

   ```python
        u = 0.5 * (1 - phi / math.pi) * w_range
        v = (1 - (theta - v_down) / (v_up - v_down)) * h_range
   ```

### PGFusionBlock & CPGNet

这里就是所有的模块组合起来的地方，将 BEV 特征和 range 特征融合，融合使用的是简单的 MLP

CPGNet 就是 PGFusionBlock 再加上一个 linear layer 预测

转换为可迭代的，以加强鲁棒性

```python
modes = [modes] if not isinstance(modes, collections.abc.Iterable) else list(modes)
```

## DistTrain & Optimizer

1. `dist.get_rank()` 获得当前 rank，`dist.get_world_size()` 可获得使用 GPU 数量（如果是单节点）

2. `dist.barrier()` 一般不会使用，当某个进程需要大量时间去单独做事情的时候会使用一下比较保险 [pytorch](https://pytorch.org/docs/stable/distributed.html#torch.distributed.barrier)

3. `dist.all_reduce()` [pytorch](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce) 可以把多个进程的结果合并起来，默认是 SUM 操作

4. [AdamW pytorch](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) 区别于 [Adam pytorch](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)，其实很简单，[AdamW](https://paperswithcode.com/method/adamw) 就是想要把 weight decay 从 EMA 当中分离出来，从而更好提供更新梯度

5. 如果想要自定义 scheduler 只需要继承 `_LRScheduler` 然后重写 `get_lr` 方法返回一个列表即可，该列表为从 0~epoch 的 lr 值。下面是 `LamdaLR` 的精简代码。如果想要获得当前 lr，可以使用 `[group['lr'] for group in self.optimizer.param_groups]`

   ```python
   class LambdaLR(_LRScheduler):
       """Sets the learning rate of each parameter group to the initial lr
       times a given function. When last_epoch=-1, sets initial lr as lr.
       """
   
       def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
           self.optimizer = optimizer
           super(LambdaLR, self).__init__(optimizer, last_epoch, verbose)
   
       def get_lr(self):
           if not self._get_lr_called_within_step:
               warnings.warn("To get the last learning rate computed by the scheduler, "
                             "please use `get_last_lr()`.")
   
           return [base_lr * lmbda(self.last_epoch)
                   for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]
   ```

   一个 [kaggle](https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook) 博客，可以借鉴里面的代码把 lr 画出来
