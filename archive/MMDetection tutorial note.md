---
title: MMDetection tutorial note
tags:
  - MMLab
  - MMDetection
categories:
  - 编程
  - OpenMMLab
abbrlink: 71f4e62
date: 2021-09-14 14:53:21
---

# MMDetection tutorial note

## Config

由于 config 是有继承机制的，所以想要查看完整的 config，可以用官方提供的 `print_config.py` 函数来查看

```shell
python tools/misc/print_config.py /PATH/TO/CONFIG
```

### 修改 config 

在使用 `tools/train.py` or `tools/test.py` 时，可以通过 `--cfg-options` 来修改此次运行的 config 而不修改 config 文件

更改字典，列表，元组都可以，只需要按照规范传入参数即可，下面给几个例子

1. 更改字典 `--cfg-options model.backbone.norm_eval=False`

2. 更改列表中的字典 `[dict(type='LoadImageFromFile'), ...]` 

   `--cfg-options data.train.pipeline.0.type=LoadImageFromWebcam`

3. 更改列表，元组 `workflow=[('train', 1)]`

   `--cfg-options workflow="[(train,1),(val,1)]"` 注意必须要加引号，且内部不能有空格

### config 结构

在 `config/_base_` 中有4个基本的组成部分

1. dataset
2. models: backbone, neck 是必须有的
3. schedule
4. default_runtime

更详细的要求和例子请直接查看 [文档](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html#config-name-style)

config 的结构其实就给学习 MMDetection 有一个明确的指导，重点就是这四个部分，怎么建立数据集、模型，训练过程中的策略是什么，在训练过程中需不需要做些其他记录和处理...

## Customize Datasets

数据可以分为两个部分处理，一个是数据本身，另一个是数据的标签信息 `annotation_file`，所以想要建立个性化的数据集，处理好这两个部分即可

对于数据本身，比如图片数据、点云数据...其实能够做的处理是比较少的，更多的是做预处理比如 data augmentation，但这些操作就不在这一部分进行，而是到下一节 Data Piplines 中进行

### 规范 annotation_file 

现在重心放在处理数据的标签信息上，核心就是把标签信息处理成为一种规范的格式就可以使用了，在 MMDetection 中 COCO format 是一种推荐的格式。COCO format 是一个 json 文件，文件中包含了整个数据集每一个样本的标签信息，其中有3个大的关键字：

- `images`: contains a list of images with their informations like `file_name`, `height`, `width`, and `id`.
- `annotations`: contains the list of instance annotations.
- `categories`: contains the list of categories names and their ID.

形如下面的内容

```json
'images': [
    {
        'file_name': 'COCO_val2014_000000001268.jpg',
        'height': 427,
        'width': 640,
        'id': 1268
    },
    ...
],

'annotations': [
    {
        'segmentation': [[192.81,
            247.09,
            ...
            219.03,
            249.06]],  # if you have mask labels
        'area': 1035.749,
        'iscrowd': 0,
        'image_id': 1268,
        'bbox': [192.81, 224.8, 74.73, 33.43],
        'category_id': 16,
        'id': 42986
    },
    ...
],

'categories': [
    {'id': 0, 'name': 'car'},
 ]
```

###  更改 config file 以匹配

更改 config file 中的 dataset/data 配置来匹配 `annotation_file`，需要明确的是三个部分：

1. `annotation_file_path` & `image_path`
2. `classes` 数据集有哪些类别
3. `num_classes` 数据集类别有多少

这三个部分在 config 中的 train, val, test 字段中均有出现，具体更改哪些部分，参考下面的代码，假设新数据集有5个类别，用字母 a~e 来表示

```python
# the new config inherits the base configs to highlight the necessary modification
_base_ = './cascade_mask_rcnn_r50_fpn_1x_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('a', 'b', 'c', 'd', 'e')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='path/to/your/train/annotation_data',
        img_prefix='path/to/your/train/image_data'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='path/to/your/val/annotation_data',
        img_prefix='path/to/your/val/image_data'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='path/to/your/test/annotation_data',
        img_prefix='path/to/your/test/image_data'))

# 2. model settings

# explicitly over-write all the `num_classes` field from default 80 to 5.
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=5),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=5),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=5)],
    # explicitly over-write all the `num_classes` field from default 80 to 5.
    mask_head=dict(num_classes=5)))
```

这里推荐 [bilibili 教程](https://www.bilibili.com/video/BV1Jb4y1r7ir?p=1)，有实例讲解了如何处理个性化数据集

文档中还提到了可以[ 建立自己的 Dataset 类](https://mmdetection.readthedocs.io/en/latest/tutorials/customize_dataset.html#reorganize-new-data-format-to-middle-format)，来完成原本的 `CocoDataset` 类的功能。只需要继承 `CustomDataset` 基类然后重写 `load_annotations(self, ann_file)` and `get_ann_info(self, idx)` 两个方法

## Customize Data Pipelines

接下来就要具体地对数据集进行处理了，需要深入到 `mmdet/datasets` 当中看看，从 `builder.py` 大致可以看出，类似于 pytorch，mmdetection 也将数据处理分为 `Dataset` 和 `Dataloader` 两个类来创建和加载数据集。这一节的重点是在 `Dataset` 类中，上一节提到的 `CocoDataset`  类就是其中一个。在上一节，处理了数据集的 annotation 规范问题，这一节需要使用 `Dataset` 类建立一个 piplines 对数据集进行一系列变换操作

### 变换操作

在 `mmdet/datasets/piplines` 中有许多的方法，用于对数据的处理。如果想要创建自己的类，步骤也是类似的：

1. 创建 `MyTransform` 类
2. 导入这个类
3. 将该类加入到 config 中，文档中的 [示例代码](https://mmdetection.readthedocs.io/en/latest/tutorials/data_pipeline.html#extend-and-use-custom-pipelines)

下面是 Faster R-CNN config 文件中 pipeline 部分的一个例子

```python
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
```

### Dataset 实现变换操作

以 `CocoDataset` 为例，查看该类的代码，并没有发现在 `mmdet/datasets/coco.py` 中有实现 pipelines。实际上，`CocoDataset` 负责实现的是对 `annotation_file` 的处理，而对于 pipelines 的实现，在其继承的 `CustomDataset` 中实现的，具体实现逻辑如下：

1. 使用 `Compose` 类将 config 文件中的所有变换操作类实例化，并将所有的类存储到一个列表中
2. 对需要处理的 data 依次使用列表中的变化操作，具体参考代码 [CustomDataset](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/custom.py) [Compose](https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/pipelines/compose.py)

## Customize Models

在 mmdetection 中是通过 registry 来管理各种模块的，当然也包括模型中的模块。而如果想要通过 registry 添加模块的逻辑都是一样的，以添加一个新的 backbone, `MobileNet` 为例

1. 定义一个新的 backbone, `mmdet/models/backbones/mobilenet.py`

2. 导入这个模块，有两种方法

   - 在 `__init__.py` 中 import 这个模块，`from .mobilenet import MobileNet`

   - 在 config 中添加 `custom_imports` 字典

   ```python
   custom_imports = dict(
       imports=['mmdet.models.backbones.mobilenet'],
       allow_failed_imports=False)
   ```

3. 在 config 中使用这个模块

在 `mmdet/tools/train.py` 中可以看到，通过 `build_detector()` 将 config 中模型的具体参数实例化 registry 中的类。下面看一个简化的 `build_detector()`

```python
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
DETECTORS = MODELS

def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
	return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
```

## Customize Runtime Settings

这一部分主要讲怎么配置 optimizer 以及训练流程

### Customize optimization settings

mmdetection 适配了所有 pytorch 中的优化器，只需要修改 optimizer 的 `type` 关键字即可

```python
# use Adam optimizer
optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
```

Optimizer 的实现不是在 mmdetection 项目中，而是在 mmcv 项目中，所以想要个性化的 optimizer 则需要自己新建文件夹进行实现。现在新建 `mmdet/core/optimizer` 文件夹，新建 `my_optimizer.py`

```python
from mmcv.runner.optimizer import OPTIMIZERS
from torch.optim import Optimizer


@OPTIMIZERS.register_module()
class MyOptimizer(Optimizer):

    def __init__(self, a, b, c)
```

 然后修改 config 文件，并 import 该模块

```python
optimizer = dict(type='MyOptimizer', a=a_value, b=b_value, c=c_value)
custom_imports = dict(imports=['mmdet.core.optimizer.my_optimizer'], 
                      allow_failed_imports=False)
```

如果想要对 optimizer 有更加精细化的操作，需要构建新的 OPTIMIZER_CONSTRUCTORS，构建逻辑都是差不多的。这里就不展开了，因为很多优化操作并不熟悉，不知道该怎么用，等有具体的例子再进一步了解

### Customize training schedules

在训练优化的过程当中，需要对训练过程进行一些微调，例如对 learning rate 进行衰减，mmdetection 也可以通过调用 hook 做到，所谓的 Hook 就是在每个 epoch or iteration 之前、之后进行的一些操作。具体更改 training schedule 的操作为：在 config 文件中配置 `lr_config` 等字段。例如默认的阶梯策略 `StepLR` 可以在指定的 epoch 调整学习率

```python
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
```

### Customize workflow

这部分基本上就使用默认的 workflow 就好，即 `workflow = [('train',1)] ` ，这将会一次一次地循环训练集，直到达到 `total_epochs/max_epochs` 如果想要训练多个 epoch 过后对模型进行 evaluation 则是调用 `EvalHook`，而这在 config 文件中对应的是 `evaluation` 字段

```python
workflow = [('train',1)]
# 每间隔一次进行评估
evaluation = dict(interval=1, metric='bbox')
```

### Customize hooks

在之前已经提到了 hooks 的作用，mmdetection 中已经实现了一些 hooks：

- og_config
- checkpoint_config
- evaluation
- lr_config
- optimizer_config
- momentum_config

这些 hooks 能够在训练过程中去提供微调、评估、记录等功能，下面简单介绍一下其中的2个

1. Checkpoint config，用于保存训练过程中的模型

   ```python
   checkpoint_config = dict(interval=1)
   ```

2. Log config，用于记录训练过程中的关键数据，可以同时使用多个 logger

   ```python
   log_config = dict(
       interval=50,
       hooks=[
           dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')
       ])
   ```

如果想要使用个性化的 hook，准备过程也是老3步了

1. 定义新的 hook

   ```python
   from mmcv.runner import HOOKS, Hook
   
   
   @HOOKS.register_module()
   class MyHook(Hook):
   
       def __init__(self, a, b):
           pass
   
       def before_run(self, runner):
           pass
   
       def after_run(self, runner):
           pass
   
       def before_epoch(self, runner):
           pass
   
       def after_epoch(self, runner):
           pass
   
       def before_iter(self, runner):
           pass
   
       def after_iter(self, runner):
           pass
   ```

2. 导入新的 hook

   ```python
   custom_imports = dict(imports=['mmdet.core.utils.my_hook'], allow_failed_imports=False)
   ```

3. 使用新的 hook

   ```python
   custom_hooks = [
       dict(type='MyHook', a=a_value, b=b_value)
   ]
   ```

对于 hook 的调用，还需要进一步了解 `Runner` 类。mmdetection 是将这些 hook config 传入到了 `runner` 实例中，并在其中创建 hook 实例。更多有关 `Runner` 的内容需要在 [mmcv](https://mmcv.readthedocs.io/en/latest/understand_mmcv/runner.html#) 里去查看

## Customize Losses

mmdetction 中损失函数是直接在模型中各个 prediction head 中使用，而不是在 forward pass 之后单独使用，所以其关键字是什么，取决于模型之中各个 prediction head 怎么接收损失函数。以 `loss_cls` 字段为例使用 `FocalLoss`

```python
loss_cls=dict(
    type='FocalLoss',
    use_sigmoid=True,
    gamma=2.0,
    alpha=0.25,
    loss_weight=1.0)
```

 ## Finetuning Models

已经训练好的模型参数已经有较好的特征提取能力，所以将这些参数用于新的数据集只需要一些微调。除了要准备好自己数据集和 config 之外，需要注意的就是调整 training schedule。由于是仅对模型进行微调，所以学习率应该设置得相对小一些，并且 epoch 也应当少一些以避免过拟合。文档给出了一个参考的例子

```python
# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7])
# the max_epochs and step in lr_config need specifically tuned for the customized dataset
runner = dict(max_epochs=8)
log_config = dict(interval=100)

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
```

## 总结

可以把 mmdetection 的使用分为两个大块来进行理解：

1. 模型的建立
2. 模型的训练

首先是模型的建立，其核心为 config & registry，要理解这两个类的作用。通过 config 将模型的所有构造细节都以字典的形式包含进来了，将这些 config 传入到 registry 中注册好的模型类中从而实现模型的实例化。从中可以看到注册和使用任何新模块的三个主要步骤：定义模型、导入模型、修改 config 文件

接着是模型的训练，部分的核心为 optimizer & runner。其中 runner 包含了不同的 hook 以在每一个 epoch/iter 之前/之后进行需要的操作，例如：调整学习率、保存模型、记录数据等等。当然这些功能也是通过 config & registry 来进行管理的

下面是一个完整的 config 文件及其简要注释说明，为 ResNet50 和 FPN 的 Mask R-CNN 的配置文件。配合之前的 tutorial，可以具体看看每个字段在 config 文件中的什么位置、有什么功能，帮助整体把握。当然也可以直接在 mmdetection github 项目上进行搜索，查找源码也是很方便的

```python
model = dict(
    type='MaskRCNN',  # 检测器(detector)名称
    backbone=dict(  # 主干网络的配置文件
        type='ResNet',  # 主干网络的类别，可用选项请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/backbones/resnet.py#L308
        depth=50,  # 主干网络的深度，对于 ResNet 和 ResNext 通常设置为 50 或 101。
        num_stages=4,  # 主干网络状态(stages)的数目，这些状态产生的特征图作为后续的 head 的输入。
        out_indices=(0, 1, 2, 3),  # 每个状态产生的特征图输出的索引。
        frozen_stages=1,  # 第一个状态的权重被冻结
        norm_cfg=dict(  # 归一化层(norm layer)的配置项。
            type='BN',  # 归一化层的类别，通常是 BN 或 GN。
            requires_grad=True),  # 是否训练归一化里的 gamma 和 beta。
        norm_eval=True,  # 是否冻结 BN 里的统计项。
        style='pytorch',  # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积。
       init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),  # 加载通过 ImageNet 与训练的模型
    neck=dict(
        type='FPN',  # 检测器的 neck 是 FPN，我们同样支持 'NASFPN', 'PAFPN' 等，更多细节可以参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/fpn.py#L10。
        in_channels=[256, 512, 1024, 2048],  # 输入通道数，这与主干网络的输出通道一致
        out_channels=256,  # 金字塔特征图每一层的输出通道
        num_outs=5),  # 输出的范围(scales)
    rpn_head=dict(
        type='RPNHead',  # RPN_head 的类型是 'RPNHead', 我们也支持 'GARPNHead' 等，更多细节可以参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/dense_heads/rpn_head.py#L12。
        in_channels=256,  # 每个输入特征图的输入通道，这与 neck 的输出通道一致。
        feat_channels=256,  # head 卷积层的特征通道。
        anchor_generator=dict(  # 锚点(Anchor)生成器的配置。
            type='AnchorGenerator',  # 大多是方法使用 AnchorGenerator 作为锚点生成器, SSD 检测器使用 `SSDAnchorGenerator`。更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/anchor/anchor_generator.py#L10。
            scales=[8],  # 锚点的基本比例，特征图某一位置的锚点面积为 scale * base_sizes
            ratios=[0.5, 1.0, 2.0],  # 高度和宽度之间的比率。
            strides=[4, 8, 16, 32, 64]),  # 锚生成器的步幅。这与 FPN 特征步幅一致。 如果未设置 base_sizes，则当前步幅值将被视为 base_sizes。
        bbox_coder=dict(  # 在训练和测试期间对框进行编码和解码。
            type='DeltaXYWHBBoxCoder',  # 框编码器的类别，'DeltaXYWHBBoxCoder' 是最常用的，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/coder/delta_xywh_bbox_coder.py#L9。
            target_means=[0.0, 0.0, 0.0, 0.0],  # 用于编码和解码框的目标均值
            target_stds=[1.0, 1.0, 1.0, 1.0]),  # 用于编码和解码框的标准方差
        loss_cls=dict(  # 分类分支的损失函数配置
            type='CrossEntropyLoss',  # 分类分支的损失类型，我们也支持 FocalLoss 等。
            use_sigmoid=True,  # RPN通常进行二分类，所以通常使用sigmoid函数。
            los_weight=1.0),  # 分类分支的损失权重。
        loss_bbox=dict(  # 回归分支的损失函数配置。
            type='L1Loss',  # 损失类型，我们还支持许多 IoU Losses 和 Smooth L1-loss 等，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/losses/smooth_l1_loss.py#L56。
            loss_weight=1.0)),  # 回归分支的损失权重。
    roi_head=dict(  # RoIHead 封装了两步(two-stage)/级联(cascade)检测器的第二步。
        type='StandardRoIHead',  # RoI head 的类型，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/standard_roi_head.py#L10。
        bbox_roi_extractor=dict(  # 用于 bbox 回归的 RoI 特征提取器。
            type='SingleRoIExtractor',  # RoI 特征提取器的类型，大多数方法使用  SingleRoIExtractor，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/roi_extractors/single_level.py#L10。
            roi_layer=dict(  # RoI 层的配置
                type='RoIAlign',  # RoI 层的类别, 也支持 DeformRoIPoolingPack 和 ModulatedDeformRoIPoolingPack，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/roi_align/roi_align.py#L79。
                output_size=7,  # 特征图的输出大小。
                sampling_ratio=0),  # 提取 RoI 特征时的采样率。0 表示自适应比率。
            out_channels=256,  # 提取特征的输出通道。
            featmap_strides=[4, 8, 16, 32]),  # 多尺度特征图的步幅，应该与主干的架构保持一致。
        bbox_head=dict(  # RoIHead 中 box head 的配置.
            type='Shared2FCBBoxHead',  # bbox head 的类别，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py#L177。
            in_channels=256,  # bbox head 的输入通道。 这与 roi_extractor 中的 out_channels 一致。
            fc_out_channels=1024,  # FC 层的输出特征通道。
            roi_feat_size=7,  # 候选区域(Region of Interest)特征的大小。
            num_classes=80,  # 分类的类别数量。
            bbox_coder=dict(  # 第二阶段使用的框编码器。
                type='DeltaXYWHBBoxCoder',  # 框编码器的类别，大多数情况使用 'DeltaXYWHBBoxCoder'。
                target_means=[0.0, 0.0, 0.0, 0.0],  # 用于编码和解码框的均值
                target_stds=[0.1, 0.1, 0.2, 0.2]),  # 编码和解码的标准方差。因为框更准确，所以值更小，常规设置时 [0.1, 0.1, 0.2, 0.2]。
            reg_class_agnostic=False,  # 回归是否与类别无关。
            loss_cls=dict(  # 分类分支的损失函数配置
                type='CrossEntropyLoss',  # 分类分支的损失类型，我们也支持 FocalLoss 等。
                use_sigmoid=False,  # 是否使用 sigmoid。
                loss_weight=1.0),  # 分类分支的损失权重。
            loss_bbox=dict(  # 回归分支的损失函数配置。
                type='L1Loss',  # 损失类型，我们还支持许多 IoU Losses 和 Smooth L1-loss 等。
                loss_weight=1.0)),  # 回归分支的损失权重。
        mask_roi_extractor=dict(  # 用于 mask 生成的 RoI 特征提取器。
            type='SingleRoIExtractor',  # RoI 特征提取器的类型，大多数方法使用 SingleRoIExtractor。
            roi_layer=dict(  # 提取实例分割特征的 RoI 层配置
                type='RoIAlign',  # RoI 层的类型，也支持 DeformRoIPoolingPack 和 ModulatedDeformRoIPoolingPack。
                output_size=14,  # 特征图的输出大小。
                sampling_ratio=0),  # 提取 RoI 特征时的采样率。
            out_channels=256,  # 提取特征的输出通道。
            featmap_strides=[4, 8, 16, 32]),  # 多尺度特征图的步幅。
        mask_head=dict(  # mask 预测 head 模型
            type='FCNMaskHead',  # mask head 的类型，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py#L21。
            num_convs=4,  # mask head 中的卷积层数
            in_channels=256,  # 输入通道，应与 mask roi extractor 的输出通道一致。
            conv_out_channels=256,  # 卷积层的输出通道。
            num_classes=80,  # 要分割的类别数。
            loss_mask=dict(  # mask 分支的损失函数配置。
                type='CrossEntropyLoss',  # 用于分割的损失类型。
                use_mask=True,  # 是否只在正确的类中训练 mask。
                loss_weight=1.0))))  # mask 分支的损失权重.
    train_cfg = dict(  # rpn 和 rcnn 训练超参数的配置
        rpn=dict(  # rpn 的训练配置
            assigner=dict(  # 分配器(assigner)的配置
                type='MaxIoUAssigner',  # 分配器的类型，MaxIoUAssigner 用于许多常见的检测器，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10。
                pos_iou_thr=0.7,  # IoU >= 0.7(阈值) 被视为正样本。
                neg_iou_thr=0.3,  # IoU < 0.3(阈值) 被视为负样本。
                min_pos_iou=0.3,  # 将框作为正样本的最小 IoU 阈值。
                match_low_quality=True,  # 是否匹配低质量的框(更多细节见 API 文档).
                ignore_iof_thr=-1),  # 忽略 bbox 的 IoF 阈值。
            sampler=dict(  # 正/负采样器(sampler)的配置
                type='RandomSampler',  # 采样器类型，还支持 PseudoSampler 和其他采样器，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8。
                num=256,  # 样本数量。
                pos_fraction=0.5,  # 正样本占总样本的比例。
                neg_pos_ub=-1,  # 基于正样本数量的负样本上限。
                add_gt_as_proposals=False),  # 采样后是否添加 GT 作为 proposal。
            allowed_border=-1,  # 填充有效锚点后允许的边框。
            pos_weight=-1,  # 训练期间正样本的权重。
            debug=False),  # 是否设置调试(debug)模式
        rpn_proposal=dict(  # 在训练期间生成 proposals 的配置
            nms_across_levels=False,  # 是否对跨层的 box 做 NMS。仅适用于 `GARPNHead` ，naive rpn 不支持 nms cross levels。
            nms_pre=2000,  # NMS 前的 box 数
            nms_post=1000,  # NMS 要保留的 box 的数量，只在 GARPNHHead 中起作用。
            max_per_img=1000,  # NMS 后要保留的 box 数量。
            nms=dict( # NMS 的配置
                type='nms',  # NMS 的类别
                iou_threshold=0.7 # NMS 的阈值
                ),
            min_bbox_size=0),  # 允许的最小 box 尺寸
        rcnn=dict(  # roi head 的配置。
            assigner=dict(  # 第二阶段分配器的配置，这与 rpn 中的不同
                type='MaxIoUAssigner',  # 分配器的类型，MaxIoUAssigner 目前用于所有 roi_heads。更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/max_iou_assigner.py#L10。
                pos_iou_thr=0.5,  # IoU >= 0.5(阈值)被认为是正样本。
                neg_iou_thr=0.5,  # IoU < 0.5(阈值)被认为是负样本。
                min_pos_iou=0.5,  # 将 box 作为正样本的最小 IoU 阈值
                match_low_quality=False,  # 是否匹配低质量下的 box(有关更多详细信息，请参阅 API 文档)。
                ignore_iof_thr=-1),  # 忽略 bbox 的 IoF 阈值
            sampler=dict(
                type='RandomSampler',  #采样器的类型，还支持 PseudoSampler 和其他采样器，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/samplers/random_sampler.py#L8。
                num=512,  # 样本数量
                pos_fraction=0.25,  # 正样本占总样本的比例。.
                neg_pos_ub=-1,  # 基于正样本数量的负样本上限。.
                add_gt_as_proposals=True
            ),  # 采样后是否添加 GT 作为 proposal。
            mask_size=28,  # mask 的大小
            pos_weight=-1,  # 训练期间正样本的权重。
            debug=False))  # 是否设置调试模式。
    test_cfg = dict(  # 用于测试 rnn 和 rnn 超参数的配置
        rpn=dict(  # 测试阶段生成 proposals 的配置
            nms_across_levels=False,  # 是否对跨层的 box 做 NMS。仅适用于`GARPNHead`，naive rpn 不支持做 NMS cross levels。
            nms_pre=1000,  # NMS 前的 box 数
            nms_post=1000,  # NMS 要保留的 box 的数量，只在`GARPNHHead`中起作用。
            max_per_img=1000,  # NMS 后要保留的 box 数量
            nms=dict( # NMS 的配置
                type='nms',  # NMS 的类型
                iou_threshold=0.7 # NMS 阈值
                ),
            min_bbox_size=0),  # box 允许的最小尺寸
        rcnn=dict(  # roi heads 的配置
            score_thr=0.05,  # bbox 的分数阈值
            nms=dict(  # 第二步的 NMS 配置
                type='nms',  # NMS 的类型
                iou_thr=0.5),  # NMS 的阈值
            max_per_img=100,  # 每张图像的最大检测次数
            mask_thr_binary=0.5))  # mask 预处的阈值
dataset_type = 'CocoDataset'  # 数据集类型，这将被用来定义数据集。
data_root = 'data/coco/'  # 数据的根路径。
img_norm_cfg = dict(  #图像归一化配置，用来归一化输入的图像。
    mean=[123.675, 116.28, 103.53],  # 预训练里用于预训练主干网络模型的平均值。
    std=[58.395, 57.12, 57.375],  # 预训练里用于预训练主干网络模型的标准差。
    to_rgb=True
)  #  预训练里用于预训练主干网络的图像的通道顺序。
train_pipeline = [  # 训练流程
    dict(type='LoadImageFromFile'),  # 第 1 个流程，从文件路径里加载图像。
    dict(
        type='LoadAnnotations',  # 第 2 个流程，对于当前图像，加载它的注释信息。
        with_bbox=True,  # 是否使用标注框(bounding box)， 目标检测需要设置为 True。
        with_mask=True,  # 是否使用 instance mask，实例分割需要设置为 True。
        poly2mask=False),  # 是否将 polygon mask 转化为 instance mask, 设置为 False 以加速和节省内存。
    dict(
        type='Resize',  # 变化图像和其注释大小的数据增广的流程。
        img_scale=(1333, 800),  # 图像的最大规模。
        keep_ratio=True
    ),  # 是否保持图像的长宽比。
    dict(
        type='RandomFlip',  #  翻转图像和其注释大小的数据增广的流程。
        flip_ratio=0.5),  # 翻转图像的概率。
    dict(
        type='Normalize',  # 归一化当前图像的数据增广的流程。
        mean=[123.675, 116.28, 103.53],  # 这些键与 img_norm_cfg 一致，因为 img_norm_cfg 被
        std=[58.395, 57.12, 57.375],     # 用作参数。
        to_rgb=True),
    dict(
        type='Pad',  # 填充当前图像到指定大小的数据增广的流程。
        size_divisor=32),  # 填充图像可以被当前值整除。
    dict(type='DefaultFormatBundle'),  # 流程里收集数据的默认格式捆。
    dict(
        type='Collect',  # 决定数据中哪些键应该传递给检测器的流程
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # 第 1 个流程，从文件路径里加载图像。
    dict(
        type='MultiScaleFlipAug',  # 封装测试时数据增广(test time augmentations)。
        img_scale=(1333, 800),  # 决定测试时可改变图像的最大规模。用于改变图像大小的流程。
        flip=False,  # 测试时是否翻转图像。
        transforms=[
            dict(type='Resize',  # 使用改变图像大小的数据增广。
                 keep_ratio=True),  # 是否保持宽和高的比例，这里的图像比例设置将覆盖上面的图像规模大小的设置。
            dict(type='RandomFlip'),  # 考虑到 RandomFlip 已经被添加到流程里，当 flip=False 时它将不被使用。
            dict(
                type='Normalize',  #  归一化配置项，值来自 img_norm_cfg。
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='Pad',  # 将配置传递给可被 32 整除的图像。
                size_divisor=32),
            dict(
                type='ImageToTensor',  # 将图像转为张量
                keys=['img']),
            dict(
                type='Collect',  # 收集测试时必须的键的收集流程。
                keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,  # 单个 GPU 的 Batch size
    workers_per_gpu=2,  # 单个 GPU 分配的数据加载线程数
    train=dict(  # 训练数据集配置
        type='CocoDataset',  # 数据集的类别, 更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/datasets/coco.py#L19。
        ann_file='data/coco/annotations/instances_train2017.json',  # 注释文件路径
        img_prefix='data/coco/train2017/',  # 图片路径前缀
        pipeline=[  # 流程, 这是由之前创建的 train_pipeline 传递的。
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(  # 验证数据集的配置
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[  # 由之前创建的 test_pipeline 传递的流程。
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(  # 测试数据集配置，修改测试开发/测试(test-dev/test)提交的 ann_file
        type='CocoDataset',
        ann_file='data/coco/annotations/instances_val2017.json',
        img_prefix='data/coco/val2017/',
        pipeline=[  # 由之前创建的 test_pipeline 传递的流程。
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        samples_per_gpu=2  # 单个 GPU 测试时的 Batch size
        ))
evaluation = dict(  # evaluation hook 的配置，更多细节请参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/evaluation/eval_hooks.py#L7。
    interval=1,  # 验证的间隔。
    metric=['bbox', 'segm'])  # 验证期间使用的指标。
optimizer = dict(  # 用于构建优化器的配置文件。支持 PyTorch 中的所有优化器，同时它们的参数与 PyTorch 里的优化器参数一致。
    type='SGD',  # 优化器种类，更多细节可参考 https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/optimizer/default_constructor.py#L13。
    lr=0.02,  # 优化器的学习率，参数的使用细节请参照对应的 PyTorch 文档。
    momentum=0.9,  # 动量(Momentum)
    weight_decay=0.0001)  # SGD 的衰减权重(weight decay)。
optimizer_config = dict(  # optimizer hook 的配置文件，执行细节请参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8。
    grad_clip=None)  # 大多数方法不使用梯度限制(grad_clip)。
lr_config = dict(  # 学习率调整配置，用于注册 LrUpdater hook。
    policy='step',  # 调度流程(scheduler)的策略，也支持 CosineAnnealing, Cyclic, 等。请从 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9 参考 LrUpdater 的细节。
    warmup='linear',  # 预热(warmup)策略，也支持 `exp` 和 `constant`。
    warmup_iters=500,  # 预热的迭代次数
    warmup_ratio=
    0.001,  # 用于热身的起始学习率的比率
    step=[8, 11])  # 衰减学习率的起止回合数
runner = dict(
    type='EpochBasedRunner',  # 将使用的 runner 的类别 (例如 IterBasedRunner 或 EpochBasedRunner)。
    max_epochs=12) # runner 总回合数， 对于 IterBasedRunner 使用 `max_iters`
checkpoint_config = dict(  # Checkpoint hook 的配置文件。执行时请参考 https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py。
    interval=1)  # 保存的间隔是 1。
log_config = dict(  # register logger hook 的配置文件。
    interval=50,  # 打印日志的间隔
    hooks=[
        # dict(type='TensorboardLoggerHook')  # 同样支持 Tensorboard 日志
        dict(type='TextLoggerHook')
    ])  # 用于记录训练过程的记录器(logger)。
dist_params = dict(backend='nccl')  # 用于设置分布式训练的参数，端口也同样可被设置。
log_level = 'INFO'  # 日志的级别。
load_from = None  # 从一个给定路径里加载模型作为预训练模型，它并不会消耗训练时间。
resume_from = None  # 从给定路径里恢复检查点(checkpoints)，训练模式将从检查点保存的轮次开始恢复训练。
workflow = [('train', 1)]  # runner 的工作流程，[('train', 1)] 表示只有一个工作流且工作流仅执行一次。根据 total_epochs 工作流训练 12个回合。
work_dir = 'work_dir'  # 用于保存当前实验的模型检查点和日志的目录文件地址。
```

## TODO

1. 学习 mmdetection 工具箱
2. 实现一个项目

