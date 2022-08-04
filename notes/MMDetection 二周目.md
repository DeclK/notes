---
title: MMDetection 二周目
tags:
  - MMDetection
categories:
  - 编程
  - OpenMMLab
abbrlink: 11abeb49
date: 2022-07-10 15:40:03
---

# MMDetection 二周目

MMDetection 其实是一个非常庞大的项目，如果想要掌握每一个细节确实太难了...现在想要快速地上手一些模型，想要了解模型的运行机制，并且在以后方便地运行自己的模型，又感觉之前整理的还是比较零零散散，没有更精炼的章法，现在重新总结尽量把逻辑讲清楚

## Installation

现在的安装也没有什么坑了，可以参考 [bilibili](https://www.bilibili.com/video/BV1NL4y1c7ki?p=2)

1. 安装 pytorch。这一步只要查找好自己的 cuda 版本，然后使用镜像就可以轻松安装了

2. 安装 mmcv-full，官方推荐使用 mim 安装

   ```shell
   pip install -U openmim
   mim install mmcv-full
   ```

3. 安装 mmdetection

   ```shell
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   pip install -v -e .
   ```

   国内也可以 clone gitee，速度飞快。官方有 repo

   ```shell
   git clone https://gitee.com/open-mmlab/mmdetection.git
   git clone https://github.com/open-mmlab/mmdetection.git
   ```

一波安装，简直丝滑

## 数据集

下载一个简单的数据集，这样就能把流程跑通，对于学习一些网络结构是非常有必要的。推荐 [VOC2007](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) 数据集，比较小巧，整体不超过1G，能够接受，[mmdet data download](https://mmdetection.readthedocs.io/en/latest/useful_tools.html#dataset-download) 提供了脚本，可以直接下载

```shell
python tools/misc/download_dataset.py --dataset-name coco2017
python tools/misc/download_dataset.py --dataset-name voc2007
python tools/misc/download_dataset.py --dataset-name lvis
```

下载完后解压

```shell
for item in *.tar;do tar -xvf $item; done;
```

得到的 `VOCdevkit` 放在 `mmdetection/data` 下就可以了

## First Train

在 mmdet 里面有很多的 config，对应着不同的数据集，不同的模型。对于 VOC 数据集，可以选择 faster rcnn 这个最基本的模型 `configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py` 

根据 [zhihu](https://zhuanlan.zhihu.com/p/102390034)，mmdetection 是将 VOC2012 & VOC2007 一起训练的，所以要更改一下数据集的配置文件 `configs/_base_/datasets/voc0712.py`

```python
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                # data_root + 'VOC2012/ImageSets/Main/trainval.txt' COMMENT IT
            ],
            # img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'], COMMENT IT AND CHANGE TO NEXT LINE
            img_prefix=[data_root + 'VOC2007/'],
            pipeline=train_pipeline)),
```

现在应该就可以开始跑了

```shell
python tools/train.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py
```

跑起来的部分输出长得像这样

```shell
2022-06-03 06:33:26,389 - mmdet - INFO - workflow: [('train', 1)], max: 4 epochs
2022-06-03 06:33:26,389 - mmdet - INFO - Checkpoints will be saved to /mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_voc0712 by HardDiskBackend.
2022-06-03 06:33:38,427 - mmdet - INFO - Epoch [1][50/7517]     lr: 1.000e-02, eta: 2:00:00, time: 0.240, data_time: 0.051, memory: 2404, loss_rpn_cls: 0.1461, loss_rpn_bbox: 0.0187, loss_cls: 0.3264, acc: 95.1289, loss_bbox: 0.1275, loss: 0.6188
2022-06-03 06:33:47,715 - mmdet - INFO - Epoch [1][100/7517]    lr: 1.000e-02, eta: 1:46:19, time: 0.186, data_time: 0.006, memory: 2404, loss_rpn_cls: 0.0788, loss_rpn_bbox: 0.0207, loss_cls: 0.3124, acc: 95.5254, loss_bbox: 0.1786, loss: 0.5905
```

## Config

在此之前一句话总结一下 config：config 包含着数据集，模型，训练的所有信息和参数。使用什么数据集，使用什么模型，使用什么样的训练流程，都是由 config 文件决定。很多时候“调参”，大多就是直接调整 config 文件完成

## Training flow

下面来读一读 `train.py`，必须看看代码才知道整个流程，这里就总结一些重点流程

1. Build config，把配置文件转化成 `Config` 类，类似于 `EasyDict`，可以把 key 当成属性用

2. Build logger & dist training process

3. Build detector，这一步需要了解 `Registry` 类。这个类感觉也是经常被人吐槽🤣，这里只总结用法和逻辑，不去纠结其他

   1. Registry 完成的功能：根据 config 创建类。mmdet 先创建一个 Registry 对象，在 `mmdet/models/builder.py` 中

      ```python
      from mmcv.cnn import MODELS as MMCV_MODELS
      from mmcv.utils import Registry
      
      MODELS = Registry('models', parent=MMCV_MODELS)	# 不必特别在意 parent, 不影响理解 Registry 逻辑
      # 'models' 就是给 Registry 取的名字，也没有特别的用处
      
      # 这些名字都指向了 MODELS 只是为了方便理解模型结构，完全可以不这么做
      BACKBONES = MODELS
      NECKS = MODELS
      ROI_EXTRACTORS = MODELS
      SHARED_HEADS = MODELS
      HEADS = MODELS
      LOSSES = MODELS
      DETECTORS = MODELS
      ```

      这个对象 `MODELS` 在之后通过装饰器，把所有的模型都注册到自己里面。所谓**通过装饰器注册**，本质上做的事情，就是把模型放在一个字典里

      ```python
      from ..builder import DETECTORS
      from .single_stage import SingleStageDetector
      
      @DETECTORS.register_module()
      class FCOS(SingleStageDetector):
      ```

      这一步就会把 `FCOS` 存储到 `MODELS/DETECTORS` 内的字典 `self._module_dict`，存储形式为

      ```python
                  name = module.__name__
                  self._module_dict[name] = module
      ```

   2. 注册好之后就可以使用 `build` 方法根据 config 创建对象了，`build_detector` 方法本质上就是调用下面这句话

      ```python
      detector = DETECTORS.build(cfg)
      
      # build() function is essentially using build_from_cfg
      
      def build_from_cfg(cfg, registry, default_args=None):
          """Build a module from config dict when it is a class configuration, or
          call a function from config dict when it is a function configuration.
      
          Example:
              >>> MODELS = Registry('models')
              >>> @MODELS.register_module()
              >>> class ResNet:
              >>>     pass
              >>> resnet = build_from_cfg(dict(type='Resnet'), MODELS)
              >>> # Returns an instantiated object
      ```
      
      `cfg` 字典中必须包含 `type` 关键字，通过 `type` 的值，在 `self._module_dict` 中找到对应的对象，然后使用其余的 `cfg` 参数创建对象。这里只举例了建造检测器，实际上任何类都可以通过注册器进行创建

4. Build dataset，原理和 builde detector 一样，都是通过 Registry 来创建 dataset 类
5. **Train detector**，这里就是最核心的函数了，单独开一节继续整理

## Train detector

整理 `train_detector` 函数的核心步骤

### Build dataloader & model & optimizer

省略这一部分的整理

1. Build dataloader
2. Build distributed model
3. Build optimizer，这里 optimizer 是作为 hook 形式存在，默认使用 `OptimizerHook`

上面的这些对象，都会被送到 runner 里运行

### Build runner

Build runner，build runner 本身没什么需要理解的。但是 runner 的功能非常多，需要了解  `HOOK` 以及 `EpochBasedRunner or IterBasedRunner`（这里我选择 `EpochBasedRunner` 整理），下面分别整理

#### Hook

[HOOK zhihu](https://zhuanlan.zhihu.com/p/238130913)，相当于是 runner 里的触发器，在特定的步骤时进行一些操作，核心代码大概长这样

```python
# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, is_method_overridden

HOOKS = Registry('hook')	# 大管家 HOOKS，都会通过这个 hook registry 对象进行注册

class Hook:		# base class for all hooks

    stages = ('before_run', 'before_train_epoch', 'before_train_iter',
              'after_train_iter', 'after_train_epoch', 'before_val_epoch',
              'before_val_iter', 'after_val_iter', 'after_val_epoch',
              'after_run')

    def before_run(self, runner):
        pass

    def after_run(self, runner):
        pass

    def before_epoch(self, runner):
        pass
   	...
```

代码来自 `mmcv.runner.hooks` 中的 `hook.py` 模块，之后的想要注册其他 hook，都要先从模块中导入管家 `HOOKS` 然后再继承基类 `Hook`， 并重载其中的方法，简单举个例子

```python
from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class CheckInvalidLossHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.
    """

    def __init__(self, interval=50):
        self.interval = interval

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            assert torch.isfinite(runner.outputs['loss']), \
                runner.logger.info('loss become infinite or NaN!')
```

这样注册了一个 `CheckInvalidLossHook`，但是光注册进 `HOOKS` 没什么用，我们最终的目的是要把 hook 运用到 runner 中

具体来说在 Runner 中定义了一个 hook 相关的 list，list 中的每一个元素就是 一个实例化的 HOOK 对象，列表的顺序是根据 hook 的 `priority` 进行排列。这里又学到了一个热知识：python 类的属性能够在外部直接定义🤣

#### BaseRunner

基类 `BaseRunner` 用于初始化相关属性：model, dataloader, optimizer, epoch, iter, **hooks**, ...其中较多的代码都是在创建 `self._hook` 属性。其功能正如上面所说，该属性是一个列表，存储实例化的 hook 对象，并根据 priority 进行排序。通过 `call_hook` 方法，在指定步骤把所有注册的 hook 都遍历一遍，而注册的 hooks 都有自己在特定步骤的功能

```python
    def call_hook(self, fn_name):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
        """
        for hook in self._hooks:
            getattr(hook, fn_name)(self)	# self is runner itself!!
```

这样就能实现在特定步骤实现特定功能啦！下面贴一段注释，根据这段注释再继续整理 `EpochBasedRunner`

```python
class BaseRunner(metaclass=ABCMeta):
    """The base class of Runner, a training helper for PyTorch.

    All subclasses should implement the following APIs:

    - ``run()``
    - ``train()``
    - ``val()``
    - ``save_checkpoint()``
```

#### EpochBasedRunner

这个类就是重点实现上面4个方法，建议直接去看原代码，基本逻辑就是：训练 + Hooks。下面只贴一些核心代码帮助理解

1. `run()` 最核心的方法，把 `train() & val()` 串起来

   ```python
           while self.epoch < self._max_epochs:
               for i, flow in enumerate(workflow):
                   mode, epochs = flow
                   epoch_runner = getattr(self, mode)
                   for _ in range(epochs):
                       if mode == 'train' and self.epoch >= self._max_epochs:
                           break
                       epoch_runner(data_loaders[i], **kwargs)
   ```

2. `train()` 训练

   ```python
       def train(self, data_loader, **kwargs):
           self.model.train()
           self.mode = 'train'
           self.data_loader = data_loader
           self._max_iters = self._max_epochs * len(self.data_loader)
           self.call_hook('before_train_epoch')
           time.sleep(2)  # Prevent possible deadlock during epoch transition
           for i, data_batch in enumerate(self.data_loader):
               self.data_batch = data_batch
               self._inner_iter = i
               self.call_hook('before_train_iter')
               self.run_iter(data_batch, train_mode=True, **kwargs)
               self.call_hook('after_train_iter')
               del self.data_batch
               self._iter += 1
   
           self.call_hook('after_train_epoch')
           self._epoch += 1
   ```

   这里必须要贴一下 `run_iter` 的代码帮助理解

   ```python
       def run_iter(self, data_batch, train_mode, **kwargs):
           if self.batch_processor is not None:
               outputs = self.batch_processor(
                   self.model, data_batch, train_mode=train_mode, **kwargs)
           elif train_mode:
               outputs = self.model.train_step(data_batch, self.optimizer,
                                               **kwargs)
           else:
               outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
   
           self.outputs = outputs
   ```

   有意思的是没有看到 `optimizer` 相关的代码，这是因为优化器被送到了 Hook 当中，在 `self.call_hook('after_train_iter')` 中调用

3. `val()` 和训练时的差不多，只不过使用的 hook 不一样，并且不对 `self._iter & self._epoch` 进行更新

4. `save_checkpoint()` 直接看下方注释即可

   ```python
   def save_checkpoint(model,
                       filename,
                       optimizer=None,
                       meta=None,
                       file_client_args=None):
       """Save checkpoint to file.
   
       The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
       ``optimizer``. By default ``meta`` will contain version and time info.
   ```

### Run

在创建好了 runner 之后，就是一通注册 hooks

```python
    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))
    
    # eval hooks
    runner.register_hook(
        eval_hook(val_dataloader, **eval_cfg), priority='LOW')
```

然后就开始训练

```python
runner.run(data_loaders, cfg.workflow)
```

## Test



## TODO

这里就基本把 mmdetection 的逻辑整理完毕，实际上真正使用的时候更加方便，之后再进行整理一下如何实战，最好做一些好玩的项目
