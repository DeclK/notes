# Train COCO on MMDetection

目的，能够跑通 coco 数据集，了解 coco 格式

更重要的是看看 detr 是怎么跑的

## 下载 COCO

虽然百度网盘极其🐶，但是这里我依然使用了百度网盘下载，需要开启一下闲置带宽优化下载。只要是热门资源下载速度都会比较快的

下载完后，解压放在如下位置

```txt
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

下载完 coco 数据集然后解压

## MMEngine

**这里整理 MMEngine 的核心部分**，我之前的整理都过于遵从于官方文档了，即太过于关注细节，依然没有建立起整体的架构

### Registry

**注册器的功能，就是利用配置文件构建类的实例**

要做到这一点，就要先把需要的类给注册到注册器中。所谓注册，本质上就是把类放到注册器内的一个字典 `_module_dict` 里，之后需要的时候取出，使用配置文件进行实例化 `Registry.build(cfg)`，如果只想获得类本身，就用 `Registry.get('key')` 即可

所有的注册器都放在了 `mmdet.registry` 当中，需要用哪个就 `import` 哪个

### Config

**配置文件定义了所使用的模型、训练、数据集**

使用 `Config` 类来完成对配置文件的处理

```python
cfg = Config.fromfile('config.py')
```

 即可用 `config.py` 文件创建一个配置类。文件即可以是 `.py` 也可以是 `.yaml`

`Config` 类可以通过关键字或者属性来调用配置文件内容，`cfg.key` 或 `cfg['key']` 都可获得其中的内容

**为了构建一个层级的配置文件，继承机制是非常有必要的**

配置文件内可以通过 `_base_` 关键字指定基础配置文件

```python
_base_ = ['list_of_base_configs.py',]
```

修改基础配置中的值也非常简单，直接在当前配置文件夹中重新定义即可，并且你不需要把字典所有的关键字都重新定义一遍，只需要定义修改的关键字即可。如果想要删除没有重新定义的关键字需要使用 `_delete_` 关键字，这样就仅剩下新定义的内容

```python
_base_ = ['optimizer_cfg.py', 'runtime_cfg.py']
optimizer = dict(_delete_=True, type='SGD', lr=0.01)
```

**其他技巧**

1. 可以通过 `{{_base_.attr}}` 来引用上级配置中的内容
2. 可以通过 `cfg.dump('config.py')` 来输出配置文件，输出形式还可以是 `.yaml`

### Runner

光看文档完全没办法理解怎么用，这里我要根据代码来自己整理

1. runner 初始化

   1. deepcopy cfg，新建属性 self.cfg

   2. 所谓 lazy init 就是指先用一个列表放 cfg 或者直接为 cfg 本身，在之后需要的时候用这些 cfg 真正的生成实例

      使用 lazy init

   3. 创建属性 traininig_related，val_related，test_related。每个 related list 为 `[xxx_dataloader, xxx_cfg, xxx_evaluator]`

   4. optim_wrapper，无操作

   5. launcher，决定分布式与否

   6. setup_env，初始化 dist 环境，新建属性 rank 和 world_size

   7. set_random_seed，新建属性 seed 和 deterministic，通过 randomness=dict(seed=None) 设置

   8. 创建 work_dir

   9. 创建 logger，logger 此时记录下环境信息和配置文件

   10. 创建属性 load_from 和 resume

   11. 创建属性 model，build_model 实例化

   12. 打包 model 为 MMDistributedDataParallel，当然也可以打包为 DDP。MMDDP 比 DDP 新定义了三分方法 tran_step, val_step, test_step

   13. 注册钩子

   14. 保存 config

2. runner.train()

   1. 检查 ori_model 是否有 _train_step。这里是对模型的基本要求。如果有 _val_loop，也得检查是否有 _val_step

   2. build_train_loop，并创建属性 train_loop。因为之前是 lazy init train loop，所以这里要 build

      一个类表明 metaclass=ABCMeta 表示该类为抽象类，不能够实例化，只能用来继承

      BaseLoop，所有 loop 的基类，需要传入 runner 和 dataloader，并且由于之前的 dataloader 是 lasy init，所以要 build dataloader, dataset, dataset sampler, batch sampler, 即真正的实例化

      pesudo collate 就是 default collate 但是不转换为 tensor

      defaultsampler 能够同时处理分布式和非分布式采样，再包一个 batchsampler 就能够处理批采样了

   3. build_optim_wrapper，创建属性 optim_wrapper，并使用 auto_scale_lr

   4. build_param_scheduler，创建属性 param_schedulers，这是对学习率的策略配置

   5. build_val_loop

   6. call_hook('before_run')

   7. 初始化模型权重，如果有预训练权重则 load

   8. self.train_loop.run()

   9. call_hook('after_run')

### train loop 运行逻辑

BaseLoop 是一个非常简单的类，只需要 runner 和 dataloader 作为初始化即可

EpochBasedTrainLoop 继承 BaseLoop 核心逻辑在 run 函数

run 将循环运行 run_epoch，并在 epoch 后判断是否需要 eval

run_epoch 是由循环 run_iter 完成，循环以 dataloader 主导

run_iter 中运行了模型的 `train_step` 步骤，在 `train_step` 中优化步已经完成了

下面写一下简化代码，去除钩子

```python
    def run(self) -> torch.nn.Module:
        """Launch training."""
        while self._epoch < self._max_epochs:
            self.run_epoch()
            if (self.runner.val_loop is not None and self._epoch >= self.val_begin and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.model.train()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)
        self._iter += 1
```

自己在写个性化 Loops 的时候最好要将这些钩子都加上，以保证结果的正确！例如 DistributedSampler 的随机种子要在各个 epoch 开始前重新设置，这里需要调用一个 DistSamplerSeedHook 完成

### train step 运行逻辑

核心代码非常简单

```python
    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

```

由于 collate_fn 使用的是一个非常简单的方法，所以**数据预处理放在了 DataPreprocessor 中**，其功能包括把数据发送到 GPU 上，数据打包，归一化，最后返回 data 字典（包含 data['inputs'] & data['data_sample']）

### 如何自己写配置文件

建议是从 `_base_` 中去继承，然后再挑选修改。总体来讲核心如下

```python
# dataset
train_pipeline = [dict(type='LoadImageFromFile')]
train_dataloader = dict(batch_size=, dataset=, pipline=train_pipline)

test_pipline = ...
val_dataloader = ...

val_evaluator = dict(type='CocoMetric', ann_file=...)

# model
model = dict(type='DETR',...)

# optimizer & scheduler
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')

optim_wrapper = dict(type='OptimWrapper', optimizer=dict(tpye='SGD', lr=0.01))
param_scheduler = dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500)

auto_scale_lr = dict(enable=False, base_batch_size=16)

# logs & hooks
default_hooks

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

visualizer = dict(type='DetLocalVisualizer', vis_backends='LocalVisBackend', name='visualizer')
```

### DataLoader 接口整理

```python
DataLoader(dataset, 
           batch_size=1, shuffle=False, sampler=None, drop_last=False,
           batch_sampler=None, 
           collate_fn=None,
           num_workers=0, pin_memory=False)
```

我把上面的参数分成了4行，其中**前两行是核心配置，控制随机性和 batch 行为**，第三行是自定义打包方法配置，第四行是加速配置

1. Sampler 和 shuffle 是互斥的，有了 sampler 后 shuffle 将不再起作用。**实际上几乎可以不用 sampler 这个参数**
2. batch_sampler 是以一个 Sampler 作为基础，再进行 group 操作。当传入 batch_sampler 后，就不用传入 sampler, batch_size, drop_last 以及 shuffle 关键字
3. num_workers 为调用线程的数量，没有固定说法设置多少最好。pin_memory 就是锁页内存，创建DataLoader时，设置pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些
4. 为了较好的可复现，mmengine 中还是用了 worker_init_fn 来给每个线程设置随机种子，这里不总结

**在后两行配置不变时，仅配置前两行即可完成对单卡和多卡（分布式）的 DataLoader 创建**

1. 单卡，直接配置第一行

   ```python
   DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)
   ```

2. 多卡，直接上 batch_sampler

   ```python
   sampler = DistributedSampler(dataset, seed=None, shuffle=False)
   batch_sampler = BatchSampler(sampler, batch_size=2, drop_last=False)
   DataLoader(dataset, batch_sampler=batch_sampler)
   ```

Sampler 的核心方法是 `__iter__`，即通过迭代不断生成 index，DistributedSampler 把数据集的总 index 分成了多个不重叠的子集，每个进程对应一个子集，然后在各自的子集中迭代生成 index。而 BatchSampler 则是生成一个 batch_size 长度的 index 序列

### Optimizer 接口整理

Pytorch 实现的 Optimizer 的输入主要由 model.parameters() 和其他超参数（如 learning_rate, weight_decay）。如果想要对特定层设置，可参考 [StackOverflow](https://stackoverflow.com/questions/51801648/how-to-apply-layer-wise-learning-rate-in-pytorch)，传入一个 list of dict 即可

mmengine 对 pytorch 优化器的包装还是比较轻的，除了 optimizer 原有的接口外，OptimWrapper 主要多了几个接口：

1. optim_wrapper.update_params(loss) 更新参数，替代 backward + step
2. optim_wrapper.get_lr() 获得学习率，替代原来的 `optimizer.param_groups[0]['lr']`
3. 加载优化器状态字典使用的原始接口 `state_dict & load_state_dict`

mmengine 中的 scheduler 和 pytorch 中的 scheduler 使用方法完全一致，但扩展了 scheduler 的使用范围，不仅仅能够对 lr 进行管理，还能对 momentum 进行管理。scheduler 的接口名称和 optimizer 的接口名称基本一致，使用 `scheduler.step()` 即可

scheduler 原理是根据当前步（last_step）和给定参数设置学习率，基本上不需要自己调整，直接参考文档 [mmengine.optim](https://mmengine.readthedocs.io/zh_CN/latest/api/optim.html) 写配置文件即可。要自己实现个性化的 scheduler 可以参考一下源码

### Dataset & DataSample

**BaseDataset 实现逻辑**

1. full_init() 方法
2. **load_data_list()，需要自己写，return a list of dict**，通常仅包含样本的路径和样本的标签

其他基本上就不需要了，接下来就是 pipline 的功能，生成完整的一个样本

通用的增强输出 PackxxInputs，需要进一步了解通用数据元素的设计

https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/data_element.html

在模型的训练/测试过程中，组件之间往往有大量的数据（images）和标签（labels）需要传递，不同的算法需要传递的数据和标签形式经常是不一样的

```python
# detection
for img, img_metas, gt_bboxes, gt_labels in data_loader:
    loss = retinanet(img, img_metas, gt_bboxes, gt_labels)
# segmentation
for img, img_metas, gt_bboxes, gt_masks, gt_labels in data_loader:
     loss = mask_rcnn(img, img_metas, gt_bboxes, gt_masks, gt_labels)
```

为了统一数据接口 mmengine 就对这**数据**和**标签**分别进行打包，该功能使用 PackxxxInputs 完成，最后输出两个对象 img & data_sample

```python
for img, data_sample in dataloader:
    loss = model(img, data_sample)
```

在实际实现过程中，mmengine 使用 DataSample 类来封装标签、预测结果信息，DataSample 由数据元素 xxxData 构成，数据元素为某种类型的预测或者标注 

#### BaseDataElement

为了更好的操作数据，实现了 BaseDataElement，其主要有如下功能

1. BaseDataElement 将数据分为 **data** 和 **metainfo** 两个部分，通过类的初始化将这两个部分构建到 BaseDataElement 中

   ```python
    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs) -> None:
           # metainfo 必须为字典
           # data 则以关键字 kwargs 直接加入
   base_data = BaseDataElement(metainfo=dict(h=1,w=2), size=100)
   ```

   二者都以 BaseDataElement 中的属性（attr）存在，区别在于 metainfo 不能够直接通过属性设置，只有 data 可以直接通过属性设置。修改 metainfo 需要使用 set_metainfo 方法

   ```python
   base_data = BaseDataElement(metainfo=dict(h=1,w=2), size=100)
   base_data.h = 2		# no!!
   base_data.set_metainfo(dict(h=2))	# yes
   base_data.size = 2					# yes
   ```

   删除属性可以直接使用 pop 方法，不管是 metainfo 还是 data 都管用

2. 实现了 new，clone，to，numpy，cuda，cpu 这些类似于张量中的方法，可以批量对 data 中的数据直接操作

3. 通过 print(BaseDataElement) 能够直观获得其中的 data 和 metainfo

#### InstanceData

- 对 `InstanceData` 中 data 所存储的数据进行了长度校验
- data 部分支持类字典访问和设置它的属性
- 支持基础索引，切片以及高级索引功能
- 支持具有**相同的 `key`** 但是不同 `InstanceData` 的拼接功能。 这些扩展功能除了支持基础的数据结构， 比如`torch.tensor`, `numpy.dnarray`, `list`, `str`, `tuple`, 也可以是自定义的数据结构，只要自定义数据结构实现了 `__len__`, `__getitem__` and `cat`.

#### DataSample

数据样本作为不同模块最外层的接口，提供了 xxxDataSample 用于单任务中各模块之间统一格式的传递。mmengine 对 xxxDataSample 的属性命名以及类型要进行约束和统一，保证各模块接口的统一性

对命名的约束是使用 @property 装饰器完成，利用 property setter 增加对属性的更改

### Default Hooks 功能

1. IterTimerHook
2. LoggerHook
3. ParamSchedulerHook
4. CheckpointHook
5. DistSamplerSeedHook
6. DetVisualizationHook，only works in test and val
7. RuntimeInfoHook

### 日志系统 MessageHub & MMLogger

`MessageHub` 的作用是在全局收集信息。收集的信息存储在 HistoryBuffer 里，这个 buffer 相当于一个队列，其最大容量为 window size，即最多缓存多少条数据，多余这个 window size，之前的数据就会被挤出去

日志系统通常使用两个功能：

1. 更新 meesage hub

   ```python
   from mmengine.logging import MessageHub
   
   message_hub = MessageHub(name='name_for_message_hub')
   message_hub.update_scalar('train/loss', loss)
   # update with dict
   message_hub.update_scalrs(log_dict)
   ```

   update_scalar 可以自动将数据转换成 python built-in 类型。要获取数据可通过下面方法

   ```python
   buffer = message_hub.get_scalar('train/loss')	# 获取 buffer
   # buffer.data 返回一个 tuple: (log_data, counts)
   # counts代表对应的数据的重复次数
   # len(log_data) == len(counts)
   buffer.data[0]	# normally, an ndarray
   buffer.mean()
   buffer.max()
   buffer.min()
   buffer.current()
   ```

2. 向文件写入日志

   ```python
   from mmengine.logging import MMLogger 
   
   logger = MMLogger.get_instance(name='mmengine', 
                                  log_level='INFO', log_file='tmp.log')
   logger.info(log_string)
   ```

**全局性质的理解**

MMLogger 和 MessageHub 都继承了 ManagerMixin，这个类的主要功能就是能够**全局调用实例**。举个例子，假设在某个地方创建了一个 MMLogger 实例，那么通过 `ManagerMixin.get_instance()` 能够在其他任何地方都能够获取这个 MMLogger 实例。该功能的实现需要通过**元类 meta class** 完成，我也不理解其中细节，模糊一点说，我们把创建的实例都保存在了元类的一个字典里面，而这是一个全局可获取的空间

### 可视化系统 Visualizer

mmengine 的 visualizer 有两个功能：

1. 常规画图

   可视化器提供了常用对象的绘制接口，例如绘制**检测框、点、文本、线、圆、多边形和二值掩码**。这些基础 API 支持以下特性：

   - 可以多次调用，实现叠加绘制需求
   - 均支持多输入，除了要求文本输入的绘制接口外，其余接口同时支持 Tensor 以及 Numpy array 的输入

   ```python
   import mmcv
   from mmengine.visualization import Visualizer
   
   file = '/mmdetection/demo/demo.jpg'
   img = mmcv.imread(file, channel_order='rgb')
   vis = Visualizer(image=img)
   vis.show()
   
   visualizer.set_image(image=image)
   visualizer.draw_texts("cat and dog", torch.tensor([10, 20]))
   visualizer.draw_bboxes(torch.tensor([72, 13, 179, 147]), edge_colors='r', line_widths=3)
   ```

   为了一些 fancy 的需求还可以将 [特征图可视化](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/visualization.html#id3)

   ```python
   visualizer.draw_featmap(feat, channel_reduction='select_max')
   visualizer.draw_featmap(feat, image, channel_reduction='select_max')
   ```

   channel_reduction 还有 squeeze_mean, None 选项，使用 None 则需要搭配 topk 参数

2. 把数据存储到 backend 中，backend 目前有三种

   1. local backend
   2. tensorboard
   3. wandb

   使用时需要在初始化时指定 backend 和 save_dir 参数

   ```python
   visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')], save_dir='temp_dir')
   ```

   可以存储的数据类型有很多

   - add_config 写配置到特定存储后端，config 必须是 Config 类
   - add_image 写图片到特定存储后端
   - add_scalar 写标量到特定存储后端
   - add_scalars 一次性写多个标量到特定存储后端

## TODO

便捷的分布式接口

Metric & Evaluator

coco api & coco metric