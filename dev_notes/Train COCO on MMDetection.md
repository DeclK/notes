#   Train COCO on MMDetection

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

**registry 在使用的使用需要注意一下 scope，scope 是根据模块所在的 package 的名字确定的，可用 `DefaultScope` 来完成相关操作**

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

光看文档完全没办法理解 runner，还是得看看代码。过完一遍后总结：Runner 就是一个大工厂，所有的组件都是其中的属性，组件与组件之间能够通过 runne 进行相互配合，完成所有的流程

#### Runner 初始化

runner 的初始化采用了一个 lazy init 的策略。所谓 lazy init 就是指先把 cfg 赋值给某个组件，如 `self.dataloader = dataloader_cfg`，在之后需要用这个组件的时候，再用 cfg 构建真正的实例

1. deepcopy cfg，新建属性 self.cfg
2. 创建属性 `self.traininig_related, self.val_related, self.test_related`。每个 related 为 `[xxx_dataloader, xxx_cfg, xxx_evaluator]`
3. 创建属性 `self.optim_wrapper`
4. 创建属性 `self._launcher`，决定是否为分布式，并创建属性 `self._distributed`
5. `self.setup_env` 初始化 dist 环境，并新建属性 `self._rank, self._world_size`
6. `self.set_random_seed`，新建属性 `self.seed, self.deterministic`，可通过 `randomness=dict(seed=None)` 配置随机种子
7. 创建 `work_dir`
8. 创建属性 `self.logger`，logger 此时记录下环境信息和配置文件
9. 创建属性 `self.load_from, self.resume`
10. 创建属性 `self.model`，并将模型打包，打包完成的事情如下:
    1. 把模型送到对应的设备上 `model.to(device)`
    2. 如果为分布式训练则将 model 打包为 `MMDistributedDataParallel`，当然也可以使用 pytorch 的 `DistributedDataParallel`，不过需要单独设置。`MMDistributedDataParallel` 继承于 DDP，并新定义了三分方法 `tran_step, val_step, test_step` 来调用 model 中定义的 `tran_step, val_step, test_step` 
11. 注册 hooks，并保存进属性 `self._hooks`
12. 输出 config，`cfg.dump(file_path)`

#### Runner.train()

1. 检查 model 是否有 `train_step` 属性/方法。这里是对模型的基本要求。如果有 `val_loop`，也得检查是否有 `val_step`
2. 创建属性 `self.train_loop`。补充知识：一个类定义时传入参数 metaclass=ABCMeta 表示该类为抽象类，不能够实例化，只能用来继承

3. 创建属性 `self.optim_wrapper`，并使用 `scale_lr` 自动缩放学习率
4. 创建属性 `self.param_schedulers`，管理学习率策略
5. 创建属性 `self.val_loop`
6. 运行钩子 `self.call_hook('before_run')`
7. 初始化模型权重，如果有预训练权重则 load
8. 运行训练循环 `self.train_loop.run()`
9. 运行钩子 `call_hook('after_run')`

### Runner 中 train_loop 逻辑

`BaseLoop` 是一个非常简单的类，只需要 runner 和 dataloader 作为初始化即可。`EpochBasedTrainLoop` 继承 `BaseLoop`，其核心逻辑在 `run` 方法

`run` 将循环运行 `run_epoch`，并在 epoch 后判断是否需要 eval

`run_epoch` 是由循环 `run_iter` 完成，循环以 dataloader 主导

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

自己在写个性化 Loops 的时候最好要将这些钩子都加上，以保证结果的正确！例如 `DefaultSampler` 的随机种子要在各个 epoch 开始前重新设置，这需要调用 `DistSamplerSeedHook` 完成

### TODO: Runner 中 val_loop 逻辑

metric 如何计算，如何传递，如何保存

### Model 中 train_step 逻辑

核心代码非常简单：数据预处理+前向损失+更新参数

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

#### DataPreprocessor

由于 collate_fn 使用的是一个非常简单的方法，所以**数据预处理放在了 DataPreprocessor 中**，其功能包括把数据发送到 GPU 上，数据打包，归一化，最后返回 data 字典（包含 data['inputs'] & data['data_sample']）

这里说明一下 DataPreprocessor **把数据发送到 GPU 上** 这个功能，写得有点隐晦：在 `BaseModel` 里为这一个功能重写了模型的 `to & cuda & cpu` 这几个方法，就是为了额外设置 DataPreprocessor 的 `device` 属性，保证了属于与模型的 `device` 是统一的，直接使用 `model.to(device)` 即可

#### parse_losses

mmengine 期望模型在训练时的输出是一个字典，`parse_losses` 将输出字典中包含 `'loss'` 键值对全都找出来放到 `log_vars` 中，然后再求和，形成最终的 `loss`，最终返回 `loss & log_vars`，前者用于反向传播，后者用于日志记录

### 如何自己写 Config 配置文件

建议是从 `_base_` 中去继承 `default_runtime.py`，然后再挑选修改。总体来讲核心如下

```python
# dataset
dataset = dict(type='COCO')
train_pipeline = [dict(type='LoadImageFromFile')]
train_dataloader = dict(batch_size=16, dataset=dataset, sampler=, pipline=train_pipline)

test_pipline = ...
val_dataloader = ...

val_evaluator = dict(type='CocoMetric', ann_file=...)

# model
model = dict(type='DETR',...)
data_preprocessor = dict(type='BaseDataPreprocessosr')

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

visualizer = dict(type='DetLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')], name='visualizer')
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

1. sampler 和 shuffle 两个参数是互斥的，有了 sampler 后 shuffle 将不再起作用。**实际上几乎可以不用 sampler 这个参数**
2. `batch_sampler` 是以一个 `Sampler` 作为基础，再进行 group 操作。当传入 batch_sampler 后，就不用传入 `sampler, batch_size, drop_last, shuffle` 关键字
3. `num_workers` 为调用线程的数量，没有固定说法设置多少最好。`pin_memory` 就是锁页内存，创建 DataLoader 时，设置 pin_memory=True，则意味着生成的Tensor数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些
4. 为了较好的可复现，mmengine 中还是用了 `worker_init_fn` 来给每个线程设置随机种子，这里不总结

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
   
   # before each epoch start
   sampler.set_epoch(epoch_number)
   ```

`Sampler` 的核心方法是 `__iter__`，即通过迭代不断生成 index，`DistributedSampler` 把数据集的总 index 分成了多个不重叠的子集，每个进程对应一个子集，然后在各自的子集中迭代生成 index。而 `BatchSampler` 则是生成一个 `batch_size` 长度的 index 序列

**mmengine 中的 DefaultSampler 能够同时处理分布式和非分布式的采样，再包一个 BatchSampler 就能够处理批采样了**，使用的 `collate_fn` 为 `pesudo_collate` 就是 pytorch 默认的 [collate function](https://pytorch.org/docs/stable/data.html#torch.utils.data.default_collate) 但是不转换数据为 tensor

### Optimizer 接口整理

Pytorch 实现的 Optimizer 的输入主要由 `model.parameters()` 和其他超参数（如 `lr, weight_decay`）。如果想要对特定层设置，可参考 [StackOverflow](https://stackoverflow.com/questions/51801648/how-to-apply-layer-wise-learning-rate-in-pytorch)，传入一个 list of dict 即可

mmengine 对 pytorch 优化器的包装还是比较轻的，除了 optimizer 原有的接口外，[OptimWrapper](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/optim_wrapper.html) 主要多了几个接口：

1. `optim_wrapper.update_params(loss)` 更新参数，替代 backward + step
2. `optim_wrapper.get_lr()` 获得学习率，替代原来的 `optimizer.param_groups[0]['lr']`
3. 加载优化器状态字典使用的原始接口 `state_dict & load_state_dict`

mmengine 中的 scheduler 和 pytorch 中的 scheduler 使用方法完全一致，但扩展了 scheduler 的使用范围，不仅仅能够对 lr 进行管理，还能对 momentum 进行管理。scheduler 的接口名称和 optimizer 的接口名称基本一致，使用 `scheduler.step()` 即可

scheduler 原理是根据当前步（last_step）和给定参数设置学习率，基本上不需要自己调整，直接参考文档 [mmengine.optim](https://mmengine.readthedocs.io/zh_CN/latest/api/optim.html) 写配置文件即可。要自己实现个性化的 scheduler 可以参考一下源码

### Dataset & DataSample

#### BaseDataset 实现逻辑

如果要自己写一个 dataset 主要考虑重写两个方法

1. `full_init()` 方法
2. **`load_data_list()`**，需要自己写，return a list of dict 并赋为属性 `self.data_list`，通常仅包含样本的路径和样本的标签

其他基本上就不需要了，接下来就是用 `__getitem__` 配合 `self.pipline`，生成完整的一个样本

```python
    def __getitem__(self, idx: int) -> dict:
        if not self._fully_initialized: self.full_init()

        data = self.prepare_data(idx)
        return data
    
    def prepare_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)
        return self.pipeline(data_info)
    
    def get_data_info(self, idx: int) -> dict:
        data_info = copy.deepcopy(self.data_list[idx])
        return data_info
```

#### PackxxxInputs

通用的增强输出 PackxxInputs，需要进一步了解通用数据元素的设计

在模型的训练/测试过程中，组件之间往往有大量的数据（images）和标签（labels）需要传递，不同的算法需要传递的数据和标签形式经常是不一样的

```python
# detection
for img, img_metas, gt_bboxes, gt_labels in data_loader:
    loss = retinanet(img, img_metas, gt_bboxes, gt_labels)
# segmentation
for img, img_metas, gt_bboxes, gt_masks, gt_labels in data_loader:
     loss = mask_rcnn(img, img_metas, gt_bboxes, gt_masks, gt_labels)
```

为了统一数据接口 mmengine 就对这**数据**和**标签**分别进行打包，该功能使用 `PackxxxInputs` 完成，最后输出的 data 只有两个关键字 `inputs & data_sample`，其中 `inputs` 一般为图像本身，而 `data_sample` 为 gt 标签，由 `DataSample` 表示

```python
for img, data_sample in dataloader:
    loss = model(img, data_sample)
```

在实际实现过程中，mmengine 使用 `DataSample` 类来封装标签、预测结果信息，`DataSample` 由数据元素 `xxxData` 构成，数据元素为某种类型的预测或者标注，继承于 BaseDataElement 类。下面从下到上介绍介绍 `DataSample`

##### BaseDataElement

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

##### InstanceData

- 对 `InstanceData` 中 data 所存储的数据进行了长度校验
- data 部分支持类字典访问和设置它的属性
- 支持基础索引，切片以及高级索引功能
- 支持具有**相同的 `key`** 但是不同 `InstanceData` 的拼接功能。 这些扩展功能除了支持基础的数据结构， 比如`torch.tensor`, `numpy.dnarray`, `list`, `str`, `tuple`, 也可以是自定义的数据结构，只要自定义数据结构实现了 `__len__`, `__getitem__` and `cat`.

##### DataSample

数据样本作为不同模块最外层的接口，提供了 xxxDataSample 用于单任务中各模块之间统一格式的传递。mmengine 对 xxxDataSample 的属性命名以及类型要进行约束和统一，保证各模块接口的统一性

对命名的约束是使用 @property 装饰器完成，利用 property setter 增加对属性的更改

### Default Hooks 功能

1. IterTimerHook，记录每一个 iteration 实用的时间

2. **LoggerHook**，日志将根据 interval 进行采样，最终输出到 terminal，并保存到日志文件和 visualization backend 中，逻辑如下

   ```python
           if self.every_n_inner_iters(batch_idx, self.interval):
               tag, log_str = runner.log_processor.get_log_after_iter(
                   runner, batch_idx, 'train')
           runner.logger.info(log_str)
           runner.visualizer.add_scalars(
               tag, step=runner.iter + 1, file_path=self.json_log_path)
   ```

   `log_processor` 是从 message hub 中获得信息，然后将信息格式化便于输出，其中 `tag` 是一个字典，`log_str` 就是将 tag 格式化后的字符串

3. **ParamSchedulerHook**，在每一个 epoch or iter 过后更新学习率

4. CheckpointHook，保存模型，optimizer，scheduler，以及一些 meta 信息（运行的 epoch or iteration 等）

5. **DistSamplerSeedHook**，`before_train_epoch` 设置随机种子 `set_epoch`

6. DetVisualizationHook，only works in test and val

7. **RuntimeInfoHook**，这里会将运行时的信息放入 message hub 当中，包括 meta，lr，loss，metrics

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

   `update_scalar` 可以自动将数据转换成 python built-in 类型。要获取数据可通过下面方法

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

### Metric & Evaluator

`Evaluator` 是一个 `Metric` 容器，包含多个 `Metric`，即可以进行多种指标的评估。同时 `Evaluator` 也增加了分布式的功能，能够将多个 GPU 上的推理结果合并起来，最终送到 CPU 上进行计算

自定义的 `Metric` 需要实现两个方法

1. `process`，这个方法的功能很简单，就是单纯的存储预测结果和标签到 `Metric` 中的 `self.results` 当中
2. `evaluate` 这个方法就是将 `self.results` 中的结果进行整合计算，最终输出一个结果**字典**

mmengine 实现了一个 `DumpResults` 的 `Metric` 类，如果需要可以将预测的结果保存，只需要指定 `out_file_path` 即可

### BaseModel 设计原则

之前介绍了模型的 `train_step`，实际上 `BaseModel` 有三个接口：

1. `train_step`
2. `val_step`
3. `test_step`

不直接使用模型的 `forward` 方法，因为各个 step 中还包含了对数据的预处理，以及模型参数更新。所以最好把 `BaseModel` 看作对模型的封装，而不是模型本身！

mmengine 要求模型的 `forward` 方法接受的参数即为 `DataLoader` 的输出 `data_batch`。如果 `DataLoader` 返回元组类型的数据 `data`，`forward` 需要能够接受 `*data` 的解包后的参数；如果返回字典类型的数据 `data`，`forward` 需要能够接受 `**data` 解包后的参数。 `mode` 参数用于控制 `forward` 的返回结果，通常会再使用一个父类来封装一层

```python
    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
```

## TODO

便捷的分布式接口

coco api & coco metric

einops for projects，我把 subway 项目的一些总结也放到里面来，因为这是一个完整的项目

position embeddings

增加一个记录 model 结构的 log