## MMDetection Dataset & Model & Runner

之前的**二周目**整理主要关注 training flow，除了 training 外，还需要进一步了解 Dataset & Model。现在要进一步对其中的**核心类**进行整理，包括他们的构建逻辑、前向方程等

## Dataset & DataLoader

和之前整理的 ONCESemiDataset 的逻辑是相似的，使用一个基类 `DatasetTemplate or CustomDataset` 处理统一格式下的数据集，对这些数据集进行数据增强。具体的数据集就由具体的子类进行处理，转化到统一的格式下

### CustomDataset

这是所有数据集的基类，有两个核心函数，`__init__ & prepare_train/test_data`

1. `__init__` 初始化数据集的基本信息

   ```python
       def __init__(self,
                    data_root,
                    ann_file,
                    pipeline=None,
                    classes=None,
                    filter_empty_gt=True,
                    test_mode=False):
           super().__init__()
           self.data_root = data_root
           self.ann_file = ann_file
           self.test_mode = test_mode
           self.CLASSES = self.get_classes(classes)
           self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
           self.data_infos = self.load_annotations(self.ann_file)
   		...
           if pipeline is not None:
               self.pipeline = Compose(pipeline)
   ```

   比较重要的是这个 `self.pipline`，这是处理整个数据集的关键，加载数据、数据增强都在 pipline 里完成。`pipline` 本身是一个 list of dict

   ```python
   train_pipeline = [
       dict(type='LoadMultiViewImageFromFiles_BEVDet',...),
       dict(type='LoadPointsFromFile',...),
       dict(type='LoadAnnotations3D',...),
       dict(type="DefaultFormatBundle"),
       dict(type="Collect",
            keys=["img", "gt_bboxes", "gt_labels"],...)
   ```

   通过 `Compose` 类以及 registry 机制把这些类别实例化，然后存储到一个列表 `transform` 里，把 data dict 依次通过所有的 transform 就能得到最终的数据集了，下面就是 `Compose` 的 `__call__` 函数

   ```python
       def __call__(self, data):
           for t in self.transforms:
               data = t(data)
               if data is None:
                   return None
           return data
   ```

2. `prepare_train/val_data` 理解了 `pipeline` 过后这里就非常简单了，直接看代码

   ```python
       def prepare_train_data(self, index):
           input_dict = self.get_data_info(index)
           if input_dict is None:
               return None
           self.pre_pipeline(input_dict)
           example = self.pipeline(input_dict)
           return example
   ```

   其中 `self.pre_pipeline`，就是给 `input_dict` 增加一些固定 key，并初始化 value（大多初始化为空列表）
   
   `__getitem__` 方法就是通过调用 `prepare_train/val_data` 来实现加载数据集
   
   ```python
       def __getitem__(self, idx):
           """Get item from infos according to the given index.
   
           Returns:
               dict: Data dictionary of the corresponding index.
           """
           if self.test_mode:
               return self.prepare_test_data(idx)
           while True:
               data = self.prepare_train_data(idx)
               if data is None:
                   idx = self._rand_another(idx)
                   continue
               return data
   ```

其实整体逻辑就如上面整理的一样，但是还有几个核心函数没有整理，下面就叙述实现个性化的数据集的步骤，在这个过程中就描述这些核心函数的功能：

1. 继承 `CustomDataset`
2. 通过 config or 直接传入数据集的类别以及其它基础信息
3. 重写以下方法以获得数据集样本：
   1. `load_annotations`，该方法将获得所有的 sample info，info 一般可包含了数据路径，样本的 gt，其他 meta 信息等
   2. `get_data_info(idx)`，这一步就获得了单个 sample 的 `input_dict`
4. 构建 `evaluate`，一般各个数据集都有自己的评测函数

### collate function

collate 的目的就是把多个 sample 合到一块去。mmdet 的 collate function 逻辑也是类似的，将字典中相同关键字的数据融合到一起，融合过程可以使用 `DataContainer` 帮助处理，这里不整理，因为这个 DataContainer 最终也会被消除，返回最原始的数据形式...最好的知道 batch content 的方式就是直接看 forward 函数里的输入，如果不清楚的地方就去具体的 pipline 看一看，不要被 mmdet 繁杂的封装给绕进去了。如果需要调试的话把 `DataLoader` 的 `num_workers` 设置为0就好，就可以使用 pdb 进行断点调试了

最后再提一下最后两个 pipline `DefaultFormatBundle & Collect`，前者就是把数据用 DataContainer 包装一下，后者就是挑选需要的关键字形成最终的 batch dict

## Model

mmdet 的模型都是用 MMDataParallel 进行封装的，也不对此进行整理。主要看模型的构建，以及前向方程

1. 模型的构建一般在 `builder` 中 import 就好，逻辑还是在二周目中整理的 registry 逻辑。下面看看示意代码

   ```python
   from ..builder import DETECTORS, build_backbone, build_head, build_neck
   
   @DETECTORS.register_module()
   class TwoStageDetector(BaseDetector):
       """Base class for two-stage detectors.
   
       Two-stage detectors typically consisting of a region proposal network and a
       task-specific regression head.
       """
   
       def __init__(self,
                    backbone,
                    neck=None,
                    ...
                    init_cfg=None):
           super(TwoStageDetector, self).__init__(init_cfg)
           self.backbone = build_backbone(backbone)
           if neck is not None:
               self.neck = build_neck(neck)
   ```

2. 模型的前向方程就显得比较绕了。我们得从 runner 中的 `run_iter` 开始跟踪...

   ```python
       def run_iter(self, data_batch, train_mode, **kwargs):
           if train_mode:
               outputs = self.model.train_step(data_batch, self.optimizer,
                                               **kwargs)
           else:
               outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
   ```

   调用的是 `train_step` 方法，这个方法不是在 `model` 中定义好的，而是由封装的 MMDataParallel 类定义的

   ```python
       def train_step(self, *inputs, **kwargs):
           inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
           return self.module.train_step(*inputs[0], **kwargs[0])
   ```

    可以理解为将之前的 DataContainer 解包，然后再调用真正模型里的 `train_step`

   ```python
       def train_step(self, data, optimizer):
           losses = self(**data)
           loss, log_vars = self._parse_losses(losses)
   
           outputs = dict(
               loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))
   
           return outputs
   ```

   这里的 optimizer 是没有被使用到的，data 可以理解为通过 `getitem` 获得的字典内容。此时就正式调用了模型的前向方程

   ```python
       def forward(self, img, img_metas, return_loss=True, **kwargs):
           if return_loss:
               return self.forward_train(img, img_metas, **kwargs)
           else:
               return self.forward_test(img, img_metas, **kwargs)
   ```

   这里就是最后一层封装啦，之后的 `forward_train` 就是最核心的前向方程函数了

3. loss 处理。一般模型前向方程返回的是一个 loss dict，包含了不同的损失函数结果，mmdet 使用 `_parse_losses` 函数去处理，下面仅贴注解

   ```python
       def _parse_losses(self, losses):
           """Parse the raw outputs (losses) of the network.
   
           Args:
               losses (dict): Raw output of the network, which usually contain
                   losses and other necessary information.
   
           Returns:
               tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                   which may be a weighted sum of all losses, log_vars contains \
                   all the variables to be sent to the logger.
           """
   ```

   最终返回的是字典里一个总 loss 以及需要放入 log 的 loss **数值**。这里也要求了，你的 loss dict 里需要求反向传播的 loss 的 key 需要包含 `loss` 关键字，例如：`cls_loss or **_loss`

## End

至此 mmdetection 的框架就已经清晰了，不得不说大的框架确实有很多封装，牺牲了一定的易用性来确保可扩展性
