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



## Model

模型建造、前向逻辑，使用 MMParallel 进行封装

## Runner

runner 前向逻辑
