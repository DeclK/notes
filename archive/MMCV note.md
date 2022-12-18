---
title: MMCV tutorial
tags:
  - MMCV
categories:
  - 编程
  - OpenMMLab
abbrlink: 81012a8b
date: 2021-09-14 16:39:47
---

# Understand MMCV

现在想要更深一步学习 mmdetection，于是必不可少地要来了解一下 MMCV 这个基础库。从之前的学习过程来看，一个思想就是一切皆为 class。配置文件有 Config class，管理模块有 Registry class，运行模型有 Runner class...下面来具体看看这些类的一些基础框架

## Config

> `Config` class is used for manipulating config and config files. It supports loading configs from multiple file formats including **python**, **json** and **yaml**. It provides dict-like apis to get and set values.

从 mmdetection 代码来看，主要是从 python 文件创建 config class

### 创建

文档举了一个例子

```python
# test.py
a = 1
b = dict(b1=[0, 1, 2], b2=None)
c = (1, 2)
d = 'string'
pre_defined = '{{ fileDirname }}'
```

从 `test.py` 创建 config class

```python
from mmcv.utils.config import Config
cfg = Config.fromfile('test.py')
print(cfg)
# print result
# Config (path: a.py): {'a': 1, 'b': {'b1': [0, 1, 2], 'b2': None}, 'c': (1, 2), 'd': 'string',
# 'pre_defined': '/home/hongkun/mmdetection'}
```

config class 也支持4个预先定义的变量 `{{ var }}`

`{{ fileDirname }}` : 当前文件路径

`{{ fileBasename }}`: 当前文件名，如 `test.py`

`{{ fileBasenameNoExtension }}`:  当前文件名，无扩展名，如 `test`

`{{ fileExtname }}`: 当前文件扩展名，如 `.py`

### 继承 inheritance

想要使用其他配置文件中的内容，只需要添加继承关键字 `_base_ = file or list_of_file` 即可

```python
# inheritance.py
_base_ = 'test.py' # or a list like, ['config_1.py', 'config_2.py']
b = dict(b2=1)
# b = dict(b3=1, _delete_=True)
e = 'new config'
```

继承之后，`inheritance.py` 除了自身的配置，还会包含 `test.py` 中的所有配置

当遇到重复的关键字时，例如两个配置文件中都有 `b` 关键字，config class 也支持修改，此时情况分为：

1. key 对应的 value 为字典，那么会将这个字典与需要继承的字典进行融合，如果想要忽略继承文件夹中的同名配置，则需要添加 `_delete_=True` 键值
2. key 对应的 value 为为其他类型，则将使用当前的 value

## Registry

registry class 是整个 MMCV 的大管家，负责将**类**进行注册并配合 config 中的信息将类实例化，这个类可以是任何已实现的类：可以是模型，也可以是其他类比如 Runner, Datasets...本质上 registry 完成的是一个映射： string --> class，这里的 string 通常在 config 中对应着 `type` 字段。你在 config 文件中只要看到了 `type='CLASS_NAME'` 那么就一定可以在项目代码中找到对应的类的实现，而 `type` 之后填写的其他关键字，正是这些类需要的初始化参数。所以大可以把 registry 看作是一个高级的字典

使用 registry 管理类需要3个步骤（一般只要2个）：

1. 创建 build function (一般不用自己创建，Registry 有自带 build_from_cfg 函数)
2. 创建 registry 类
3. 把待管理的类注册到 registry 中，即可使用 registry 管理模块

### Example

假设要管理一个 `Converter class`，准备在 `Converters` 文件夹中实现。先尝试用一个 python 文件： `builder.py` 实现 registry 的管理功能，并通过 registry 实例化 Converter

```python
# Converters/builder.py
from mmcv.utils import Registry

# 1. create a build function
def build_converter(cfg, registry, *args, **kwargs):
    cfg_ = cfg.copy()
    converter_type = cfg_.pop('type')
    if converter_type not in registry:
        raise KeyError(f'Unrecognized converter type {converter_type}')
    else:
        converter_cls = registry.get(converter_type)

    converter = converter_cls(*args, **kwargs, **cfg_)
    return converter


# 2. create a registry for converters
CONVERTERS = Registry('Converter', build_func=build_converter)
# if use default build_func: CONVERTERS = Registry('Converter')


# 3. use the registry to manage the module
@CONVERTERS.register_module()
class Converter1(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
 

# use this converter through configs
converter_cfg = dict(type='Converter1', a='a_value', b='b_value')
converter = CONVERTERS.build(converter_cfg)
print('converter attribute:', converter.a, converter.b)
```

上面就是 registry 的工作逻辑：先建造 registry 类，再将待管理的类通过装饰器注册到 registry 类中，最后使用 cfg 建造实例

如果去看 mmdetection 的代码会发现，其 `builder.py` 是写得非常简单的，并没有包含 `build_func` 以及具体管理的类。是因为其使用的是默认的 `build_func=build_from_cfg` 并且将其他类用单独文件保存，之后通过 `__init__.py` ，将所有的类注册到 builder 中的 registry 类中

### 层级注册 Hierarchy Registry

这里我更想叫它继承注册。在 registry 类中提供了 parent 参数，parent 必须也是 registry 类。这样子 registry 就能够继承 parent registry 的 build_func 来进行实例化，并且每一个类既能够使用注册在自己名下的类，通过指定”族谱“也能够也使用其他 registry 管理的类。但这个”族谱“是按照 OpenMMLab 项目划分的，不是我们能修改的，所以个这继承注册的意义是在于 OpenMMLab 不同项目之间的模型都可以调用，例如 mmdetection 可以调用 mmclassification 中的模型。请前往文档参考 [更多](https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html#hierarchy-registry)

## Runner

Runner 是管理训练过程的类，有两个子类 `EpochBasedRunner` and `IterBasedRunner`。Runner 也能够在 train 和 val 两种模式中切换，和 hook 结合可以在训练过程中有更多拓展功能

 ### EpochBasedRunner

该 runner 的工作流是基于 epoch 的，每一个 epoch 将包含多个 iterations。比如当 `workflow = [('train', 2), ('val', 1)]` 该 runner 就会运行2次训练集1次验证集。下面看看其运行的基本逻辑

```python
# the condition to stop training
while curr_epoch < max_epochs:
    # traverse the workflow.
    # e.g. workflow = [('train', 2), ('val', 1)]
    for i, flow in enumerate(workflow):
        # mode(e.g. train) determines which function to run
        mode, epochs = flow
        # epoch_runner will be either self.train() or self.val()
        epoch_runner = getattr(self, mode)
        # execute the corresponding function
        for _ in range(epochs):
            epoch_runner(data_loaders[i], **kwargs)
```

如果是 train 模式，那么将调用类似如下函数

```python
# Currently, epoch_runner could be either train or val
def train(self, data_loader, **kwargs):
    # traverse the dataset and get batch data for 1 epoch
    for i, data_batch in enumerate(data_loader):
        # it will execute all before_train_iter function in the hooks registered. You may want to watch out for the order.
        self.call_hook('before_train_iter')
        # set train_mode as False in val function
        self.run_iter(data_batch, train_mode=True, **kwargs)
        self.call_hook('after_train_iter')
   self.call_hook('after_train_epoch')
```

以上两段代码就展示了 runner 的基本逻辑：遍历整个训练集，并在每次遍历前后运行 hook 进行拓展操作

### IterBasedRunner

与 `EpochBasedRunner` 类似，但是是基于 iteration 的 runner 实现，一个 iteration 将包含一个 batchsize。如果定义了 `workflow = [('train', 2), ('val', 1)]` 那么将会循环运行2个训练集 iterations 和1个验证机 iteration。下面也看看其运行逻辑

```python
# Although we set workflow by iters here, we might also need info on the epochs in some using cases. That can be provided by IterLoader.
iter_loaders = [IterLoader(x) for x in data_loaders]
# the condition to stop training
while curr_iter < max_iters:
    # traverse the workflow.
    # e.g. workflow = [('train', 2), ('val', 1)]
    for i, flow in enumerate(workflow):
        # mode(e.g. train) determines which function to run
        mode, iters = flow
        # iter_runner will be either self.train() or self.val()
        iter_runner = getattr(self, mode)
        # execute the corresponding function
        for _ in range(iters):
            iter_runner(iter_loaders[i], **kwargs)
```

### 使用 Runner

一般来讲使用 runner 来进行训练有4个基本步骤：

1. 初始化 dataloader, model, optimizer, etc.
2. 初始化 runner：将 config, model, optimizer, etc. 传入 Runner 中进行实例化
3. 注册 training hooks & customized hooks
4. 开始训练：`runner.run(data_loaders, cfg.workflow)`

具体示例代码请移步 [MMCV Runner](https://mmcv.readthedocs.io/en/latest/understand_mmcv/runner.html#a-simple-example)

## File IO

mmcv 给导入和输入文档设有 API，可以导入不同格式的文档，例如 json, yaml, pkl。下面来看看如何导入一个 json 文件为字典

```json
// file.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

```python
import mmcv

file = mmcv.load('file.json')
# load data from a file-like object
with open('file.json', 'r') as f:
    data = mmcv.load(f, file_format='json')
print(type(file))
print(file)
# <class 'dict'>
# {'version': '0.2.0', 'configurations': [{'name': 'Python: Current File', 'type': 'python', 'request': 'launch', 'program': '${file}', 'console': 'integratedTerminal', 'justMyCode': False}]}
```

也可以使用 `mmcv.dump` 来转化文件或者输出文件

```python
import mmcv

# 将字典转化为字符串
info = dict(name='Declan', age=23)
dump_info = mmcv.dump(info, file_format='yaml')
print(dump_info, type(dump_info))
# age: 23
# name: Declan
# <class 'str'>

# 将文件保存为 info.yaml
mmcv.dump(info, 'info.yaml')
```

但是以上的 API 不能导入 txt 格式文件，需要使用 `mmcv.list_from_txt` or `mmcv.dict_from_txt` 将 txt 文件导入为列表或者字典

```a.txt
1 cat
2 dog cow
3 panda
```

```python
mmcv.dict_from_file('a.txt')
# {'1': 'cat', '2': ['dog', 'cow'], '3': 'panda'}
mmcv.list_from_file('a.txt')
# ['1 cat', '2 dog cow', '3 panda']
```

## Data Process

> This module provides some image processing methods, which requires `opencv` to be installed.

### Image

#### Read/Write/Show

对于图片的基本操作：读、写、展示，分别使用 `imread`，`imwrite`，`imshow` 三个 API

```python
import mmcv

# Read & Write
img = mmcv.imread('test.jpg')
mmcv.imwrite(img, 'out.jpg')

# Show
mmcv.imshow('tests/data/color.jpg')
# Show with bboxes
bboxes = np.array([[0, 0, 50, 50], [20, 20, 60, 60]])
mmcv.imshow_bboxes(img, bboxes)

# Show with ndarray
for i in range(10):
    img = np.random.randint(256, size=(100, 100, 3), dtype=np.uint8)
    mmcv.imshow(img, win_name='test image', wait_time=200)
```

#### More

对于图片还有更多的操作就不在这里列举了，例如色域转换、剪裁、旋转等，参考 [MMCV 文档](https://mmcv.readthedocs.io/en/latest/understand_mmcv/data_process.html#data-process)

### Video

关于视频模块主要实现3个功能：

1. 读取视频
2. 编辑视频
3. 处理光流文件

#### VideoReader

该 API 能够过得视频的一些基本信息，并能够视频中的每一帧进行索引或遍历

```python
video = mmcv.VideoReader('test.mp4')

# obtain basic information
print(len(video))
print(video.width, video.height, video.resolution, video.fps)

# iterate over all frames
for frame in video:
    print(frame.shape)

# read the next frame
img = video.read()

# read a frame by index
img = video[100]

# read some frames
img = video[5:10]
```

该模块内置方法 `cvt2frames` 可以将 video 转化为 images，同时 `mmcv.frames2video` 也可以将 images 转化为 video

```python
# split a video into frames and save to a folder
video = mmcv.VideoReader('test.mp4')
# convert a video to frame images
video.cvt2frames('out_dir')

# generate video from frames
mmcv.frames2video('out_dir', 'test.avi')
```

如果熟悉 ffmpeg 的话，通过 ffmpeg 能够实现更多格式的转化，可以使用 python 中的 `os.system` 执行命令行

```python
import os
os.system('ffmpeg -version')
```

mmcv 中也有一些接口借用了 ffmpeg 以实现对视频的更多操作

```python
# cut a video clip
mmcv.cut_video('test.mp4', 'clip1.mp4', start=3, end=10, vcodec='h264')

# join a list of video clips
mmcv.concat_video(['clip1.mp4', 'clip2.mp4'], 'joined.mp4', log_level='quiet')

# resize a video with the specified size
mmcv.resize_video('test.mp4', 'resized1.mp4', (360, 240))

# resize a video with a scaling ratio of 2
mmcv.resize_video('test.mp4', 'resized2.mp4', ratio=2)
```

#### Optical flow

mmcv 也对光流文件的处理提供支持，但我对于其了解不多，暂时不作整理。这里贴一个链接 [MMCV Optical flow](https://mmcv.readthedocs.io/en/latest/understand_mmcv/data_process.html#optical-flow)

## TODO

### KITTI 数据集

KITTI数据集有40多个G，相比起来是一个比较小的数据集

对于数据集的讲解可以参考 [CSDN](https://blog.csdn.net/u013086672/article/details/103913361)

下载数据集可以在 [GRAVITI](https://gas.graviti.cn/dataset/data-decorators/KITTIObject) 下载，需要注册一个账户，但是速度很快

