---
title: MMDetection Quick Run
tags:
  - MMDetection
categories:
  - 编程
  - OpenMMLab
abbrlink: 6f4f5bd8
date: 2021-08-17 11:31:01
---

# MMDetection Quick Run

整理自 [MMDetection 官方文档](https://mmdetection.readthedocs.io/en/latest/index.html)

## Part 1, Inference and train with existing models and standard datasets

> MMDetection provides hundreds of existing and existing detection models in [Model Zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html), and supports multiple standard datasets, including Pascal VOC, COCO, CityScapes, LVIS, etc.

这一部分将简要介绍，如何使用 MMDetection 对 Model zoo 中的模型进行测试和训练

### Inference with existing models

> By inference, we mean using trained models to detect objects on images. In MMDetection, a model is defined by a configuration file and existing model parameters are save in a checkpoint file.

这里介绍了 MMDetection 模型表示的基本逻辑：`configuration file`+ `checkpoint file` 

文档以 Faster R-CNN 为例子，如何进行目标检测推理

> To start with, we recommend [Faster RCNN](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn) with this [configuration file](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) and this [checkpoint file](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth). It is recommended to download the checkpoint file to `checkpoints` directory.

Q: config files 名字的意义是什么？在 config 对应的文件夹中有 markdown 文档进行说明，例如 [faster rcnn](https://github.com/open-mmlab/mmdetection/tree/master/configs/faster_rcnn) 

#### High-level APIs for inference

在之前也见到过这个代码，现在再来看一看

```python
from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
video = mmcv.VideoReader('video.mp4')
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1)
```

上面的代码中用到了是这三个 API:

1. 异步推理 `async_inference_detector`
2. 绘制 `show_result`
3. 推理 `inference_detector`

这里 `show_result` 的参数请查看 [原代码](https://github.com/open-mmlab/mmdetection/blob/a1cecf63c713c53941b8dcf8a9d762baf8511f2c/mmdet/models/detectors/base.py)，如果需要显示的话要添加参数 `show=True`。但在用 vscode 连接远程服务器的情况下，即使设置了 `show=True` 也不会展示结果，可能因为服务器不支持 GUI，替代的方法就是在 Jupyter notebook 或者 interactive window 中运行

#### Asynchronous interface - supported for Python 3.7+

这一节没有理解清楚，只稍微留了点印象：使用异步接口理论上能够加速推理/训练过程

##### 异步

首先什么是 [异步思维](https://blog.csdn.net/lemon4869/article/details/107145903)，在链接中的理解来说：异步就是不必等待推理结束才开始下一张图像的预处理。但看原代码说的异步到底是什么意思，我也不太理解，以后再回来整理 asyncio 相关知识吧

##### async & await

什么是 Asynchronous interface [知乎链接](https://zhuanlan.zhihu.com/p/353857526)

什么是 asyncio [知乎链接](https://blog.csdn.net/permike/article/details/110821246)

Q: 这里需要更多的 python 知识，我得回去补充，比如 with 关键字什么意思？整个异步的过程是怎么样的，能否把整个时间与事件弄清楚？

#### Demo

文档给了3个demo：

1. Image Demo
2. Webcam Demo
3. Video Demo

简单看了一下代码，看来还需要补充总结 OpenCV 的一些基础知识

### Test existing models on standard datasets

> MMDetection supports multiple public datasets including COCO, Pascal VOC, CityScapes, and [more](https://github.com/open-mmlab/mmdetection/tree/master/configs/_base_/datasets). This section will show how to test existing models on supported datasets.

#### Prepare datasets

推荐在项目之外建立数据集，然后以软链接的形式放到项目中，且目录结构要根据 config files 规放置（或者你自己修改 config files），举个几个例子

```
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
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
```

#### Test existing models

> We provide testing scripts for evaluating an existing model on the whole dataset (COCO, PASCAL VOC, Cityscapes, etc.). 

可以在单个 GPU 上测试，也可以在多个 GPU 上进行分布测试

```shell
# single-gpu testing
python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}] \
    [--show]

# multi-gpu testing
bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}]
```

mmdetection 提供对无标记数据集进行测试，但数据集需要符合 COCO format，如果不是 COCO format 例如 VOC 数据集，需要使用 [script in tools](https://github.com/open-mmlab/mmdetection/tree/master/tools/dataset_converters/pascal_voc.py) 转化

mmdetection 提供 batch inference，能够一次推理多个样本，在 config files 中修改 `sample_per_gpu` 即可，或者使用 `--cfg-options data.test.samples_per_gpu=2`

还有很多参数可以调整，具体请看源代码，文档也给出了很多 [例子](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html#examples)

### Train predefined models on standard datasets

> This section will show how to train *predefined* models (under [configs](https://github.com/open-mmlab/mmdetection/tree/master/configs)) on standard datasets i.e. COCO.

> **Important**: 训练的默认学习率的配置为 8个 GPU 和 2 img/gpu，也就是 batch size 为16，如果使用不同的 batch size 那么就要按照线性缩放原则更改学习率 e.g., `lr=0.01` for 4 GPUs * 2 imgs/gpu and `lr=0.08` for 16 GPUs * 4 imgs/gpu.

#### Prepare datasets

准备过程和上一节 Test 部分是一致的。建议先将模型下载好，以防网不好导致报错

#### Training on a single GPU

在单个 GPU 上训练，基本使用方法

```shell
python tools/train.py \
    ${CONFIG_FILE} \
    [optional arguments]
```

训练过程中的 log 和 checkpoints 都会被存放到 `work_dir` 当中，也可以用 `--work-dir` 重新指定

默认每12个 epoch 使用验证集对模型进行一次评估 evaluation，train.py 还接受一些常用参数：

- `--no-validate` (**not suggested**): Disable evaluation during training.
- `--work-dir ${WORK_DIR}`: Override the working directory.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--options 'Key=value'`: Overrides other settings in the used config.

文档提到两个参数的区别：`--resume-from & --load-from` 前者不仅加载参数权重，也加载 optimizer 的状态，主要用于训练被突然打断后接着训练。后者只加载参数权重，主要用于 finetuning

#### Training on multiple GPUs

同样，基本用法

```shell
bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]
```

可选参数和上面训练单个 GPU 是一样的

##### Launch multiple jobs simultaneously

在训练当中，通常会有多任务的存在，mmdetection 也可以实现，但现在能力有限，实在不能理解具体的实现逻辑，留个 [原文档链接](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html#training-on-multiple-gpus)

提到了很多新东西，nodes, port, slurm, pytorch launch utility...如果能够在实践中理解这些概念更好了

## Part 2, Train with customized datasets

>  We use the [balloon dataset](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon) as an example to describe the whole process.

基本步骤为：

1. Prepare the customized dataset
2. Prepare a config
3. Train, test, inference models on the customized dataset.

### Prepare the customized dataset

mmdetection 支持对 COCO format 数据集进行训练，将不同格式的数据集转化为 COCO format 即可。可能也可以对 config file 进行配置，来适配不同格式的数据集，但这一部分文档没有提及

#### COCO annotation format

> The necessary keys of COCO format for instance segmentation is as below, for the complete details, please refer [here](https://cocodataset.org/#format-data).

```json
{
    "images": [image],
    "annotations": [annotation],
    "categories": [category]
}


image = {
    "id": int,
    "width": int,
    "height": int,
    "file_name": str,
}

annotation = {
    "id": int,
    "image_id": int,
    "category_id": int,
    "segmentation": RLE or [polygon],
    "area": float,
    "bbox": [x,y,width,height],
    "iscrowd": 0 or 1,
}

categories = [{
    "id": int,
    "name": str,
    "supercategory": str,
}]
```

balloon dataset format 形如下

```json
{'base64_img_data': '',
 'file_attributes': {},
 'filename': '34020010494_e5cb88e1c4_k.jpg',
 'fileref': '',
 'regions': {'0': {'region_attributes': {},
   'shape_attributes': {'all_points_x': [...,],
    'all_points_y': [...,],
    'name': 'polygon'}}},
 'size': 1115004}
```

文档提供了函数将 balloon dataset JSON 格式转化为 COCO format，[原文档](https://mmdetection.readthedocs.io/en/latest/2_new_data_model.html#coco-annotation-format)

### Prepare a config

> The second step is to prepare a config thus the dataset could be successfully loaded. 

假设以 balloon dataset 训练 Mask R-CNN with FPN，config 在 `configs/balloon` 命名为  `mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py`

```python
# The new config inherits a base config to highlight the necessary modification
_base_ = 'mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('balloon',)
data = dict(
    train=dict(
        img_prefix='balloon/train/',
        classes=classes,
        ann_file='balloon/train/annotation_coco.json'),
    val=dict(
        img_prefix='balloon/val/',
        classes=classes,
        ann_file='balloon/val/annotation_coco.json'),
    test=dict(
        img_prefix='balloon/val/',
        classes=classes,
        ann_file='balloon/val/annotation_coco.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
```

### Train & Test Model

配置好了数据集过后，基本上就和 Part 1 中的训练和测试方法一样

To train a model with the new config, you can simply run

```shell
python tools/train.py configs/balloon/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py
```

To test the trained model, you can simply run

```shell
python tools/test.py configs/balloon/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py \
work_dirs/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py/latest.pth \
--eval bbox segm
```

## Part 3, Train with customized models and standard datasets

> In this note, you will know how to train, test and inference your own customized models under standard datasets.

使用 cityscapes dataset 训练个性化 Cascade Mask R-CNN R50 model，使用 [`AugFPN`](https://github.com/Gus-Guo/AugFPN) 替代 `FPN` 作为 neck，并且加上 `Rotate` or `Translate` 作为数据增强

基本步骤比 Part 2 多一个准备 customized model:

1. Prepare the standard dataset
2. Prepare your own customized model
3. Prepare a config
4. Train, test, and inference models on the standard dataset

### Prepare the standard dataset

基本流程，也是需要将 cityscapes dataset 转为 COCO format，mmdetection 也提供转化脚本 `tools/dataset_converters/cityscapes.py`

```shell
pip install cityscapesscriptspython tools/dataset_converters/cityscapes.py ./data/cityscapes --nproc 8 --out-dir ./data/cityscapes/annotations
```

### Prepare your own customized model

#### 1. Define a new neck (e.g. AugFPN)

首先新建一个文件 ``mmdet/models/necks/augfpn.py``

```python
from ..builder import NECKS@NECKS.register_module()class AugFPN(nn.Module):    def __init__(self,                in_channels,                out_channels,                num_outs,                start_level=0,                end_level=-1,                add_extra_convs=False):        pass    def forward(self, inputs):        # implementation is ignored        pass
```

注意这里用了一个装饰器，复习 python 的时候可以联系起来

####  2. Import the module

导入模块有两种方法：

1. 在  `mmdet/models/necks/__init__.py` 导入

   ```python
   from .augfpn import AugFPN
   ```

2. 在 config 文件中更改

   ```python
   custom_imports = dict(
       imports=['mmdet.models.necks.augfpn.py'],
       allow_failed_imports=False)
   ```

   这样就避免去更改原文件

#### 3. Modify the config file

将 neck 加入到 config file 中

```python
neck=dict(
    type='AugFPN',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    num_outs=5)
```

可以看出，如果想要更改别人的模型，mmdetection 都是以 config file 为核心，而不去更改原文件

如果要完成 AugFPN, Rotate, Translate 三个修改的话，在文档中给出了参考的 [config file](https://mmdetection.readthedocs.io/en/latest/3_exist_data_new_model.html#prepare-a-config)

### Train & Test Model

训练和测试基本使用方法在之前已经介绍过，这部分也一样

To train a model with the new config, you can simply run

```shell
python tools/train.py configs/cityscapes/cascade_mask_rcnn_r50_augfpn_autoaug_10e_cityscapes.py
```

To test the trained model, you can simply run

```shell
python tools/test.py configs/cityscapes/cascade_mask_rcnn_r50_augfpn_autoaug_10e_cityscapes.py \work_dirs/cascade_mask_rcnn_r50_augfpn_autoaug_10e_cityscapes.py/latest.pth --eval bbox segm
```

## 补充

### .pth文件里有什么

要回答一个问题，什么是 [.pth 文件 知乎](https://zhuanlan.zhihu.com/p/84797438)，在 mmdetection 框架中可以看到 pth 文件包含了两个部分 `meta & state_dict`

其中 `meta` 保存了模型的配置信息，列出其 key：

1. mmdet_version
2. **config**
3. CLASSES
4. epoch
5. iter
6. mmcv_version
7. time

而 `state_dict` 保存了模型参数值，列出部分 key：

backbone.conv1.weight
backbone.bn1.weight
backbone.bn1.bias
backbone.bn1.running_mean
backbone.bn1.running_var
backbone.bn1.num_batches_tracked
backbone.layer1.0.conv1.weight
backbone.layer1.0.bn1.weight
backbone.layer1.0.bn1.bias
backbone.layer1.0.bn1.running_mean
backbone.layer1.0.bn1.running_var
backbone.layer1.0.bn1.num_batches_tracked

.......

### 如何理解 Registry

在学习 mmdetection 过程中一直又一个问题：如何使用 config + registry 返回一个 nn.Module 类的模型，这些模型是如何被注册到 Registry 类当中的？

阅读资源：

1. [mmdetection 源码阅读笔记](https://blog.csdn.net/sinat_29963957)

2. [mmdetection 整体构建流程](https://zhuanlan.zhihu.com/p/337375549)

3. [mmdetection之model构建](https://blog.csdn.net/wulele2/article/details/114139512)

4. [MMCV 官方文档](https://mmtracking.readthedocs.io/en/latest/)，这个文档相当有用，是 mmlab 的通用基础库，要好好学习一下

5. [python \__all__变量](https://www.cnblogs.com/hester/p/10546235.html#:~:text=Python%E4%B8%AD%E4%B8%80%E4%B8%AApy%E6%96%87%E4%BB%B6%E5%B0%B1%E6%98%AF%E4%B8%80%E4%B8%AA%E6%A8%A1%E5%9D%97%EF%BC%8C%E2%80%9C__all__%E2%80%9D%E5%8F%98%E9%87%8F%E6%98%AF%E4%B8%80%E4%B8%AA%E7%89%B9%E6%AE%8A%E7%9A%84%E5%8F%98%E9%87%8F%EF%BC%8C%E5%8F%AF%E4%BB%A5%E5%9C%A8py%E6%96%87%E4%BB%B6%E4%B8%AD%EF%BC%8C%E4%B9%9F%E5%8F%AF%E4%BB%A5%E5%9C%A8%E5%8C%85%E7%9A%84__init__.py%E4%B8%AD%E5%87%BA%E7%8E%B0%E3%80%82%20%E5%A6%82%EF%BC%9A%E5%85%A8%E5%B1%80%E5%8F%98%E9%87%8F%EF%BC%8C%E5%87%BD%E6%95%B0%EF%BC%8C%E7%B1%BB%E3%80%82%20%E5%A6%82%E4%B8%8B%EF%BC%8Ctest1.py%E5%92%8Cmain.py,%3F%20%3F%20%E4%B8%A4%E4%B8%AA%E6%96%87%E4%BB%B6%E5%9C%A8%E5%90%8C%E4%B8%80%E4%B8%AA%E7%9B%AE%E5%BD%95%E4%B8%8B%E3%80%82%20%E9%82%A3%E4%B9%88%E5%9C%A8%E6%A8%A1%E5%9D%97%E4%B8%AD%E7%9A%84__all__%E5%8F%98%E9%87%8F%E5%B0%B1%E6%98%AF%E4%B8%BA%E4%BA%86%E9%99%90%E5%88%B6%E6%88%96%E8%80%85%E6%8C%87%E5%AE%9A%E8%83%BD%E8%A2%AB%E5%AF%BC%E5%85%A5%E5%88%B0%E5%88%AB%E7%9A%84%E6%A8%A1%E5%9D%97%E7%9A%84%E5%87%BD%E6%95%B0%EF%BC%8C%E7%B1%BB%EF%BC%8C%E5%85%A8%E5%B1%80%E5%8F%98%E9%87%8F%E7%AD%89%EF%BC%8C%E5%A6%82%E6%9E%9C%E6%8C%87%E5%AE%9A%E4%BA%86%E9%82%A3%E4%B9%88%E5%8F%AA%E8%83%BD%E6%98%AF%E6%8C%87%E5%AE%9A%E7%9A%84%E9%82%A3%E4%BA%9B%E5%8F%AF%E4%BB%A5%E8%A2%AB%E5%AF%BC%E5%85%A5%EF%BC%8C%E6%B2%A1%E6%9C%89%E6%8C%87%E5%AE%9A%E9%BB%98%E8%AE%A4%E5%B0%B1%E6%98%AF%E5%85%A8%E9%83%A8%E5%8F%AF%E4%BB%A5%E5%AF%BC%E5%85%A5%EF%BC%8C%E5%BD%93%E7%84%B6%E7%A7%81%E6%9C%89%E5%B1%9E%E6%80%A7%E5%BA%94%E8%AF%A5%E9%99%A4%E5%A4%96%E3%80%82)
6. 在 B 站上找到了一个官方 [学习讲座系列](https://space.bilibili.com/630319191/channel/detail?cid=179690&ctype=0)，这个讲座不太和我胃口，又在 B 站上找了一个西交的 [教学视频](https://www.bilibili.com/video/BV1Jb4y1r7ir?p=1)，这个视频就比较友好，对我帮助更大

我一直都很不理解这些类是怎么注册的，为什么有的代码似乎没有“明显”地运行，但是这些模块就已经被注册到 Registry 当中了？看完 MMCV 文档之后才大概明白了，重点就在于 `__init__.py` 文件以及 `__all__` 变量。可以看到在许多文件夹下都有 `__init__.py` 文件，例如 `mmdet/models/__init__.py`, `mmdet/models/backbones/__init__.py`

如果 import 文件夹/模块下有 init.py 文件，那么就会在 import 该模块之前先运行 init.py 文件，就是在这个文件之中完成了各个模块的注册，其中 `__all__` 变量包含了需要导入的模块列表

具体看一下其中的内容

```python
from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                      ROI_EXTRACTORS, SHARED_HEADS, build_backbone,
                      build_detector, build_head, build_loss, build_neck,
                      build_roi_extractor, build_shared_head)
from .dense_heads import *  # noqa: F401,F403
from .detectors import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .plugins import *  # noqa: F401,F403
from .roi_heads import *  # noqa: F401,F403

__all__ = [
    'BACKBONES', 'NECKS', 'ROI_EXTRACTORS', 'SHARED_HEADS', 'HEADS', 'LOSSES',
    'DETECTORS', 'build_backbone', 'build_neck', 'build_roi_extractor',
    'build_shared_head', 'build_head', 'build_loss', 'build_detector'
]

```

### Bug  & vscode debugger

捣鼓了一天了在服务器，怎么都不能够搞好 xrdp，还差点把电脑弄出点毛病...最后用 `sudo dpkg --purge xrdp` 删除了没有下载完全的包才弄好了。我认为实验室的服务器需要一个大更新，一个是系统升级，另一个是清理内存，现在大概还有350G的空间剩余，最重要的是服务器的网络是个大问题！很多资源都无法直接下载，有些资源下载起来非常吃力

放一个 [nvidia-smi讲解链接](https://www.jianshu.com/p/ceb3c020e06b)，理解一下各个指标是在说什么

#### vscode debugger 

在使用 vscode debug 的时候发现有的代码并不会 step in，后来发现这些代码都是通过 pip/conda install 下载的库中的代码。而 vscode 的 debugger 会默认设定只在 'MyCode' 中进行 debug，简单的说，如果不是“自己”写的代码，vscode debugger 是不会 step in 的

解决办法：在 debugger 的 json 文件中设置 `"justMyCode: false"`

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            // add the next line into your .json file
            "justMyCode": false
        }
    ]
}
```

## TODO

1. 还需要整理 MMCV 以了解整个 OpenMMLab 的运行逻辑 

2. 对于如何训练自己的网络还要继续看文档中的 Tutorial 部分
