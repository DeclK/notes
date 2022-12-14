# Detectron2 & Detrex

目前最好的检测器就是基于 DETR，非常有必要进行深入了解，所以选择 [detrex](https://github.com/IDEA-Research/detrex) 进行学习，原因如下：

1. 实现了许多 DETR-based 检测模型
2. 基于 [detectron2](https://detectron2.readthedocs.io/en/latest/)

我也一直想学习 detectron2 的框架，和 mmdet 比较一下，看看各自的优劣在哪里

## Install

使用了 pytorch 1.13 的 docker image，安装还是比较丝滑。只用了 `pip install` 就搞定了

## COCO

了解 coco 数据集的格式

### 下载 COCO

虽然百度网盘极其🐶，但是这里我依然使用了百度网盘下载，需要开启一下闲置带宽优化下载。只要是热门资源下载速度都会比较快的

这里下载 coco2017 train/val images & train/val annotations

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

### Annotations

1. `instances_train2017.json`，对应**目标检测、分割任务的训练集标注文件**
2. `captions_train2017.json`，对应图像描述的训练集标注文件
3. `person_,keypoints_train2017.json`，对应人体关键点检测的训练集标注文件

重点关注 instances json 文件，其核心内容有三个：

1.  images
2. annotations
3. categories

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
    ...
 ]
```

### pycocotools

`pip install pycocotools`

```python
coco.loadAnns
coco.loadImgs
```

evaluation

TODO

