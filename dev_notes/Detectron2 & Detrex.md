# COCO

目的有俩：

1. 熟悉对图像的预处理
2. 熟悉 COCO api 以及 evaluation

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

## COCO in MMDet

