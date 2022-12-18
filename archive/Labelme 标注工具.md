---
title: Labelme 标注工具
tags:
  - Labelme
categories:
  - 编程
  - OpenMMLab
abbrlink: be6c9228
date: 2021-09-28 15:39:29
---

# Labelme 标注工具

[labelme github](https://github.com/wkentaro/labelme)

下载直接 `pip install labelme`

参考 [bilibili](https://www.bilibili.com/video/BV1jV411U7zb)

## 使用

### 简单设置

<img src="Labelme 标注工具/屏幕截图 2021-09-17 115035.png" style="zoom: 50%;" />

1. 打开 Save Automatically
2. Change Output Dir，选定输出文件夹
3. Save With Image Data

### 开始标注

选择图片所在文件夹即可开始标注，可以使用多边形逐步将物体包围，也可以使用一些简单的集合图形进行标注。标注完成后在当前文件夹即可发现 json 文件

### 文件整理

1. 对未标注的文件进行清理，因为自己没把所有图片都标注。对已标注的文件重新命名，使得数据集列表更连贯
2. 修改 json 文件中的信息，以匹配修改的文件名，即修改 `imagePath` 字段

### 数据转化

使用 [Tony607](https://github.com/Tony607)/**[labelme2coco](https://github.com/Tony607/labelme2coco)** 中的脚本进行转化，将所有的 json 文件转化为一个 coco format json 文件，非常方便

```cmd
python labelme2coco.py image_dir
```

`image_dir` 为之前标注所在的文件夹。而且作者还提供了一个 `COCO_Image_Viewer.ipynb` 可以将结果可视化显示出来，只需要简单修改一下其中的文件夹路径即可

```python
annotation_path = "trainval.json"
image_dir = "your_image_dir"
```

## TODO

COCO api 学习

