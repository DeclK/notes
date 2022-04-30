---
title: MMDetection 项目
tags:
  - MMDetection
categories:
  - 编程
  - OpenMMLab
abbrlink: acef3112
date: 2021-09-28 15:40:03
---

# MMDetection 项目

现在正式开始 MMDetection 项目！之前有一个想法，看看能不能用 MMDetection 来检测网球比赛中的网球球速，使用了 Faster-RCNN 模型试了下目标检测，发现对于网球这种小物体根本检测不出来，在视频中就是非常小的一个像素点。那么能不能使用自己标注的数据集来训练一个小目标检测（仅网球）的网络，来对网球视频进行检测？现在来进行具体的试验

## 数据集

使用 Labelme 进行标注，并转化为 COCO 数据集。将数据集按照如下结构排列

```
mmdetection
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

其中 annotations 放置 coco format json 文件，其他文件夹放置图片数据集

## 准备 config 文件

由于标注文件是有做 segmentation 的，选择 Mask-RCNN 作为模型

1. base config: [Mask-RCNN-R101-FPN](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/mask_rcnn_r101_fpn_mstrain-poly_3x_coco.py) 

2. checkpoint: [model](https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_mstrain-poly_3x_coco/mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244-5675c317.pth)

放置在 `/mmdet/config/mask_rcnn` 下，具体 config 文件放在文末

## 训练

目前还是不能跑起来，预计是 config 文件和数据集 CLASSES 的原因，还有要注意 config 的继承关系！

1. 由于继承机制，使用了 `RepeatDataset` 不对原数据集类型进行修改，不然报错

   ```python
   TypeError: CocoDataset: __init__() got an unexpected keyword argument 'times'
   ```

2. 遇到报错

   ````python
   AssertionError: The `num_classes` (1) in Shared2FCBBoxHead of MMDataParallel does not matches the length of `CLASSES` 80) in RepeatDataset
   ````

   依旧是由于 `ReapeatDataset` 造成的，由于对这个类不够了解，所以频繁报错😅这里的逻辑是因为没有指定 `classes`，由于原 coco 数据集有80个类，自己的类别不一定是原 COCO 数据集相同。既然是 `RepeatDataset` 那么一定是重复了自己定义的数据集，那就看看定义的数据集中是否指定了 `classes`。结果一看，果然没有指定，加上就解决了

   ```python
   data = dict(
       samples_per_gpu=2,
       workers_per_gpu=2,
       train=dict(
           type='RepeatDataset',
           times=3,
           dataset=dict(
               type='CocoDataset',
               ann_file='/home/chenhongkun/mmdetection/data/coco/annotations/train.json',
               img_prefix='/home/chenhongkun/mmdetection/data/coco/train2017',
               # classes=('tennis', ),
               pipeline=[
                   dict(type='LoadImageFromFile'),
                   dict(
                       type='LoadAnnotations',
                       with_bbox=True,
                       with_mask=True,
                       poly2mask=False),
                   dict(
                       type='Resize',
                       img_scale=[(1333, 640), (1333, 800)],
                       multiscale_mode='range',
                       keep_ratio=True),
                   dict(type='RandomFlip', flip_ratio=0.5),
                   dict(
                       type='Normalize',
                       mean=[123.675, 116.28, 103.53],
                       std=[58.395, 57.12, 57.375],
                       to_rgb=True),
                   dict(type='Pad', size_divisor=32),
                   dict(type='DefaultFormatBundle'),
                   dict(
                       type='Collect',
                       keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
               ]),
           classes=('tennis', ),
           ann_file='/home/chenhongkun/mmdetection/data/coco/annotations/train.json',
           img_prefix='/home/chenhongkun/mmdetection/data/coco/train2017'),
       val=dict(...)
       test=dict(...)
   ```

训练了一个 epoch 来看看结果，发现很难检测到小的物体。而且经过了70个 epoch 的训练，最后 Loss 没有持续下降，感觉优化停止了。现在尝试增加 batch 数量，下一个手段就是减少学习率。我也在思考会不会是 anchor 的问题，因为 anchor 太大了根本检测不到这么小的物体

现在修改了 anchor 大小效果不错，继续训练...最后可视化结果来看有许多重复的结果，NMS 的阈值需要再调一下，而且现在的预测值非常低，不知道为什么，不过位置还是可以接受

<img src="MMDetection 项目/image-20210920231709443.png" style="zoom:50%;" />

## 工具箱

1. `print_config.py`，打印完整 config 文件

   ```python
   python tools/misc/print_config.py ${CONFIG} [-h] [--options ${OPTIONS [OPTIONS...]}]
   ```

2. `analyze_logs.py`，可以将日志中的记录值绘制成曲线图

   ```python
   python tools/analysis_tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
   ```

   > `tools/analysis_tools/analyze_results.py` calculates single image mAP and saves or shows the topk images with the highest and lowest scores based on prediction results. [link](https://mmdetection.readthedocs.io/en/latest/useful_tools.html#result-analysis)

   ```python
   python tools/analysis_tools/analyze_results.py \
         ${CONFIG} \
         ${PREDICTION_PATH} \
         ${SHOW_DIR} \
         [--show] \
         [--wait-time ${WAIT_TIME}] \
         [--topk ${TOPK}] \
         [--show-score-thr ${SHOW_SCORE_THR}] \
         [--cfg-options ${CFG_OPTIONS}]
   ```

3. `dataset_converters` 工具箱能够将不同的数据集格式转为 coco format，比如使用 `image2coco.py` 能够将图片生成没有标签的 coco format json 文件

4. [DetVisGUI project](https://github.com/Chien-Hung/DetVisGUI/tree/mmdetection)，为一个可视化项目，能够将检测结果 `result.pkl` 可视化

## 完整 config 文件

```python
checkpoint_config = dict(interval=5)    # mark
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')]) # mark
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/home/declan/vscode/mmlab_test/mmdetection/checkpoints/mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244-5675c317.pth'
resume_from = None
workflow = [('train', 1)]
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,  # mark
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='CocoDataset',
            ann_file='/home/declan/vscode/mmlab_test/mmdetection/data/coco/annotations/train.json',
            img_prefix='/home/declan/vscode/mmlab_test/mmdetection/data/coco/train2017',
            classes=('tennis', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True,
                    poly2mask=False),
                dict(
                    type='Resize',
                    img_scale=[(1333, 640), (1333, 800)],
                    multiscale_mode='range',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
            ]),
        classes=('tennis', ),
        ann_file='/home/declan/vscode/mmlab_test/mmdetection/data/coco/annotations/train.json',
        img_prefix='/home/declan/vscode/mmlab_test/mmdetection/data/coco/train2017'),
    val=dict(
        type='CocoDataset',
        ann_file='/home/declan/vscode/mmlab_test/mmdetection/data/coco/annotations/val.json',
        img_prefix='/home/declan/vscode/mmlab_test/mmdetection/data/coco/val2017',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('tennis', )),
    test=dict(
        type='CocoDataset',
        ann_file='/home/declan/vscode/mmlab_test/mmdetection/data/coco/annotations/test2017.json',
        img_prefix='/home/declan/vscode/mmlab_test/mmdetection/data/coco/test2017',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        classes=('tennis', )))
evaluation = dict(interval=1, metric=['bbox', 'segm'])
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[10, 20, 30, 40, 50])  # mark try 70 epoch
runner = dict(type='EpochBasedRunner', max_epochs=100)
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[1], # mark important
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.01),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.01),
            max_per_img=100,
            mask_thr_binary=0.5)))
classes = ('tennis', )
work_dir = './work_dirs/mask_rcnn'
gpu_ids = [7]
```

