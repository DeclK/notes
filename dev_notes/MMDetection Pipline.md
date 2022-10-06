# MMDetection Pipline

## LoadImageFromFile

主要加入 `results['img']`，通过 `results['img_info']['filename']` 获得图片路径

`results['img'].shape = (357, 500, 3), type is np.ndarray`

## LoadAnnotations

主要加入 `results['gt_bboxes'] & results['gt_labels']`，实际上就是对 `results['ann_info']` 做一个 copy，把里面的相关信息摘出来，在重新放到 `results` 里面

## Resize

根据 `img_scale = (1000, 600)` 参数缩放图片，通常只有一个边能够达到缩放的大小，另一个边根据情况等比例缩放，同时还需要缩放 bboxes 等标签

不同 size 图片在进行 collate_fn 时会进行合并，合并方式：左上角对齐，batch tensor 的大小为该 batch 最大的 H & W

## RandomFlip

根据 `flip_ratio = 0.5` 进行随机水平翻转

## Normalize

根据 `img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)` 进行归一化处理

本质是通过 OpenCV 完成，原始图片应该是 `BGR` 格式

## Pad

两种pad模式：(1) pad到固定大小 (2) pad使得能被整数（比如resnet要求输入能被32整除）

## DefaultFormatBundle

把数据从 `np.ndarray` 转换为 `Tensor`，将图像转置为 `(3, H, W)`，然后再套一个 DataContainer（基本无用

## Collect

保留指定的 keys & meta_keys

## MultiScaleFlipAug

Test time augmentation

通常不会做 TTA，但在这一步又有封装，所以这个类在 `test_pipline` 中一定会有...