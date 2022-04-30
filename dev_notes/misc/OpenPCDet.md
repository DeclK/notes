# OpenPCDet

OpenPCDet 并没有像 MMDetection 一样有非常详尽的文档，只能自己去看源码了，这也是对自己的一种锻炼🤔这篇笔记打算对 OpenPCDet 的运行流程进行一个总结，并且对 SECOND 和 Voxel R-CNN 的相关代码进行详细研究

## Test.py

一些重要的参数

```python
--cfg_file
--batch_size
--workers
--ckpt
--start_epoch
--save_to_file
```

其中对于 `--workers` 的作用需要更多的理解，参考 [CSDN](https://blog.csdn.net/jiongjiongxia123/article/details/112850223)，个人理解就是 CPU 作为搬运工 workers 提前将需要的 batch 放入内存 RAM 中（非显存），然后 GPU 处理完一个 batch 过后来取

## EasyDict

项目写了一个 `EasyDict` 类作为更好用的字典，能够将 `key` 直接作为 `attribute` 进行调用

```python
# Get attributes

>>> d = EasyDict({'foo':3})
>>> d['foo']
3
>>> d.foo
3
```

当然还有其他特性，这里就不介绍了

`__class__` & `__dict__` & `__setattr__` & `__setitem__` 

模型的参数是以 [yaml](https://www.runoob.com/w3cnote/yaml-intro.html) 格式

[class](https://luobuda.github.io/2015/01/16/python-class/) [dict](https://www.jianshu.com/p/c390d591ce65) [all & init](https://stackoverflow.com/questions/44834/can-someone-explain-all-in-python)

`__all__` 定义在 `__init__.py` 中，代表着仅在这一层 package，你想要暴露的接口，外部如果从这个包里 import，只能 import `__all__` 列表中包含的模块。如果确实希望导入某个模块，但该模块不再 `__all__` 中，可以通过具体的路径进行导入 

```python
import yaml

with open(file, mode='r') as f:
    cfg_dict = yaml.load(f, Loader=FullLoader)
```

## pathlib

一个好用的文件路径库，一些基本使用参考 [知乎](https://zhuanlan.zhihu.com/p/33524938)

更详细的总结 [知乎](https://zhuanlan.zhihu.com/p/139783331)

## logging

用于记录日志 [知乎](https://zhuanlan.zhihu.com/p/360306588)





https://zhuanlan.zhihu.com/p/417286741

https://zhuanlan.zhihu.com/p/152120636

可以按照以下五个模块进行系统学习

![image-20211104205646529](OpenPCDet/image-20211104205646529.png)

## KITTI 数据集处理

<img src="OpenPCDet/image-20211105141924194.png" style="zoom:50%;" />

https://blog.csdn.net/weixin_44128857/article/details/108516213

https://zhuanlan.zhihu.com/p/99114433



numpy fromfile, tofile

skimage.io 读取图像

构建了 Object3d 类存储目标/标签



Calibration 类，目前不知道怎么用，可能需要可视化来帮助一下

<img src="OpenPCDet/image-20211105214225556.png" alt="image-20211105214225556" style="zoom:50%;" />



<img src="OpenPCDet/image-20211105214139494.png" alt="image-20211105214139494" style="zoom: 50%;" />

## 分布式训练

https://zhuanlan.zhihu.com/p/113694038

打包数据集，打包模型



tqdm 包用于展示进度条

https://zhuanlan.zhihu.com/p/163613814

set_postfix update refresh

# 如何学习一个 Python 项目

1. 学习 `main.py`
2. 学习模块，学习类

python 的变量是非常灵活的，有时候并不知道哪些变量代表着什么，如果有一个抽象的定义，能否让你理解这个模块的功能是最好的

## 可视化

tensorboard summarywriter

https://www.bilibili.com/video/BV1Qf4y1C7kz

https://pytorch.org/docs/stable/tensorboard.html



info.pkl 文件格式，主要是一些标签信息，没有原始点云信息

```shell
key: point_cloud
        num_features:4
        lidar_idx:000000
--------------------------
key: image
        image_idx:000000
        image_shape:[ 370 1224]
--------------------------
key: calib
        P2:...
        R0_rect:...
        Tr_velo_to_cam:...
--------------------------
key: annos
        name:['Pedestrian']
        truncated:[0.]
        occluded:[0.]
        alpha:[-0.2]
        bbox:[[712.4  143.   810.73 307.92]]
        dimensions:[[1.2  1.89 0.48]]
        location:[[1.84 1.47 8.41]]
        rotation_y:[0.01]
        score:[-1.]
        difficulty:[0]
        index:[0]
        gt_boxes_lidar:[[ 8.73138046 -1.85591757 -0.65469939  1.2         0.48        1.89
  -1.58079633]]
        num_points_in_gt:[377]
--------------------------
```

calib 矩阵是用于转换点云坐标的，将点云坐标从一个坐标系转移到另一个坐标系，比如从激光雷达相机坐标系，转移到RGB相机坐标系，R0_rect 矩阵是由于在长时间的运行中（车子在开），坐标系之间的相对位置会发生变化



## 优化策略 onecycle

https://blog.csdn.net/xys430381_1/article/details/89102866

one cycle 策略到底是个啥



use road plane 应该不会特别影响结果

The ground plane is just for GT sampling augmentation, and it also almost doesn't affect the final results much.



Field of view 视场，表现视野范围

https://blog.csdn.net/gbmaotai/article/details/104835991



evaluation

model structure



想要绘图，直接用 matplotlib 绝对够用，不论是画矩形框还是添加文字

opencv 可以当作业余爱好复现一下别人的项目

## 模型学习

frame_id
gt_boxes
points
use_lead_xyz
voxels
voxel_coords
voxel_num_points
image_shape



collections 模块

https://zhuanlan.zhihu.com/p/110476502



collate_fn

https://zhuanlan.zhihu.com/p/361830892



读取 pickle 文件 pkl

使用 np.fromfile 读取 bin 文件为 ndarray，注意 dtype 要与数据一致 # 已经总结到 cheatsheet 当中



numpy

nonzero 可以返回非零值

## data_dict 跟踪

1. build_dataloader

   ```python
   train_set, train_loader, train_sampler = build_dataloader(
       dataset_cfg=cfg.DATA_CONFIG,
       class_names=cfg.CLASS_NAMES,
       batch_size=args.batch_size,
       dist=dist_train, workers=args.workers,
       logger=logger,
       training=True,
       merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
       total_epochs=args.epochs
   )
   ```

   1. init kitti dataset

      ```python
          dataset = __all__[dataset_cfg.DATASET](
              dataset_cfg=dataset_cfg,
              class_names=class_names,
              root_path=root_path,
              training=training,
              logger=logger,
          )
      ```

      1. init dataset template

         ```python
         super().__init__(
             dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
         )
         ```

         ```python
                 Args:
                     root_path: none
                     dataset_cfg: cfg.DATA_CONFIG
                     class_names: pedestrian car cyclist
                     training:
                     logger:
                 看看数据集基类的结构：
                 Attributes:
                     - dataset config
                     - training
                     - class_names
                     - root_path, 通常在 dataset config 中已经指定 DATA_PATH
                     - 自己的 logger
                     - 点云范围 point cloud range
                     - 点云特征编码器 point feature encoder
                     - 数据增强器 sampling & rotation
                     - 数据处理器 voxelization
         ```

         1. init Point Feature encoder

            ```python
            POINT_FEATURE_ENCODING: {
                encoding_type: absolute_coordinates_encoding,
                used_feature_list: ['x', 'y', 'z', 'intensity'],
                src_feature_list: ['x', 'y', 'z', 'intensity'],
            }
            ```

         2. init DataAugmentor

            ```python
                    Args:
                        - root_path: DATA_PATH
                        - augmenter_config: DATA_AUGMENTOR 以 KITTI 为例
                        -----------------------------------------------
                        DATA_AUGMENTOR:
                            DISABLE_AUG_LIST: ['placeholder']
                            AUG_CONFIG_LIST:
                                - NAME: random_world_flip
                                ALONG_AXIS_LIST: ['x']
                                ...
                                ...
                        -----------------------------------------------
                    Attributes:
                        - data_augmentor_queue，数据增强器组成的列表
            ```

            除了将 augmentor 组成列表外，还去除了不符合要求的 gt 标签

            ```shell
            2021-11-10 08:07:48,480   INFO  Database filter by min points Car: 14357 => 13532
            2021-11-10 08:07:48,482   INFO  Database filter by min points Pedestrian: 2207 => 2168
            2021-11-10 08:07:48,483   INFO  Database filter by min points Cyclist: 734 => 705
            2021-11-10 08:07:48,527   INFO  Database filter by difficulty Car: 13532 => 10759
            2021-11-10 08:07:48,535   INFO  Database filter by difficulty Pedestrian: 2168 => 2075
            2021-11-10 08:07:48,538   INFO  Database filter by difficulty Cyclist: 705 => 581
            ```

            对 kitti_dbinfos_train.pkl 中的内容进行过滤

            ```python
                        - NAME: gt_sampling
                        USE_ROAD_PLANE: True
                        DB_INFO_PATH:
                            - kitti_dbinfos_train.pkl
                        PREPARE: {
                            filter_by_min_points: ['Car:5', 'Pedestrian:5', 'Cyclist:5'],
                            filter_by_difficulty: [-1],
                        }
            
                        SAMPLE_GROUPS: ['Car:20','Pedestrian:15', 'Cyclist:15']
                        NUM_POINT_FEATURES: 4
                        DATABASE_WITH_FAKELIDAR: False
                        REMOVE_EXTRA_WIDTH: [0.0, 0.0, 0.0]
                        LIMIT_WHOLE_SCENE: True
            ```

         3. init data porcessor

            ```python
                        DATA_PROCESSOR:
                            - NAME: mask_points_and_boxes_outside_range
                            REMOVE_OUTSIDE_BOXES: True
            
                            - NAME: shuffle_points
                            SHUFFLE_ENABLED: {
                                'train': True,
                                'test': False
                            }
            
                            - NAME: transform_points_to_voxels
                            VOXEL_SIZE: [0.05, 0.05, 0.1]
                            MAX_POINTS_PER_VOXEL: 5
                            MAX_NUMBER_OF_VOXELS: {
                                'train': 16000,
                                'test': 40000
                            }
            ```

      2. 获得 split 文件路径如 ImageSets/train.txt

      3. 获得样本列表 self.sample_id_list

      4. 加载 kitti_info_train.pkl 信息 self.kitti_infos = []

   2. 分布式打包数据集

      ```python
      sampler = torch.utils.data.distributed.DistributedSampler(dataset)
      ```

   3. init Dataloader

      ```python
      dataloader = DataLoader(
          dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
          shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
          drop_last=False, sampler=sampler, timeout=0
      ```

2. build networks

   ```python
   model_info_dict = {
       'module_list': [],
       # 下面两个 featrue 初始化都是 4 (x, y, z, intensity)
       'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
       'num_point_features': self.dataset.point_feature_encoder.num_point_features,
       'grid_size': self.dataset.grid_size,
       'point_cloud_range': self.dataset.point_cloud_range,
       'voxel_size': self.dataset.voxel_size,
       # 有的没有 downsample 为 None
       'depth_downsample_factor': self.dataset.depth_downsample_factor
   }
   ```

   ```python
   self.add_module(module_name, module)
   ```

   初始化网络，add_module 将模块作为子模块，self.module_list 包含了所有模块\

3. build optimizer

4. （如果有）加载 checkpoint

5. build scheduler

   ```python
   lr_scheduler, lr_warmup_scheduler = build_scheduler(
       optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
       last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
   )
   ```

6. train a model

   ```python
   train_model(
       model,
       optimizer,
       train_loader,
       model_func=model_fn_decorator(),
       lr_scheduler=lr_scheduler,
       optim_cfg=cfg.OPTIMIZATION,
       start_epoch=start_epoch,
       total_epochs=args.epochs,
       start_iter=it,
       rank=cfg.LOCAL_RANK,
       tb_log=tb_log,
       ckpt_save_dir=ckpt_dir,
       train_sampler=train_sampler,
       lr_warmup_scheduler=lr_warmup_scheduler,
       ckpt_save_interval=args.ckpt_save_interval,
       max_ckpt_save_num=args.max_ckpt_save_num,
       merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch
   )
   ```

   1. 获取数据 dataloader_iter = iter(data_loader)	`next(dataloader_iter)`

      这里无法使用调试获得 next 里面的细节，只能自己一个一个看了！！先上`data_dict` 结果，可以看到结果是有增强的，比如 gt_boxes, points

      ```shell
      frame_id
      (4,)
      gt_boxes
      (4, 42, 8)
      points
      (89649, 5)
      use_lead_xyz
      (4,)
      voxels
      (64000, 5, 4)
      voxel_coords
      (64000, 4)
      voxel_num_points
      (64000,)
      image_shape
      (4, 2)
      batch_size
      4
      ```

      1. 先看 `__getitem__` 方法，本质是根据 self.kitti_infos 列表中得到点云和标签的信息，然后通过数据增强和数据处理，存储为 `data_dict` 字典，作为最终训练的数据。所以这个 `data_dict` 就是核心中的核心，非常重要

          ```python
                  --------------------------
                  key: point_cloud
                          num_features:4
                          lidar_idx:000000
                  --------------------------
                  key: image
                          image_idx:000000
                          image_shape:[ 370 1224]
                  --------------------------
                  key: calib
                          P2:...
                          R0_rect:...
                          Tr_velo_to_cam:...
                  --------------------------
                  key: annos
                          name:['Pedestrian']
                          truncated:[0.]
                          occluded:[0.]
                          alpha:[-0.2]
                          bbox:[[712.4  143.   810.73 307.92]]
                          dimensions:[[1.2  1.89 0.48]]
                          location:[[1.84 1.47 8.41]]
                          rotation_y:[0.01]
                          score:[-1.]
                          difficulty:[0]
                          index:[0]
                          gt_boxes_lidar:[[ 8.73138046 -1.85591757 -0.65469939  1.2         0.48        1.89
                  -1.58079633]]
                          num_points_in_gt:[377]
                  --------------------------
          ```

          现在准备一个 input_dict 作为 prepare_data 的输入，input dict 的内容大致和 kitti_inofs 相同，有两点操作需要注意一下：

          1. 对 **gt boxes** 做了一些处理，一个是将中心、大小、旋转角三个特征合并为一个 ndarray，另一个是坐标系的转换，从 camera 坐标系转化为 lidar 坐标系
          2. 获得原始点云 **points**，并去除不是 FOV 视场角中的点，这将减少大量的点云数据，大约从 100k -> 20k
          
      2. 再看 prepare_data，基本由以下三个部分组成

          ```python
          data_dict = self.prepare_data(data_dict=input_dict)
          ```
      
          1. data_augmentor.forward
      
              1. gt_sampling
      
                  除了之前的对点少的 gt 进行筛选之外，这里还有新的操作，就是将所有的 gt database 作为一个大样本池，进行采样 `SAMPLE_GROUPS: ['Car:20','Pedestrian:15', 'Cyclist:15']` 然后将这些点放入的到点云场景中，要去除不满足要求的样本，例如重合样本
      
              2.  random rotation & filp & scaling 
      
          2. 给 boxes 加入类别特征（one hot），至此 boxes 维度由7变为8
      
          3. point_feature_encoding.forward
      
              ![image-20211110183032199](OpenPCDet/image-20211110183032199.png)
      
              ![image-20211110183045377](OpenPCDet/image-20211110183045377.png)
      
              基本上没有变化，data_dict 只多了一个 used_xyz
      
          4. data_processor.forward
      
              ![image-20211110184159897](OpenPCDet/image-20211110184159897.png)
      
              ![image-20211110184217297](OpenPCDet/image-20211110184217297.png)
      
              ![image-20211110184313679](OpenPCDet/image-20211110184313679.png)
      
              主要是使用 spconv 进行体素化，data_dict 多了一些内容
      
              ```python
              # voxels: [M, max_points, ndim] float tensor. only contain points.
              # coordinates: [M, 3] int32 tensor. zyx format.
              # num_points_per_voxel: [M] int32 tensor.
              ```
      
              此时应该还没有对体素特征进行处理
              
          5. post process, collate_fn，继续对样本进行处理：
      
              1. 给 'points' 和 'voxel_coords' 加入特征 batch_idx 在最前面，表明是这个点是来自于该 batch 中的第几个 sample
              2. 由于 batch 中每个 gt 标签个数可能不一样
      
      3. 根据 epoch 更新学习率，model.train，optimizer.zero_grad()
      
      4. **计算 loss**
      
      5. loss.backward() & optimizer.step()

## result.pkl

```shellname (3,)
name (3,)
['Car' 'Car' 'Car']
-------------------------------------
truncated (3,)
[0. 0. 0.]
-------------------------------------
occluded (3,)
[0. 0. 0.]
-------------------------------------
alpha (3,)
[-7.9 -4.6 -4.3]
-------------------------------------
bbox (3, 4)
[[764.6 173.1 811.6 209.8]
 [392.1 181.5 416.8 201.7]
 [217.8 188.6 273.4 213.2]]
-------------------------------------
dimensions (3, 3)
[[3.5 1.4 1.5]
 [3.9 1.5 1.6]
 [4.1 1.4 1.6]]
-------------------------------------
location (3, 3)
[[  7.1   1.4  29.3]
 [-16.7   2.3  58.5]
 [-23.5   2.5  46.5]]
-------------------------------------
rotation_y (3,)
[-7.7 -4.9 -4.7]
-------------------------------------
score (3,)
[0.5 0.5 0.2]
-------------------------------------
boxes_lidar (3, 7)
[[29.6 -7.1 -0.5  3.5  1.5  1.4  6.1]
 [58.8 16.7 -0.8  3.9  1.6  1.5  3.3]
 [46.8 23.5 -1.1  4.1  1.6  1.4  3.1]]
-------------------------------------
frame_id 000001
