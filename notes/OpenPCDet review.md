---
title: OpenPCDet review
tags:
  - OpenPCDet
categories:
  - 编程
  - OpenMMLab
mathjax: true
abbrlink: 6b4af7d3
date: 2021-12-19 22:20:38
---

# OpenPCDet review

现在想要打造自己的网络了，就要对这个框架完全的熟悉。虽然之前把代码都过了一遍，但是熟悉度还是不够，打算重新整理过。目标就是完成 SA-SSD 在 OpenPCDet 框架下的复现。贴一个 [Shaoshuai Shi](https://www.zhihu.com/people/yilu-kuang-shuai) 本人在知乎的解读 [OpenPCDet: Open-MMLab 面向LiDAR点云表征的3D目标检测代码库](https://zhuanlan.zhihu.com/p/152120636)

## 零碎的知识

### 库

之前在看代码的时候遇到了一些有用的第三方库，总结如下：

1. pathlib， [知乎](https://zhuanlan.zhihu.com/p/33524938)

2. logging， [知乎](https://zhuanlan.zhihu.com/p/360306588)

3. tqdm，[知乎](https://zhuanlan.zhihu.com/p/163613814)

4. tensorboard，[bilibili](https://www.bilibili.com/video/BV1Qf4y1C7kz) [pytorch](https://pytorch.org/docs/stable/tensorboard.html)

5. pdb（推荐 ipdb，相当于是 pdb + ipython），[知乎](https://zhuanlan.zhihu.com/p/37294138)

6. 分布式训练，[知乎](https://zhuanlan.zhihu.com/p/113694038) [bilibili](https://www.bilibili.com/video/BV1xZ4y1S7dG/?spm_id_from=333.788)，关于指定 GPU, CUDA_VISIBLE_DEVICES [CSDN](https://blog.csdn.net/alip39/article/details/87913543) 

7. collections, pickle

8. screen，用于新建窗口，让命令在后台运行，即使退出会话程序也不会停止

   ```shell
   screen -ls              # 查看所有screen
   screen -S <screen-name> # 创建screen，并命名
   screen -d				# detach or Ctrl + A + D
   screen -r <screen-name> # 进入screen
   screen -X quit          # 删除screen，但没有指定会话
   screen -X -S [session you want to kill] quit #删除screen，指定会话
   screen -wipe            # 清除dead screens
   ```

### 重要的参数

```python
--cfg_file
--batch_size
--workers
--ckpt
--start_epoch
--save_to_file
```

其中对于 `--workers` 的作用需要更多的理解，参考 [CSDN](https://blog.csdn.net/jiongjiongxia123/article/details/112850223)，个人理解就是 CPU 作为搬运工 workers 提前将需要的 batch 放入内存 RAM 中（非显存），然后 GPU 处理完一个 batch 过后来取

### EasyDict

项目用了一个 `EasyDict` 类作为更好用的字典，能够将 `key` 直接作为 `attribute` 进行调用

### Yaml

模型的参数是以 [yaml](https://www.runoob.com/w3cnote/yaml-intro.html) 格式，使用如下方法导入

```python
import yaml

with open(file, mode='r') as f:
    cfg_dict = yaml.load(f, Loader=FullLoader)
```

### Python

1. 关于 python 类中的内置属性：[class](https://luobuda.github.io/2015/01/16/python-class/) [dict](https://www.jianshu.com/p/c390d591ce65)

2. 关于  [all & init](https://stackoverflow.com/questions/44834/can-someone-explain-all-in-python)：`__all__` 定义在 `__init__.py` 中，代表着在这一个 package，你想要暴露的接口/函数/类（但也要先导入到 package 中），外部如果从这个包里 import，只能 import `__all__` 列表中包含的接口。如果确实希望导入某个接口，但该模块不在 `__all__` 中，可以通过具体的路径进行导入 

   如果是嵌套的包内调用，需要从包的根路径开始调用

## KITTI Dataset

### kitti_infos_train.pkl

这里应该还有2个 pkl 文件 `kitti_infos_val.pkl & kitti_dbinfos_train.pkl` 也很重要，这些 pkl 文件保存了 KITTI 的数据信息便于使用 python 进行操作，下面看看其中存储了什么

```shell
# kitti_infos_train.pkl & kitti_infos_val.pkl 都以如下形式存储
point_cloud 
        num_features 
        lidar_idx 
image 
        image_idx 
        image_shape (2,)
calib 
        P2 (4, 4)
        R0_rect (4, 4)
        Tr_velo_to_cam (4, 4)
annos 
        name (1,)
        truncated (1,)
        occluded (1,)
        alpha (1,)
        bbox (1, 4)
        dimensions (1, 3)
        location (1, 3)
        rotation_y (1,)
        score (1,)
        difficulty (1,)
        index (1,)
        gt_boxes_lidar (1, 7)
        num_points_in_gt (1,)
```

每一个 pkl 文件是一个列表，列表的成员是一个层级字典，保存了对应样本的信息（将 `ndarray` 的形状在关键字后标出）

`kitti_dbinfos_train.pkl` 有点不一样，它是一个字典，关键字就是标签类别

```shell
Pedestrian 
Car 
Cyclist 
Van 
Truck 
Tram 
Misc 
Person_sitting
```

每一个字典又对应了一个列表，列表的成员又是一个字典，以 Pedestrian 为例

```shell
name 
Pedestrian
-------------------------------------
path 
gt_database/000000_Pedestrian_0.bin
-------------------------------------
image_idx 
000000
-------------------------------------
gt_idx 
0
-------------------------------------
box3d_lidar (7,)
[ 8.7 -1.9 -0.7  1.2  0.5  1.9 -1.6]
-------------------------------------
num_points_in_gt 
383
-------------------------------------
difficulty 
0
-------------------------------------
bbox (4,)
[712.4 143.  810.7 307.9]
-------------------------------------
score 
-1.0
-------------------------------------
```

### DatasetTemplate

这是一个数据集的基础类，先看其参数和基本属性

```python
class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path: 一般为 None
            dataset_cfg: cfg.DATA_CONFIG
            class_names: pedestrian car cyclist
            training: True
            logger:
        Attributes:
            - dataset config
            - training
            - class_names
            - root_path 通常为 dataset config 中的指定 DATA_PATH
            - logger
            - point cloud range 点云范围 
            - point_feature_encoder 点云特征编码器 
            - data_augmentor 数据增强器 sampling & rotation
            - data_processor 数据处理器 voxelization
        """
```

在初始化的过程中，只有在 `data_augmentor` 中的 `gt_sampling` 实际处理了数据，即对 gt database 进行筛选，去除点少的和困难的样本

该类中还定义了两个重要的方法，在之后调用：1. `prepare_data(self, data_dict)` 2. `collate_batch(batch_list)` 

#### prepare_data

`prepare_data(self, data_dict)` 通过该方法准备数据。数据的采样、增强和体素化都是在这一方法中进行调用。需要注意的是返回的 voxel 坐标排列为 zyx 而不是 xyz，这是由于 spconv 的设计导致的

```python
def prepare_data(self, data_dict):
        ################# IMPORTANT #################
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                (poped) gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
                batch_size: 在 dataloader collate_fn 中加入
        """
		# voxels: [M, max_points, ndim] float tensor. only contain points.
        # coordinates: [M, 3] int32 tensor. zyx format. #### note zyx not xyz !!!####
        # num_points_per_voxel: [M] int32 tensor.
```

#### collate_batch

`collate_batch(batch_list)` 用于传入 `DataLoader` 中的 `collate_fn`，这样就能定义每一个 batch 返回的具体形式（为一个字典），关于 [collate_fn](https://zhuanlan.zhihu.com/p/361830892)，核心的工作如下：

1. 将 batch 每个 sample 的数据 concat
2. 在 points, voxel_coords... 中增加 batch 维度以区分是来自哪个 sample
3. batch 中每个 sample 的 boxes 数量不一样，需要将形状统一然后合并起来

```python
frame_id (4,)
gt_boxes (4, 36, 8)	# 在 prepare_data 中加入了类别特征
points (94168, 5)
use_lead_xyz (4,)
voxels (64000, 5, 4)
voxel_coords (64000, 4) # bzyx order
voxel_num_points (64000,)
image_shape (4, 2)
```

当然作为一个 `Dataset` 的子类，需要定义 `__len__ & __getitem__` 方法，这些方法没有在 `DatasetTemplate` 中实现，而是在更具体的类中实现比如 `KittiDataset`

### kitti_dataset.py

这个文件用于定义 `KittiDataset` 类，这个类比较大，因为该类实现了对原始 KITTI 数据集的处理函数，`kitti_infos_train.pkl` 就是使用这些函数生成的。先看看该类的初始化

```python
class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        Attributes:
        	- split_dir & sample_id_list: 根据训练集/验证集确定 sample 列表
            - kitti_infos: 来自 train/test pkl 文件，加载后为一个列表，列表中的元素为形如下面的字典
            ### 当然还有基类中的属性，这里不再重复 ###
        """
```

#### getitem

其核心函数 `__getitem__`，返回一个字典，其实大部分都是返回了 `kitti_infos_train.pkl` 中的信息，特别的操作就是增加了点云信息，并选取了视场角中的点 `get_fov_flag`

```python
# 获得原始点云 points (N, 4)
if "points" in get_item_list:
    points = self.get_lidar(sample_idx)
    # 仅取视角中的点，其余点放弃
    if self.dataset_cfg.FOV_POINTS_ONLY:
        pts_rect = calib.lidar_to_rect(points[:, 0:3])
        fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
        points = points[fov_flag]
        input_dict['points'] = points
```

`__getitem__` 中在最后调用了基类 `prepare_data` 方法，对点云进行采样、增强和体素化，最终返回一个字典包含了每个样本的信息

关于 calib 是什么作用，可以参考这一篇博客 [Kitti Calib](https://medium.com/test-ttile/kitti-3d-object-detection-dataset-d78a762b5a4)。对于坐标转换的矩阵还是比较好理解的，难以理解的是修正矩阵，通过观察我发现修正矩阵的值非常接近于一个单位矩阵，所以我猜测这是为了修正由于路况颠簸而导致的相机微小抖动/旋转

#### generate_prediction_results

除了核心函数外，该类还实现了一个静态方法 `generate_prediction_results`，这个方法的目的是将预测的 bbox, scores, label 等结果转化为 kitti 原始标签格式，用于之后的评估

```python
def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
    """
    Args:
        batch_dict:
            frame_id:
        pred_dicts: list of pred_dicts
            pred_boxes: (N, 7), Tensor
            pred_scores: (N), Tensor
            pred_labels: (N), Tensor
        class_names:
        output_path:

    Returns:
    	- annos: a list, contains batch prediction results, consists of dict
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
    """
```

#### evaluation

使用 kitti 的评价标准对预测结果进行评估，返回一个元组 `ap_result_str, ap_dict` 其实二者的数据是一致的，前者将保存到日志中方便查看，后者将记录到 tensorboard 中

之后有需要可以整理一下其中的 `box_utils & common_utils & calib` ，里面有哪些操作是常用的，不用自己造轮子 

## Detector

下面总结如何建造一个模型，代码传入了三个参数：模型配置，类别，数据集（KittiDataset instance）

```python
__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN
}

def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model

def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
```

通过模型配置中的关键字段创建模型实例，例如：`SECONDNet`。检测器都有一个共同的基类 `Detector3DTemplate`  下面看看它具有什么功能，然后再结合具体模型学习

### Detector3DTemplate

先看初始化函数

```python
class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        # 用于记录 epoch
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head'
        ]
```

该类的核心方法有三个主要功能：

1. 构造网络结构方法
2. 后处理方法，对 NMS 算法和 recall 数据都在这个部分实现
3. 载入参数方法，将 checkpoint 中的参数载入模型中

这里主要对前两个方法进行整理

#### build_networks

该方法最终返回一个 `module_list` 保存各个结构的模块，在之后的向前传播路径中，数据将按顺序经过各个模块。`model_info_dict` 还用于记录每个模块输出的特征数/通道数之类的信息，以便于传入下一个模块进行初始化，当然初始化过程还要结合 `self.model_cfg` 传入必要参数

```python
def build_networks(self):
    # 创建一个 module info dict 储存模型中的各个模块与模型信息
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
    for module_name in self.module_topology:
        module, model_info_dict = getattr(self, 'build_%s' % module_name)(
            model_info_dict=model_info_dict
        )
        self.add_module(module_name, module)
        return model_info_dict['module_list']
```

#### post_processing

该方法仅在测试的时候被调用，其功能是使用 NMS 算法筛选出最终的选框

```python
def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                (not often) multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:
        		- pred_dicts: 实际是一个列表，其成员是字典
                	- pred_boxes
                    - pred_scores
                    - pred_labels
```

以上就是模板检测器的主要功能，下面看看 SECOND 类是怎么在基类上创建的

### SECOND

其实 SECOND 类的实现在基类之上是比较简单的，主要需要定义三个部分：

1. 调用基类的 `build_networks` 实现模型构建
2. 定义前向方程

模型的构建在初始化中完成，只用两行代码就完成了

```python
class SECONDNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
```

#### forward

前向方程 `forward` 也比较简单，就是将数据按顺序输入各个模块中，最后根据模式返回损失函数或者预测结果，其中损失函数一般定义在 

`dense_head` 当中

```python
def forward(self, batch_dict):
    for cur_module in self.module_list:
        batch_dict = cur_module(batch_dict)

    # 每一个 batch 的 loss (cls_loss + reg_loss +...)
    # tb_dict 是 tensor.item() 用于 tensorboard 可视化
    if self.training:
        loss, tb_dict, disp_dict = self.get_training_loss()

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict, disp_dict
    else:
        pred_dicts, recall_dicts = self.post_processing(batch_dict)
        return pred_dicts, recall_dicts
 
def get_training_loss(self):
    disp_dict = {}

    loss_rpn, tb_dict = self.dense_head.get_loss()
    tb_dict = {
        'loss_rpn': loss_rpn.item(),
        **tb_dict
    }

    loss = loss_rpn
    return loss, tb_dict, disp_dict
```

要深入学习还得看每个子模块的具体实现，继续痛苦的源码阅读，Voxel R-CNN 我来了...
