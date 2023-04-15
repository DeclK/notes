# NuScenes Dataset

NuScenes 数据集是论文中常用的数据集，他没有 waymo 那么大，但是又比 KITTI ONCE 标注更多，并且支持多个任务。我觉得掌握该数据集的表示形式非常重要，对于自动驾驶场景的数据理解会很有帮助

我希望了解几个方面：

1. 数据集的 sample 由哪些部分组成，各个部分有什么作用？尤其是相机参数以及连续帧的部分，我一直想要了解
2. nuscenes_info_pkl 是怎么生成的，做了哪些处理
3. nuscenes 数据集的任务包含哪些

## 下载 NuScenes Dataset

对于数据集可以直接去官网下载 [nuScenes download](https://www.nuscenes.org/download) 现在 Asia 有服务器，速度也挺不错的！建议找个网速好的地方，我在学校宿舍里有 10 MB/s，但在学院办公室只有 1 MB/s 不到

在下载页面有几个可下载的：

1. nuPlan，这个没接触过

2. nuImages，2D 选框

3. nuScenes-panoptic，全景分割标签

4. **nuScenes-lidarseg**，语义分割标签，语义分割与全景分割的区别 [CSDN](https://blog.csdn.net/qq_29893385/article/details/90213699)

   <img src="NuScenes Dataset/image-20230412143754283.png" alt="image-20230412143754283" style="zoom: 25%;" />

5. **Full dataset v1.0**，完整数据集（但不包含上述部分）

由于整个数据集太大了，可以先下载 mini 数据集熟悉结构

## NuScenes

### 文件结构

解压过后

```python
- nuscenes
	- maps		# 地图, png 文件，基本不用
	- samples	# 样本，一个场景中的关键帧
	- sweeps	# 样本，与 samples 结构一致，一个场景中的非关键帧，不包含标注
	- v1.0-trainval	# or v1.0-mini，为 json 数据，类似于 COCO 中的 json
    				# 包含了文件路径，标签，类别，传感器参数等等
```

可以看看 samples & sweeps 的文件结构

```python
- samples
	- CAM_BACK			# 6个相机采集的图像
    - CAM_BACK_LEFT
    - CAM_BACK_RIGHT
    - CAM_FRONT
    - CAM_FRONT_LEFT
    - CAM_FRON_RIGHT
    - LIDAR_TOP			# 激光雷达采集数据，因为雷达是放在车的最上面的
    - RADAR_BACK_LEFT	# 毫米波雷达采集数据
    - RADAR_BACK_RIGHT
    - RADAR_FRONT
    - RADAR_FRONT_LEFT
    - RADAR_FRONT_RIGHT
```

采集车的示意图，这个示意图非常重要，**标明了各个传感器的坐标系**，在之后使用 `calibrated_sensor` 以及 `ego_pose` 时可以帮助理解！

<img src="NuScenes Dataset/data.9ef46c59.png" alt="data" style="zoom: 25%;" />

### 数据读取

nuScenes 发布了 `nuscenes-devkit` 用于方便调取样本，类似于 `pycocotool`，官方提供了 [Colab](https://colab.research.google.com/github/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuscenes_tutorial.ipynb) 用于学习基本操作

下面将逐个介绍 nuScenes 中的样本元素，并利用 `nuscenes-devkit` 获得这些元素

#### scene

nuScenes 一共包含了 1000 个 scenes，其中 850(=700 + 150) 个 scenes 为 trainval 数据集，剩余 150 个 scenes 为 test 数据集。每一个 scenes 为一个 20s 左右的采集片段，并且以 2Hz 的频率对片段进行标记

mini 数据集为 10 个 scenes

```python
from nuscenes.nuscenes import NuScenes

# 创建 database
nusc = NuScenes(version='v1.0-mini', dataroot='/github/The-Eyes-Have-It/data/nuscenes', verbose=True)

# 列出 scenes
nusc.list_scenes()
# output
"scene-0061, Parked truck, construction, intersectio... [18-07-24 03:28:47]   19s, singapore-onenorth, #anns:4622"

# 获得 scenes meta, scenes 是一个 list of dict
# 每一个 dict 包含 scene 的 token, name, first_sample_token...
scenes = nusc.scene
print(scenes[0])

# output
{'token': 'cc8c0bf57f984915a77078b10eb33198',
 'log_token': '7e25a2c8ea1f41c5b0da1e69ecfa71a2',
 'nbr_samples': 39,
 'first_sample_token': 'ca9a282c9e77460f8360f564131a8af5',
 'last_sample_token': 'ed5fc18c31904f96a8f0dbb99ff069c0',
 'name': 'scene-0061',
 'description': 'Parked truck, construction, intersection, turn left, following a van'}
```

**重要！nuScenes api 有一个非常常用的 `nusc.get(name, token)` 可以通过 token 获得数据集中的任意数据的 meta 信息。**例如，上述的 `scene` 也可以通过 `token` 获得

```python
nusc.get('sample', 'cc8c0bf57f984915a77078b10eb33198')
# equal with: scenes[0]
```

所谓的 meta 信息，就是一切**除图像数据以外的信息**，例如数据路径、采集时间、tokens、labels 等等

#### sample

sample 的定义是：在一个 scene 中所标注的关键帧。如前所述，关键帧的标注频率为 2Hz，即半秒就标注一次。所以一个 scene 大约有 40 个 samples。

在 nuScenes 中每一个 sample/key frame 都有自己的 token，在 scene 中就列有 `first_sample_token & last_sample_token`。通过 token 就可以获得 sample 的 meta 数据

可以看到包含了**传感器数据以及标签的 token！**

```python
my_scene = nusc.scene[0]
first_sample_token = my_scene['first_sample_token']

# nusc 中提供了不少 render 函数
nusc.render_sample(first_sample_token)

# 获得 sample
my_sample = nusc.get('sample', first_sample_token)

# output
{'token': 'ca9a282c9e77460f8360f564131a8af5',
 'timestamp': 1532402927647951,
 'prev': '',
 'next': '39586f9d59004284a7114a68825e8eec',
 'scene_token': 'cc8c0bf57f984915a77078b10eb33198',
 'data': {'RADAR_FRONT': '37091c75b9704e0daa829ba56dfa0906',
  'RADAR_FRONT_LEFT': '11946c1461d14016a322916157da3c7d',
  'RADAR_FRONT_RIGHT': '491209956ee3435a9ec173dad3aaf58b',
  'RADAR_BACK_LEFT': '312aa38d0e3e4f01b3124c523e6f9776',
  'RADAR_BACK_RIGHT': '07b30d5eb6104e79be58eadf94382bc1',
  'LIDAR_TOP': '9d9bf11fb0e144c8b446d54a8a00184f',
  'CAM_FRONT': 'e3d495d4ac534d54b321f50006683844',
  'CAM_FRONT_RIGHT': 'aac7867ebf4f446395d29fbd60b63b3b',
  'CAM_BACK_RIGHT': '79dbb4460a6b40f49f9c150cb118247e',
  'CAM_BACK': '03bea5763f0f4722933508d5999c5fd8',
  'CAM_BACK_LEFT': '43893a033f9c46d4a51b5e08a67a1eb7',
  'CAM_FRONT_LEFT': 'fe5422747a7d4268a4b07fc396707b23'},
 'anns': ['ef63a697930c4b20a6b9791f423351da',
  '6b89da9bf1f84fd6a5fbe1c3b236f809',
  '924ee6ac1fed440a9d9e3720aac635a0',
   ...]}
```

`nusc` 中的所有性质既可以通过 token 获得，也可以通过列表获得，这个列表为 `nusc` 的一个属性

```python
nusc.sample[0]
nusc.sample_data[0]
```

#### sample_data

`sample_data` 包含了具体的传感器采集数据的 meta 信息，例如获得前视摄像头的 meta 数据。包含了：图像路径，相机参数，自车姿势等等

```python
cam_token = my_sample['data']['CAM_FRONT']
cam_meta = nusc.get('sample_data', cam_token)

# output
{'token': 'e3d495d4ac534d54b321f50006683844',
 'sample_token': 'ca9a282c9e77460f8360f564131a8af5',
 'ego_pose_token': 'e3d495d4ac534d54b321f50006683844',
 'calibrated_sensor_token': '1d31c729b073425e8e0202c5c6e66ee1',
 'timestamp': 1532402927612460,
 'fileformat': 'jpg',
 'is_key_frame': True,
 'height': 900,
 'width': 1600,
 'filename': 'samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg',
 'prev': '',
 'next': '68e8e98cf7b0487baa139df808641db7',
 'sensor_modality': 'camera',
 'channel': 'CAM_FRONT'}
```

需要注意的是这里的 `sample_data` 包含了 sweep 的信息，而不仅仅是关键帧的信息，你可以查看下一个 `sample_data` 的 meta

```python
nusc.get('sample_data', '68e8e98cf7b0487baa139df808641db7')

# output
{'token': '68e8e98cf7b0487baa139df808641db7',
 'sample_token': '39586f9d59004284a7114a68825e8eec',
 'ego_pose_token': '68e8e98cf7b0487baa139df808641db7',
 'calibrated_sensor_token': '1d31c729b073425e8e0202c5c6e66ee1',
 'timestamp': 1532402927662460,
 'fileformat': 'jpg',
 'is_key_frame': False,
 'height': 900,
 'width': 1600,
 'filename': 'sweeps/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927662460.jpg',
 'prev': 'e3d495d4ac534d54b321f50006683844',
 'next': '512015c209c1490f906982c3b182c2a8',
 'sensor_modality': 'camera',
 'channel': 'CAM_FRONT'}
```

其中 `calibrated_sensor` 以及 `ego_pose` 可用于投影和校正，通过这两个 issue  [sensor](https://github.com/nutonomy/nuscenes-devkit/issues/744) [ego pose](https://github.com/nutonomy/nuscenes-devkit/issues/841) 可以理解如何使用它们

- `calibrated_sensor` 中所描述的转换，是当前 timestamp，sensor 坐标系到自车坐标系的转换。下面举一个例子，将 LIDAR 坐标系中的点云转换到全局坐标系当中

  ```python
  from pyquaternion import Quaternion
  
  # cs means calibrated sensor, pc means point cloud
  
  # transform the pointcloud to the ego vehicle frame for the timestamp of the sweep
  cs_record = self.nusc.get('calibrated_sensor', token) 
  pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
  # rotate function
  # self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])
  
  # transform from ego to the global frame
  pc.translate(np.array(cs_record['translation'])) 
  # translate function
  # self.points[:3, :] = self.points[:3, :] + x[:, None]
  ```

- `ego_pose` 中存储是自车坐标系与全局坐标系 global coords 之间的转换。global coords 的原点定义在 map 的左上角 [global coord](https://github.com/nutonomy/nuscenes-devkit/issues/803)，ego car 原点就是 IMU 坐标系，还是如这个示意图所示

  <img src="NuScenes Dataset/data.9ef46c59.png" alt="data" style="zoom: 25%;" />

  官方描述为：**We express extrinsic coordinates relative to the ego frame, i.e. the midpoint of the rear vehicle axle.** 

#### sample_annotation

包含了 instance, attribute, category, bbox 等信息与 meta，所有的标注信息都是在 global coordinate system 下表示

```python
my_annotation_token = my_sample['anns'][7]
my_annotation_metadata =  nusc.get('sample_annotation', my_annotation_token)

# output
{'token': '36d52dfedd764b27863375543c965376',
 'sample_token': 'ca9a282c9e77460f8360f564131a8af5',
 'instance_token': 'f4b2632a2f9947da9f7959a3bd0e322c',
 'visibility_token': '1',
 'attribute_tokens': ['a14936d865eb4216b396adae8cb3939c'],
 'translation': [372.664, 1129.247, 0.672],
 'size': [0.689, 1.77, 1.709],
 'rotation': [0.994910649043305, 0.0, 0.0, -0.10076110569177861],
 'prev': '',
 'next': '86214ec54d034a839ee1f400719d49b2',
 'num_lidar_pts': 1,
 'num_radar_pts': 0,
 'category_name': 'vehicle.bicycle'}

nusc.render_annotation(my_annotation_token)
```

<img src="NuScenes Dataset/image-20230413160335452.png" alt="image-20230413160335452" style="zoom: 33%;" />

其中：

- rotation & translation 类似于 calibrated sensor，描述的是物体中心

- **instance** 代表的是一个具体的车，这个车会在多个帧中出现，这个标注信息对于跟踪任务是必须的

  ```python
  my_instance_token = my_annotation_metadata['instance_token']
  instance_meta = nusc.get('instance', my_instance_token)
  
  # output
  {'token': 'f4b2632a2f9947da9f7959a3bd0e322c',
   'category_token': 'fc95c87b806f48f8a1faea2dcc2222a4',
   'nbr_annotations': 35,
   'first_annotation_token': '36d52dfedd764b27863375543c965376',
   'last_annotation_token': '92ff17eaebd84b33b671d9fb700c7bb6'}
  ```

- attribute 代表的是物体的状态，例如自行车上有人没人，这个物体是动态的还是静止的。一个物体可以有多个 attributes，也可以一个没有

  ```python
  my_attribute_token = my_annotation_metadata['attribute_tokens'][0]
  attribute_meta = nusc.get('attribute', my_instance_token)
  
  # output
  {'token': 'a14936d865eb4216b396adae8cb3939c',
   'name': 'cycle.with_rider',
   'description': 'There is a rider on the bicycle or motorcycle.'}
  ```

- visibility 代表物体的可见状态

## 补充

### quaternion

[万向锁解释](https://www.bilibili.com/video/BV1Nr4y1j7kn)，四元数能够很好的解决万向锁（对比欧拉角）以及差值（对比矩阵旋转）

实际上 nuScenes 在进行旋转运算的时候依然是使用的矩阵旋转，四元数只是充当了计算旋转矩阵的中间媒介

```python
from pyquaternion import Quaternion

Quaternion([w, x, y, z]).rotation_matrix
```

四元数的表示与正交矩阵表示是等价的，参考 [wiki zh](https://zh.wikipedia.org/wiki/%E5%9B%9B%E5%85%83%E6%95%B0%E4%B8%8E%E7%A9%BA%E9%97%B4%E6%97%8B%E8%BD%AC)，[wiki en](https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation) 的描述更加详细，我尝试去看，但是还是没办法自然理解，目前只能将四元数的三维旋转作为一个神奇的公式

给定一个点 $p=(X,Y,Z)$ 以及一个四元数 $q=(w, x, y, z)=w+xi+yj+jk$，我们可以用如下算法计算

1. 将点表示为纯四元数 $p=Xi+Yj+Zk$
2. 计算 $qpq^{-1}$，其中 $q^{-1}=\frac{1}{||q||}q^*=\frac{1}{||q||}(w-xi-yj-zk)$

四元数中实部描述了旋转角，虚部描述了旋转轴，并且旋转方向遵循右手法则

### get_sample_data

在 nuScenes 接口中一个常用的方法就是 `nusc.get_sample_data(sample_data_token)`，该方法能够返回数据路径、bboxes，相机内参。并且能够将 bbox 转换到对应的传感器坐标当中

```python
    def get_sample_data(self, sample_data_token: str,
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,	# rest params are not often used
                        selected_anntokens: List[str] = None,
                        use_flat_vehicle_coordinates: bool = False) -> \
            Tuple[str, List[Box], np.array]:
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        :param sample_data_token: Sample_data token.

        :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """
```

其中 `Box` 是 nuScenes 自己定义的数据结构，定义了简单的旋转、平移、获得角点的操作。但通常在处理数据为 pkl 文件时，直接存储 7 维的 bbox 信息

