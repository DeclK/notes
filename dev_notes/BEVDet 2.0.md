# BEVDet 2.0

想要学习的点：

- [ ] CUDA
- [ ] BEV Aug
- [ ] 时序数据处理（总感觉这里有更好的选择...
- [ ] 把 TEHI 做大做强！

## 阅读论文笔记

### BEVDet

- BEV augmentation

  在训练过程中作者发现了过拟合现象，原因可能在于从图像空间转移到BEV空间时，图像的增广作用并不能同样使用于BEV空间。为解决这个问题提出了在 BEV 空间下的增广

- Scaled-NMS 以移除重叠选框

### BEVDet4D

- Align features

  需要将连续两帧的图像进行连接，需要对两个帧的特征空间进行统一。似乎是预测物体的移动来让特征对齐

### BEVPoolv2

- TODO

## 代码阅读

### Install

1. pytorch 1.11+cu113

2. mmlab related

   ```shell
   pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
   
   pip install mmdet==2.25.1 mmsegmentation==0.25.0
   ```

3. other packages

   ```shell
   pip install pycuda \
       lyft_dataset_sdk \
       networkx==2.2 \
       numba==0.53.0 \
       numpy \
       nuscenes-devkit \
       plyfile \
       scikit-image \
       tensorboard \
       trimesh==2.35.39
   ```

4. project itself

   ```shell
   pip install -v -e .
   ```

5. Dataset

   首先将 `tools/create_data_bevdet.py` 中的 trainval 改成 mini，因为我只有 mini 在手上。然后

   ```python
   python tools/create_data_bevdet.py
   ```

   使用 `samples_per_gpu=1` 成功跑通！说明 BEVDet 还是非常友好的

### Dataset

