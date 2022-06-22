# Experiment Results

## SA-SSD Structure

基本放弃使用 SA-SSD 结构

1. SA-SSD reproduce，更深的网络和更宽的通道都没法提升

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |72.49       |83.26       |65.65       |50.53       |
   |Pedestrian  |22.87       |25.22       |21.19       |16.40       |
   |Cyclist     |58.86       |69.79       |53.47       |37.67       |
   |mAP         |51.41       |59.42       |46.77       |34.86       |
   ```

## SECOND

### Supervised

1. baseline

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |72.61       |83.54       |66.00       |50.55       |
   |Pedestrian  |29.60       |33.76       |26.28       |17.83       |
   |Cyclist     |58.23       |69.43       |53.05       |35.87       |
   |mAP         |53.48       |62.24       |48.44       |34.75       |
   
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |73.27       |84.39       |66.07       |51.41       |
   |Pedestrian  |29.17       |33.41       |25.62       |16.98       |
   |Cyclist     |58.12       |68.96       |52.25       |35.98       |
   |mAP         |53.52       |62.25       |47.98       |34.79       |
   ```

2. res backbone baseline

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |73.96       |85.76       |69.30       |54.34       |
   |Pedestrian  |34.75       |39.80       |30.52       |19.17       |
   |Cyclist     |59.79       |70.80       |54.93       |39.55       |
   |mAP         |56.17       |65.45       |51.59       |37.69       |
   ```

### SSL

1. semi second

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |74.55       |85.11       |68.41       |55.19       |
   |Pedestrian  |31.30       |35.71       |27.47       |18.54       |
   |Cyclist     |60.12       |70.32       |55.39       |37.83       |
   |mAP         |55.32       |63.71       |50.42       |37.19       |
   
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |74.77       |87.07       |68.40       |54.10       |
   |Pedestrian  |32.31       |36.22       |27.73       |19.37       |
   |Cyclist     |60.03       |71.24       |55.25       |37.05       |
   |mAP         |55.70       |64.84       |50.46       |36.84       |
   ```

2. nms 0.7，和期望的一样，pedestrian & cyclist 效果下降

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |74.40       |85.05       |68.24       |53.98       |
   |Pedestrian  |30.38       |34.83       |26.05       |17.78       |
   |Cyclist     |59.36       |69.25       |54.73       |37.72       |
   |mAP         |54.71       |63.04       |49.67       |36.49       |
   ```
   
3. 使用了 se-ssd augmentation 后有了更好的 teacher 模型

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |75.59       |85.96       |70.37       |55.43       |
   |Pedestrian  |33.39       |38.14       |29.01       |18.73       |
   |Cyclist     |62.11       |73.63       |56.07       |39.87       |
   |mAP         |57.03       |65.91       |51.82       |38.01       |
   ```

    不清楚是不是这个 augmentation 起了作用，下一个实验去掉这个 augmentation 看看

4. 去掉 se-ssd augmentation，对 teacher & student 都使用常规的数据增强，效果更好

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |75.74       |86.50       |70.41       |55.11       |
   |Pedestrian  |34.30       |39.13       |30.89       |18.35       |
   |Cyclist     |62.34       |73.06       |56.67       |38.87       |
   |mAP         |57.46       |66.23       |52.66       |37.45       |
   ```

## CG-SSD

### Supervised Anchor-based

1. v1.0, aux module 除行人外均有提升

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |75.41       |85.68       |69.84       |55.62       |
   |Pedestrian  |28.21       |31.26       |25.84       |15.28       |
   |Cyclist     |60.89       |72.47       |55.15       |37.90       |
   |mAP         |54.84       |63.14       |50.28       |36.27       |
   ```

2. v1.0 + residule，残差模块不仅在 backbone 上使用，而且在 CGAM 和 BEV 的融合上有使用，并且减少了 1 个 head block

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |75.48       |85.37       |71.51       |56.33       |
   |Pedestrian  |36.82       |42.23       |31.15       |19.46       |
   |Cyclist     |62.00       |72.96       |57.76       |40.70       |
   |mAP         |58.10       |66.85       |53.47       |38.83       |
   ```

   后面再看自己的代码还是有点问题，路径分为了两条，shared conv 也多了一个，之后可能要改进一下

   ```python
           # aux module
           aux_feat = self.aux_module(data_dict)
           corner_hm, corner_offset = aux_feat['hm'], aux_feat['corner']
           spatial_features_2d = data_dict['spatial_features_2d']
           neck1 = self.neck_conv1(spatial_features_2d)
           neck1 = torch.cat((neck1, corner_hm, corner_offset), dim=1)
           neck2 = self.neck_conv2(neck1)
           spatial_features_2d = spatial_features_2d + neck2   # residule structure
   ```

3. v1.0 + cia backbone

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |73.78       |85.92       |68.15       |51.58       |
   |Pedestrian  |20.79       |22.06       |19.36       |16.32       |
   |Cyclist     |58.93       |71.57       |52.90       |36.17       |
   |mAP         |51.16       |59.85       |46.80       |34.69       |
   
   CIA-SSD
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |73.05       |86.04       |65.15       |50.05       |
   |Pedestrian  |29.81       |34.60       |25.36       |16.44       |
   |Cyclist     |59.07       |70.77       |53.03       |37.53       |
   |mAP         |53.98       |63.81       |47.85       |34.67       |
   ```

4. 改动了 database，并且改动了 anchor head aux

   ```python
           # aux module
           aux_feat = self.aux_module(data_dict)   # consider put this after neck1
           corner_hm, corner_offset = aux_feat['hm'], aux_feat['corner']
           spatial_features_2d = data_dict['spatial_features_2d']
           neck1 = self.neck_conv1(spatial_features_2d)    # consider not use double neck
           neck1 = torch.cat((neck1, corner_hm, corner_offset), dim=1)
           neck2 = self.neck_conv2(neck1)
           spatial_features_2d = spatial_features_2d + neck2   # residule structure
   
           
    # modified       
           # aux module
           aux_feat = self.aux_module(data_dict)   # consider put this after neck1
           corner_hm, corner_offset = aux_feat['hm'], aux_feat['corner']
           spatial_features_2d = data_dict['spatial_features_2d']
           neck = torch.cat((spatial_features_2d, corner_hm, corner_offset), dim=1)
           neck = self.neck_conv(neck)
           spatial_features_2d = spatial_features_2d + neck   # residule structure
   ```

   得到结果大有提升，看来残差的结构是有效的

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |75.27       |86.23       |69.47       |54.29       |
   |Pedestrian  |31.23       |35.98       |27.14       |15.40       |
   |Cyclist     |59.66       |71.15       |54.53       |38.04       |
   |mAP         |55.39       |64.45       |50.38       |35.91       |
   ```

5. 增加 channel，行人指标反而大退步

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |75.17       |85.62       |69.70       |54.27       |
   |Pedestrian  |23.85       |26.35       |21.88       |14.26       |
   |Cyclist     |60.23       |71.76       |54.77       |37.58       |
   |mAP         |53.08       |61.24       |48.78       |35.37       |
   ```

6. 之后将 database 保存下来，训练速度快了很多，不过这也让我 debug 很久...

   ```txt
   adjust input channel, single head, multi channel 5*3 + 2*3, right corner
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |73.18       |83.64       |67.47       |51.90       |
   |Pedestrian  |29.30       |32.51       |26.34       |19.19       |
   |Cyclist     |59.05       |71.09       |52.27       |36.86       |
   |mAP         |53.85       |62.41       |48.69       |35.98       |
   
   use conv to fuse feature, single head, simple channel 3 + 2, wrong corner
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |75.27       |86.23       |69.47       |54.29       |
   |Pedestrian  |31.23       |35.98       |27.14       |15.40       |
   |Cyclist     |59.66       |71.15       |54.53       |38.04       |
   |mAP         |55.39       |64.45       |50.38       |35.91       |
   ```

7. 最后还是用最简单的实现达到了效果，增加的 channel 数量 (5 + 2) * 4

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |73.59       |84.32       |67.32       |52.77       |
   |Pedestrian  |39.07       |44.97       |33.23       |21.95       |
   |Cyclist     |63.57       |74.89       |58.66       |40.69       |
   |mAP         |58.74       |68.06       |53.07       |38.47       |
   
   cia backbone
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |73.03       |83.83       |67.31       |49.47       |
   |Pedestrian  |41.66       |47.79       |35.46       |21.94       |
   |Cyclist     |62.44       |73.77       |57.93       |39.89       |
   |mAP         |59.04       |68.46       |53.57       |37.10       |
   ```


8. 使用一个 residule block 在 aux feature 后，想看看感受野扩大之后的结果，没有提升。为方便实验用的 anchor head

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |73.49       |84.01       |67.37       |52.35       |
   |Pedestrian  |39.35       |46.53       |32.32       |20.11       |
   |Cyclist     |61.80       |72.82       |57.69       |40.12       |
   |mAP         |58.21       |67.79       |52.46       |37.53       |
   ```

9. 使用原始特征的结果也差不多，使用的 anchor head

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |73.23       |85.20       |67.16       |52.24       |
   |Pedestrian  |39.16       |45.54       |33.02       |18.39       |
   |Cyclist     |62.94       |73.16       |58.30       |39.81       |
   |mAP         |58.44       |67.97       |52.83       |36.81       |
   ```

10. 仅使用 corner 2 center 的信息也能行，不整体还是要差一些，还是不要搞些花里胡哨了，搞完这一波就该上路了

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |72.82       |84.19       |64.89       |49.57       |
   |Pedestrian  |39.78       |45.99       |33.63       |21.62       |
   |Cyclist     |61.84       |73.52       |55.80       |39.93       |
   |mAP         |58.15       |67.90       |51.44       |37.04       |
   ```

### Supervised Anchor-free

1. CG-SSD 在 centerpoint 上的效果好像都不怎么好

   ```txt
   before shared conv 3 corners
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |77.92       |87.72       |72.50       |60.89       |
   |Pedestrian  |49.75       |58.42       |40.28       |25.50       |
   |Cyclist     |68.12       |78.04       |62.91       |46.81       |
   |mAP         |65.26       |74.73       |58.57       |44.40       |
   aux weight cls = 1 loc = 2
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |77.80       |86.16       |73.10       |58.43       |
   |Pedestrian  |48.48       |55.72       |40.48       |24.49       |
   |Cyclist     |68.35       |78.09       |62.86       |47.91       |
   |mAP         |64.87       |73.32       |58.81       |43.61       |
   
   
   after shared conv and use extra fuse conv 3 corners
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |77.99       |86.31       |72.73       |60.87       |
   |Pedestrian  |48.68       |56.14       |40.10       |25.71       |
   |Cyclist     |67.58       |77.99       |62.37       |44.84       |
   |mAP         |64.75       |73.48       |58.40       |43.81       |
   
   before prediction layer
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.06       |87.93       |72.85       |60.43       |
   |Pedestrian  |48.70       |56.14       |40.46       |25.45       |
   |Cyclist     |68.05       |78.07       |62.57       |46.13       |
   |mAP         |64.94       |74.05       |58.63       |44.00       |
   aux weight cls = 1 loc = 2
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |77.61       |85.88       |72.54       |60.66       |
   |Pedestrian  |49.56       |58.02       |40.45       |25.02       |
   |Cyclist     |68.12       |77.87       |62.85       |46.90       |
   |mAP         |65.09       |73.92       |58.62       |44.19       |
   ```

2. 尝试 detach

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.12       |88.10       |72.84       |60.57       |
   |Pedestrian  |49.50       |58.06       |40.54       |24.36       |
   |Cyclist     |67.76       |77.32       |63.16       |46.65       |
   |mAP         |65.13       |74.49       |58.85       |43.86       |
   ```

3. 尝试在 share conv 之后加入，没有 fuse conv

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.25       |87.98       |72.96       |61.20       |
   |Pedestrian  |48.88       |57.38       |40.00       |25.09       |
   |Cyclist     |68.22       |78.19       |63.08       |45.85       |
   |mAP         |65.12       |74.52       |58.68       |44.05       |
   ```

4. 尝试调整 aux weight 0.5

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.13       |87.98       |72.91       |60.78       |
   |Pedestrian  |49.24       |56.19       |42.20       |26.91       |
   |Cyclist     |68.47       |78.17       |64.02       |46.55       |
   |mAP         |65.28       |74.11       |59.71       |44.75       |
   ```

5. 尝试调整 aux weight: cls = 1, loc = 2

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.41       |86.69       |74.67       |60.30       |
   |Pedestrian  |50.57       |58.65       |41.54       |24.22       |
   |Cyclist     |68.25       |78.16       |63.33       |46.09       |
   |mAP         |65.74       |74.50       |59.85       |43.54       |
   ```
   
   在此基础上使用 class specific nms，再提一点点
   
   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.30       |86.61       |74.63       |60.25       |
   |Pedestrian  |51.44       |59.63       |42.67       |24.45       |
   |Cyclist     |68.27       |78.01       |64.05       |46.45       |
   |mAP         |66.00       |74.75       |60.45       |43.72       |
   ```
   
6. With Better fusion mechanism

### SSL

1. cia backbone, anchor head

   ```txt
   baseline
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |73.03       |83.83       |67.31       |49.47       |
   |Pedestrian  |41.66       |47.79       |35.46       |21.94       |
   |Cyclist     |62.44       |73.77       |57.93       |39.89       |
   |mAP         |59.04       |68.46       |53.57       |37.10       |
   
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |76.54       |86.83       |71.01       |56.64       |
   |Pedestrian  |41.96       |48.46       |35.02       |21.40       |
   |Cyclist     |65.40       |76.53       |59.82       |42.31       |
   |mAP         |61.30       |70.61       |55.28       |40.12       |
   ```

   也出现了对行人提升不高的情况

2. res spconv, anchor head

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.28       |86.89       |72.99       |60.04       |
   |Pedestrian  |42.28       |48.78       |34.29       |20.71       |
   |Cyclist     |67.41       |77.65       |62.55       |44.03       |
   |mAP         |62.65       |71.11       |56.61       |41.60       |
   ```

### Joint Inference

尝试 RepPoints v2 里面的 joint inference，经过一些尝试都没办法提点，放弃

## CenterPoint

### Supervised

1. OpenPCDet implementation

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |77.58       |87.50       |73.22       |61.41       |
   |Pedestrian  |48.24       |58.61       |39.69       |22.42       |
   |Cyclist     |68.54       |79.07       |63.73       |46.27       |
   |mAP         |64.78       |75.06       |58.88       |43.37       |
   
   NMS thresh = 0.01
   
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |77.70       |86.33       |72.40       |60.70       |
   |Pedestrian  |49.41       |58.11       |40.44       |24.06       |
   |Cyclist     |67.71       |77.95       |62.80       |46.19       |
   |mAP         |64.94       |74.13       |58.55       |43.65       |
   ```

2. ONCE Benchmark implementation，对于行人有更好的表现，但是其他两类都差的多

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |67.30       |80.62       |59.76       |42.15       |
   |Pedestrian  |50.14       |57.46       |42.31       |25.80       |
   |Cyclist     |63.86       |74.75       |58.22       |40.13       |
   |mAP         |60.43       |70.94       |53.43       |36.03       |
   ```

### IoU

1. first edition

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |79.12       |89.50       |73.71       |60.28       |
   |Pedestrian  |46.37       |56.22       |37.90       |24.41       |
   |Cyclist     |66.76       |78.11       |60.58       |44.24       |
   |mAP         |64.08       |74.61       |57.39       |42.98       |
   ```

2. only iou regression

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.00       |88.25       |72.52       |59.98       |
   |Pedestrian  |48.67       |55.93       |41.44       |24.16       |
   |Cyclist     |66.59       |77.43       |61.63       |46.52       |
   |mAP         |64.42       |73.87       |58.53       |43.55       |
   ```

3. only iou rectify

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.90       |89.33       |73.13       |61.00       |
   |Pedestrian  |47.12       |56.44       |39.43       |23.16       |
   |Cyclist     |66.88       |78.05       |61.35       |44.06       |
   |mAP         |64.30       |74.61       |57.97       |42.74       |
   
   # just loss, no rectify
   
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |77.71       |86.49       |71.04       |60.07       |
   |Pedestrian  |49.71       |58.29       |41.42       |24.46       |
   |Cyclist     |66.55       |77.33       |62.27       |45.10       |
   |mAP         |64.66       |74.04       |58.24       |43.21       |
   ```

4. 不粗暴地使用 CIA-SSD 中的公式，选择如下公式
   $$
   score = S^{1-\lambda}·IoU^{\lambda}
   $$
   对所有类别使用 $\lambda=0.5$ 有不错的效果，不知道结合 cgam 能不能又更多的提升

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.53       |88.75       |72.82       |60.85       |
   |Pedestrian  |51.53       |60.22       |42.40       |24.81       |
   |Cyclist     |67.35       |78.26       |62.77       |45.70       |
   |mAP         |65.80       |75.74       |59.33       |43.79       |
   ```

5. 使用 multi class nms 又有提升，依然使用 $\lambda = 0.5$

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |77.85       |87.58       |72.23       |60.78       |
   |Pedestrian  |53.47       |62.67       |44.53       |26.22       |
   |Cyclist     |67.79       |77.94       |63.00       |46.15       |
   |mAP         |66.37       |76.06       |59.92       |44.38       |
   ```

6. 经过搜索过后获得更优的模型

   ```txt
   RECTIFIER: [0.8, 0.8, 0.8, 0.4, 0.6]
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.85       |89.30       |73.09       |60.94       |
   |Pedestrian  |53.04       |62.03       |43.74       |25.53       |
   |Cyclist     |68.13       |78.30       |63.28       |46.30       |
   |mAP         |66.67       |76.55       |60.04       |44.26       |
   
   RECTIFIER: [0.0, 0.0, 0.0, 0.0, 0.0]
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |77.69       |87.89       |72.11       |60.05       |
   |Pedestrian  |51.09       |59.20       |42.52       |25.07       |
   |Cyclist     |67.25       |77.26       |62.85       |45.56       |
   |mAP         |65.34       |74.78       |59.16       |43.56       |
   ```

### IoU Regression



### SSL

1. **semi** centerpoint, single score，行人更差

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |79.42       |88.70       |73.99       |62.32       |
   |Pedestrian  |47.39       |56.91       |39.18       |22.21       |
   |Cyclist     |68.72       |79.23       |63.91       |46.45       |
   |mAP         |65.18       |74.95       |59.03       |43.66       |
   ```

   **semi** centerpoint, single score, iou match，更差

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.18       |88.20       |74.02       |61.99       |
   |Pedestrian  |47.23       |57.01       |38.30       |20.98       |
   |Cyclist     |68.49       |78.87       |63.67       |46.60       |
   |mAP         |64.63       |74.69       |58.66       |43.19       |
   ```

2. **semi** centerpoint, all scores，结果差

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.38       |88.52       |73.93       |62.23       |
   |Pedestrian  |47.03       |56.60       |38.31       |21.91       |
   |Cyclist     |68.52       |79.15       |63.43       |46.97       |
   |mAP         |64.64       |74.76       |58.55       |43.70       |
   ```

   semi centerpoint, all scores, iou match，稍微好一点，但总体依然更差

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |79.25       |88.44       |74.11       |62.00       |
   |Pedestrian  |47.34       |57.09       |38.21       |21.94       |
   |Cyclist     |68.72       |79.24       |63.81       |46.30       |
   |mAP         |65.10       |74.92       |58.71       |43.41       |
   ```

3. semi centerpoint, **size loss**，差

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.32       |88.50       |73.79       |61.53       |
   |Pedestrian  |46.49       |56.87       |38.17       |21.06       |
   |Cyclist     |69.07       |79.59       |64.13       |47.15       |
   |mAP         |64.63       |74.99       |58.70       |43.25       |
   ```

   **center loss**，稍好

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.58       |88.69       |74.39       |62.13       |
   |Pedestrian  |47.93       |57.64       |39.42       |22.53       |
   |Cyclist     |69.15       |79.60       |64.34       |46.73       |
   |mAP         |65.22       |75.31       |59.38       |43.80       |
   ```

4. semi centerpoint, nms 0.01 稍好

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.49       |86.92       |73.03       |61.50       |
   |Pedestrian  |48.76       |57.27       |40.49       |22.90       |
   |Cyclist     |68.41       |78.51       |63.10       |46.11       |
   |mAP         |65.22       |74.24       |58.87       |43.50       |
   ```

5. 使用不同的 nms score thres 0.25, 0.6, 甚至测试了 score 区间

   ```txt
   nms score 0.6
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.75       |88.70       |73.08       |61.83       |
   |Pedestrian  |48.09       |57.03       |38.65       |21.78       |
   |Cyclist     |68.34       |78.40       |62.94       |46.42       |
   |mAP         |65.06       |74.71       |58.22       |43.34       |
   
   nms score 0.25
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.29       |88.05       |72.76       |61.49       |
   |Pedestrian  |49.13       |57.47       |40.78       |24.10       |
   |Cyclist     |67.75       |77.59       |62.53       |46.58       |
   |mAP         |65.06       |74.37       |58.69       |44.06       |
   
   nms score 0~0.1 0.5~1
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.06       |87.89       |72.40       |60.83       |
   |Pedestrian  |48.17       |55.62       |39.34       |23.36       |
   |Cyclist     |67.81       |77.71       |63.04       |45.49       |
   |mAP         |64.68       |73.74       |58.26       |43.23       |
   ```

6. 使用了数据增强，获得了最好的 teacher model，nms score = 0.1，nms thresh = 0.01

   ```txt
   teacher
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.60       |88.55       |73.01       |61.57       |
   |Pedestrian  |49.92       |57.44       |41.79       |25.24       |
   |Cyclist     |68.32       |78.32       |63.06       |46.88       |
   |mAP         |65.61       |74.77       |59.28       |44.57       |
   
   student
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.37       |86.87       |72.77       |61.11       |
   |Pedestrian  |48.71       |57.03       |40.44       |24.82       |
   |Cyclist     |68.30       |78.35       |62.87       |46.99       |
   |mAP         |65.13       |74.08       |58.69       |44.31       |
   ```

7. 使用了更大的学习率 learning rate 0.003，似乎没有想想中的效果

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.14       |86.52       |72.84       |60.65       |
   |Pedestrian  |48.78       |57.35       |40.00       |23.22       |
   |Cyclist     |67.08       |77.37       |62.13       |45.79       |
   |mAP         |64.67       |73.75       |58.33       |43.22       |
   ```

8. 使用 epoch 40 & warm up 12 epoch，依然是 teacher model 表现更好，但是没有出现之前的最好的情况

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.38       |88.36       |72.67       |61.42       |
   |Pedestrian  |49.37       |57.82       |41.08       |22.82       |
   |Cyclist     |67.67       |77.84       |62.39       |46.14       |
   |mAP         |65.14       |74.67       |58.71       |43.46       |
   ```

9. 使用 longer  epoch 60 & learning rate 0.003 & warm up 8 epoch，终于出现和之前最好差不多的模型

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |78.67       |86.90       |73.21       |63.14       |
   |Pedestrian  |49.45       |56.68       |41.76       |24.19       |
   |Cyclist     |68.43       |79.26       |62.84       |46.30       |
   |mAP         |65.52       |74.28       |59.27       |44.54       |
   ```

10. 在 9 的基础上尝试加入其他损失，试一试 center loss

    ```txt
    |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
    |Vehicle     |78.66       |88.47       |73.08       |63.05       |
    |Pedestrian  |49.29       |56.84       |41.27       |22.31       |
    |Cyclist     |68.24       |78.03       |63.89       |46.32       |
    |mAP         |65.40       |74.45       |59.41       |43.89       |
    ```

    确实是效果更低了，在大量的 unlabeled sample 下 center loss 都没有作用，现在少量的 augmentation 可能作用也不大
    
11. 尝试了 se-ssd augmentaion 效果不大好，并且又出现了 teacher 更好的情况

    ```txt
    student
    |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
    |Vehicle     |61.19       |77.13       |50.03       |41.79       |
    |Pedestrian  |36.31       |46.06       |26.07       |11.44       |
    |Cyclist     |57.07       |68.84       |50.55       |34.14       |
    |mAP         |51.52       |64.01       |42.22       |29.12       |
    
    teacher
    |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
    |Vehicle     |78.17       |87.78       |72.98       |61.12       |
    |Pedestrian  |48.17       |56.52       |40.54       |22.88       |
    |Cyclist     |68.18       |77.99       |63.02       |46.63       |
    |mAP         |64.84       |74.09       |58.85       |43.54       |
    ```

12. lr 0.001 epoch 60 warm up 8，还是用 lr 0.003 好

    ```txt
    |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
    |Vehicle     |78.38       |88.10       |72.79       |62.61       |
    |Pedestrian  |47.75       |56.13       |39.34       |22.33       |
    |Cyclist     |68.30       |78.22       |62.67       |47.62       |
    |mAP         |64.81       |74.15       |58.27       |44.19       |
    ```


## PillarNet

1. PillarNet spconv 2D + neck v2

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |75.66       |85.68       |70.37       |56.65       |
   |Pedestrian  |23.87       |26.41       |21.63       |16.39       |
   |Cyclist     |57.95       |69.03       |51.93       |34.58       |
   |mAP         |52.50       |60.38       |47.98       |35.87       |
   ```

2. PillarNet small pillar size 0.05

   ```txt
   |AP@50       |overall     |0-30m       |30-50m      |50m-inf     |
   |Vehicle     |73.69       |85.72       |67.74       |53.46       |
   |Pedestrian  |28.27       |32.25       |23.12       |14.95       |
   |Cyclist     |59.29       |71.11       |53.07       |36.76       |
   |mAP         |53.75       |63.03       |47.98       |35.06       |
   ```

3. with deeper PillarVFE
