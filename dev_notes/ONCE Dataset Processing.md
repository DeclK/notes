## ONCE Dataset Processing

为了加速训练过程，我需要加入自己关注的 database 信息，所以必须要学习如何构建自己的 database info

之前在看 `ONCEDataset` 的时候就看到 `__getitem` 就停止了，对于 `get_infos & create_ground_truth_database` 这两个方法并没有仔细看了，现在是必须要上了，在然后就是 `DatabaseSampler` 的功能整理一下，就基本打通了数据集的处理流程，然后就可以加入自己想要的特征了

## ONCEDataset

### get_infos

生成各个 split 生成 info 信息，在之后将这些 info 存储为 `once_infos_***.pkl` 文件

1. input params，基本不需要

2. output，一个列表 `all_infos`，存储了所有 frames 的信息

3. used functions

   1. `process_single_sequence`，返回一个 list of dict，每一个 dict 代表一个 frame，里面字典包括很多信息

      ```python
                      frame_dict = {
                          'sequence_id': seq_idx,
                          'frame_id': frame_id,
                          'timestamp': int(frame_id),
                          'prev_id': prev_id,
                          'next_id': next_id,
                          'meta_info': meta_info,
                          'lidar': pc_path,
                          'pose': pose,
                          'annos':
                          	{
                                  'name':
                                  'boxes_3d':
                                  'boxes_2d':
                                  'num_points_in_gt':
                              }
                      }
      ```

      最终的 `all_infos` 就是多个 `process_single_sequence` 的结果合在一起，形成一个大的 list of dict

### create_groundtruth_database

这一步是为 data augmentation 准备的，只针对训练集生成 database。把每个类别的物体都放在了一起，在之后好针对类别进行采样

1. input params，主要有两个：`info_path & split`，具体一点来说就是 `once_infos_train.pkl & train`。

2. output 可以认为有两个：

   1. `once_dbinfos_train.pkl` 是一个 dict of list of dict，存储所有类别的信息

      ```python
                      db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                                  'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
      ```

   2. `'%s_%s_%d.bin' % (frame_id, names[i], i)`，直接存储每个场景的每个物体

