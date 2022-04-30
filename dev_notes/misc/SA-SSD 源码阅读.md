# SA-SSD 源码阅读

主要对于 SA-SSD 的辅助网络架构感兴趣，参考 [CSDN](https://blog.csdn.net/qq_39732684/article/details/105147497)

## SpMiddleFHD

### init

```python
class SpMiddleFHD(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=4,
                 num_hidden_features=128,
                 ):

        super(SpMiddleFHD, self).__init__()

        print(output_shape)
        self.sparse_shape = output_shape

        # 稀疏 3d 卷积
        self.backbone = VxNet(num_input_features)
        # 6个 Conv2d 对 BEV 特征进行特征提取
        self.fcn = BEVNet(in_features=num_hidden_features, num_filters=256)

        self.point_fc = nn.Linear(160, 64, bias=False)
        self.point_cls = nn.Linear(64, 1, bias=False)
        self.point_reg = nn.Linear(64, 3, bias=False)
```

### forward

```python
    def forward(self, voxel_features, coors, batch_size, is_test=False):

        # voxel mean features 
        points_mean = torch.zeros_like(voxel_features)
        points_mean[:, 0] = coors[:, 0]
        points_mean[:, 1:] = voxel_features[:, :3]

        coors = coors.int()
        x = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        # 获得 backbone 特征 x & 不同分辨率特征 middle
        x, middle = self.backbone(x)

        x = x.dense()
        N, C, D, H, W = x.shape
        x = x.view(N, C * D, H, W)

        x, conv6 = self.fcn(x)

        if is_test:
            return x, conv6
        else:
            # auxiliary network
            # vx_feat: (num_non-empty_voxels, channels) vx_nxyz: (num_non-empty_voxels, 4) lidar 坐标系
            vx_feat, vx_nxyz = tensor2points(middle[0], (0, -40., -3.), voxel_size=(.1, .1, .2))
            # 获得插值特征 (num_points, features)
            p0 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

            vx_feat, vx_nxyz = tensor2points(middle[1], (0, -40., -3.), voxel_size=(.2, .2, .4))
            p1 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

            vx_feat, vx_nxyz = tensor2points(middle[2], (0, -40., -3.), voxel_size=(.4, .4, .8))
            p2 = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat)

            pointwise = self.point_fc(torch.cat([p0, p1, p2], dim=-1))
            point_cls = self.point_cls(pointwise)
            point_reg = self.point_reg(pointwise)

            return x, conv6, (points_mean, point_cls, point_reg)

```

### aux target & aux loss

```python
    def build_aux_target(self, nxyz, gt_boxes3d, enlarge=1.0):
        center_offsets = list()
        pts_labels = list()
        for i in range(len(gt_boxes3d)):
            # 对每一个 batch 进行遍历
            boxes3d = gt_boxes3d[i].cpu()
            idx = torch.nonzero(nxyz[:, 0] == i).view(-1)
            new_xyz = nxyz[idx, 1:].cpu()

            boxes3d[:, 3:6] *= enlarge

            pts_in_flag, center_offset = pts_in_boxes3d(new_xyz, boxes3d)
            pts_label = pts_in_flag.max(0)[0].byte()

            # import mayavi.mlab as mlab
            # from mmdet.datasets.kitti_utils import draw_lidar, draw_gt_boxes3d
            # f = draw_lidar((new_xyz).numpy(), show=False)
            # pts = new_xyz[pts_label].numpy()
            # mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], color=(1, 1, 1), scale_factor=0.25, figure=f)
            # f = draw_gt_boxes3d(center_to_corner_box3d(boxes3d.numpy()), f, draw_text=False, show=True)

            pts_labels.append(pts_label)
            center_offsets.append(center_offset)

        center_offsets = torch.cat(center_offsets).cuda()
        pts_labels = torch.cat(pts_labels).cuda()

        return pts_labels, center_offsets

    def aux_loss(self, points, point_cls, point_reg, gt_bboxes):

        N = len(gt_bboxes)

        pts_labels, center_targets = self.build_aux_target(points, gt_bboxes)

        rpn_cls_target = pts_labels.float()
        pos = (pts_labels > 0).float()
        neg = (pts_labels == 0).float()

        pos_normalizer = pos.sum()
        pos_normalizer = torch.clamp(pos_normalizer, min=1.0)

        cls_weights = pos + neg
        cls_weights = cls_weights / pos_normalizer

        reg_weights = pos
        reg_weights = reg_weights / pos_normalizer

        aux_loss_cls = weighted_sigmoid_focal_loss(point_cls.view(-1), rpn_cls_target, weight=cls_weights, avg_factor=1.)
        aux_loss_cls /= N

        aux_loss_reg = weighted_smoothl1(point_reg, center_targets, beta=1 / 9., weight=reg_weights[..., None], avg_factor=1.)
        aux_loss_reg /= N

        return dict(
            aux_loss_cls = aux_loss_cls,
            aux_loss_reg = aux_loss_reg,
        )
```

