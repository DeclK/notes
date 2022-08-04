# BEVDet

## 相机内外参

关于相机内外参 [zhihu](https://zhuanlan.zhihu.com/p/389653208)，这篇博客写得已经非常清晰了，只需要记住小孔成像的一个比值公式就好。不同的项目可能会有不同的矩阵表达形式，但大体逃不出博客中的逻辑。小孔成像（已补偿翻转）的公式为
$$
\frac{Z}{f}=\frac{X}{X^{\prime}}=\frac{Y}{Y^{\prime}}
$$
这样就可以表示像素化坐标 $u, v$ 为：
$$
\begin{array}{l}
u=\alpha X^{\prime}+c_{x} \\
v=\beta Y^{\prime}+c_{y}
\end{array}
$$
其中 $\alpha,\beta, c_x, c_y$ 代表着对小孔成像的图像的缩放和平移。把上面的式子带入得
$$
\begin{array}{l}
u=\alpha f \frac{X}{Z}+c_{x}=f_{x} \frac{X}{Z}+c_{x} \\
v=\beta f \frac{Y}{Z}+c_{y}=f_{y} \frac{Y}{Z}+c_{y}
\end{array}
$$
使用更简洁的矩阵公式表达为
$$
Z\left(\begin{array}{l}
u \\
v \\
1
\end{array}\right)=\left(\begin{array}{ccc}
f_{x} & 0 & c_{x} \\
0 & f_{y} & c_{y} \\
0 & 0 & 1
\end{array}\right)\left(\begin{array}{c}
X \\
Y \\
Z
\end{array}\right)=\mathbf{K P}
$$
在像素坐标里加入一维，可以方便之后做矩阵运算

相机的位置由旋转矩阵 R 和平移向量 t 来描述
$$
\mathbf{P}=\mathbf{R} \mathbf{P}_{\mathbf{w}}+\mathbf{t}
$$
其中 $P_w$ 就是世界坐标。这个关系是比较一般性的，表达的是两个坐标系之间的相互转换

## BEVDet View Transformer

关于 lift splat shoot 的讲解视频 [bilibili](bilibili.com/video/BV16T411g7Gc)，基本上 BEVDet 就是使用这篇论文作为特征提取，然后接入一个检测头进行训练即可

直接贴一个前向方程，根据代码进行注释笔记

```python
    def forward(self, input):
        x, rots, trans, intrins, post_rots, post_trans = input
        # post_rots & post_trans 记录的是数据增强的操作，之后会逆转数据增强的效果，
        # 获得 frustum 在点云坐标系下的位置
        # rots & trans 应该是记录了 cam 到 lidar 坐标之间的变换
        # 使用 intrins 将像素坐标转换到 cam 坐标
        
        B, N, C, H, W = x.shape	# N 代表 cam 个数
        x = x.view(B * N, C, H, W)
        
        x = self.depthnet(x)
        # depthnet 输出维度为 self.D + self.numC_trans
        # self.D 代表离散化深度，self.numC_trans 代表预测的特征维度
        
        depth = self.get_depth_dist(x[:, :self.D])	# 等价于 softmax
        
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        # 获得 frustum 在 lidar 坐标系下的位置 (B, N, D, H, W, 3)
        img_feat = x[:, self.D:(self.D + self.numC_Trans)]

        # Lift
        # 升维，做 outer product，即 broadcast
        volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        # 把所有 cam 的 frustum 都转移到 lidar 体素坐标系中，
        # 并且落在相同 voxel 的特征将做 sum 操作
        bev_feat = self.voxel_pooling(geom, volume)
        if self.image_view_supervision:
            return bev_feat, [x[:, :self.D].view(B,N,self.D,H,W), x[:, self.D:].view(B,N,self.numC_Trans,H,W)]
        return bev_feat
```

这里我把 ego 坐标系换成了 lidar 坐标系便于理解，关于什么是 ego 可以看这个 [issue](https://github.com/nutonomy/nuscenes-devkit/issues/487)

## 可提升的地方

1. frustum 转移到 lidar 坐标系过后是近处稠密远处稀疏的，可以更针对地进行采样
2. 可以加入一些中间监督来对 view transformer 进行训练
3. 对经过 view transformer 过后的点云再次进行 BEV 增广（这在 BEVDet 中已经提出）
4. 做 `voxel_pooling` 的时候可以使用更好的 pooling 策略，而不是简单的 sum
5. 一定要使用更大的感受野去处理（deformable convolution, big kernel, transformer）
6. 可以加入时序信息