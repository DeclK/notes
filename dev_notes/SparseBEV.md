# SparseBEV

[arxiv](https://arxiv.org/abs/2308.09244) [github](https://github.com/MCG-NJU/SparseBEV) [zhihu](https://zhuanlan.zhihu.com/p/654821380)

## Concept

1. FPN

   其实之前整理过 FPN，在 YOLO 的笔记里面，现在又复习了一次。其实霹导画的图一步总结到位了，只不过在 SparseBEV 里面没有 extra conv，即：输入多少个 feature map 就输出多少个 feature map

2. input image with sweeps

   输入图像的 shape 为 $(B, T, N, C, H, W)$，其中 T 为帧数，在 SparseBEV 中为8，N 为相机数量。在 T 张图片中，只有1张为有标签的样本，剩余7张为该标记样本之前的7帧图片，如果不足7张则以最早的图片重复填充

3. SASA, scale adaptive self attention 用于控制感受野大小，从而获得 multi-scale 特征，这相当于起到了 FPN 的作用

   <img src="SparseBEV/image-20240103112937228.png" alt="image-20240103112937228" style="zoom:67%;" />

4. **Orinal DINO model**

5. Hungarian Loss

6. DN

7. SASA

8. Query Sampling

9. Spatial Temporal Mixing

## Question

1. In this work, we argue that the self attention can play the role of BEV encoder, since queries are defined in BEV space.

   应该有更好的 encoder 方式，还有什么可以考虑的？

2. 似乎使用了自己写的 deformable attention，是不是可以用两阶段的方案？

3. temporal alignment 是怎么做的？

   简单的堆叠 +  channel mix + token mix，肯定有更简单时空方案

4. 为什么要对 decoder 使用 shared params？

5. SparseBEV 没有使用 encoder 来进一步对特征进行加强，在 DETR 系列中都有一个 encoder 来提取 neck 输出的特征