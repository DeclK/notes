# TPVFormer

这篇笔记主要记录其中的 Attention 机制，包括 CVHA & ICA，理解了这里就能轻松理解 BEVFormer 的编码。本次整理采用下面的层级形式来总结，这样的形式还可以方便生成思维导图，而且很方便在整理过程中找到想查询的变量在哪个模块

## TPVFormerHead

- **首先对 tpv queries 进行初始化**

  - HW, ZH, WZ,使用的 embedding 的方式设置 query  `(B, HW, C)`

  - 将 FPN 获得的 feature list 展平并进行连接，形成 value  ` (B*N_cam, flatten_length, C)`。实际上不要把这里看作展开最好，仍然理解为二维图像是最好，反正 deformable attention 又会还原回去

  - 给 value 加入 cam & scale embeddings

- TPVFormerEncoder

  - `get_ref_points`

    - 3D points，生成 `(D, HW, 3)` 的张量（HW平面）代表三维栅格坐标，其中 D 为 Z 方向的采样点数量。因为 TPVFormer 有三个平面，所以这样的坐标也要生成3个：HW, ZH, WZ。用于 Image Cross Attention

    - 2D points，生成 `(B, HW, 1, 2)` 的张量，即二维空间的归一化栅格坐标，同样也是要生成3个。用于 Cross View Attention

  - **`point_sampling`** 

    - 将 3D points 通过 lidar2img 矩阵 `(B, N_cam, 4, 4)`，投影到 img 坐标系中，获得这些点的像素坐标 `(N_cam, B, HW, D, 2)`，也称为 `reference_points_cam`

    - 计算 `tpv_mask` 记录了哪些点在 image 中，哪些点在 image 外 `(N_cam, B, HW, D)`

  - TPVFormerLayer

    - 一个 Layer 就是一个 transformer block，**输入就是 QKV**。唯一特殊的点就是要处理3个平面，所以 query 是一个 list `[(bs, num_query, embed_dims) * 3]`

    - TPV-Cross View Attentnion
      - CVHA 是用于三个视图特征交互的模块。本质上就是将 3 个 query 都展开，变为一个长 query，就代表了 3 个 level，然后做一个 deformable attention 进行信息交换

    - Image Cross Attention

      - ICA是用于获得三个视图特征的模块，与 BEVFormer 中的 attention 是一致的，只不过需要完成的是 3 个平面的编码。

      - **Deformable Attention is ALL about reference points**，记住这句话！可变注意力都是以 reference points 为核心进行展开，query 仅仅是计算 sampling offsets & attention weights 的工具人！

      - 为了减少不必要的注意力，reference points cam 中有许多点是在相机之外的，我们需要去除它们。使用一个新的张量 `reference_points_rebatch & queries_rebatch` 来存储

        query `[(B, HW, C), (B, ZH, C), (B, WZ, C)]`

        value `(B*N_cam, flatten_length, C)`

      - for each tpv plane

        - `query (HW, C)`
        - `query_rebatch (B*N_cam, max_len, C)`，这个 `max_len` 是所有 cam 中的最多参考点。虽然 rebatch 后的数量看似比 query 要多，但实际上很多都是重复的 query。`query_rebatch` 是利用 `tpv_mask` 计算得到的，过滤了在 D 维度上没有一个点落在图像内的像素。
        - `reference_rebatch (B*N_cam, max_len, D, 2)`，可以看到这里没有 `num_levels` 维度，实际上在原始的 deformable attention 中该维度也仅仅是重复而已，所以不必在意形式
        - 之后就可以利用 `query_rebatch` 计算 `sampling_offsets (B*N_cam, max_len, D, num_levels, num_heads, 2)` 以及 `attention (B*N_cam, max_len, D, num_levels, num_heads)`，经过加权过后就能够获得新的 `query_rebatch`
        - 最后需要将 `query_rebatch` 还原回原始的 `query`
        
        虽然上述过程是写在一个循环当中的，但实际上三个平面是可以并行完成的，但这里为了方便理解仍然使用循环表述。并且我认为需要对 attention weights 也进行 mask，以获得更合理的结果
    
      TPVFormerLayer 总结：就是根据空间内的 3d reference points 进行采样加权，获得 query

## Aggregator

其实非常简单，简单理解就是 MLP😎。要获得点的特征，只需要获得3个平面对应的 pixel，把这3个特征加起来就行了。然后就愉快输入 MLP 进行分类
