# CUDA Programming 10.1

由于 CUDA Programming 10 的笔记太杂乱了，创建一个 10.1 版本，对其中的内容进行重新梳理。核心目标是梳理 Hopper GPU 编程

reference code: [DeepGemm](https://github.com/deepseek-ai/DeepGEMM/tree/main) [CalebDu/Awesome-Cute](https://github.com/CalebDu/Awesome-Cute/tree/main)

## TMA

reference blog:

- [Deep Dive on the Hopper TMA Unit for FP8 GEMMs – PyTorch](https://pytorch.org/blog/hopper-tma-unit/) [tma tutorial](https://research.colfax-intl.com/tutorial-hopper-tma/)

- [CUTLASS Tutorial: Mastering the NVIDIA® Tensor Memory Accelerator (TMA)](https://research.colfax-intl.com/tutorial-hopper-tma/)

TMA 其实是 Hopper 架构中新引入的硬件单元，其功能是在 global memory 和 shared memory 中传输数据。有了 TMA 过后，有几个优势：

1. 能够节省数据传输中的 register（这些 register 通常用于寻址计算），所以 register 可以更多分配给 gemm，以计算更大的矩阵。同时还能再传输数据过程中直接完成 swizzle 计算和简单的 reduction 工作

   <img src="CUDA Programming 10/fg2-3.png" alt="A100-style data movement vs H100 with TMA.  TMA hardware eliminates the need for a large amount of threads and registers participating in bulk data transfers." style="zoom:50%;" />

   也许随着 GPU 的发展，之后的 register 只会参与 CUDA core 的计算，对于 tensor core 的计算数据将不会使用 register

2. 能够以单线程发起传输，简化了线程对数据的划分问题，同时也节省了线程资源、register 资源（`pipeline_states` 中用于同步的 phase, stage idx）

### tma descriptor

也叫做 `CUtensorMap`。如前面所述，tma 的功能是 global mem 和 shared mem 之间的数据传输。从 global mem -> shared mem 的传输就是 tma load；反之就是 tma store。r不管是 tma load or tma store，都是由 tma descriptor 发起 

在 cute 中使用 `make_tma_copy` 的方式来构建 tma descriptor（实际上是一个 tiled copy 对象），其中有5个重要参数

```c++
make_tma_copy(CopyOp                  const& copy_op,
              Tensor<GEngine,GLayout> const& gtensor,
              SLayout                 const& slayout,
              CTA_Tiler               const& cta_tiler,
              Cluster_Size            const& cluster_size)
```

1. `copy_op`，一共有三种选择：store & load & multicast load，以定义该 copy 功能：是从 gmem -> smem 还是从 smem -> gmem；如果是 gmem -> smem 是否使用 multicast（cluster size > 1）

2. `gtensor`，tensor 在 gmem 当中的表示。

3. `slayout`，所定义的 smem layout 表示。通常配合 `cta_tiler` 以让 `slayout` 就是 tma load & store 的 tiler 基本单位，只能以 slayout 的整数倍数据进行 copy

4. `cta_tiler`，每一次 tma copy 的基本单位。通常就是 size of `slayout` 

5. `cluster_size`，一个 cute `Int`，代表 size of cluster layout。如果 `cluster_size > 1` 的话，`copy_op` 必须是 multicast load，同时在 load 时，会把数据划分成 `cluster_size` 份，每一个 cluster 会 load 各自的数据，最后通过 multicast 的方式共享各自的数据，让每一个 cta 都拥有完整的数据。这个参数似乎只针对 load，对于 store 没有

   下面简单介绍一下 cluster 的概念

   > Cluster in Hopper From DeepSeek
   >
   > 线程块集群允许你将多个**线程块（Thread Blocks）** 组织成一个更大的协作单元。这些线程块可以分布在多个 SM 上，但能保证被**并发调度**，并且支持高效的跨 SM 协作与数据共享。在传统的**线程 (Thread) → 线程块 (Thread Block) → 网格 (Grid)** 结构基础上，Hopper 新增了 **线程块集群 (Thread Block Cluster)** 这一层级，介于线程块和网格之间。运行时设置通过 CUDA 内核启动 API `cudaLaunchKernelEx`来配置集群维度。

   如果 cluster dim 设置为 `(x=2, y=1, z=1)`，对应着在 x dim 上，每两个 cta 构成一个 cluster。通过 `cute::block_rank_in_cluster();` 可以直接获得 cta 在当前 cluster 当中的 cluster id

### tma load

有了 cuTensorMap 过后，可以利用 `cute::copy` 来进行 tma load，命令如下

```c++
  if (is_tma_thread) {
    // mbar
    __shared__ uint64_t tma_load_mbar; // init of mbar is omitted here
    // tma tensor
    auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));
    // cta data slice
    auto tma_load_per_cta = tma_load.get_slice(cluster_id);
	
    copy(tma_load.with(tma_load_mbar, mcast_mask),
         tma_load_per_cta.partition_S(gmem_tensor_coord_cta),
         tma_load_per_cta.partition_D(smem_tensor));
  }
```

可以看到（对比 Ampere copy）有四个显著不一样的地方：

1. mbarrier

   tma load 需要传入一个 mbarrier (memory barrier)，mbarrier 是 tma 同步的最核心的数据，其被定义在 shared memory 当中。通过 mbarrier 可以来判断：copy 是否能够开始 (empty barrier)，以及 copy 是否已经结束 (full barrier)。在上述代码中的 `tma_load_mbar` 充当的就是 full barrier 的角色

2. tma tensor

   tma 进行 copy 时使用的是坐标（coordinate）而不是偏移（offset），所以需要有一个专门的 tensor 来表示，通过 `get_tma_tensor(shape(gmem_tensor))` 即可获得。tma tensor 与普通 tensor 最大的区别在于其 stride 是一个 vector 而不是一个 scaler

   ```python
   Tensor(
   	shape=(M, N),
       stride=(1, M)
   )
   TMA_Tensor(
       shape=(M, N)，
       stride=(1@0, 1@1)
   )
   1@0 = (1, 0)	# a vector
   1@1 = (0, 1)
   
   # layout function
   Tensor(1, 2) = 1 * 1 + 2 * M
   
   # TMA layout function
   Tensor(1, 2) = 1 * (1, 0) + 2 * (0, 1) = (1, 2)
   ```

   其中 `x@y` 意味着数字 x，在向量中的第 y 个位置，向量的维度等于 shape 的维度。此时 layout function 的输出变成了一个 vector，也就是我们需要的坐标。这也是 `make_identity_layout` 的原理

3. get slice

   在 Ampere 架构时，我们传入的是 thread id，从而通过 `partition_S/D` 获得每个线程应有的数据。而这里传入的是 cluster id，也是通过 `partition_S/D` 获得每个 cluster 应有的数据。如果 cluster size = 1 的话，说明没有 multicast，每一个 cta 所获得的都是完整的数据；反之 cluster size > 1 的话，一个 clustere 内的各个 cta 将获得部分数据

4. mcast mask

   这也是为了配合 cluster multicast 所设计的，我们可以选择 cluster 内的哪些 cta 参与到 multicast 的过程当中，只有选中的 cta 才会将数据进行共享。该 mask 是一个 bit mask，总共为 16-bit，值为1则为参与，反之不参与。通常所有的 cluster 都会共享自己的数据，所以 mcast mask 的设置也相对固定，直接设置为 `(1 << size(cluster_layout)) - 1`

### tma store



## Cluster

sync

## wgmma

## Warp Specialization

register 动态分配（producer 少，consumer 多）

## mbarrier & pipeline

sync for tma 都是利用 mbarrier & pipeline 完成

能否用相同的 pipeline 思想，在 sm80 中实现？似乎不需要，可以直接使用 wait one in-flight 算法

除此之外还有一个 named barrier 不知道有什么作用？

## fence & visibility

在之前的很多小节里都触及了 fence sync，这里再统一总结一下

## Scheduler

persistant warp

## Coorperative & PingPong

## GEMM 实践

如何构建 producer & consumer 完成高效的 gemm

multi-stage in epilogue, 这个 multi-stage 形式也在 Ampere Gemm 当中出现过，我想称之为 wait one in-flight pipeline multi-stage

## Question

1. 即使是 cute 也有很重的包装/抽象和 layout 计算，但实际上很多包装和计算都是没用的。而 DeepGemm 的包装非常非常少，基本就是 PTX 包了一层，然后使用了一些 cutlass 中的小组件。但是 DeepGemm 的优点也是缺点：极少的包装导致算法的可读性 maybe 差了一点

   可能这也是现在出现各个 DSL 的原因：构建自己熟悉的抽象，然后构建高效的算法。而如果要解决这两个问题就必须要学会阅读 PTX，构建自己的功能抽象；然后熟悉各个流水线算法的流程，以清晰的代码、文档逻辑进行展现

   阅读 PTX 可能是更难的，我现在能做的是熟悉 cute 当中的 PTX 抽象，理解其功能以方便构建高效算法，好消息是 cute 可能在可读性和可用性上也在努力（python DSL），这也是 Tri Dao 最近在 GPU mode 上推荐的，[link](https://zhuanlan.zhihu.com/p/1951029558313714196)。虽然 DeepGemm 的封装很少，但是要理解其用法也没那么容易，因为 CUDA 的 doc 并不好看。总之这个领域对新手从来都不是友好的