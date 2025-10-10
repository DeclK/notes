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

有了 cuTensorMap (i.e. `tma_load`) 过后，可以利用 `cute::copy` 来进行 tma load，实现 gmem -> smem 的数据传输，命令如下

```c++
// tma_load is cuTensorMap  
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

tma store 的使用和 tma load 类似，其功能是将数据从 smem copy 到 gmem，但是同步和 tma load 有显著区别。

```c++
// cuTensorMap is tma_store
if (is_tma_thread) {
    // tma tensor
    auto gmem_tensor_coord = tma_store.get_tma_tensor(shape(gmem_tensor));
    // cta data slice
    auto tma_store_per_cta = tma_store.get_slice(cluster_id);
	
    copy(tma_store.with(tma_load_mbar, mcast_mask),
         tma_store_per_cta.partition_S(gmem_tensor_coord_cta),
         tma_store_per_cta.partition_D(smem_tensor));
	// commit
    tma_store_arrive();
  }
// wait latest 0 commit
tma_store_wait<0>();
```

可以看到 tma store 和 tma load 都需要 tma tensor & slice 操作，但是不需要 mbarrier 来进行同步。其使用的是 `tma_store_arrive` 和 `tma_store_wait` 来进行同步操作，这里类似 Ampere 中的 async copy 同步 `cp_async_fence & cp_async_wait`，对 copy 操作进行 commit 和 wait

## wgmma

[blog1](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) [blog2](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)

一个 warp group 是由 4 个连续的 warps 构成，i.e. 128 个连续的线程。而 wgmma 就是由一个 warp group 协作执行的 mma，其支持更大的矩阵分块计算。wgmma 有几个特点：

1. 矩阵形状：基本计算形状为 `m64nNk16`，其中 N 可以是 8 的倍数，范围从 8 到 256
2. 异步执行：遵循异步的 [consistency model](https://docs.nvidia.com/cuda/archive/12.3.2/parallel-thread-execution/index.html#program-order-async-operations)，wgmma 不遵循 cuda 代码当中的 program order，和 Ampere mma 不同的是，需要使用额外的 fence 操作来保证 mma 执行的顺序在 smem 完成写入之后。这种异步执行的操作都是在  [async proxy](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-proxy) 中完成的，tma operations 也是
3. 操作数存储：操作数矩阵 **B 必须存储在共享内存（SMEM）** 中。操作数矩阵 A 可以位于共享内存或寄存器内存（RMEM）中，而累加器矩阵 C 则始终保存在寄存器中
4. 数据类型支持：WGMMA 支持多种数据类型，包括 FP16、BF16、TF32、FP8（E4M3 和 E5M2 格式）以及整数格式（如 U8/S8），并在 FP32 或 FP16 中进行累加。wgmma 没有 4-bit 运算单元，即：不支持 fp4/int4 的矩阵运算

SM90 MMA atoms 在 cute 中都标记为 `SM90_MxNxK_XYZ_SS` or `SM90_MxNxK_XYZ_RS`

- `X` and `Y` 是操作数的数据类型
- `Z` 是累加器的数据类型
- `MxNxK` 是计算 mma 的 tile 大小，M 始终是 64，N 是 8~256 的任意 8 的倍数，K 是 32 bytes 对应的数据类型的个数，例如 fp16 mma 则 K 是 16

wgmma 的构建和 mma 的构建是类似的，都有 `AtomLayoutMNK` and `PermutationMNK` 

```c++
TiledMMA tiled_mma = make_tiled_mma(
                         SM90_64x64x16_F16F16F16_SS{},
                         Layout<Shape<_2,_1,_1>>{}
);
```

除此之外 wgmma 会对 smem 的排布有要求，cute 中有直接的接口可以生成符合要求的 smem layout `ss_smem_selector` + `tile_to_shape`，smem selector 传入参数为 major, datatype, tile size

```c++ 
  using SmemLayoutAtomA =
      decltype(ss_smem_selector<GmmaMajorA, ABtype,
                                decltype(cute::get<0>(CtaTile{})),	
                                decltype(cute::get<2>(CtaTile{}))>());
  using SmemLayoutA = decltype(tile_to_shape(
                                  SmemLayoutAtomA{},
                                  make_shape(shape<0>(CtaTile{}), shape<2>(CtaTile{}), Int<Stage>{}),
                                  Step<_1, _2, _3>{}));// LayoutLeft{}
```

对于 K major 来说，只有4种合法的 smem layout

```c++
Layout_K_INTER_Atom_Bits  = ComposedLayout<Swizzle<0,4,3>, smem_ptr_flag, Layout<Shape<_8, _128>,Stride< _128,_1>>>;
Layout_K_SW32_Atom_Bits   = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape<_8, _256>,Stride< _256,_1>>>;
Layout_K_SW64_Atom_Bits   = ComposedLayout<Swizzle<2,4,3>, smem_ptr_flag, Layout<Shape<_8, _512>,Stride< _512,_1>>>;
Layout_K_SW128_Atom_Bits  = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag, Layout<Shape<_8,_1024>,Stride<_1024,_1>>>;
```

也不是任意的 cta tile 都能找到合适的 smem layout，其必须要求 K 维度的大小必须是 multiple of 16/32/64/128 byte

## Warp Specialization

现在的 tensor core 计算能力远强于数据的运输能力，所以一切的优化都围绕着如何打满 tensor core 的算力。这个过程叫做 "feading the beast"。总体上有两种优化技巧

1. 有效的 threadblock scheduling，以提升 L2 cache hits。这一点在 scheduler 当中体现
2. 流水线并行，overlap copying with math computation

对于流水线并行在 Ampere 架构使用的是 multi-stage 流水线，我也称之为 wati one in-flight 算法。而到了 Hopper 架构，则大力推广了 warp specialization 算法，也即 producer-consumer 算法。一部分 warp 作为 producer，向 buffer 当中输入数据；异步分 warp 作为 consumer，计算 buffer 当中的数据。二者的本质都是想要利用 compute 来掩藏 copy 的时间，我认为 producer-consumer 模型更加简洁直观，但是需要更多的同步操作以避免 racing

warp specialization 的表现形式很简单，在 CUDA 当中就是一个 if-else 分支

```c++
if (warp_group_idx == WarpGroupCnt - 1) {
  // producer
  // alloc 40 register for tma load
  cutlass::arch::warpgroup_reg_dealloc<40>();
  // elect 1 thread issue tma load
  if (warp_idx_in_group == 0 && elect_one_sync()) {
    producer(param, shared_storage, block_rank_in_cluster);
  }
} else {
  // consumer
  // alloc 232 register for mma compute
  cutlass::arch::warpgroup_reg_alloc<232>();
  ws_consumer(param, shared_storage);
}
```

这里还体现了 tma 节省寄存器的优点，给 producer 分配了较少的 register，而给 consumer 分配更多 register

之前的疑问：在之前的 SIMT 编程思想下，写 if-else 分支是效率比较低的行为。为什么在 warp specialization 就可以被允许了？

> From Kimi
>
> 经典 SIMT 以 32 线程的 warp 为最小调度单位，同一个 warp 里的线程只要条件不同就会顺序执行 if 和 else 两段指令，造成浪费。
> Hopper 的 warp specialization 把粒度拉大到「整个 warp-group」（通常是 128 线程甚至 4-warp-group 的 512 线程）。也就是说，只要一个 warp-group 里的所有线程都走同一条路径，就不会出现传统意义上的 divergence。

之前的疑问：对于 producer 来说，只有一个 thread 在进行操作，那其他的 thread 是不是就没有工作了？在此情况下还会给他们一起分配寄存器之类的资源吗（根据 SIMT 编程原则）？

> From Kimi
>
> 是的，**整个 warp group 都会进入 producer 分支**，但**真正干活的只有 warp group 里被 `elect_one_sync()` 选出来的那一个 thread**，其余 127 个 thread 在这条代码路径上就是“空转”

## mbarrier & pipeline

在 GEMM 计算中最核心的需求就是打满算力，而打满算力的核心就是高效的流水线，高效流水线的核心则是准确的同步机制。我将在这一小节里讨论如何利用 mbarrier 建立 producer-consumer 流水线模型的同步机制

mbarrier 将分为两类

- full barrier

  维护 shared memory 是否完成写入的状态。如果未完成写入，则对应 shared memory consumer 无法运行

- empty barrier

  维护 shared memory 是否完成计算的状态。如果未完成计算，则对应 shared memory producer 无法运行

首先我将建立一个清晰的 producer-consumer 模型，然后我再来介绍如何使用同步机制保证流水线模型的正确运行

对于有 N 个 stage 的流水线来说，就有 N 个 full barrier & empty barrier pair

同步的本质是什么？我个人认为核心的目标是避免 racing 产生，即当多个操作访问相同资源时，必须利用同步机制来确定执行顺序

sync for shared memory 都是利用 mbarrier & pipeline 完成。

能否用相同的 pipeline 思想，在 sm80 中实现？似乎不需要，可以直接使用 wait one in-flight 算法

## fence & visibility

在之前的很多小节里都触及了 fence sync，这里再统一总结一下

- `warpgroup_fence_operand(acc)`

  [[QST] What can go wrong without cute::warpgroup_fence_operand(accum) in GEMM](https://github.com/NVIDIA/cutlass/discussions/1375)

  我的理解：在 wgmma launch 过后，会先进行计算（此时称为 in-flight 状态，指令还未完成）。在这个期间，编译器可能会改变其他操作的执行顺序，让其他操作在 wgmma 的计算过程中，使用这些 acc。为了保证这些 acc 不被其他操作所占用，必须使用 `warpgroup_fence_operand(acc)` 来保护这些寄存器，一直等warpgroup_fence_operand(acc)待，直到 wgmma 计算完成过后写入到其中

  ```
  warpgroup_fence_operand(acc);
  ...
  wgmma(..., acc);
  ...
  warpgroup_fence_operand(acc);
  ```

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

2. AwesomeCute 和 DeepGemm 当中的矩阵乘法实现是不一样的，我认为 DeepGemm  的实现跟简洁，少了一些 in-flight 操作，还没测试过性能

3. 如何对 wait one in-flight pipeline 进行合理抽象/模块化，通过定义 pipeline 的各个模块以形成高效的 pipeline 代码

4. tensor core 与 cta 之间的关系是什么？一个 cta 可以调动多个 tensor core 吗？

   > From Kimi
   >
   > **每个 warp 一次只能使用一个 Tensor Core**，但一个 CTA 可以包含多个 warp，因此**一个 CTA 可以并发地使用多个 Tensor Core**（取决于其 warp 数量与调度情况）。

   A100 有 108 个 SM processor，每一个 SM 有 4 个 tensor core。而 H100 有 132 个 SM processor，每一个 SM 同样也有 4 个 tensor core，所以共有 528 个 tensor core

5. pingpong 会使用 scheduler 同时计算两个 tile 的 gemm，但是 producer 只有一个，shared memory 是如何管理的？