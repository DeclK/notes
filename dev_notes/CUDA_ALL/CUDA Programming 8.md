# CUDA Programming 8

之前的 CUDA Programming 7 已经很长了，本来想就在 7 就终结。但是没办法，需要弄清楚的事情仍然很多！这篇笔记要整理的是 cutlass cute 当中的基础概念

感觉现在即使是 DeepSeek 也对 cutlass 没有太多的头绪，我只能够不断地从 zhihu 博客中学习，配合以 DeepSeek 进行一些简单的推理来构建清晰的 cutlass 如何抽象 GPU 的模型

- ldmatrix

  how a thread carries data, this needs allocate registers for each thread

  为什么使用 LDS 需要使用 4 条指令加载，而使用 ldmatrix 只需要一条指令就能够加载完成从 shared memory -> register 的过程

  通过各个线程提供的寄存器可以完成warp level的矩阵表示和存储，利用Tensor Core则可以完成高效的存储在寄存器上的矩阵计算。就数据从共享内存到寄存器的加载方面而言，可以通过SIMT意义下的LDS（load shared）来完成，但是由于数据是分布在不同的线程的寄存器上连续性方面不友好。

  如何理解上面这个句子中的不连续性？

  [tensorcore中ldmatrix指令的优势是什么？](https://www.zhihu.com/question/600927104/answer/3029266372)

  每一个 thread 可以最多操作 128 bit 的 register/shared memory 数据吗？如何让这个说法更加准确

- mma

  为什么 mma 的 register 数据分布和 shared memory 数据分布不一样，是 scatter 的，又如何保证计算的正确性？连续的数据被打散了

- tiledmma

  print latex to see data and thread

  在扩展时是否有限制，显然不能无限制地进行扩展，Threads 数量和 Vectorized Register 数量都是有限制的

  如何去推理整个 permute layout 的数据分布？ 

  为什么在 reed 的 gemm-simple 中 permutemnk 是 (1, 2, 1) 而不是具体的 shape，其中是否有什么隐藏等价变换？

- vectorized register 是可以变化的 [nv-doc](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type)

  对于 16x8x16 的 mma 来说，每一个 thread 持有一个 vectorized register，该 vectorized register 包含了 4 个 `.f16x2` 的 registers，也就是说每一个 register 涵盖了 2 个 fp16/bf16 的元素。再换句话说，每一个 thread 将会持有 8 个 fp16 元素，按照 `[a0, a1, ..., a7]` 排列，在 `print_latex` 中展示为 `[V0, V1, ..., V7]`

  为什么需要 vectorized register 这样的形式？

## TiledCopy

- 为什么在修改 `static constexpr int kMmaEURepeatN = 1;` 过后 multi-stage gemm 结果出现了变化？

- `cp_async_fence & cp_async_wait`

- `make_tiled_copy`

  global -> shared memory & shared memory -> register 使用的是不同的 atom
  
  next: understand shared memory to register copying

- `SM75_U32x4_LDSM_N`

  每个线程一次加载 **4个32位元素**（如4个`float32`或8个`half16`），still 128 bit for one thread。但这里的 32x4 并不是指矩阵形状

  为什么需要 mma？因为需要把 128 bit 的数据分配到不同的线程当中（see the mma latex layout）

  <img src="CUDA Programming 8/image-20250320234644102-1742921406550-10.png" alt="image-20250320234644102" style="zoom:50%;" />

## CUTE Pipeline

< img src="CUDA Programming 8/v2-f9c13c984a5d8364e2d67e592cf7ddbf_1440w.jpg" alt="img" style="zoom:67%;" />

Reed 在知乎中对于流水线的描述非常多，但是排版不怎么行。我需要整理一下，不然没办法好好理解

其中比较明显的是各个长方形，代表着数据的异步加载：

1. 浅绿色长方形代表：全局内存到共享内存的数据搬运 $G^i \rarr S^i$ ，上标 $i$ 代表的是第 $i$ 个 Tile 的数据（我称之为大 k 循环）。也就是下图里的 $s_A$ 

   <img src="CUDA Programming 7/image-20241203160014235.png" alt="image-20241203160014235" style="zoom: 67%;" />

2. 浅橙色长方形代表：共享内存到寄存器的数据搬运 $S_j \rarr R_j$，下标 $j$ 代表的是第 $j$ 个小 k 循环（Tile 内循环）

   <img src="CUDA Programming 8/image-20250321150524731.png" alt="image-20250321150524731" style="zoom: 80%;" />

   由于我在 1 中画的图的 k = 8，基本上没有循环的必要。但在日常的使用当中，k 的取值一般会比这个要大。例如 k = 32，此时一个 TiledMMA 一次只够处理 k = 16，所以需要计算两次才能够完成矩阵乘法运算

3. 深绿色的长方形代表：TiledMMA 利用寄存器上的数据进行矩阵计算
4. 黑色实线之间代表：完成一个 Tile 的矩阵运算（完整的小 k 循环）。并且黑色实线上方使用了曲线虚线进行了连接，代表完成了一个 Tile 计算之后继续计算下一个 Tile
5. 黑色虚线代表：进行 `cp_async_wait`，等待 shared memory 搬运完毕

整个流水线的关键步骤：

1. 首先将 `Stage - 1` 个全局内存到共享内存的加载任务异步地发布出去（发布过后不进行等待，直接执行之后的任务）

2. 等待 $S^0$ 的数据完成加载

3. 在进入小 k 循环之前，首先从 $S^0$ 中取出第一个小 k 循环所需要的数据，将其发送到寄存器上 $S_0\rarr R_0$

4. 此时正式进入到小 k 循环，可以分为 4 个要点：

   1. 发射异步读取新 Tile 的任务请求，即图中的 $G^3 \rarr S^3$
   2. 从共享内存中异步读取下一个小 k 循环所需要的数据 $S_j\rarr R_j$
   3. 执行第一个小 k 循环矩阵运算
   4. 重复步骤 2~3 直到当前小 k 循环完成

   需要注意的是，在做最后一个小 k 循环时，我们需要读取下一个 Tile 中的第一个小 k 循环数据，该操作需要使用 `cp_async_wait ` 来保证下一 Tile 的数据已经完全加载到 shared memory 当中。这也是图中的虚线所表达的含义

## Epilogue

经过一系列大小 k 循环过后，所有的结果都被存储到了累加器 `tCrD` 当中。我们需要将累加器中的结果，全部都运输到 global memory 当中存储起来。但直接完成这件事并不是最优选项，因为会造成不连续的数据写入，这样会导致存储时需要更多的内存事务，而不能使用向量化存储指令（STG.128）

<img src="CUDA Programming 8/v2-ddece7971d1161bbf7c7fa8022859993_1440w.jpg" alt="v2-ddece7971d1161bbf7c7fa8022859993_1440w" style="zoom: 50%;" />

针对这个问题，cute中（实质为cutlass中），专门提供了Epilogue来通过共享内存作为中间媒介。先将寄存器数据存储到共享内存，然后再从共享内存中以更连续、更高位宽的形式存储到全局内存中去。

为什么需要用 `kSmemLayoutCBatch_` 来控制 epilogue shared memory 大小？

为什么在进行 `make_tiled_copy_C` 的时候 `tiled_mma` 似乎使用了 permutationMNK 的作用，但是在 `thr_mma.partition_fragment_A` 进行 partition 的时候，permutationMNK 感觉没有什么作用？这应该还是涉及到 layout algebra

为什么要使用 pipe 来进行 copy？

## Layout Algebra

From layouts definition to Logical division

admissible for complementation 要求

1. for all $1 \le i \le \alpha$, the product $N_{i-1}d_{i-1}$ divides $d_i$
2. the product $N_\alpha d_\alpha$ divides $M$

这两条基本上可以合为一条来看待，这似乎保障了整个 mapping 的可分性和递增性质

1. 递增性质比较好理解，这种情况只发生于 $$N_{i-1}d_{i-1} \gt d_i$$，此时随着 $x$ 的增加，并不保证 $f_L(x)$ 的增加
2. divides 保证了可分性？

Definition 2.11 & 2.13 保证了 composition 的合理性

> From Grok
>
> **What’s Layout Algebra About?**
>
> In layout algebra, we’re dealing with how data is organized in memory using multi-dimensional grids. A layout (like  $A$ or $B$) has a **shape** (the dimensions of the grid, e.g., $ (2, 3) $) and a **stride** (how many memory slots you jump to move along each dimension). When we compose two layouts, $A \circ B $, we’re basically saying, “Use $ B $ to pick spots in memory, then apply $ A $’s layout to those spots.” For this to work cleanly, the pieces have to fit together like a puzzle.
>
> - $ A $ has a shape $ \mathbf{S} = (M_0, M_1, \ldots, M_\alpha) $ and strides defining how it maps indices to memory.
> - $ B $ has a shape $ N $ (could be a single number or a tuple) and a stride $ r $, which says how $ B $ steps through memory.
>
> The composition $ A \circ B $ should itself be a valid layout with a clear shape and stride. That’s where these conditions come in.

这一段话理解了，基本上就茅塞顿开了。将 composition 分成两步来看：

1. 不考虑 $N$，只考虑 $r$。每一个 element 都是以间隔 $r$ 排开，当这个间隔 $r$ 要排开在 shape A 当中时，势必当 A 的 shape 与 $r$ 匹配比较好（align）。那什么样的情况会比较匹配呢？当各个 element 能够在 A 的 shape 中“整齐”地排布时，是我们比较喜欢的。所谓“整齐”用更数学的说法就是：shape A 能够被 $r$ 分割（divisible）。但这仍然不够严谨，所以有了 notes 当中的 definition 2.11。其保证了 shape A 能够将 $r$ 间隔元素很好地进行排列

   Example:

   ```c++
   Layout A  = (2, 3):(xx, xx)
   Layout B1 = (xx,):(2,)
   Layout B2 = (xx,):(3,)
   ```

   此时我对不关注的对象用 `xx` 来表示。

   ```python
   # A's natural layout
   0 2 4
   1 3 5
   
   # elements 0,2,4,... using B1's stride
   x x x
   1 3 5
   
   # elements 0,3,... using B2's stride
   x 2 4 
   1 x 6
   ```

   显然 `B2` 的 stride 和 shape A 并不匹配，看起来杂乱

2. 考虑 $N$。现在要考虑具体的 $N$ 所带来的影响了。因为 shape A 被 stride r 分割过后还剩了一部分子空间。如果具体的 $N$ 个 elements 不能够整齐地在 shape A 中排布也不是我们希望看到的。所以我们要求剩余的子空间能够被 N 弱分割（weakly divisible）

   通常来说对于 dim = 2 的 A 来说，这是很好满足的，因为 $i=\alpha$ 很容易被满足。不过我们也可以强行构造出不满足弱分割的情况

   Example:

   ```python
   Layout A  = (2, 3):(xx, xx)
   Layout B1 = (2):(1)
   Layout B2 = (3):(2)
   Layout B3 = (5):(2)
   Layout B4 = (5):(1)
   ```

   仍然以上面的可视化方式

   ```python
   # A's natural layout
   0 2 4
   1 3 5
   
   # elements 0,1 using B1
   x 2 4
   x 3 5
   
   # elements 0,2,4 using B2
   x x x
   1 3 5
   
   # elements 0,2,4,6,8 using B3
   x x x x x
   1 3 5
   
   # elements 0,1,2,3,4 using B4
   x x x
   x x 5
   ```

   即使 B3 超出了 A shape 的范围，根据定义我们仍然认为其是有效的弱分割，可以想象为其是 A shape 的扩展（extension）。对于 B4 而言，$N=5$ 是无法被 shape $(2, 3)$ 进行弱分割的，所以会看到不整齐的排列

上方我们一直都没有考虑到 A 的 stride，都是仅要求了 A 的 shape

## Question

1. 为什么在修改 `static constexpr int kMmaEURepeatN = 1;` 过后 multi-stage gemm 结果出现了变化？改变 `kMmaPN = 4 * kMmaEURepeatN` 也会改变结果

   为什么还要在 N 这个维度的 PermutationMNK 做扩展？在我看来这并没有什么意义，所带来的收益是什么？

2. 在使用 `cp_async_wait` 之后是否一定要添加 `__syncthreads`？

   > From DeepSeek:
   >
   > **是的，通常需要添加 `__syncthreads`**
   >
   > `cp_async_wait<>` 仅确保当前线程发起的异步拷贝操作（如 `cp.async`）完成，它不保证其他线程的异步操作状态。`__syncthreads()` 是线程块级别的同步，确保所有线程对共享内存的写入对其他线程可见

3. 既然 copy 都是异步的，那在进行 mma 的时候是如何确认数据已经搬运完成的呢？ 

   Reed 在评论区有回复

   > 是的，是异步的。gpu的指令发射和执行是异步的。ldmatrix指令发射后就可以发射后面的mma指令了，只需要等待ldmatrix发射结束，并不需要等待ldmatrix执行结束。这样ldmatrix和mma就同时工作起来了。至于依赖是通过scoreboard来决策的。ldmatrix和mma能同时执行的前提是他们在寄存器层面没有依赖，如果有依赖，scoreboard会保证ldmatrix发射+执行结束才执行mma发射。

   意思就是当 mma 一定会确保所使用的数据是通过 ldmatrix 执行结束过后的数据

4. `tCrA & tCrA_view` 似乎是共用的数据

   > From DeepSeek
   >
   > **`tCrA_view` 是 `tCrA` 的视图**，两者共享内存，`retile_D` 仅调整访问方式
0 / 3