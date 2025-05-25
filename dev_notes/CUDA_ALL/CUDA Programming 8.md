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

  `cp_async_fence`

  - 这是一个内存屏障（fence）操作，用于标记当前所有已提交的异步拷贝（`cp.async`）任务的完成点。
  - 它的作用是确保在该 `fence` 之前的所有 `cp.async` 操作（即从全局内存到共享内存的异步拷贝）被视为一个批次，后续的 `cp.async_wait` 可以对这些批次进行同步。
  - 它并不阻塞线程，只是标记一个任务提交的边界。

  `cp_async_wait`

  - 这是一个同步操作，用于等待之前提交的异步拷贝任务完成。
  - 参数 `N` 表示“等待除了最新的 `N` 个批次之外的所有批次完成”。例如：
    - `cp_async_wait<0>`：等待所有之前提交的异步拷贝完成。
    - `cp_async_wait<1>`：允许最多 1 个批次的异步拷贝未完成（即等待除最新提交的 1 个批次外的其他所有批次完成）。
  - 通常用于实现流水线的同步，确保数据在计算之前已经加载到共享内存。

- `make_tiled_copy`

  global -> shared memory & shared memory -> register 使用的是不同的 atom

  next: understand shared memory to register copying

- `SM75_U32x4_LDSM_N`

  每个线程一次加载 **4个32位元素**（如4个`float32`或8个`half16`），still 128 bit for one thread。但这里的 32x4 并不是指矩阵形状

  为什么需要 mma？因为需要把 128 bit 的数据分配到不同的线程当中（see the mma latex layout）

  <img src="CUDA Programming 8/image-20250320234644102-1742921406550-10.png" alt="image-20250320234644102" style="zoom:50%;" />

## CUTE Pipeline

<img src="CUDA Programming 8/v2-f9c13c984a5d8364e2d67e592cf7ddbf_1440w-1744706945438-9.jpg" alt="v2-f9c13c984a5d8364e2d67e592cf7ddbf_1440w.jpg (1191×336)" style="zoom:80%;" />

Reed 在知乎中对于流水线的描述非常多，但是排版不怎么行。我需要整理一下，不然没办法好好理解

其中比较明显的是各个长方形，代表着数据的异步加载：

1. 浅绿色长方形代表：全局内存到共享内存的数据搬运 $G^i \rarr S^i$ ，上标 $i$ 代表的是第 $i$ 个 Tile 的数据（我称之为大 k 循环）。也就是下图里的 $s_A$ 

   <img src="CUDA Programming 7/image-20241203160014235.png" alt="image-20241203160014235" style="zoom: 67%;" />

2. 浅橙色长方形代表：共享内存到寄存器的数据搬运 $S_j \rarr R_j$，下标 $j$ 代表的是第 $j$ 个小 k 循环（Tile 内循环）

   <img src="CUDA Programming 8/image-20250321150524731.png" alt="image-20250321150524731" style="zoom: 80%;" />

   由于我在 1 中画的图的 k = 8，基本上没有循环的必要。但在日常的使用当中，k 的取值一般会比这个要大。例如 k = 32，此时如果一个 TiledMMA 一次只够处理 k = 16，则需要计算两次才能够完成矩阵乘法运算

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
   
   #✅ elements 0,1 using B1
   x 2 4
   x 3 5
   
   #✅ elements 0,2,4 using B2
   x x x
   1 3 5
   
   #✅ elements 0,2,4,6,8 using B3
   x x x x x
   1 3 5
   
   #❌ elements 0,1,2,3,4 using B4
   x x x
   x x 5
   ```

   即使 B3 超出了 A shape 的范围，根据定义我们仍然认为其是有效的弱分割，可以想象为其是 A shape 的扩展（extension）。对于 B4 而言，$N=5$ 是无法被 shape $(2, 3)$ 进行弱分割的，所以会看到不整齐的排列

上方我们一直都没有考虑到 A 的 stride，都是仅要求了 A 的 shape

- 我需要使用 python 来表达出各种 layout algebra 的计算，来帮助自己进行理解以及实战演练。彻底理解 complement & composition & divide & product 的清晰作用。我看 swizzle 也就是用了一个 composition 就完成了 layout algebra 运算，所以这对理解 swizzle 至关重要

## MMA Atom

> FROM [CuTe-Tiled-MMA](https://leimao.github.io/blog/CuTe-Tiled-MMA/)
>
> The shared memory configuration has to be compatible with the tiled MMA we configured later.

[TN & NT & TT & NN](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0x_gemm_tutorial.md#aside-m-major-n-major-k-major)

这里和 reed 的描述不符合，我认为上述稳当描述更为正确。其中 TN 为最为主流的 mma atom 方式，在 sm80 以及之后的 sm 版本中只看见了 TN mma atom

1. 数据需要以 Layout Right 进行看待
2. 但是 TV -> MN 仍然是以 Layout Left 来进行看待。不过由于我们将 mma 看做一个整体，所以不必纠结于具体线程获得的具体数据，而是要保证整个 warp 获得了对应的 tile 数据

## Copy Atom

make tiled copy with tiled_mma

1. use tiled mma's tv layouts
2. use tiled mma's mn size

在 r2s 的选择上时，在 regiter 视角上，每一个 t 有 4 个 register，并且在 n 方向上进行了重复，所以一个 thread 总共拥有 register (4, 4, 8)。但是这 4 个数据并不是连续的。` ((_2,_2),_4,_8):((_1,_2),_4,_16)`，可以看到实际上只有两个两个是连续的。所以当我们使用 4 作为 tiler 去搬运到 shared memory 的时候应该是行不通的。这是我想到的最合理的解释，不过仍然需要进行打磨

对于 mma atom TN 也符合我的这个猜想，可以看到 tv layouts 都是沿着 K mode 连续，因为 elements 就是沿着 K mode 连续的。如果我们更改了 A or B or C 的 layouts，改变了其连续 mode 的位置，copy atom 就会报错

我之前以 position 来描述 layout 的输出，看了 reed 知乎过后，应该以 offset 来描述会更合适：即距离初始距离的位置，这是一个更为精确的描述。这个 offset 可以就是一维内存当中的位置，也可以是 logical offset，该 logical offset 能够转化成为多维的 natural coordinate，例如 (T, V)

From high level to see how the TiledCopy is going to partition the data

From low level to determine whether the copy atom is able to complete one tiled copy

有几个核心的元素需要特别清晰的定义：

1. TiledLayout_TV, TilerMN
2. ~~AtomNumThr, AtomNumVal~~
3. AtomLayoutDst & AtomLayoutSrc

第一个是元素是在 Tiled Level，第二个元素和第三个元素都是在 Atom Level，实际上第三个元素包含了第二个元素，那么我们就只看第三个元素好了

现在有三个 layouts 以及一个 MN 需要我们关心

- 在 Tile level

  MN 就是一个 Tile 所覆盖的数据逻辑区域。通常我们使用 TiledLayout_TV 来切分这个逻辑区域，让每一个 thread 负责该 MN 区域中的一部分。这个 MN 逻辑区域作为 tensor 的 entry，通过 tensor layout，映射到内存 offset 当中，不过目前我们应该只聚焦在 MN 区域中，即不去考虑 tensor layout，而只考虑 tensor shape

  TilerMN: copy tensor 所使用的 tiler mn 大小

  TiledLayout_TV: 对一个 tiler mn 使用 tv 来进行划分，构建 tv -> mn 的 layout 映射，该 layout 针对于 dstination

  所谓 tv 就是 threads & values，定义了 threads 所拥有的各自 values，threads 数量为一个 block 所包含的 threads 数量。一个 thread 可以拥有多个 values，例如 2/4/8 个。根据 MN 大小的变化，一个 thread 所拥有的 value 也随之增减。

  这个 tv 可能来自于其他 layouts，例如 tiledmma，但是 tiled copy 仍然希望采用这个 layouts 来完成 copy 任务

- 在 copy atom level

  所谓 atom 就是最小的不可分割元素。而 copy atom 就代表了 copy 所能最小的单位 copy 任务。在 s2r 的过程中有较大的 copy atom，以 warp 为单位，一次 copy atom 将完成 (32, 8) 个元素的 copy，并且 src & dst 所对应的 layout tv 还是不一样的（这是为了符合 mma atom 的需求）。而在 r2s 的过程中则使用的较小 copy atom，以单个 thread 为单位，一个 thread 将完成 2 个元素的 copy，一个 warp 则完成 (32, 2) 个元素的 copy

  在 copy atom level 没有定义类似 TilerMN 这样的 MN，所以理解 copy atom layout 就不能理解成为 tv 对 mn 的分割，也不能理解成为 tv -> mn 的 layout 映射。那 copy atom layout tv 是映射到哪个区域去了呢？

  ```c++
  template <class Copy_Atom,
            class LayoutCopy_TV,  // (tid,vid) -> coord   [Need not be 2D...]
            class ShapeTiler_MN>  // coord space
  struct TiledCopy : Copy_Atom
  {
    // Layout information from the CopyAtom
    using AtomThrID     = typename Copy_Atom::ThrID;        // thrid -> thr_idx
    using AtomLayoutSrc = typename Copy_Atom::ValLayoutSrc; // (thr,val) -> offset
    using AtomLayoutDst = typename Copy_Atom::ValLayoutDst; // (thr,val) -> offset
    using AtomLayoutRef = typename Copy_Atom::ValLayoutRef; // (thr,val) -> offset
  ```

  在 TiledCopy 的注释中，对 LayoutCopy_TV 有明确的注释：`(tid, vid) -> coord`，也就是我所理解的 tv -> mn 映射。而 copy atom layout 写为 `(thr, val) -> offset`，并没有明确说明这是向 coord 的映射，而只是一个最抽象的 offset。

  我们需要在 TiledLayout_TV 和 AtomLayout_TV 之间建立起联系，这个联系更具体的说：如何将 tiled layout tv 进行切分，使用 atom layout tv 来构建出 tiled layout tv。整个过程几个困难：

  1. src & dst atom layout 并不相同，元素从 src copy 到 dst 过后 tv id 可能会发生改变

     实际上这个 offset 并没有指向 mn，也没有指向一维内存。我认为这个 offset 更像是一个逻辑 id，其核心功能并不是描述 src or dst tv 会去获得某个位置的元素，而是作为一个中间桥梁联系 src & dst tv。更具体地来说，src & dst 如果其 layout 相同，那么所有的 src (tid,vid) 和 dst (tid,vid) 的 id 都是一样，这意味着，从 src (tid, vid) 的元素就会 copy 到 dst (tid, vid)。但是当 src & dst layout 不相同，那么 src (tid, vid) 算出来的 id 应该是对应 dst 不一样的 (tid', vid')，从 src (tid, vid) 的元素就会 copy 到 dst (tid', vid')。例如 offset = 2 的元素，在 src 是由 tv(0, 2) 拥有，而在 dst 则是由 tv(1, 0) 拥有

     于是根据这个 id 关系我们可以构建一个 src tv -> dst tv 的映射关系 `src_tv2dst_tv`。该映射也可以用 layout 的形式来表示。通过该映射关系，能够轻松获得 src tv 中的元素在 dst tv 当中的位置

  2. AtomLayout TV 和 TiledLayout_TV 之间，T 和 V 都不一样，通常 atom layout tv <= tiled layout tv

     这个问题很好解决，只需要通过重复 atom tv 来覆盖 tiled layout tv 即可。这可以直接使用 zipped divide 完成，计算得到需要重复的 thread & value（注意，我们此时直接将 atom tv shape 作为了 tiler，没有关注其 stride，正如之前所说其映射结果是一个抽象的 offset，核心是联系 src tv）

     此时问题就来了：应该使用 src atom tv 还是 dst atom tv 呢？这里我们就要将 TiledLayout_TV 进行更加清晰的定义了：TiledLayout_TV 实际上是 dst tiled layout tv。为什么没有一个 src tiled layout tv 呢？我认为答案是不需要。当确定好了以上三个 tv layouts 过后，通过映射关系，src tiled layout tv 就已经是确定的了（更具体的说：src tiled layout tv -> mn 是确定的）

     ```c++
     layout_dst = zipped_divide(dst_tiled_layout_tv, dst_atom_layout_tv.shape) // ((t, v), (rest_t, rest_v))
     layout_src = layout_dst.compose(src_tv2dst_tv)	// ((t, v), (rest_t, rest_v))
     src_tiled_layout_tv = zip(layout_src) // ((t, rest_t), (v, rest_v))
     ```

     再定义一个 src tiled layout tv 则会引起冲突，如果映射到了不同的 mn，那么通过 copy 过后，这个元素将会发送到错误的 dst tv 上，从而映射到错误的 mn 上

     此时 src tiled layout tv 将会对 mn 进行正确的分割，从而让每个线程获得正确的 mn 元素。

     AtomLayoutSrc: 描述 copy atom src tv 映射到的 offset (一种逻辑 id)，与 dst 逻辑 id 对应，将 (tid, vid) -> mn 所拥有元素发送到 id 相同的 dst (tid', vid') -> mn 上

     AtomLayoutDst: 描述 copy atom dst tv 映射到的 offset (一种逻辑 id)，与 src 逻辑 id 对应

考虑一个 S2R 的 copy 过程



考虑一个 R2S 的 copy 过程

在进行 copy 的时候应当考虑两个 tensor，一个是 source tensor 一个则是 destination tensor

在 copy 的时候数据连续性是 copy 可行的重要条件

## partition

将 PermutationMNK 作为 tiler 来划分 tensor，实际上做的是 logical divide

partition 通常有两个核心元素，围绕这两个核心元素进行 zipped divide & compose

1. tiler 大小，通常由 `ShapeMNK` 来表示
2. tv layouts，通常由 `Layout_TV` 来表示

以 tiled mma partition_A 为例

围绕这两个核心进行了三大步骤

1. 用 atom shape MK (16, 16) 对 tensor 进行 zipped divide，生成了 `a_tensor`

   `((AtomM, AtomK), (RestM, RestK))`

2. 用 atom tv layouts 对 `a_tesnor` 进行 layouts transform，生成 `tv_tensor`

   `tv_tensor = a_tensor.compose(AtomLayoutA_TV(), _)`

   tv tensor 的形状为 `((T, V), (RestM, RestK))`，也就是将 MK 转换成了 TV

   为什么不直接用 TV layout 进行 compose，而是要先进行 zipped divide？是因为要调整 modes 的顺序，本质上是一种 rearange，从而以 tile 的视角看待 tensor。然后直接对 tile 进行 compose，这样就非常方便了

3. 对 `(RestM, RestK)` 进行 grouping，这其实是 EU repeat 的体现。在实际操作中是用 vmnk 作为 tiler 来对后面两个维度进行 zipped divide，生成 `thr_tensor`

   `thr_tensor = zipped_divide(tv_tensor, thr_tile)`

   thr tensor 的形状为 `((T, EU_RepeatM, EU_RepeatK), (V, RestM', RestK'))`

此时第一个维度全部是 thread，可以直接通过 index 来进行获取，生成 partitioned tensor `(V, RestM', RestK')`

## copy

focus on copy of tensor B，focus on s2r process

B is natural layout (128, 32)

对于 copy 来说就变得似乎更复杂？没错是的，gemm is all about copy...

copy 涉及到了两个 atom layout：一个是 mma atom layout，另一个则是 copy atom layout，我需要弄明白这两者之间的关系。另外由于是使用 ldmatrix，所以一个 thread 上的数据会被发送到多个 thread 当中，这也添加了模型的复杂性

1. 一个 copy atom 会做什么？

当在看 position 时一定要以 position 的顺序去理解，而不要回到 natural index 顺序，除非正在进行 compose，此时 position 才是 natural index

一定要搞清楚目的：position = x 的 element 和 coord = (x,x) 的 element 是不一样的问题。coord = (x,x) 还需要通过 layout function 转换成 position 才能得到 element。也就是说当输入为 coord 时，获得的一定是 position，当输入为 position 时，通过 tensor 首地址就可以获得 element value。position = 1 的 tensor value 无论 layout 如何变换，永远都会等于那个元素！

回答问题：为什么一定保证 register 一定获得自己的元素？

这是由 copy atom 的 tv layout & mma atom 的 tv layout 共同决定的，通常二者是紧密关联的，二者不能任意改变。准确来说，copy atom tv layout 应当根据 mma tv atom 需求，并考虑 ldmatrix 的功能来进行精确的编排，从而使得下面这张图能够发生：

<img src="CUDA Programming 8/v2-5a2257c2bea9b2f6652cfe579444f3bb_720w.webp" alt="img" style="zoom:67%;" />

即 copy atom thread 的数据需要分配到不同 thread 的 register 当中。以 t0 为例子，其分配的 shared memory 数据将被分配到 t0,t1,t2,t3 所拥有的 register 当中。而 t0 register 数据来自于 t0, t8, t16, t24 当中的 shared memory 数据

## Swizzle

reed swizzle https://zhuanlan.zhihu.com/p/671419093

> 回顾之前的介绍我们知道描述逻辑空间我们可以使用 [Layout（本质是函数）](https://zhuanlan.zhihu.com/p/661182311)，而为了避免 bank 冲突，cute 中定义了 swizzle 抽象，swizzle 的本质也是函数，swizzle 作用在 layout 上，即函数作用在函数上，复合函数复合的定义。Layout 的作用是给定坐标返回 offset，而 swizzle 的作用则是给定 offset 返回 bank conflict free 的 offset。即
> $$
> offset_{\text{no-conflict}}=Swizzle(Layout(coord))
> $$

通过 swizzle 获得了新的 layout，将 (M, N) -> offset 的位置进行改变。所以当在进行 read & write 时，会将数据读写到 swizzled position 从而避免 bank conflict

swizzle 不同于普通的 layout algebra，没办法用之前的 composition 来统一表达，但其本质仍然是函数映射。通过 B, M, S 三个参数来完全表示。最小单元为 $2^B$，而这个单元就是从 layout offset 顺序进行 group 和排序

## 图解 Efficient GEMM

- 我需要将整个 gemm 进行图解来方便自己对过程进行把控和推导

## Question

1. 为什么在修改 `static constexpr int kMmaEURepeatN = 1;` 过后 multi-stage gemm 结果出现了变化？改变 `kMmaPN = 4 * kMmaEURepeatN` 也会改变结果

   观察到

   ```c++
     auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);  // (CPY, CPY_N, CPY_K), (8, 4, 2) = ((8, 1), 128/32, 32/16)
                                                      // (CPY, CPY_N, CPY_K), (8, 8, 2) if set EURpeatN = 1, tCrB = (4, 16, 2)
   ```

   PermutationMNK 这将改变 TiledCopy 进行划分的情况（如果 TiledCopy 使用 TiledMMA 来进行划分的话），但是不改变 TiledMMA 对数据的划分（如果只是简单扩张，没有真正地进行 permute），quote [github-issue](https://github.com/NVIDIA/cutlass/discussions/1345#discussioncomment-8485429)

   > This doesn't actually affect the partitioning of input/output tensors because, by convention, only a single atom is ever partitioned out. It will affect the output of `tile_size` and `get_layoutC_MN` and `get_layoutC_TV` etc, which could affect any `TiledCopy` that rely on those partitioning patterns by being built on this TiledMMA.

   真正出问题在于 G2S & S2G 的 copy 是写死了用 128 个 thread 进行 copy，所以在使用 EURepeat = 1 过后，线程数量直接减少，但是由于 copy layout 已经写死，每一个 thread 所获得的数据都不变（实际上应该增加），所以结果错误

   而在 kMmaPN 更改过后，由于和 epilogue 当中的 copy 代码也会产生冲突。申请了更大的 shared memory 空间，一个 shared memory pad 现在有 `(32, 64, 2)` 个数据，在 copy 数据的时候大小和顺序都需要进行更改

   这告诉我们：如果手写 kernel 的话，通常所设计的 kernel 对于参数的要求非常死，改变一点都会出错。要写一个 robust kernel 还是得用这些三方库的封装来完成，这样就不需要考虑太多 mma atom & copy atom 之间的合作关系


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

5. copy 是异步进行的，但是每一次 copy 应该是使用线程的。那么当在发射多个 copy 命令的时候，是否会发生线程不够的情况？如何解决这样的疑问？

6. 在 layout algebra 中使用 zipped divide，有时候会将 divide 过后的 modes 进行合并，为什么能进行这样的合并？这个合并在特定情况下是一定会发生的吗？

   合并（coalesce）操作会将能够合并的 mode 进行合并，对于矩阵乘法来说，在进行简单 tile 的时候，这种合并几乎是一定会发生的。简单来说：将矩阵先 divide 为 4 份，再 divide 为 2 份，等价于直接 divide 成为 8 份。CUTE 在实现 compose 的时候会在 return layout 时去做合并操作。所以会经常看到一些 modes 在 logical divide & zipped divide 时被合并掉

   ```c++
     thrfrg_A(ATensor&& atensor) const
     {
       CUTE_STATIC_ASSERT_V(rank(atensor) >= Int<2>{});
       //CUTE_STATIC_ASSERT_V(size<0>(atensor) % size<0>(TiledShape_MNK{}) == Int<0>{});
       //CUTE_STATIC_ASSERT_V(size<1>(atensor) % size<2>(TiledShape_MNK{}) == Int<0>{});

       // Reorder the tensor for the TiledAtom
       auto t_tile = make_tile(get<0>(PermutationMNK{}),
                               get<2>(PermutationMNK{}));
       auto t_tensor = logical_divide(atensor, t_tile);  // (PermM,PermK), (32, 16)
       // t_tensor ((_32,_4),(_16,_2)):((_32,_1024),(_1,_16))

       // Tile the tensor for the Atom
       // the AtomShape_MNK would be always the same
       // the AtomLayoutMNK only changes the thr_layout_vmnk_
       auto a_tile = make_tile(make_layout(size<0>(AtomShape_MNK{})),
                               make_layout(size<2>(AtomShape_MNK{})));  // a_tile (16:1, 16:1)
       auto a_tensor = zipped_divide(t_tensor, a_tile);  // ((AtomM,AtomK),(RestM,RestK))
                                         // ((_16,_16),(_8,_2)):((_32,_1),(_512,_16))
       print(a_tensor);print("\n");
       // Transform the Atom mode from (M,K) to (Thr,Val)
       // Thr is (4,8) and Val is (2,2,2) for each thread
       auto tv_tensor = a_tensor.compose(AtomLayoutA_TV{},_); // ((ThrV,FrgV),(RestM,RestK))
       // atom layout A TV: ((4, 8), (2, 2, 2)) : ((32, 1), (16, 8, 128))
       // tv_tensor: (((4, 8), (2, 2, 2)), (8, 2))
       // the rest axis is to replicate the AtomLayout, because they are just shifting it, shift does not permute!

       // Tile the tensor for the Thread
       auto thr_tile = make_tile(_,
                                 make_tile(make_layout(size<1>(thr_layout_vmnk_)),
                                           make_layout(size<3>(thr_layout_vmnk_))));
                                   // thr_layout_vmnk_ (_32,_2,_2,_1):(_1,_32,_64,_0)
                                   // thr_tile (_,(_2:_1,_1:_0))
       auto thr_tensor = zipped_divide(tv_tensor, thr_tile); // ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK)))
       // thr_tensor (((_4,_8),(_2,_1)),((_2,_2,_2),(_4,_2))):(((_2,_32),(_512,_0)),((_1,_256,_8),(_1024,_16))
       print(thr_tensor);

       return thr_tensor;
     }
   ```

7. 将 layout 和 tv_tensor 进行 compose 过后会得到什么？

   ```python
   # atom layout: ((4, 8), (2, 2, 2)) : ((32, 1), (16, 8, 128))
   # atom_tenosr: ((_16,_16),(_8,_2)):((_32,_1),(_512,_16))
   
   # compose on the 1st mode
   tv_tensor = compose(get<0>(atom_tensor), atom_layout)
   
   # tv_tensor: (((_4,_8),(_2,_2,_2)),(_8,_2)):(((_2,_32),(_1,_256,_8)),(_512,_16))
   ```

   我们实际上在做的事情，就是将 `atom_layout` 映射到 `atom_tensor` 当中的位置，用更形象的话来说，就是将 `atom_tensor` 分配给 `atom_layout`。可以看到分配结果就是 `tv_tensor`，通过 natural index 可以找到对应的 `atom_layout` 元素和 `tv_tensor`，这样就构建起了桥梁。总结：use the B's natural index to get the correct position in A。下面我举2个例子，就是从 `thrfrg_A` 中找的具体代码。其功能就是将 A tensor compose tv tensor

   ```python
   ### Example 1 ###
   # atom_layout (ThrV, FrgV) ==> ((4, 8), (2, 2, 2)) : ((32, 1), (16, 8, 128))
   atom_layout = Layout([4, 8, 2, 2, 2], [32, 1, 16, 8, 128])
   
   # a_tensor (M, K) ==> (16, 16) : (32, 1)
   A_tensor = Layout([16, 16], [32, 1])
   
   # tv_tensor (ThrV, FrgV) ==> (((_4,_8),(_2,_2,_2))):(((_2,_32),(_1,_256,_8)))
   tv_tensor = compose(a_tensor, atom_layout)
   

   ### Example 2 ###
   # permuted a_tensor (ThrV, FrgV) ==> (((_2,_4,_2),_16)):(((_32,_128,_64),_1))
   A_tensor_permute = Layout([2, 4, 2, 16], [32, 128, 64, 1])

   # tv_tensor (ThrV, FrgV) ==> (((_4,(_2,_4)),(_2,_2,_2))):(((_2,(_32,_128)),(_1,_64,_8)))
   tv_tensor = compose(a_tensor_permute, atom_layout)
   ```

   可以看到无论 A tensor 进行如何 permute，最终的 `tv_tensor` 的 shape 都可以视作 `(4,8,2,2,2)`，只是其中的会有一些分解，但是完全不妨碍进行 index to coordinate 的变换。所以说 compose 的功能就是一种位置转换 (position transform)：保持 LayoutB 的整体 shape 不变，将 B 中各个元素的位置转换到 A 中的位置。通过这种转换带来的便利：能够以 shape B 视角去获得 A 当中的元素，这样的功能就是 tiler or divide 的基础。

   在 compose 过程中，我们实际上不会太去在乎 A layout 中包含的 position (layout function 结果) 是什么，我们主要关注其 shape 是否能够容纳 B's position (compose admissibility)，所以我以变量 $x_i$ 来表示，代表其可以随意变换。只要 B 的 layout 不变，那其所对应的 A coordintate 是不会改变的。B 中元素所代表的变量则控制了其所选择的 A

   我得明确一下我的术语：layout function & position 混用了起来, position is the output of the layout function `Layout(coordinate)`

   <img src="CUDA Programming 8/image-20250525152311595.png" alt="image-20250525152311595" style="zoom:50%;" />

8. 如果对 data 进行 Permute，所分配的 layout 会发生什么变化？总感觉在进行 layout 分配的时候，会有一些重复操作。但是按理来说所有的操作都应当是必须的，因为所有的变量都在之后被使用，并且都需要这些信息。另外在连续地进行 compose & divide 的过程中是否会出现无法进行 compose 的情况？（e.g. permute the input）我是否可以把 compose 中的输入一直视为一个整体，这样能够快速地理解这个 layout algebra 做了什么事情

   logical divide shrink the admissibility of the original tensor, but the M does not change, makes the input domain being the same

   following this thread, A compose B does not change the input domain, which keeps the same as B, because the M keeps being the same as B

   we should understand the admissibility as: the stride can be divide by the prev M, and the next M can be divided by c, and the rest of the space can be divided by the single shape. and there is a special case: the stride == prev M, then the next M would definitely be divided by c, because c == 1, and then only requires the rest of the space can be divided by the single shape. These requirements are easy to be satisfied, if everything is in $2^n$

9. 为什么要用单个 atom 来进行 compose，为什么不用 tiled atom 来进行 compose？

   实际上是可以的，这样做实际上更符合我对 tiled mma 的理解。不过这样做需要先对 atom layout 进行 product。而 cute 代码是先做单个 atom 的 compose，然后再做一个 tiler 的 zipped divide。或许我应该拿出一个我认为合适的 layout algebra 转换

   integer as unique position, when natural layout, most of the time is logical position. when strided/permuted layouts, it can be represented in memory positions

10. **retile 的作用是什么？为什么会需要 retile**

    如果不进行 retile，copy 会报错

    ```shell
    error: static assertion failed with "Copy_Traits: dst failed to vectorize into registers. Layout is incompatible with this CopyOp."
   ```

    一个简单的理由：tCrB 是按照 mma atom 来分配的，而 copy 则是按照 copy atom 分配，但是在真正 copy 的时候要求 tCrB 是按照 copy atom 的 reference shape 进行的，所以需要将其 retile 成为 copy atom 对应的形状
    
    本质是因为 copy atom 和 mma atom 不一致导致：copy atom 一次可以搬运两个 mma atom 所需要的数据。所以需要计算 mma atom value 与 copy atom value 之间的关系。通常这个关系就是在 M or N 方向上的重复，通过在 M or N 方向上进行 zipped divide 即可切分出 copy atom value 所对应的区块，然后再用 copy atom val layout 进行 compose，就能够将 tCrB 的 shape retile 为 copy atom 所需要的 shape
    
    为什么是 (8, 4, 2) 而不是 (8, 8, 1)，这取决于 copy atom val 是对应的 mma atom val 的哪个部分。在 matrix B 的例子当中，是在 N 方向进行重复，copy atom 的 8 个 value 对应的是 mma atom 4x2 个 value（同一个 t）
    
    an extention: 如果 copy atom 所采取的 layout tv 和 tiled mma atom layout tv 不一致的话应该如何去做分配呢？例如 copy atom tv 是在 K 上做 repeat，而 tiled mma atom 是在 N 上做 repeat
    
    此时 tCrA 仍然是 (4, 8, 2)，但是需要进行重新进行 tile：zipped divide with (4, 1, 2) -> ((4, 1, 2), (1, 8, 1))，然后再用 copy atom layout v 去 compose，获得 ((4, 2), 8, 1)，这样才会生成一个 (8, 8, 1) 的 tCrB，虽然从整理来看，从 copy 中划分出来的 thread value 包含了所有的 mma atom 所需要的元素，但如果 copy 的顺序不正确仍然是错误的。
    
    gemm 当中有2个 retile (s2r & r2s)，都可以从 mma atom 视角与 copy atom 视角之间的转换来进行。connection：copy atom value is repeated mma atom value layout. This transform can be done easily by a simple tiler

11. 为什么 `tCrA` 的 layout 是连续的，而不是像我想想中的 atom layout 中不连续的分布

    ```c++
    tCrA shape = ptr[16b](0x7f424cfffae0) o ((_2,_2,_2),_4,_2):((_1,_2,_4),_8,_32)
    tCrB shape = ptr[16b](0x7f424cfffb60) o ((_2,_2),(_2,_4),_2):((_1,_2),(_4,_8),_32)
    tCrD shape = ptr[16b](0x7f424cfffbe0) o ((_2,_2),_4,_8):((_1,_2),_4,_16)

    tAsA shape = smem_ptr[16b](0x7f424e000040) o ((_8,_1),_4,_2,_3):((_1,_0),_1024,16,_4096)
    tCrA_view shape = ptr[16b](0x7f424cfffae0) o ((_8,_1),_4,_2):((_1,_0),_8,_32)

    tBsB shape = smem_ptr[16b](0x7f424e006040) o ((_8,_1),_4,_2,_3):((_1,_0),_1024,16,_4096)
    tCrB_view shape = ptr[16b](0x7f424cfffb60) o ((_8,_1),_4,_2):((_1,_0),_8,_32)
   ```

    因为这只是申请的 Register，而没有进行 copy，所以呈现出来 natural layout
    
    我发现 `tAsA & tArA_view` 的 layout 在不同的 thread idx 下也是一样的，但是其首地址是不一样的。这是因为对其使用了 coordinate 进行了 index，所以取出来的 tensor layout 一样但是首地址不一样

12. 验证了 permute tiler 过后对结果并没有影响。那么在进行 mma 计算的时候应该如何看待这种 permute？似乎这里有一个现象: layout & memory 之间其实没有必然关联，因为 memory 是不会去改变的，变的是 layout，也就是 thread 是如何获得 memory 当中的 value。问题在于如果 thread 获得 memory 中的 value 改变，那么 mma 结果为何还是正确的？

    position can be seen as the corresponding elements, they are tied up

    shape is just a entry where we can take the element with slicing conveniently, each shape square store a position & elements

    shape is how we "see" the matrix

    when permuting the layout, we can see the elements are shifted to different square of shape, but the relationship between elemtn and position is not changing

    ```python
    Layout([4, 2], [2, 1])
    Layout([4, 2], [1, 4])

    #      0      1
    #      2      3
    #      4      5
    #      6      7
    # ====================
    #      0      4
    #      1      5
    #      2      6
    #      3      7
    ```

13. retile_D & retile_S

    这似乎只对 rA & rB 进行，而 rA & rB 在我们的 case 里是 simple layout，retile 变得非常简单且直观

    其本质作用就是将 tensor_D or tensor_S 转换成 copy atom layout，使得 copy 过程中 source & destination 的 layouts 对齐。用例子来理解

    copy atom layouts 和 tiledmma layouts 是保持一致的，但是会考虑到 permutation mnk

    **为什么 retile 和 partition 会有相同的结果？输入不同，但结果相同？**

    从第一性原理来理解 retile 到底做了什么事情，在进行 cuda 变成需要在几个不同的视角进行切换讨论：

    1. thread level
    2. atom level, num threads = warp
    3. tile level, num threads = the whole block
    4. previous level is trying to define what an operation can do, now we have to fit these ops into the whole tensor, see what it can do

    在阅读 cuda 代码的时候一定要弄清楚自己是在哪些 level 进行讨论

    1. thread level in a atom tensor
    2. thread level in whole tensor
    3. tile level to partition the tensor

14. 遇到新问题：partition_S 所获得的 thread tensor 和 partition_B 所获得的 thread tensor 并不一致，那经过 copy 过后是否就凌乱了呢？

    partition_B is correct, and the tCrB gets the same value of partition_B

    问题可能在于 ldmatrix 命令，其本身就是会将数据分配到不同的线程当中，那么问题就变成了：如何确保经过 ldmatrix 将数据分配到对应线程当中？

15. right inverse 的作用是什么？

    在 reed [zhihu](https://zhuanlan.zhihu.com/p/662089556) 中有提到定义，但是没有解释具体作用和使用场景

    layout function: natural coord -> position

    right inverse: position -> natural coord

    我应该定义为 logical position & physical position。那么 right inverse 会有什么作用呢？举一个例子

    ```c++
      tidfrg_S(STensor&& stensor)
      {
        CUTE_STATIC_ASSERT_V(rank(stensor) >= rank(Tiler_MN{}), "Rank of tensor to be partitioned too small.");

        // Tile the stensor and compute the (src-thr, src-val) -> (ref-thr, ref-val) layout
        #if 0
        if (thread(0, 0)) {
          print("==> TilerMN");print(Tiler_MN{});print("\n");
          print("==> Right inverse of AtomLayoutRef");print(right_inverse(AtomLayoutRef{}));print("\n");
          print("==> AtomLayoutSrc");print(AtomLayoutSrc{});print("\n");
          print("==> AtomLayoutRef");print(AtomLayoutRef{});print("\n");
          print("==> AtomLayoutDst");print(AtomLayoutDst{});print("\n");
        }
        #endif
        return tile2thrfrg(zipped_divide(stensor,Tiler_MN{}), right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{}));
      }

    ```

    这里是 copy atom 在分配数据的时候会碰到的代码。其中就有一个 ` right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{}))`，这里 `AtomLayoutRef` 实际上就是 `AtomLayoutDst`

    OK，现在来应用 `right_inverse` 本身的，概念：将 physical position -> logical position。在我们这个 case 当中 AtomLayoutDst/AtomLayoutSrc 就是一个 TV layouts，logical position 就是 (Tid, Vid)，而 physical position 代表对应元素在内存当中的位置。现在使用 right inverse 过后就有了映射：内存位置 -> TV。现在用这个 inverse 来 compose AtomLayoutSrc 就能获得这样的映射：Src (Tid, Vid) -> 内存位置 -> Dst (Tid, Vid)

    也就是说我们建立起了 Src (Tid, Vid) -> Dst (Tid, Vid) 的映射关系，其桥梁是共同的内存位置，该映射关系称之为 `src2ref`

    OK，顺着这个映射我们再进一步思考，这样的映射能有什么功能？在 s2r copy 当中，我们需要将 shared memory 当中的数据 copy 到 register 当中，并且符合 mma atom 的数据 layout 需求，所以 mma atom 有一个特定的 layout `atom_layout_TV`，该 layout 作为映射 MMA (Tid, Vid) -> 内存位置

    那此时将 `atom_layout_tv` 和 `src2ref` 进行 compose 会获得什么结果呢？我们就能够获得 src tv 会分配得到的 memory position，此时的分配结果就能够保证，通过 copy atom 过后，register 能够获得自己所需要的元素

    ```c++
    // atom_layout_TV ((atom_tid,atom_val),(rest_tid,rest_val))
    // (((_4,_8),(_2,_2,_2)),((_2,_2),_1)):(((_64,_1),(_32,_256,_16)),((_0,_8),_0))

    // src2ref (t, v) of source layout
    // ((_8,_4),(_2,_4)):((_4,_64),(_32,_1))
    auto trg_layout_TV = atom_layout_TV.compose(src2ref, _);
    ```

    证明：为什么 copmose 过后的结果一定符合 register 所需要的

    1. src2dst 的映射保证了，src (t_i,v_j) 所对应的元素 `x` 一定就是 dst (t_i', v_j') 所对应的元素 `x`
    2. 该元素具体是什么由 atom laytou tv (t_i', v_j') 决定。这里就隐含了一个条件，atom layout tv 和 dst layout tv 在 shape 上是一一对应的

    所以 `trg_layout_tv` 所获得的结果 (t, v) -> (m,n) 一定能够满足 register 要求。用上方的例子说：atom layout tv (t_i', t_j') 需要元素 x，通过给 src (t_i, v_j) 分配元素 x，就能完成所需 copy 结果

    **it seems that there is a fact**: the logical tv positions are often being the same, this makes sense, because what we actually changing is the content of value (physical position)

    layout algebra 的本质的确是从 map from integer to integer，但是为了更好地理解其本质，我们还是应当为这些 integer 赋予合理的物理意义。例如 tv integer to memory integer；tv integer to tv integer; logical shape to logical shape

    我终于理解了注释中的 `*((atom_tid,atom_val),(rest_tid,rest_val)) -> (m,n)`，这个箭头到底代表什么！这就是代表的 layout 的映射关系：将 atom tv integer 映射到 tensor element

16. A few useful methods:
    1. append
    2. prepend
    3. product_each
    4. size
    5. get
    6. rank
    7. group_modes
    8. recast

       1. upcast

          by mode operation

          For stride = 1, shape_div(N, 1) = N, so shape becomes shape / N.

          For stride > 1, stride becomes stride / N. there are special cases where the stride are less than N, then the stride becomse 0, shape becomes 1 (which can be coalesce)

       2. downcast

17. stride = 0 就可以达到重复抽样的目的

18. change of B's layout to natural layout would cause error, why?

    连续性要求

19. 尝试过了 pipe = 1，对于我们的 copy 效率没有太多的影响。说明以 (32, 32) 作为一个 chunk，已经能够很高效地进行运输，不会发生 global memory 内存不连续的情况（以 128 bit 为例，只需要连续 8 个 fp16 就能够达成 coalesced copy）

20. 为什么 mma atom 的要求是 TN，而输入的 tensor 的形状都是 layout T，这不会造成错误吗？

    TN 并不是 CUTE 中所声明的 TN，只是延续了 blas 当中的传统。具体的要求在 MMA Atom 部分已经讨论清楚

21. 我看到在从 shared memory -> register 的 copy 当中使用的是 ldmatrix 这样的指令，这个指令能够将自己线程的内容发送到其他线程当中，一个 thread 将会 copy 8 个元素。而在进行 register -> shared memory 的过程中确没有看到对应的指令，而是用的简单的 universal copy，每一个 thread copy 两个元素。这样的话 copy 效率是否就下降了呢？

    > From DeepSeeek
    >
    > - **读取更需要优化**：在GEMM中，从Shared Memory读取数据的频率远高于写入（例如，一个Tile可能被多次读取但只需写入一次）。因此，优化读取（`ldmatrix`）的收益更大。
    > - **写入的合并性容易保证**：只要线程按规则写入（如每个线程写连续地址），`st.shared`本身就能实现合并写入，无需额外的线程间协作。

22. 在 cute-gemm main loop 当中，为什么会在 if 之外写 `cp_async_fence()`

    ```c++
        // if this is the first small k iter, load the next tile of the big k iter
        if (itile_to_read < ntile) {
          cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read), tAsA_copy(_, _, _, ismem_write));
          cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read), tBsB_copy(_, _, _, ismem_write));
          // cp_async_fence(); // it must be outside the if condition, or the next cp_async_wait will not work as expected
          ++itile_to_read;
          ismem_write = (ismem_write + 1) % kStage;
        }
        cp_async_fence();
    ```

    如果在 if 内写的话，之后的一个 `cp_asnyc_wait` 在等待最后一个 tiled 的时候就不会起作用，少一个 fence，这会造成输出错误

## TODO

1. cutlass api learn
2. cmake learn
3. cutlass 学习路线总结
4. 两个大笔记的梳理，有 2w 字，还是需要好好把第一阶段总结下的，哪些弄清楚的了，哪些没弄清楚，如何学习的。实际上我们还是把 cute-gemm 都学透了的，基本上没遗留大问题
