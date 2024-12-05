# CUDA Programming 7

有了之前 CUDA Programming 1-6 的铺垫，对于 CUDA 基础应该有了一定的了解。现在想要干一些有趣的事情

想必绝大部分在深度学习领域的人都听说过 [FlashAttention](https://github.com/Dao-AILab/flash-attention)，其是一个 fast & memory efficient 的 attention 实现。除此之外，也有非常多的人听说过 [FlashInfer](https://github.com/flashinfer-ai/flashinfer)，该项目也是一个用于加速 LLM serving 的 library。而你去查看他们的仓库时都会发现一个共同的 3rdparty 仓库：[CUTLASS](https://github.com/NVIDIA/cutlass)

我的本意是想要将 FlashInfer 仓库进行深入的学习，但显然在这之前还有大量的 cutlass 知识需要做铺垫。所以我先进行 cutlass 的学习，再进行 flashinfer 的学习

但问题在于：cutlass 似乎没有特别好的教材来帮助入门。目前我有一些切入口：

1. CUDA MODE lecture 15 introduced cutlass a little bit
2. CUTLASS examples
3. CUTE tutorial

通过学习 cutlass 的例子来掌握其用法，**掌握用法是最主要的需求，也是最直接的反馈**

Other zhihu references:

- [Reed's zhihu posts](https://www.zhihu.com/people/reed-84-49/posts), and its gemm code [github](https://github.com/reed-lau/cute-gemm)
- [CUTLASS CuTe实战(一)-基础](https://zhuanlan.zhihu.com/p/690703999)
- [CUTLASS CuTe实战(二)-应用](https://zhuanlan.zhihu.com/p/692078624)
  - [github](https://github.com/zeroine/cutlass-cute-sample)
- [cutlass cute 101](https://zhuanlan.zhihu.com/p/660379052)
- A collective repo which gathers a lots of blogs and kenel impl [CUDA-Learn-Notes](https://github.com/DefTruth/CUDA-Learn-Notes), not suitable for system learning, can be used for look-up table if you are trying to seek for some topic

## Install & Hello World

Good news! CUTLASS does not need to be built!!!

> CUTLASS is a header-only template library and does not need to be built to be used by other projects. Client applications should target CUTLASS's `include/` directory in their include paths.

但为了运行一些 example code，我们必须要进行编译，才能 run 起来

Hello World in CUTE

[sgemm_1.cu](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_1.cu)

[quick start guide](https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md)

[quick start guide-cute](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md)

根据 [quick start guide](https://github.com/NVIDIA/cutlass/blob/main/media/docs/quickstart.md) 中提示，我先试用 cmake 并且指定自己的 compute capacity

```shell
# see your compute capacity with command:
# nvidia-smi --query-gpu=compute_cap --format=csv
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=86
```

似乎应该使用 `-DCUTLASS_NVCC_ARCHS=80`? 即使我的 compute capacity 是86。Another tip: 我尝试在 WSL 上进行 cmake，结果卡住了 `(pid, sts) = os.waitpid(self.pid, wait_flags)`，应该是多线程在 WSL 上不太管用，把 unit test 取消编译就好 ` cmake .. -DCUTLASS_NVCC_ARCHS=86 -DCUTLASS_ENABLE_TESTS=OFF`

在运行完过后可以看到 `build/Makefile` 中会有许多我们可 build 的文件，其中 `examples` 下的所有文件都可以在里面搜到 (e.g. `00_basic_gemm`)，而我要使用的就是 `sgemm_1` 

```shell
make sgemm_1 -j8
```

运行完过后就可以在 `build/examples/cute/tutorial/sgemm_1` 找到，运行

```shell
# under the `build` dir
./examples/cute/tutorial/sgemm_1
```

```shell
➜  cutlass git:(main) ./build/examples/cute/tutorial/sgemm_1
M = 5120
N = 5120
K = 4096
C = A^N B^T
Using device 0: NVIDIA GeForce RTX 3080  (SM86, 68 SMs)
CUTE_GEMM:     [11452.9]GFlop/s  (18.7506)ms
```

该代码的源码就在 `./examples/cute/tutorial/sgemm_1.cu`。于此同时我测试了一下同等条件下 cuBLAS 的 latency 只需要 10.89 ms，所以上面的 sgemm 仍然是一个非常慢 gemm，说明优化空间还非常大👀

## A Closer Look

现在来看下 `sgemm_1.cu` 到底干了什么事情也许是个不错的选择，该文件也就是一个 400 多行的代码，我来贴一部分代码

```c++
// to be paste later...
// not so convinient to see the whole picture
```

可以看到整个代码分为4个部分：

1. `gemm_device` 是 cutlass gpu kernel 的核心实现
2. `gemm_nt & gemm_tn`，调用 `gemm_devie` 完成矩阵乘法
3. `gemm` 就是一个 wrapper，包裹 `gemm_nt & gemm_tn`，其区别于 `cute::gemm`，或许应该换一个名字
4. `main` 即为 host 主程序的运行，包含测量 Flops & latency

###  gemm_device

在此之前，我从未学习过如何使用 GPU 来进行矩阵运算，只对简单的 reduce & tranpose 做过学习。那么问题来了：为何矩阵乘法适合于并行运算？这是因为可以独立地计算每一个元素
$$
C_{ij}=\Sigma_k A_{ik}B_{kj}
$$
这个公式可以用图像非常形象地表达

<img src="CUDA Programming 7/image-20241128111348874.png" alt="image-20241128111348874" style="zoom: 50%;" />

蓝色部分的矩阵乘积结果，由绿色和黄色部分的矩阵的点积和得到。`gemm_device` 所采用的就是这样朴素的思维，比不过 cuBLAS 也是情有可原的，下面具体地讨论整个过程，即：如何将数据分配到各个 block/warp/thread 当中，并进行计算，此处参考 [cutlass cute 101](https://zhuanlan.zhihu.com/p/660379052) [0x_gemm_tutorial.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_gemm_tutorial.md)

- What does this code trying to do logically?

  这基本上就是在教你如何用并行的思维去处理矩阵乘法。我先从逻辑视角完成这件事情，然后再将这些逻辑映射到代码当中，看下代码的具体实现

  为了简单起见，我们首先定义问题为最简单的单精度矩阵乘法：$C = AB$，他们形状分别为 `A(M,K) & B(K,N), C(M,N)`，为了让问题更加具象化，我们以 `M=N=5120, K=4096` 为例子（这也是 cutlass 代码例子中所使用的数值）

  一个不错的划分视角是以矩阵 $C$ 为核心：我们将矩阵 C 进行划分，以 $(128,128)$ 的方块作为划分单位，去单独求解每一个方块中的矩阵乘法结果。从 CUDA 编程的角度来讲：我们让一个 block 去处理一个 $(128,128)$ 的矩阵乘法结果

  <img src="CUDA Programming 7/image-20241203155920569.png" alt="image-20241203155920569" style="zoom:80%;" />

  好，现在就可以集中精力来处理每一个 block 应当如何计算了，即：计算一个 $(128,128)$ 的矩阵 C 应该如何做到？首先我们需要获取矩阵 A & B 中对应的数据，分别获得对应的 $(128, 4096)$​​ 大小的数据（即上图中的 `gA & gB`，`gC` 也在图里，`g` 代表的是 global memory，请忽略图中错误的分块数量，因为实在画不了...）

  其实现在就可以做矩阵乘，得到我们想要的结果：
  $$
  gC=gA \times gB
  $$
  但是这里是 GPU！我们如果每次都从 global memory 读取数据的话，时延会非常大，所以我们需要先把数据读到 shared memory 里面，这样读取数据做计算的时候会更快。但是 shared memory 又没有这么大的空间，每一个 block 需要分配~ 100K*4Byte 大小的空间，这太多了！好消息是：我们还可以将这个问题继续进行切分

  我们沿着 K 轴方向，把 4096 切分成为 $(512,8)$ 的形状，我们每次做 $(128,8)$ 大小的矩阵乘法，然后进行了累加也能够得到相同的结果。经过切分过后我们的计算过程变为了
  $$
  gC=\Sigma_{i=1}^{512}sA\times sB
  $$
  <img src="CUDA Programming 7/image-20241203160014235.png" alt="image-20241203160014235" style="zoom:80%;" />

  最后我们就需要考虑将这个 block level 的问题，切分到 thread level 上，考虑每一个 thread 应该干的工作。考虑一个 block 有 256 个 thread，我们将这个 thread 排列成为 $(16,16)$ 形状，这样一次就可以处理 $(16,16)$ 大小的矩阵。

  为了方便用图形表示，我没办法画这么大的图，我将用形状 `gC.shape=(8,8) & sA.shape=sB.shape=(8,4) & thread.shape=(4,4)` 来示意每一个线程所要分配的数据，以及最终计算结果 [reference: math-partitioning](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0x_gemm_tutorial.md#math-partitioning)

  <img src="CUDA Programming 7/image-20241203160059324.png" alt="image-20241203160059324" style="zoom: 67%;" />

  我在 threads 中选择了5个 thread 来进行表示，每一个 thread 用不同的颜色。虽然眼花缭乱的，但是就是矩阵乘法所需要的数据，专注一个颜色的就OK。第一行红色和橙色的方块都需要读 `sA` 中第一行的数据，所以我多画了一行出来，`sB` 中多出来的一列也是这个道理。每一个 thread 将会去读取 $sA(2,4)$ 以及 $sB(2,4)$ 大小的矩阵用于矩阵乘法

  所以当理解了图中的线程分配方式过后，理解我们例子当中的：将 $(128, 128)$ 大小的矩阵分配到 $(16,16)$ 大小的 threads 中，就非常容易了。每一个单独的 thread 总共会处理 $gC(8,8)$ 大小的矩阵，并且每一次都会获取 $sA(8,8)$ 以及 $sB(8,8)$​ 大小的数据用于矩阵乘法

- What are the cutlass codes?

  经过上面的逻辑分析整个切分脉络已经很清晰了：

  1. 将整个矩阵进行切分，划分成为多个 blocks
  2. 再将矩阵乘法沿着 K 轴进行切分，以满足 shared memory 要求，利用循环累加的方式完成矩阵乘法
  3. 最后将矩阵使用 threads 队列进行划分，给每一个 thread 分配数据，是最终矩阵乘法的执行者

  我直接贴代码了，整个过程跟着注释看也非常的清晰明了

  ```c++
  // Setup params for an NT GEMM
  // Use m-major smem sA, n-major smem sB, and mn-major threads tA|tB
  template <class TA, class TB, class TC,
            class Alpha, class Beta>
  void
  gemm_nt(int m, int n, int k,
          Alpha alpha,
          TA const* A, int ldA,
          TB const* B, int ldB,
          Beta beta,
          TC      * C, int ldC,
          cudaStream_t stream = 0)
  {
    using namespace cute;
  
    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);                     // (M, N, K)
  
    // Define NT strides (mixed)
    auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
    auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
    auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)
  
    // Define CTA tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<  8>{};
    auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  
    // Define the smem layouts (static)
    auto sA = make_layout(make_shape(bM, bK));                 // (m,k) -> smem_idx; m-major
    auto sB = make_layout(make_shape(bN, bK));                 // (n,k) -> smem_idx; n-major
    auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major
  
    // Define the thread layouts (static)
    auto tA = make_layout(make_shape(Int<32>{}, Int< 8>{}));   // (m,k) -> thr_idx
    auto tB = make_layout(make_shape(Int<32>{}, Int< 8>{}));   // (n,k) -> thr_idx
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));   // (m,n) -> thr_idx
  
    dim3 dimBlock(size(tC));
    dim3 dimGrid(size(ceil_div(M, bM)),
                 size(ceil_div(N, bN)));
    gemm_device<<<dimGrid, dimBlock, 0, stream>>>
        (prob_shape, cta_tiler,
         A, dA, sA, tA,
         B, dB, sB, tB,
         C, dC, sC, tC,
         alpha, beta);
  }
  
  template <class ProblemShape, class CtaTiler,// CTA means Cooperative Thread Array
            class TA, class AStride, class ASmemLayout, class AThreadLayout,
            class TB, class BStride, class BSmemLayout, class BThreadLayout,
            class TC, class CStride, class CSmemLayout, class CThreadLayout,
            class Alpha, class Beta>
  __global__ static
  __launch_bounds__(decltype(size(CThreadLayout{}))::value)
  void
  gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
              TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
              TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
              TC      * C, CStride dC, CSmemLayout sC_layout, CThreadLayout tC,
              Alpha alpha, Beta beta)
  {
    using namespace cute;
    //
    // Full and Tiled Tensors
    //
  
    // Represent the full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)
  
    // Get the appropriate blocks for this thread block
    // so when try to tile it with step<_1, X, _1>, it will first tile it in a sub-tensor mA_tiled ((BLK_M, BLK_K), (m, k))
    // then we try to slice it with the coord (blockIdx.x, blockIdx.y, _) in the rest-mode(second-mode)
    // the _ means that for the k dimension, we will take all the elements, similar : in the pytorch
    // so we are actually doing this: gA = mA_tiled[:, :, x, :]
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (x,y,_)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k) i.e. (128, 8, 512)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k) i.e. (128, 8, 512)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)   i.e. (128, 128)
  
    // Shared memory buffers
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K) i.e. (128, 8)
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K) i.e. (128, 8)
  
    //
    // Partition the copying of A and B tiles across the threads
    //
  
    // local_partition is a lot like local_tile, first we tile the tensor, then we still slice in the first mode
    // gA_tiled = ((THR_M, THR_K), (thr_m, thr_k, k))
    // Note that the threadIdx.x will be converted into a coord (x, y) automatically
    // we slice it at the fist-mode (tile-mode) gA = gA_tiled[x, y, :, :, :]
  
    // gA is the tiled tensor in this block, and tA is this thread
    // so tAgA basically means the single tensor allocated for a single tensor
    Tensor tAgA = local_partition(gA, tA, threadIdx.x);                  // (thr_m,thr_k,k) i.e. (4, 1, 512)
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);                  // (thr_m,thr_k)   i.e. (4, 1)
  
    Tensor tBgB = local_partition(gB, tB, threadIdx.x);                  // (thr_n,thr_k,k) i.e. (4, 1, 512)
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)   i.e. (4, 1)
  
    //
    // Define A/B partitioning and C accumulators
    //
  
    // Partition sA(BLK_M,BLK_K) by the rows of tC(THR_M,THR_N)
    // (128, 8) -> ((16), (8, 8)) -> (8, 8)
    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});   // (8, 8)
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X,_1>{});   // (8, 8)
    // (128, 128) -> ((16, 16), (8, 8)) -> (8, 8)
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1,_1>{});   // (8, 8)
  
    // Allocate the accumulators -- same shape/layout as the partitioned data
    Tensor tCrC = make_tensor_like(tCgC);                                // (8, 8)
    // Clear the accumulators
    clear(tCrC);
  
    // TUTORIAL: Example of a simple mainloop that read tiles of data into shared memory,
    //           and then computes on those tiles.
    //   copy(.) operates on the global and shared memory via the tA|tB partitioning
    //   gemm(.) operates on the shared and register memory via the tC partitioning
  
    auto K_TILE_MAX = size<2>(tAgA); // k = K // BLK_K = 4096 // 8 = 512 (in this case)
  
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
      // Copy gmem to smem with tA|tB thread-partitioned tensors
      // tAgA (thr_m, thr_k, k) i.e. (4, 1, 512) in our case
      // tAsA (thr_m, thr_k) i.e. (4, 1) in our case
      // Note that whole block would use the shared memory, (4, 1) actually will multiply (THR_M, THR_K) i.e. (32, 8)
      // and then fill up entire shared memory (128, 8)
      copy(tAgA(_,_,k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
      copy(tBgB(_,_,k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)
  
      cp_async_fence();        // Label the end of (potential) cp.async instructions
      cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
      __syncthreads();         // Wait for all threads to write to smem
  
      // Compute gemm on tC thread-partitioned smem
      gemm(tCsA, tCsB, tCrC);            // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)
      __syncthreads();         // Wait for all threads to read from smem
    }
  
    //
    // Epilogue
    //
  
    axpby(alpha, tCrC, beta, tCgC);
  }
  ```

## CUTE Tutorials

主要总结 Concepts

### Layout

[layout](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md)

- Dynamic and static int

  在介绍 layout 之前先简要介绍下 cutlass 中表示 int 的方式。有两种，一种就是 dynamic int，另一种是 static int

  ```c++
  int{2};   // dynamic
  2;		  // also dynamic
  Int<2>{}; // static
  _2{};	  // equal with Int<2>{}, using cute::_2 = cute::Int<2> 
  ```

- What is Layout?

  > A `Layout` is a pair of (`Shape`, `Stride`). Semantically, it implements a mapping from any coordinate within the Shape to an index via the Stride. 
  >
  > In CuTe, `Layout` is a first class citizen.

  正如上面所说，layout 是由 shape 和 stride 组成的 pair，而 shape 和 stride 本身是 tuple (IntTuple to be exact)，layout 提供了一套抽象，能够很好地描述 multi-dimensional array 的排布，并且方便我们对排布进行变换和操作。当你去 print 一个 cutlass layout 时，就会获得  `shape:stride` 这样的表示

  那么这就引出了接下来的问题：what is shape and stride?

- Shape & Stride

  Shape 其实没什么好介绍的，就是描述 layout 的形状的。和 pytorch 中的 tensor.shape 类似，不同的是 Shape 也可以嵌套: `(make_shape(2, make_shape (2,2))`，这样我们就有一个形状 `(2, (2, 2))` 的 shape。其实嵌套的 shapr or stride 没什么 fancy 的东西，通常用来表示 sub layout/tensor，在我们在进行 tiling or dividing 的时候会比较有用，你其实就把嵌套的括号打开，把这个看作一个多维 tensor 也 OK 的

  Stride 可以说是描述 layout 排布的关键所在，它告诉了我们元素之间的间隔到底是多少。我们可以表示一个 column-major 的矩阵如下

  ```c++
  // Shape :  (4,2)
  // Stride:  (1,4)
    0   4
    1   5
    2   6
    3   7
  ```

  而表示一个 row-major 的矩阵只需要将 stride 反过来即可

  ```c++
  // Shape :  (4,2)
  // Stride:  (2,1)
    0   1
    2   3
    4   5
    6   7
  ```

  在 cutlass 中也称 column-major 为 LayoutLeft，而 row-major 为 LayoutRight。如果你在构建 layout 的时候不指定 stride，将默认使用 LayoutLeft。用一个更加通用的表示 LayoutLeft

  ```c++
  // Shape : (x1, x2, x3, ..., xn)
  // LayoutLeft: (1, x1, x1·x2, ..., x1·x2·x3·...·xn-1)
  ```

- Creation & Use

  可以使用 `make_layout & make_shape & make_tuple` 来创建 layout

  ```c++
  Layout s8 = make_layout(Int<8>{});
  Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
  Layout s2xd4_a = make_layout(make_shape (2,4), make_stride(12, 1}));
  ```

  创建完 layout 过后可以通过 cutlass 内建方法来查看 layout 的一些性质：rank, depth, shape, stride, size

- **Coordinates & Index Conversion/Mapping**

  Coordinates & Index Conversion 可以说就是 layout 最重要的核心逻辑！一共有两种类型的 Mapping:

  1. the map from an input coordinate to the corresponding natural coordinate via the `Shape`
  2. the map from a natural coordinate to the index via the `Stride`

  上面出现了两个新的概念：

  1. input coordinate
  2. natural coordinate

  coordinate 对我来说是熟悉的，其实就是 tensor 的索引，或者说空间中的坐标，加一个 natural 就变得比较迷惑了。个人理解：与其说是 natural coordinate 不如说是 natural layout，因为 coordinate 本身没有什么好变化的，只有当 coordinate 配合 stride 时才有更多的区分。**而 natural layout 其实就是 LayoutLeft**！

  那么什么是 input coordinate? 个人理解：其实是 cutlass 对 natural coordinate 的拓展，或者说泛化。对于一个 n-D shape 的 coordinate，可以转换为其他维度的 coordinate。在 tutorial 中举了一个 3-D shape `(3, (2, 3))` 的 coordinate 在 2-D 和 1-D coordinate 之间的转换

  | 1-D  | 2-D     | Natural     |      | 1-D  | 2-D     | Natural     |
  | ---- | ------- | ----------- | ---- | ---- | ------- | ----------- |
  | `0`  | `(0,0)` | `(0,(0,0))` |      | `9`  | `(0,3)` | `(0,(1,1))` |
  | `1`  | `(1,0)` | `(1,(0,0))` |      | `10` | `(1,3)` | `(1,(1,1))` |
  | `2`  | `(2,0)` | `(2,(0,0))` |      | `11` | `(2,3)` | `(2,(1,1))` |
  | `3`  | `(0,1)` | `(0,(1,0))` |      | `12` | `(0,4)` | `(0,(0,2))` |
  | `4`  | `(1,1)` | `(1,(1,0))` |      | `13` | `(1,4)` | `(1,(0,2))` |
  | `5`  | `(2,1)` | `(2,(1,0))` |      | `14` | `(2,4)` | `(2,(0,2))` |
  | `6`  | `(0,2)` | `(0,(0,1))` |      | `15` | `(0,5)` | `(0,(1,2))` |
  | `7`  | `(1,2)` | `(1,(0,1))` |      | `16` | `(1,5)` | `(1,(1,2))` |
  | `8`  | `(2,2)` | `(2,(0,1))` |      | `17` | `(2,5)` | `(2,(1,2))` |

  那么这些维度的 coordinate 之间是怎么转化的呢？很显然他们应该都有一个相同的 id，使得他们本质是等价的，这个 id 就是 **index**

  计算 index 的方式很简单：就是 coordinate 和 stride 的内积 (inner product)

  ```c++
  // Shape: (M, N, K)
  // Stride: (1, M, MN), LayoutLeft/natural layout
  // Coord: (i, j, k)
  index = i*1 + j*M + k*MN
  ```

  这个公式就能够完成从 coordinate 到 index 的映射，并且通过这个关系，我们也能够在多个维度之间转换得到等价的坐标

  NOTE: 其实 stride 的定义可以非常灵活，例如之前定义过 layout

  ```c++
  Layout s2xd4_a = make_layout(make_shape (2,4), make_stride(12, 1}));
  // (2,4):(12,1)
  ```

  显然这个 stride 非常地不 natural，所计算得到的 index 远远超过了 natural layout 中最大的 index

- Congruent

  所谓的 congruent 就是指的 shape 和 stride 他们 tuple profiles 是一样的。换句话说：shape 有几个维度，stride 就有几个维度

  ```c++
  (2, 2):(1, 2) // congruent
  (2, 2):(3)	// not congruent
  ```

- Print

  cutlass cute 提供了一些 print 函数帮助我们查看 layout 信息，例如 `print_layout & print_latex`

### Layout Algebra

Only focus on the [Divide](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md#division-tiling) and [Product](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md#product-tiling) section

### Tensor

Basically the application of Layout

### Algorithms

Functions that we can use

- [layout algebra](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md)
- tensor
- algorithms

### MMA Atom

> An "MMA atom" in the CUTLASS library refers to a basic building block or operation unit for performing these matrix multiply-accumulate operations efficiently on NVIDIA GPUs.
>
> The term "atom" in MMA atom is used metaphorically to describe a fundamental unit or building block of computation within this context.



- CTA Cooperative Thread Array

  The simplest of the tutorial examples covers the basics of partitioning the global memory into tiles across the CTAs (also called threadblocks in CUDA), partitioning the data tiles across the threads of each CTA, and writing a mainloop using `cute::copy` and `cute::gemm`.

  - `CtaTiler`. A CuTe [tiler concept](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md#composition-tilers) that determines how to extract a tile of data from the problem shape.
  - At the highest level, the work is distributed across CTAs. In principle, each CTA's tile could come from the input tensors in many different ways. Many [CuTe `Tiler`s](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md#composition-tilers) could be used to tile the data, but for these cases it is sufficient to simply use the shape of the desired CTA tile.

- zipped_divide

  本来想就看下 zipped_divide，但可以顺手把 logical divide 给学了，都在 [layout algebra](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md) 里面

- drawing from each level: how gemm is divided from block to thread, and we have different layout to deal with computation (global mem & shared mem)

- From a bigger picture: an intuitive way to think about layout algebra, where is the bottleneck to understanding the process?

Key functions

- `make_tensor`

- `make_coord`

- `make_gemm_ptr`

- `make_smem_ptr`

- `make_shape`

- `make_strid`

- `make_layout`

- `size`

- `local_tile`

  what is `Step` trying to do?

- `local_partition`

  seems like there are reload functions of `local_partition`

- `copy`

- `gemm`

Everything is layout in cutlass: problem layout, block layout, thread layout, memory layout, **use layout algebra to solve them in a unified view**

Other reference

- [CUDA MODE lecture 15](https://www.bilibili.com/video/BV1QZ421N7pT?spm_id_from=333.788.videopod.episodes&p=15) checked, pretty useful
- [Reed's zhihu posts](https://www.zhihu.com/people/reed-84-49/posts), not checked
- [CUTLASS CuTe实战(一)-基础](https://zhuanlan.zhihu.com/p/690703999), not checked

## CUTLASS in Practice

- improve cutlass gemm  [zhihu](https://zhuanlan.zhihu.com/p/707715989) [Reed's zhihu posts](https://www.zhihu.com/people/reed-84-49/posts)
- pick up cutlass examples: interested in all kinds of gemm and kernel fusion
- [CUTLASS CuTe实战(二)-应用](https://zhuanlan.zhihu.com/p/692078624) [github](https://github.com/zeroine/cutlass-cute-sample) this gives examples on optimze gemm and fusing kernel, and most importantly, it gives examples on how to use ncu & nsys to analyize the performance
- cutlass in flash attention

## Questions

- How to Debug? I often see colored squares in screen

- cute and cutlass, which should we choose?

- How to fuse kernels?

- when using `cmake .. -DCUTLASS_NVCC_ARCHS=86` does this equal to `cmake .. -DCUTLASS_NVCC_ARCHS=80`

- CUTLASS assert functions

- dynamic & static difference✅

  dynamic is the value you can get at runtime, the static is the value that can be determined at compile time.

- what is nested tensor mode

- sgemm_2 is 3ms faster than sgemm_1 (15ms v.s. 18 ms), still 5ms to go from 10ms (cuBLAS), how to exceed it? [zhihu](https://zhuanlan.zhihu.com/p/707715989)

- what is mode and major

  seems like mode refers to axis...

- Layout compatibility is not well explained: Why Shape 24 is compatible with Shape (24), but Shape (24) is **NOT** compatible with Shape 24.

- There are 2 async functions that I don't quite understand

  ```c++
  cp_async_fence();        // Label the end of (potential) cp.async instructions
  cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
  __syncthreads();         // Wait for all threads to write to smem
  ```
