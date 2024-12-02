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

似乎应该使用 `-DCUTLASS_NVCC_ARCHS=80`? 即使我的 compute capacity 是86

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

- What does this code trying to do?

  这基本上就是在教你如何用并行的思维去处理矩阵乘法。我先从逻辑视角完成这件事情，然后再将这些逻辑映射到代码当中，看下代码的具体实现

  为了简单起见，我们首先定义问题为最简单的矩阵乘法：$C = AB$，他们形状分别为 `A(M,K) & B(K,N), C(M,N)`，为了让问题更加具象化，我们以 `M=N=5120, K=4096` 为例子（这也是 cutlass 代码例子中所使用的数值）

  一个不错的划分视角是以矩阵 $C$ 为核心：我们将矩阵 C 进行划分，以 $(128,128)$ 的方块作为划分单位，去单独求解每一个方块中的矩阵乘法结果。从 CUDA 编程的角度来讲：我们让一个 block 去处理一个 $(128,128)$ 的矩阵乘法结果

  TODO: 插图

  好，现在就可以集中精力来处理每一个 block 应当如何计算了

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

## CUTE Tutorials

- [CUDA MODE lecture 15](https://www.bilibili.com/video/BV1QZ421N7pT?spm_id_from=333.788.videopod.episodes&p=15)
- [Reed's zhihu posts](https://www.zhihu.com/people/reed-84-49/posts)
- [CUTLASS CuTe实战(一)-基础](https://zhuanlan.zhihu.com/p/690703999)

主要总结 Concepts

- [layout algebra](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/02_layout_algebra.md)
- tensor
- algorithms

## CUTLASS in Practice

- improve cutlass gemm  [zhihu](https://zhuanlan.zhihu.com/p/707715989) [Reed's zhihu posts](https://www.zhihu.com/people/reed-84-49/posts)
- pick up cutlass examples: interested in all kinds of gemm and kernel fusion
- [CUTLASS CuTe实战(二)-应用](https://zhuanlan.zhihu.com/p/692078624) [github](https://github.com/zeroine/cutlass-cute-sample) this gives examples on optimze gemm and fusing kernel, and most importantly, it gives examples on how to use ncu & nsys to analyize the performance
- cutlass in flash attention

## Questions

- How to Debug? I often see colored squares in screen
- cute and cutlass, which should we choose?
- How to fuse kernels?
- What are the basic tools? and maybe basic types?
- when using `cmake .. -DCUTLASS_NVCC_ARCHS=86` does this equal to `cmake .. -DCUTLASS_NVCC_ARCHS=80`
- CUTLASS assert functions
- dynamic & static difference
- what is nested tensor mode
- why do we try to divide the K dimension into smaller k?
- sgemm_2 is 3ms faster than sgemm_1 (15ms v.s. 18 ms), still 5ms to go from 10ms (cuBLAS), how to exceed it? [zhihu](https://zhuanlan.zhihu.com/p/707715989)
- what is mode and major

  seems like mode refers to axis...
