# CUDA Tutorial

整体的学习路径如下：

1. ~~Basic Concept - 2day~~
2. Cases
   1. ~~VectorAdd - 0.5 day~~
   2. **Trilinear Interpolation - 0.5 day (pytorch extension pipline)**
   3. **BEV Pool - 0.5 day**
   4. **Plugin Exercise, use bev pool as example - 0.5 day**
   5. Deformable Attention - 2 days
   6. Flash Attention - 2-days
3. Projects
   1. **Cutlass learning - 2days**
   2. GPU Puzzles - 0.5 days
   3. hpc-kernel-libs - 0.5 days
   4. Faster Transformer Swin - 1 days
4. Problems
   1. 给出Res50 在A10 上的理论分析，同时利用nsight compute 验证理论模型
   2. 实现Gemm 不使用第三方库，到达cublas 的90%的性能

我打算首先通过学习基本语法，掌握 CUDA 中的关键概念，然后再通过常用的算子，补充额外的知识点。选择关键问题集中火力处理，这样就不被冗长的文档所困扰

然后通过学习两个项目进一步增长实战经验（细节！）

最后完成 mentor 布置的任务

## 基础概念

参考 [zhihu-CUDA编程入门极简教程](https://zhuanlan.zhihu.com/p/34587739) 进行整理

来一张 CPU / GPU 芯片设计图

<img src="CUDA Tutorial/gpu-devotes-more-transistors-to-data-processing.png" alt="The GPU Devotes More Transistors to Data Processing" style="zoom: 33%;" />

目前我能理解的 CPU GPU 的区别，`/` 左侧代表一般设备，右侧代表高端设备

|                      | CPU                    | GPU                 |
| -------------------- | ---------------------- | ------------------- |
| 核(Core, NOT kernel) | 核少，一般6核/高端64核 | 核多，一般2K/高端4K |
| TFLOPS               | 0.2 / 1                | 10 / 100            |
| 内存/显存带宽        | 30 GB/s / 100 GB/s     | 400 GB/s / 1 TB/s   |
| 控制流               | 强                     | 弱                  |

1. 核多是 GPU 适合于计算密集任务的最重要原因，虽然 CPU 在一个核计算快，但是架不住 GPU 核多，这也是 GPU TFLOPS 大的最主要原因

   CPU 在单核快于 GPU 很大部分是因为 clock speed 更快，即频率更快。而 GPU 由于计算密集，想要达到这样的频率，可能会使得 GPU 发热非常严重

2. GPU 内存带宽（也叫显存）大于 CPU 内存带宽的原因之一是：Memory Bus Width 更大。Memory Bus Width 基本上描述了显存与 GPU 之间的数据通道的宽度。可以将其想象为一个高速公路：这个“公路”上的“车道”越多（即Memory Bus Width越大），能够同时传输的“车辆”（数据）就越多

   这也是为什么 GPU 显存更贵的原因，这也是显存无法像内存一样做到 1TB 的原因之一（Highway is expensive!）

3. 控制流强弱，最直观的体现就是 CPU 用于控制的芯片区域更大（黄色部分）更本质的原因是 CPU 的设计注重于复杂串行指令执行，为了让串行更快，人们设计了很多复杂的控制流，例如调整指令顺序、分支预测等等来加速串行指令的执行

**总之 GPU 设计的一切都是为了计算密集型任务/大规模并行任务！**基于 CPU+GPU 的异构计算平台可以优势互补，CPU 负责处理逻辑复杂的串行程序，而 GPU 重点处理数据密集型的并行计算程序，从而发挥最大功效

**CUDA (Compute Unified Device Architecture)**，是由 NVIDIA 创建的并行计算平台，其提供了相应的程序接口，能够让我们对 GPU 进行操作以获得计算加速

### Host and Device

在 CUDA 中，**host** 和 **device** 是两个重要的概念，我们用 host 指代 CPU 及其内存，而用 device 指代 GPU 及其内存。CUDA 程序中既包含 host 程序，又包含 device 程序，它们分别在 CPU 和 GPU 上运行。同时，host 与 device 之间可以进行通信，这样它们之间可以进行数据拷贝。

典型的CUDA程序的执行流程如下：

1. 分配 host 内存，并进行数据初始化；
2. 分配 device 内存，并从 host 将数据拷贝到 device 上；
3. 调用 CUDA 的**核函数（kernel）**在 device 上完成指定的运算；
4. 将 device 上的运算结果拷贝到 host 上；
5. 释放 device 和 host 上分配的内存

### Kernel

Kernel 是在 device 上线程中并行执行的**函数**，核函数用 `__global__` 符号声明，在调用时需要用 `<<<grid, block>>>` 来指定 kernel 要执行的线程数量，在 CUDA 中，每一个线程都要执行核函数，并且每个线程会分配一个唯一的线程号 thread ID，这个 ID 值可以通过核函数的内置变量 `threadIdx` 来获得

这里我还说明一下这里的核函数与之前提到的核数量的区别 (kernel & core)

> The Kernel: It is a software layer that sits between the hardware and the user-level software. The kernel manages system resources, orchestrates the execution of processes, handles input/output (I/O) operations, provides security mechanisms, and more.
>
> The Core: The core refers to the processing unit within a central processing unit (CPU). A CPU typically consists of multiple cores. Cores are responsible for carrying out the actual computational work of the CPU.

总结：核函数是软件，调用硬件；CPU/GPU 核是硬件，为计算真正发生的地方

### 函数类型

由于 GPU 实际上是异构模型，所以需要区分 host 和 device 上的代码，在 CUDA 中是通过函数类型限定词开区别 host 和 device 上的函数，主要的三个函数类型限定词如下：

1. `__global__` 函数。使用 `__global__` 限定的函数，通常也成为核函数（kernel），其表明该函数是从 host 中调用在 device 上执行，返回类型必须是`void`，不支持可变参数参数。并且由于是异步进行，host 不会等待 kernel 执行完成才进行下一步
2. `__device__` 函数。表明该函数为 device 代码，只能运行在 GPU 上，其只能由 `__global__` 函数调用或者由另一个 `__device__` 函数调用
3. `__host__` 函数。表明该函数为 host 代码，只能运行在 CPU 上，退化为平时所写的普通 C++ 代码

### Grid, Block, Thread

首先我需要定义：在 GPU 中什么是 thread？

> In GPU computing, a thread refers to an individual unit of execution that performs a specific task or set of instructions based on a given kernel. 
>
> Threads are the fundamental building blocks of parallel execution on a GPU.

我的理解：在 kernel 发起的并行计算中，thread 就是最小执行单元

有了上述基本概念，我们只需要发起大量的 thread 就能完成并行计算了。为了更方便对 thread 进行管理，我们抽象除了 block & grid 两个逻辑概念，二者就是对 thread 的打包：

1.  一个 block 可包含多个 thread，通常由于资源限制一个 block 最多支持 1024 个 thread
2. 一个 grid 可包含多个 block

其示意图如下

<img src="CUDA Tutorial/v2-aa6aa453ff39aa7078dde59b59b512d8_720w.png" alt="img" style="zoom: 67%;" />

可以看到，示意图使用了矩阵式的打包方式。图中，一个 block 由一个 `(x, y) = (5, 3)` 形状的 thread 矩阵构成；一个 grid 由一个 `(x, y) = (3, 2)` 的 block 矩阵构成（这里使用像素坐标系）。这样的定义是非常自然的，非常方便我们对输入张量进行 element-wise 操作

而实际上我们不仅可以用二维矩阵来进行打包，还可以用一维、三维的矩阵来进行打包，其他维度的打包方式是不可行的，此时你需要重新考量你的代码实现

我们通过 `dim3` 这个类型来定义打包方式，其可看作3个无符号整数 `(x, y, z)` 成员的结构体变量，在定义时，缺省值初始化为1

```c++
dim3 grid(3, 2);	// 2-dim pack, our example
dim3 block(5, 3);
    
dim3 grid(3);	// 1-dim pack, usually we directly use int
dim3 block(5);
    
dim3 grid(3, 2, 2);	// 3-dim pack 
dim3 block(5, 3, 4);
```

kernel在调用时也必须通过语法 `<<<grid, block>>>` 来指定kernel所使用的线程结构

```c++
kernel_function<<<grid, block>>>(params...);
```

这里要注意：`grid` 实际上填入的是组成 grid 的 block 形状，而不是 grid 的形状；同理 `block` 代表的是组成 block 的 thread 形状。一个更确切的表示为

```c++
kernel_function<<<blocksPerGrid, threadsPerBlock>>>(params...);
```

#### Example

下面使用2个简单例子来说明：

1. 一维张量的 element-wise add
2. 二维张量的 element-wise add

首先是一维张量的 element-wise add

```c++
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x; // get thread ID
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```

可以看到我们需要一个独特的 thread ID 来告知每个线程其对应的数据位置，下面是一个更准确的解释

> **Each thread that executes the kernel is given a unique *thread ID* that is accessible within the kernel through built-in variables.**

`threadIdx` 和 `dim3` 应该都是 CUDA 编程中的内置类型/变量

以上内容很好理解，下面来看二维张量的情况，这里就要使用 `dim3` 来定义 `threadsPerBlock` 以操作二维张量

```c++ 
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;	// get threadID at x axis
    int j = blockIdx.y * blockDim.y + threadIdx.y;	// get threadID at y axis
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
```

这里多了两个变量 `blockIdx` 和 `blockDim`，其中 `blockIdx` 和 `threadIdx` 是相似的意义，代表了 block 的位置；而 `blockDim` 实际上就代表了其维度大小，在本例中 `blockDim.x = 16`，因为我们定义了 `dim3 threadsPerBlock(16, 16)`

由此就非常好理解 `int i & int j` 的含义了，他们就是该 thread 在**全局的位置**。自己画一个图就明白了，并注意 index 一定是从0开始的🤔

### SP, Warp, SM

参考 [zhihu-理解CUDA中的thread,block,grid和warp](https://zhuanlan.zhihu.com/p/123170285)

这部分是对 GPU 的硬件介绍，并且将硬件概念与软件概念进行联系与区分

**SP(Streaming Processor)：**流处理器， 是 GPU 最基本的**处理**单元，也叫做 **CUDA core**

**SM(Streaming MultiProcessor)：流式多处理器**， 一个 SM 由多个 CUDA core 以及共享内存、寄存器、调度器等硬件组成，CUDA core 在 SM 中占据了大部分的比例。**每个 SM 根据 GPU 架构不同有不同数量的 CUDA core**，例如 4090 有 128 个 SM，一个 SM 有128 个 CUDA core（一共16384个 CUDA core），现在还诞生了新的概念：**Tensor Core**，其专注于矩阵运算与混合精度运算

以上两个概念就是从硬件出发来看待 GPU，现在我们需要把硬件和 thread/block/grid 联系起来，thread/block/grid 可看作从软件角度出发来看待 GPU，他们的联系如下：

1. 一个 CUDA core 可以执行一个 thread
2. 一个 block 只能在一个 SM 上执行，不能跨 SM

以上两点基本上是非常好理解的，也是正确的！但我还需要补充的是：实际上 GPU 在执行 thread 的时候，并不是以 thread 为单位，而是以 warp 为单位

**Warp：**线程束，一个 warp 包含 32 个 thread，是 GPU 最基本的**执行**单元，是 SIMT 架构 (Single-Instruction, Multiple-Thread) 的直接体现

引入了 warp 概念过后，就有两个疑问需要解决：

1. 为什么 warp 是最基本的执行单元，之前不是说 thread 才是最小的执行单元吗？

   这里玩了个文字游戏：小和基本😂但我们还是讲清楚其中的区别即可。这里仅用1个例子即可说明：一个 warp 包含 32 个线程，但是当你想要并行处理 33 个线程的时候，你必须要启动 2 个 warp 来处理，此时会启动 64 个线程，其中运行的 33 个线程占用了 33 个 CUDA core，为活跃状态（active），而剩下的 31 个线程也占用了 31 个 CUDA core，其计算的结果没有意义，称为不活跃状态（inactive）

   虽然上述过程看起来没有效率，浪费了 CUDA core，实际上以 warp 作为基本执行单元相比于 single-thread 作为执行单元更高效的：

   1. 有更少的 instruction。以 thread 为基本执行单元就要为每一个 thraed 配置一个 instruction，降低了效率
   2. 更快的内存获取。以 warp 为单位进行内存访问能一次性获得更多的连续空间

2. warp 也是打包 thread，block 也是打包 thread，二者的区别是什么？

   warp 是物理执行层面上的打包，而 block 是逻辑层面上的打包。二者的打包也并不冲突，我们可以通过下面的公式计算一个 block 中所包含的 warp 数量
   $$
   WarpPerBlock=\text{ceil}(\frac{ThreadPerBlock}{WarpSize})
   \\
   WarpSize=32
   $$
   所以通常 block 所含的 thread 数量一般为 32 的倍数

   而为什么说 block 是逻辑层面上的打包并行，是因为 block 可以打包很多个 thread，但我们之前说过，一个 block 只能在一个 SM 上执行，而一个 SM 只包含 128 个 CUDA core，**这就会面临硬件资源不够的情况**。当线程需要使用比可用资源更多的 CUDA core 或者寄存器或共享内存时，它们可能需要等待其他线程完成，以便空出资源

总结下来就是这样一张图（自己根据理解绘制，与实际有出入）

<img src="CUDA Tutorial/image-20230822145030453.png" alt="image-20230822145030453" style="zoom:67%;" />

使用下面代码检查了 A10 的一些配置

```c++
#include <iostream>

int main(){
  int dev = 0;
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, dev);
  std::cout << "GPU Device" << dev << ": " << devProp.name << std::endl;
  std::cout << "Num SM: " << devProp.multiProcessorCount << std::endl;
  std::cout << "Shared Memory Per Block: " << devProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
  std::cout << "Max Threads Per Block: " << devProp.maxThreadsPerBlock << std::endl;
  std::cout << "Max Threads Per SM: " << devProp.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "Max Warps Per SM: " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
}
// GPU Device0: NVIDIA A10
// Num SM: 72
// Shared Memory Per Block: 48 KB
// Max Threads Per Block: 1024
// Max Threads Per SM: 1536
// Max Warps Per SM: 48
```

## Hello World

参考 [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) 进行整理

在 C++ 中我们使用了 g++ 编译器来将 C++ 语言翻译（更准确说是编译）成为了 CPU 能够执行的二进制文件，对于 CUDA 编程也是类似的。我们需要使用一个编译器，来将所写的 CUDA 代码 翻译/编译成为 GPU 能够执行的文件，而这个编译器就是 **nvcc**，之前对 nvcc 的了解仅限于使用 `nvcc -V` 查看 CUDA 版本🤣

nvcc 对于 C/C++ 是兼容的，所以即使是纯粹的 C/C++ 代码也可以编译

```c++
// hello_world.cu

#include <iostream>
int main(){
    std::cout << "Hello World!" << std::endl;
}
```

使用 nvcc 编译并运行

```shell
nvcc hello_world.cu -o hello_world
./hello_world
```

恭喜你！完成了第一个 CUDA 代码🎉🎉🎉

## VectorAdd

### multi-thread

下面写一个 `VectorAdd` CUDA 代码，其功能是将两个长度为 N 的数组相加。

```c++
#include <iostream>
#include <math.h>

__global__ void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}
int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the GPU
  add<<<1, 256>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
}
```

以上的 `main` 函数主题并不难理解，这里主要说一下核函数 `add`：

1. 首先获得了线程 id，以及每个 block 的线程数 `blockDim.x`
2. 每一个线程，以 `blockDim.x` 作为步长，进行相加操作

之前我还在疑惑为什么要使用 id 这样的方法，通过这个例子就解除了疑惑：id 本身没有太多意义，其就是一个 int 类型，**关键在于我们需要利用 id 来将任务进行合理的划分**，使得整个任务通过多线程得到完成。在这个例子里我们使用 id 作为获得数据的 index，然后通过设置步长，使得整个过程没有重叠。但实际上我最初的想法是使用一个线程来处理连续的一块数组，其实这个想法也是可以实现的！不过没这么简洁

```c++
__global__ void my_add(int n, float *x, float *y)
{
  // my solution, each thread would add consecutive elements
  int index = threadIdx.x;
  int stride = n / blockDim.x;
  int start = index * stride;
  // spetial case for the last thread
  if (index == blockDim.x - 1)
    stride = n - start;
  for (int i = start; i < start + stride; i++)
    y[i] = x[i] + y[i];
}
```

### multi-block

上面的 kernel 只用了多线程，blcok 数量只有一个，我们可以使用多个 block 来进一步加速。我们只需要调整 index 以及步长即可完成

```c++
#include <iostream>
#include <math.h>

__global__ void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // ********** utilize the block ***********
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  std::cout << "numBlocks: " << numBlocks << std::endl;
  add<<<numBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
}
```

这张示意图将广为流传，其中 `4096 = N / blockDim = 2 ** 20 / 256`

<img src="CUDA Tutorial/cuda_indexing.png" alt="&quot;Figure" style="zoom:67%;" />

然后使用 nvcc 编译并运行

```shell
nvcc add.cu -o add
./add
# output
# Max error: 0
```

### No cudaMallocManaged

在以上代码里我们使用 `cudaMallocManaged` 方法来分配内存，该方法能够让 CPU 和 GPU 同时分配内存，这样写感觉很方便，但是写出来的代码并不快。因为我是在 A10 上进行运行，而所阅读的教程已经是 6 年前的教程，但是通过 profile 发现，我堂堂 A10 的运行速度竟然没有教程中 GT 740 快😨

后来使用了另一个版本，是使用 `cudaMalloc & cudaMemcpy` 单独对 GPU 操作内存，速度就起飞了

```c++
#include <iostream>

__global__ void add(float* x, float * y, float* z, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 20;
    int nBytes = N * sizeof(float);
    
    // cpu malloc
    float *x, *y, *z;
    x = (float*)malloc(nBytes);
    y = (float*)malloc(nBytes);
    z = (float*)malloc(nBytes);

    // init
    for (int i = 0; i < N; ++i)
    {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    // gpu malloc
    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, nBytes);
    cudaMalloc((void**)&d_y, nBytes);
    cudaMalloc((void**)&d_z, nBytes);

    // copy data from host to device, i.e. CPU -> GPU
    cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);

    // kernel config
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // kernel launch
    add <<< gridSize, blockSize >>>(d_x, d_y, d_z, N);

    // copy data from device to host, i.e. GPU -> CPU
    cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);

    // check result
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    std::cout << "Max Error: " << maxError << std::endl;

    // release device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    // release host memory
    free(x);
    free(y);
    free(z);

    return 0;
}
```

使用 `cudaMallocManaged` 会降低 kernel 的运算速度，（为什么？）速度更慢。使用 nsys 来对程序运行时进行分析

```shell
nsys profile --stats true  -o profile -f true --trace cuda ./add
```

两种方式的速度比较如下

```c++
// ** CUDA GPU Kernel Summary (gpukernsum):

// cuMalloc 
Time (%)  Total Time (ms)  Instances  Avg (ms)  Med (ms)  Min (ms)  Max (ms)	Name
--------  ---------------  ---------  --------  --------  --------  --------  -----------------------------------
   100.0           0.0242          1    0.0242    0.0242    0.0242    0.0242	add(float *, float *, float *, int)
    
// cuMallocManaged
Time (%)  Total Time (ms)  Instances  Avg (ms)  Med (ms)  Min (ms)  Max (ms)	Name                
--------  ---------------  ---------  --------  --------  --------  --------  -----------------------------------
   100.0           1.9878          1    1.9878    1.9878    1.9878    1.9878   add(float *, float *, float *, int)

```

可能 cuMallocManaged 计算了数据从 cpu 输送到 gpu 的时间

同时这里再解释一下使用双重指针 pointer to pointer `void**` 的意义：**非常简单，因为我们需要改变指针的地址**

我们在 cpu 上创建了一个 `float* d_x = nullptr` 指针，然后我们在 gpu 上也分配了一段相应的内存，这时候，假设 gpu 内存的首地址就为 `d_x_gpu`。显然我们需要将 `d_x` 的值修改为 `d_x_gpu`，而且这个修改是在函数中完成的，那么就需要指针的指针。就像修改变量的值，我们需要变量的指针一样 
