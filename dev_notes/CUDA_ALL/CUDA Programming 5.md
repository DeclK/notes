# CUDA Programming 5

## 原子函数的合理使用

在之前的教材中，写了一个 reduce 的归约函数，但是其实是不完整的：其仅仅将每个 block 内的线程进行了累加，各个 block 累加的结果发送到 CPU 上再进行最后的累加

```c++
void __global__ reduce_shared(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    __shared__ real s_y[128];
    s_y[tid] = (n < N) ? d_x[n] : 0.0;
    __syncthreads();

    for (int half_block = blockDim.x >> 1; half_block > 0; half_block >>= 1)
    {

        if (tid < half_block)
        {
            s_y[tid] += s_y[tid + half_block];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[bid] = s_y[0];
    }
}
```

为什么不直接在 GPU 上进行累加呢？这是因为我们每个 block 累加的结构都存放到了 `s_y[0]` 当中，当时我们没办法操控各个 block 的 shared memory 进行正确的累加。我们希望做的操作如下：

```c++
if (tid == 0) {
    d_y[0] = s_y[0] + d_y[0]
}
```

即：将各个 block 中的 `s_y[0]` 都叠加到 `d_y[0]` 上。我们同样面临多线程相互影响的问题：多个线程同时在运行，threadA 读取了 `d_y[0]` 在进行计算的过程中，threadB 同时也在读取 `d_y[0]` 进行计算，他们拿到的都是通一个 `d_y[0]`，那么他们再写入累加过后的 `d_y[0]` 时，必定不可能得到正确的结果。所以我们需要获得一个正确的结果，就需要各个线程在进行操作的时候不能相互影响，也就是要排队进行操作：threadA 做完操作过后，threadB 再做它自己的操作，这样的操作也就是原子函数

> 一个线程的原子操作可以在不受其他线程的 任何操作的影响下完成对某个（全局内存或共享内存中的）数据的一套“读-改-写”操作

所以我们只需要将上面的代码改成原子操作就行了

```c++
if (tid == 0)
{
	atomicAdd(&d_y[0], s_y[0]);
}
```

在实际的编程过程中，无必要则不使用原子操作，因为这会引起排队操作而使程序降速。但是在这个场景中原子操作有相当大的加速作用，在我自己的机器上从 3ms 加速到了 1ms

教材还有一个例子：计算节点邻居，来说明原子函数的使用，这里我就不做整理了

## 线程束基本函数与协作组

- SIMT 模式

  > SIMT, or Single Instruction, Multiple Threads, is a key concept in NVIDIA's CUDA programming model used for parallel computing on GPUs (Graphics Processing Units). It is inspired by SIMD (Single Instruction, Multiple Data) but designed to fit the execution model of GPUs more naturally.
  >
  > While similar to SIMD, which operates on multiple data points with a single instruction, SIMT allows each thread to have its own instruction stream. The term "SIMT" stems from this ability to manage divergence through branch predication and masking.
  >
  > If there is divergence (e.g., different threads within a warp taking different branches of an if-else condition), it serializes the execution of divergent paths, potentially leading to performance reduction known as warp divergence.

  所以 SIMT 更多地代表着 GPU 处理分支发散的特性：在同一时刻，同一个线程束中的线程只能执行一个共同的指令或者闲置，而不会同时处理两种指令。值得强调的是，分支发散是针对同一个线程束内部的线程的。如果不同的线程束执行条件语句的不同分支，则不属于分支发散

- 线程束内的同步函数 `__syncwarp`

  `__syncwarp` 是比 `__syncthread` 更为轻便的同步函数，而且更为灵活，可以通过设置 mask 来控制哪些线程参与同步，而那些不需要

  以之前的 reduce kernel 为例，当折半归约折到 warpsize 大小的时候就可以使用 `__syncwarp` 来替代 `__syncthreads`

  ```c++
  void __global__ reduce_syncwarp(const real *d_x, real *d_y, const int N)
  {
      const int tid = threadIdx.x;
      const int bid = blockIdx.x;
      const int n = bid * blockDim.x + tid;
      extern __shared__ real s_y[];
      s_y[tid] = (n < N) ? d_x[n] : 0.0;
      __syncthreads();
  
      for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
      {
          if (tid < offset)
          {
              s_y[tid] += s_y[tid + offset];
          }
          __syncthreads();
      }
  
      for (int offset = 16; offset > 0; offset >>= 1)
      {
          if (tid < offset)
          {
              s_y[tid] += s_y[tid + offset];
          }
          __syncwarp();
          // __syncthreads();
      }
  
      if (tid == 0)
      {
          atomicAdd(d_y, s_y[0]);
      }
  }
  ```

  其实这里 offset 的条件可以设置到 `offset>=64`，最后的结果也是正确的。实测下来比原来的 `__syncthreads` 快 5%，教材中说快 10%

- 利用洗牌函数 shuffle 完成 Reduce

  线程束洗牌函数作为 CUDA 的内建函数，可以直接调用，我们所使用的是 `__shfl_down_sync` [cuda programming guide reference](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions)

  该函数的参数与功能

  > ```c++
  > T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
  > ```
  >
  > Copy from a lane with higher ID relative to caller
  >
  > Threads within a warp are referred to as *lanes*, and may have an index between 0 and `warpSize-1` (inclusive)

  自己理解：洗牌函数根据参数中的 `delta` 以及当前的线程id `threadIdx` 去获得线程 `threadIdx + delta` 中对应的 `var` 值，并返回这个 `var` 值。这提供了线程束内的交流渠道，利用这个渠道，我们就可以直接完成 warpSize 内的 reduce，所以把上面的 `__syncwarp` 部分替换为 `__shfl_down_sync` 可得到

  ```c++
  void __global__ reduce_shfl(const real *d_x, real *d_y, const int N)
  {
      const int tid = threadIdx.x;
      const int bid = blockIdx.x;
      const int n = bid * blockDim.x + tid;
      extern __shared__ real s_y[];
      s_y[tid] = (n < N) ? d_x[n] : 0.0;
      __syncthreads();
  
      for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
      {
          if (tid < offset)
          {
              s_y[tid] += s_y[tid + offset];
          }
          __syncthreads();
      }
  
      real y = s_y[tid];
  
      for (int offset = 16; offset > 0; offset >>= 1)
      {
          y += __shfl_down_sync(FULL_MASK, y, offset);
      }
  
      if (tid == 0)
      {
          atomicAdd(d_y, y);
      }
  }
  ```

  可以看到，有几处改动：

  1. 将 shared memory `s_y[tid]` 移动到 register 变量 `y` 当中，寄存器是比共享内存更快的存储，能用则用。实测下来使用寄存器可能又会快 10% 
  2. 去掉了 `if(tid < offset)` 的判定，这是由洗牌函数的特性决定的，其能够自动处理线程读写竞争的问题，所以不再考虑 `tid > offset` 的线程先完成导致 `tid < offset` 线程结果出错的问题
  3. 不再需要额外的 sync 函数，这也是洗牌函数的特性，会自动同步线程束内的线程

  除此之外还有其他的洗牌函数这里仅列举，不过多介绍：

  ```c++
  T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
  T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
  T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
  T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
  ```

- 协作组 cooperative group

  这其实是一种新的 CUDA 变成范式，将整个 thread block grid 显式地在代码里表示出来，能够进行直观的操作。下面的代码就是让 cooperative group 去替代之前的 sync & shuffle 指令

  ```c++
  void __global__ reduce_cg(const real *d_x, real *d_y, const int N)
  {
      const int tid = threadIdx.x;
      const int bid = blockIdx.x;
      const int n = bid * blockDim.x + tid;
      extern __shared__ real s_y[];
      s_y[tid] = (n < N) ? d_x[n] : 0.0;
      __syncthreads();
  
      for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
      {
          if (tid < offset)
          {
              s_y[tid] += s_y[tid + offset];
          }
          __syncthreads();
      }
  
      real y = s_y[tid];
  
      thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
      for (int i = g.size() >> 1; i > 0; i >>= 1)
      {
          y += g.shfl_down(y, i);
      }
  
      if (tid == 0)
      {
          atomicAdd(d_y, y);
      }
  }
  
  ```

  可以看待有几处改动：

  1. `this_thread_block()` 显式地创建了 block，这个 block 代表的其实就是该线程所在的 block，我们可以通过该对象的方法获得一些内建变量

     ```c++
     thread_block group = this_thread_block();
     group.thread_index(); // threadIdx
     group.dim_threads(); // blockDim
     group.group_index(); // blockIdx
     group.group_dim(); // gridDim
     group.num_threads(); // total threads
     group.thread_rank(); // Rank of the calling thread within [0, num_threads)
     ```

  2. 使用 `thread_block_tile` 将该 block 切分为更小的 tile (tile group)

     每一个 tile group 也有自己方法来获得相关信息

     ```c++
     thread_block_tile<32> tile_group = tiled_partition<32>(this_thread_block());
     tile_group.num_threads();
     tile_group.thread_rank();
     ```

     同时 tile group 还能使用线程束函数 `shfl_down, any`  等等

- 大数误差

  教材在进行 reduce 的时候，使用了 1e8 个 `1.23` 数进行相加，最终使用 cpu 进行累加得到的结果误差非常大，而 GPU 获得的误差也很大，但是相对误差会小一点 ~0.5%

  ```c++
  // cpu
  sum = 33554432.000000.
  // gpu
  sum = 123633392.000000.
  // my impl gpu, using cooperative group, bad performance
  sum = 124658344.000000.
  ```

- 优化 reduce 线程利用率

  之前的 reduce 代码，每一次都是折半 reduce，所以会有很多线程都是空闲的，线程利用率非常低
  $$
  (\frac{1}{2}+\frac{1}{4}+\frac{1}{8}+...+\frac{1}{128}) \div7≈\frac{1}{7}
  $$
  其实我们可以指定一些空间，然后所有的线程都往这些空间里累加数据，然后再对这些固定空间进行折半 reduce，这样的话线程利用率就大大提高了
  
  我使用以下示意图来直观表示为什么这样做会更快
  
  <img src="CUDA Programming 5/image-20241119174035300.png" alt="image-20241119174035300" style="zoom:80%;" />
  
  左侧就是我们之前的方式，右侧是优化过后的方式。我们假设 GPU 只能 launch 一个 block，并且数据量足够的多，此时就能清楚地计算时延。可以明显的看到在 half reduce 的时间显著的减少了，因为数据提前进行了累加，只需要对单个 block 的数据进行 half reduce
  
  ```c++
  void __global__ optimized_reduce(const real *d_x, real *d_y, const int N)
  {
      const int tid = threadIdx.x;
      const int bid = blockIdx.x;
      extern __shared__ real s_y[];
  
      real y = 0.0;
      const int stride = blockDim.x * gridDim.x;
      for (int n = bid * blockDim.x + tid; n < N; n += stride)
      {
          y += d_x[n];    // sum(d[:, bid, tid])
      }
      s_y[tid] = y;
      __syncthreads();
  
      for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
      {
          if (tid < offset)
          {
              s_y[tid] += s_y[tid + offset];
          }
          __syncthreads();
      }
  
      y = s_y[tid];
  
      thread_block_tile<32> g = tiled_partition<32>(this_thread_block());
      for (int i = g.size() >> 1; i > 0; i >>= 1)
      {
          y += g.shfl_down(y, i);
      }
  
      if (tid == 0)
      {
          atomicAdd(d_y, y);
      }
  }
  ```
  
  运行一下程序得到结果
  
  ```shell
  Time = 0.647755 ms.
  sum = 123007480.000000.
  ```
  
  相比于之前的结果 1.8 ms 提升了非常多，而且精度也提升了非常多，因为如此一来我们进行原子相加的次数也显著减少了，而原子操作是低精度的操作
  
  教材使用了两个 kernel 来直接避免使用 atomicAdd 操作，只需要把上面的 `atomicAdd` 改成如下部分
  
  ```c++
      if (tid == 0)
      {
          d_y[bid] = y;
      }
  ```
  
  即，把每个 block reduce 得到的结果放回全局内存，此时我们通过两个 kernel launch
  
  ```c++
  reduce_cp<<<GRID_SIZE, BLOCK_SIZE, smem>>>(d_x, d_y, N);
  reduce_cp<<<1, 1024, sizeof(real) * 1024>>>(d_y, d_y, GRID_SIZE);
  ```
  
  第一次将数组归约到 `GRID_SIZE` 大小，第二次归约为1
  
  ```shell
  Time = 0.653058 ms.
  sum = 123000064.000000.
  ```
  
  可以看到精度进一步提升！因为我们没有使用任何的原子操作
  
  最后提一下教材中说：不要让一个线程去处理相邻的数据，这样一定会导致非合并访问，GPT 解释如下
  
  > CUDA architecture achieves optimal performance when memory access is coalesced, meaning consecutive threads access consecutive memory addresses. If a single thread accesses scattered, non-coalesced data (such as neighboring elements), it can lead to inefficient memory access patterns and reduced performance.
  
  现在对合并访问有更直观的理解：consecutive threads access consecutive memory
  
- 使用静态全局内存优化 reduce

  之前都是使用动态全局内存，会比静态全局内存更慢，教材选择直接分配一块静态的全局内存给 `d_y`，这样节省了内存的分配与释放

  ```c++
  __device__ real static_y[GRID_SIZE];
  ```

  可以在 kernel 中直接使用该变量，但为了不更改之前的代码我们可以使用 `cudaGetSymbolAddress` 来将这块内存的指针指向我们定义好的变量

  ```c++
  real *d_y;
  CHECK(cudaGetSymbolAddress((void**)&d_y, static_y));
  ```

  该操作将再次优化 10%

  ```shell
  Time = 0.576535 ms.
  sum = 123000064.000000.
  ```

## Question

- 为什么教材中使用了两次 kernel launch 获得的精度比之前要高

  > 这是因为，在使用两个核函数时，将数组 `d_y` 归约到最终结果的计算也使用了折半求和，比直接累加（使用原子函数或复制到主机再累加的情形）要稳健

  总之就是原子函数和 CPU 大数计算都有较大的误差

- 如何使用跨 blockDim 的方式改写 reduce? 这是教材留下的思考题

- 显然当我们拥有很多的数据的时候，会有一些 block 在等待中，此时 atomicAdd 会在干什么呢？
