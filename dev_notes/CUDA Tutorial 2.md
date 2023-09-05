# CUDA Tutorial 2

针对以下概念进行整理，以便于在今后更好利用 CUDA 加速 GPU 运行：

1. GPU 存储层次结构
2. 数据格式（FP32/FP6/BF16）
3. CUDA stream
4. Cublas (闭源）GPU相关的线性代数库
5. **Cutlass 利用GPU存储层次的高性能GEMM库**

## GPU 存储层次

参考 [CUDA doc](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-interface)	[blog](https://face2ai.com/CUDA-F-4-1-%E5%86%85%E5%AD%98%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0/)	[zhihu](https://zhuanlan.zhihu.com/p/108019839)

Grid cluster 这个概念似乎很少被提及，相当于再次对 grid 进行打包的逻辑，为了方便理解下图，我在此提一下

> CUDA threads may access data from multiple memory spaces during their execution as illustrated by [Figure 6](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy-memory-hierarchy-figure). Each thread has private local memory. Each thread block has shared memory visible to all threads of the block and with the same lifetime as the block. Thread blocks in a thread block cluster can perform read, write, and atomics operations on each other’s shared memory. All threads have access to the same global memory.

<img src="CUDA Tutorial 2/memory-hierarchy.png" alt="Memory Hierarchy" style="zoom: 15%;" />

可以看到 memory 分为了四个等级：

1. **Local Memory** 
2. **Shared Memory** within block (Shared Memory between threads)
3. **Distributed Shared Memory** (Shared Memory between blocks)
4. **Global Memory** (Shared Memory bewteen all GPU kernels)

四个等级的划分依然是按照 thread block grid cluster 打包概念来的。

> There are also two additional read-only memory spaces accessible by all threads: the constant and texture memory spaces.

除此之外还有两个 memory 概念：constant & texture memory，二者提供更快的访问速度，可作为中间缓存 cache 方案

​	

大容量高速的内存不仅造价高，而且不容易生产

一个基本的存储模型

<img src="CUDA Tutorial 2/1-3.png" alt="img" style="zoom: 33%;" />

速度最快的是寄存器，他能和cpu同步的配合；接着是缓存，在CPU片上；然后是主存储器，现在常见的就是内存条，显卡上也有内存芯片；然后是硬盘，这些内存设备的速度和容量相反，越快的越小，越慢的越大

上面是官方的存储结构，下面是博客中所提到的存储结构，也是一张常见的网图，但是我相信 [blog](https://face2ai.com/CUDA-F-4-1-%E5%86%85%E5%AD%98%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0/) 是原始作者

<img src="CUDA Tutorial 2/1-5.png" alt="img" style="zoom:50%;" />

相比于官方的内存，忽略掉了 Distributed Shared Memory 这一个层级，应该是将 blocks 之间的共享内存和全局的共享内存，进行了统一处理

### Registers



寄存器无论是在CPU还是在GPU都是速度最快的内存空间，但是和CPU不同的是GPU的寄存器储量要多一些，**而且当我们在核函数内不加修饰的声明一个变量，此变量就存储在寄存器中**

### Local Memory

核函数中符合存储在寄存器中但不能进入被核函数分配的寄存器空间中的变量将存储在本地内存中

编译器可能存放在本地内存中的变量有以下几种：

- 使用未知索引引用的本地数组
- 可能会占用大量寄存器空间的较大本地数组或者结构体
- 任何不满足核函数寄存器限定条件的变量

**本地内存实质上是和全局内存一样在同一块存储区域当中的**，虽然图里是单独画出来的。其访问特点——高延迟（慢慢慢），低带宽。**(TODO, 把各个存储的速度做一个实际对比)**

Local Memory本身在硬件中**没有特定的存储单元**，而是从Global Memory虚拟出来的地址空间。Local Memory是为寄存器无法满足存储需求的情况而设计的，主要是用于存放单线程的大型数组和变量。Local Memory是线程私有的，线程之间是不可见的。由于GPU硬件单位没有Local Memory的存储单元，所以，针对它的访问是比较慢的。从上面的表格中，也可以看到跟Global Memory的访问速度是接近的。

### Shared Memory

在核函数中使用如下修饰符的内存，称为共享内存：

```cpp
__share__
```

每个SM都有一定数量的由线程块分配的共享内存，**共享内存是片上内存（on chip）**，跟主存相比，**速度要快很多**，也是延迟低，带宽高，并且可以随机访问。其类似于**一级缓存**，但是可以被编程(甚至在GPU 可认为 shared memory就是 L1。（从上面的内存模型中可以看到，主存是在缓存下面的，所以速度更慢）

因为共享内存是块内线程可见的，所以就有竞争问题的存在，也可以通过共享内存进行通信，当然，为了避免内存竞争，可以使用同步语句：

```cpp
void __syncthreads();// barrier
```

使用共享内存的时候一定要注意，不要因为过度使用共享内存，而导致SM上活跃的线程束减少，也就是说，一个线程块使用的共享内存过多，导致其他的线程块没办法被SM启动，这样影响活跃的线程束数量。

SM中的一级缓存，和共享内存共享一个64k的片上内存（不知道现在的设备有没有提高），他们通过静态划分，划分彼此的容量，运行时可以通过下面语句进行设置：

```
cudaError_t cudaFuncSetCacheConfig(const void * func,enum cudaFuncCache);
```

这个函数可以设置内核的共享内存和一级缓存之间的比例

### Constant Memory

每个SM都有专用的常量内存缓存，常量内存使用：

```
__constant__
```

修饰，常量内存在核函数外，全局范围内声明，对于所有设备，只可以声明64k (bit?)的常量内存，常量内存静态声明，并对同一编译单元（Grid?）中的所有核函数可见。

叫常量内存，显然是不能被修改的，这里不能被修改指的是被核函数修改，**主机端代码是可以初始化常量内存的，不然这个内存谁都不能改就没有什么使用意义**了，常量内存，被主机端初始化后不能被核函数修改，初始化函数如下：

```
cudaError_t cudaMemcpyToSymbol(const void* symbol,const void *src,size_t count);
```

同 cudaMemcpy的参数列表相似，从src复制count个字节的内存到symbol里面，也就是设备端的常量内存。多数情况下此函数是同步的，也就是会马上被执行。
当线程束中所有线程都从相同的地址取数据时，常量内存表现较好，比如执行某一个多项式计算，系数都存在常量内存里效率会非常高，但是如果不同的线程取不同地址的数据，常量内存就不那么好了，因为**常量内存的读取机制是：**
**一次读取会广播给所有线程束内的线程。**所以读取不同地址时，会以串行方式读取，而不是以并行方式读取，花费时间为一次读取的 32x，所以这样的速度甚至不如读取全局内存。 

Constant Memory类似于Local Memory，也是没有特定的存储单元的，只是Global Memory的虚拟地址。因为它是只读的，所以简化了缓存管理，硬件无需管理复杂的回写策略。Constant Memory启动的条件是同一个warp所有的线程同时访问同样的常量数据。

### Texture Memory

Texture Memory是GPU的重要特性之一，Texture Memory实际上也是Global Memory的一部分，但是它有自己专用的只读cache。在每个SM的只读缓存中缓存，纹理内存是通过指定的缓存访问的全局内存。这个cache在浮点运算很有用，Texture Memory是针对2D空间局部性的优化策略，所以thread要获取2D数据就可以使用texture Memory来达到很高的性能。从读取性能的角度跟Constant Memory类似。一般来说，下方的内存在 CPU 中是不连续的，所以一般不会存在在缓存当中，但是通过 texture memory 可以一起缓存下来

> Specifically, texture caches are designed for graphics applications where memory access patterns exhibit a great deal of ***spatial locality\***. **In a computing application, this roughly implies that a thread is likely to read from an address “near” the address that nearby threads read, as shown in Figure**

<img src="CUDA Tutorial 2/1.png" alt="img" style="zoom:50%;" />

但在 deep learning 中似乎用的比较少

**TODO: Texture & Constant Memory 他们的缓存是 L1 还是 L2？这个问题有意义吗？**

### Global Memory

Global Memory在某种意义上等同于GPU显存，kernel函数通过Global Memory来读写显存，全局内存GPU上最大的内存空间，延迟最高，使用最常见的内存。**Global Memory是kernel函数输入数据和写入结果的唯一来源**。

全局内存可以动态声明，或者静态声明，可以用下面的修饰符在设备代码中静态的声明一个变量：

```
__device__
```

我们前面声明的所有的在GPU上访问的内存都是全局内存，或者说到目前为止我们还没对内存进行任何优化。
因为全局内存的性质，当有多个核函数同时执行的时候，如果使用到了同一全局变量，应注意内存竞争。

### GPU Cache

GPU上有4种缓存：

1. 一级缓存
2. 二级缓存
3. 只读常量缓存
4. 只读纹理缓存

每个SM都有一个一级缓存，所有SM公用一个二级缓存。一级二级缓存的作用都是被用来存储本地内存和全局内存中的数据，也包括寄存器溢出的部分。

每个SM有一个只读常量缓存，只读纹理缓存，它们用于设备内存中提高来自于各自内存空间内的读取性能。

### 总结

对五种存储（除了 texture 以外）进行整理

|     存储器      |    修饰符     |   变量名称   |
| :-------------: | :-----------: | :----------: |
|    Register     |               |  float var   |
|  Local Memory   |               | float var[5] |
|  Shared Memory  |  \__share__   |  float var*  |
|  Global Memory  |  \__device__  |  float var*  |
| Constant Memory | \__constant__ |  float var*  |

对六种存储的特征进行整理

|     存储器      | 片上/片外 | 缓存 | 存取 |     范围      | 生命周期 |
| :-------------: | :-------: | :--: | :--: | :-----------: | :------: |
|    Register     |   片上    | N/A  | R/W  |   一个线程    |   线程   |
|  Local Memory   |   片外    | Yes  | R/W  |   一个线程    |   线程   |
|  Shared Memory  |   片上    | N/A  | R/W  | 块内所有线程  |    块    |
|  Global Memory  |   片外    | Yes  | R/W  | 所有线程+主机 | 主机配置 |
| Constant Memory |   片外    | Yes  |  R   | 所有线程+主机 | 主机配置 |
| Texture Memory  |   片外    | Yes  |  R   | 所有线程+主机 | 主机配置 |

各个 memory 的速度





## 内存访问

我们本文研究的就是这一个线程束的内存访问，不同线程的内存请求，其目标位置的不同，可以产生非常多种情况。所以本篇就是研究这些不同情况的，以及如何实现最佳的全局内存访问。

<img src="CUDA Tutorial 2/1-1.png" alt="1-1" style="zoom:50%;" />

全局内存是一个逻辑层面的模型，我们编程的时候有两种模型考虑：**一种是逻辑层面的**，也就是我们在写程序的时候（包括串行程序和并行程序），写的一维（多维）数组，结构体，定义的变量，这些都是在逻辑层面的；**一种是硬件角度**，就是一块DRAM上的电信号，以及最底层内存驱动代码所完成数字信号的处理。
L1表示一级缓存，每个SM都有自己L1，但是L2是所有SM公用的，除了L1缓存外，还有只读缓存和常量缓存，这个我们后面会详细介绍。

我们本文研究的就是这一个线程束的内存访问，不同线程的内存请求，其目标位置的不同，可以产生非常多种情况。所以本篇就是研究这些不同情况的，以及如何实现最佳的全局内存访问。
注意：访问可以是加载，也可以是存储。注意我们说的都是读取，也就是加载过程，写或者叫做存储是另外一回事！（**TODO：怎么理解这是另外一回事？**

我们把一次内存请求——也就是从内核函数发起请求，到硬件响应返回数据这个过程称为一个内存事务（加载和存储都行）。

这里总结一下内存事务的优化关键：用最少的事务次数满足最多的内存请求



内存访问有以下特点：

- 是否使用缓存：一级缓存是否介入加载过程
- 对齐与非对齐的：如果访问的第一个地址是32的倍数（前面说是32或者128的偶数倍，这里似乎产生了矛盾，为什么我现在也很迷惑）
- 合并与非合并，访问连续数据块则是合并的







## Matrix Multiplication Example

[bilibili](https://www.bilibili.com/video/BV1kx411m7Fk?p=9)

TODO: 我们使用 malloc 是将变量放在哪儿了？Global or Local

TODO: 一些硬件资源极限的计算

TODO: 在 shared memory 中使用 _syncthreads 的作用？ Mds[i] = Mds[i] + Mds[i + 1]

**TODO: shared memory 与 global memory 测速**

TODO: shared Memory **bank 冲突**

TODO: 什么是访存模式，什么是随机访存，什么是访存合并



## Matrix Transpose

shared memory 优化，实现合并

## 存储优化

[bilibili](https://www.bilibili.com/video/BV1kx411m7Fk?p=12)



## CUDA Runtime APIs

> The CUDA Runtime API provides functions for memory management, device initialization, device queries, context management, and launching kernel functions.
>
> The runtime is implemented in the `cudart` library, which is linked to the application, either statically via `cudart.lib` or `libcudart.a`, or dynamically via `cudart.dll` or `libcudart.so`. 

这些 CUDA Runtime APIs 是 CUDA 编程的一部分，这些 APIs 由 `cudart` 库提供，该库在 `nvcc` 默认的路径中，所以可直接调用。之前所使用的 `cudaMalloc` 就是其中一个接口，用于分配 GPU 内存

## Points

线程发散的原理

跳动的访问，不理解为什么会低效，换句话说，连续访问为什么高效（**合并访存**

SM 资源分割（需要理解载入/驻扎和执行的区别，寄存器数量与 threads 所需寄存器 占用率计算器 如何理解occupancy calculator

指令优化：

1. 例如使用位运算来替代2的指数。通常先担心存储优化最后来看指令优化
2. 循环展开 `#pragma unroll SIZE`，缺点：可扩展性差
3. 精度减半
