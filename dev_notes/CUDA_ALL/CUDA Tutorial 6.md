# CUDA-Programming

对整个 cuda 的基础知识进行梳理和总结。之前看过不少的教程，对于基础部分应该只有少数查漏补缺的

通过 《CUDA 编程：基础与实践》作为大纲完成 Concept-Layout-Question-style 总结

- 核函数参数将作为 local memory 存储在各自的线程当中，而核函数参数使用 const int N，内存应该是存放在常量内存中，但是常量内存的大小总量是有限的。

- `__syncthreads` 能够将一个 block 中的线程进行同步，但是不能对不同 block 的线程进行同步。**通常如果发现在核函数中线程之间的读写可能存在依赖**，就需要使用！

- 为了避免 bank 冲突的章节中，是否能够采用访问行线程束，而不是采用访问列的线程束；答案是不行的，因为如果合并读取那就意味着是行的读取，而写入的时候就必须是列的写入，这会导致写入不合并

- shared memory 不一定能完全提高性能！具体显卡还需要具体测试，这可能就是为什么 TensorRT 需要测试各种算子速度的原因了

- 在同一个 warp 里的线程，执行顺序是一定的，但是不同 warp 执行顺序会不一样

  ```c++
  #include <stdio.h>
  
  /****** try BLOCK_SIZE=64, it would be different *****/
  const unsigned BLOCK_SIZE = 16;
  
  void __global__ test_warp_primitives(void);
  
  int main(int argc, char **argv)
  {
      test_warp_primitives<<<1, BLOCK_SIZE>>>();
      CHECK(cudaDeviceSynchronize());
      return 0;
  }
  
  void __global__ test_warp_primitives(void)
  {
      int tid = threadIdx.x;
      printf("%2d ", tid);
  }
  ```

- `__shfl_sync` 是一个洗牌指令（可能因为它开头为 shuffle 的原因吧），其作用是在线程束内进行通信计算。在介绍洗牌指令前，先介绍一个叫 lane id 的概念：对于一个线程束，包含32个线程，这个线程束可以被分为多个均匀 lane，每一个 lane 的长度 w 可以为 2, 4, 8, 16, 32

  举一个例子，一个线程束可以分成为 4 个 w=8 的 lane，每一个 lane 中的线程 id 为 0, 1, 2, 3, 4, 5, 6, 7

  下面介绍 `__shfl_sync(mask, v, srcLane, w)` 的作用：

  mask 代表哪些线程将参与同步，在线程束中 `lane_id=srcLane` 的线程，会将自己线程中的变量 v，广播到自己所属的 lane 当中。w 就是 lane 的宽度

- 在周斌的讲义中说道，尽量避免使用原子函数的使用，应该用什么方法进行替代？原子函数将对线程进行排队，一个线程一个线程轮流操作，从而保证读取数据不产生冲突和错误。但是线程的排队顺序没有讲究，会是混乱的，这将导致一些随机性

- 最后的 reduce 优化！相当于要使用固定的“桶”去装一个很大很大的数组，这样就能使用 shared memory 进行快速的叠加，然后再对这些固定的“桶内”进行折半叠加，然后再对所有的桶进行折半叠加。相比于用变化的桶的数量，在桶内进行折半叠加，我们用 shared memory 的叠加更快，本质上是增加了线程利用率！

- 为什么单精度的 reduce 误差这么大！？

- 在进行多个 cudaStream 程序所受的限制：

  1. 所有 cudaStream 的核函数所使用的线程超过了**某一个数值**，这时瓶颈在于 GPU 的计算资源不够。即不能再支持更多的 cuda stream (kernel function)，再增加就得排队，就得等

     在我们的测试结果中，这个**某一个数值**并不是 GPU 所允许的最多线程数量。并且这个数值要小得多，这和教材中的现象是一致的。此时这个线程数阈值为 1024 * 16

     一旦超过了这个线程数，速度就会发生显著增加。这个数值我猜测应该与 GPU 能够同时运算（不是驻扎）的线程数量相关

     <img src="CUDA Tutorial 6/image-20231030160124772.png" alt="image-20231030160124772" style="zoom:50%;" />

     当不使用 cuda stream 来同时运行多个 kernel function，我直接增大了 thread numbers，获得了和上面类似的阶梯图像，不过最大耗时要少一些。我减少了 kernel function 的运行时间，增加了线程数量，最大线程数量为 1024 *128

     <img src="CUDA Tutorial 6/image-20231031105256286.png" alt="image-20231031105256286" style="zoom:50%;" />

     这说明之前的猜测是正确的，我可以计算出在当前核函数条件下，最大的 active thread 大约在 8 * 1024 = 8192 附近。但这个阶梯在运行初期是不稳定的，其中原理不得而知

     一般来说提供足够多的线程有利于掩藏获取数据延迟，但是这里我们使用的核函数就是简单的加法，所有的数据在一开始就分配好了，不需要额外的获取数据。所以这个实验更能说明最大的 active thread 数量

  2. GPU 支持的同时并发的核函数数量是有限的，对不同的 Compute Capacity 设备这个数值是不一样的。在我当前的 3080 上显然这个数值是 128

     <img src="CUDA Tutorial 6/image-20231030114048468.png" alt="image-20231030114048468" style="zoom:50%;" />

     超过 128 过后，时间就变成两倍了。注意这里我调低了每一个 block 的线程数量(=32)以寻找最多的 cuda stream 数量	

- 