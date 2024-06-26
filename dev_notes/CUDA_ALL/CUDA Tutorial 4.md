# CUDA Tutorial 4

## E11 基本优化

基本思路：

有效的并行算法 + 针对 GPU 架构的优化 = 最优性能

### 思路1：并行规约

两两求和，指数递减 log(n) 复杂度

注意代码：

1. 代码
2. 线程释放（warp 分割）。**减少分支发散，让 warp 尽早完工**

TODO: 如何理解 warp 切换设备没有代价？

TODO: 许多 warps 在一起可以**隐藏延迟**？我的理解是足够多的任务，能够达到延迟隐藏，隐藏延迟可能需要一个专题以及一些 CUDA 例子

## E12 存储优化

1. 数据传输带宽（Host to Device）远小于 global memory 带宽，两个数量级

   1. 减少传输
   2. 组团传输，避免小块数据频繁传输
   3. 内存传输与计算时间重叠

2. **访存合并**

   global memory 带宽大，但是延迟高

   global memory 默认缓存于一级缓存，可以设置编译器禁用一级缓存（为什么要禁用？数据如何决定是否留在缓存？）

   [合并访问](https://face2ai.com/CUDA-F-4-3-%E5%86%85%E5%AD%98%E8%AE%BF%E9%97%AE%E6%A8%A1%E5%BC%8F/) 合并访问本质上就是说一次内存事务满足线程束（多个线程）的访存需求，因为一次内存事务可以访问的最小粒度包含多个 bit，可以分给这些线程。再换句话说，如果多个线程访问的内存地址是一段连续的，那么就能够尽量少的使用内存事务数量来完成访存操作

   > 内存事务：内存事务（Memory Transaction）是指GPU或其他处理器从内存中读取或写入数据的操作过程。它可以是由一个或多个内存访问请求组成的逻辑单元，用于执行读取或写入操作。
   >
   > 在内存事务中，处理器发送读取或写入请求到内存控制器，并等待内存控制器返回所需的数据或完成写入操作。一次内存事务通常包括以下几个步骤：
   >
   > 1. 发送请求；2. 地址解码；3. 数据传输；4. 完成事务

3. shared memory 用来避免不满足合并条件的访存，进行重排顺序，从而支持合并寻址

### shared memory

shared memory 很快，比 global memory 快上百倍

线程可以通过 shared memory 写作

读入 shared memory 重排顺序 ，从而支持合并寻址**（TODO: 如何重新排序？）**



架构

1. 很多线程访问存储器，所以存储器被划分为 banks
2. 连续的 32-bit 存储空间被分配 banks 当中
3. 每个 bank 每个周期可以相应一个地址
4. 对同一个 bank 进行多个并发访存将导致 bank 冲突，冲突的访存事务需要排队，进行串行执行

什么是 bank？

对于一块连续的地址 0，1，2，3，4，5，6，7，8，... 假设我们有 4 个 bank，

那么 

bank0 对应 0，4，8，...

bank1 对应 1，5，9，...

bank2 对应 2，6，10，...

bank 对应的不是连续的地址，而是跳跃的地址，跳跃间隔为 bank 数量

如果没有 bank 冲突，shared memory 和 register 寄存器一样快！

**快速情况**

1. warp 内所有线程访问同一个地址时没有冲突， 因为有广播机制
2. warp 内所有线程访问不同 banks，没有冲突

慢速情况

多个线程访问同一个 bank 但不同的地址



例子：矩阵转置，读写总有一个是不合并的（数据空间不连续，横着连续，竖着不连续）

通过 shared memory 进行重新排布，并且还需要**解决 bank 冲突**，技巧是加入一个 Empty 占位符，就能解决



### texture memory

1. 特别适用于无法合并访存的场合
2. 线性、双线性、三线性差值
3. 针对越界寻址
4. 以整数或归一化小数作为坐标

### SM 资源的动态分配

以 G80 为例

1. block number：8
2. thread number：768 per block（上下文空间分割导致）
3. registers：8K registers/ 32K bit memory
4. shared memory：16K

这是一个多约束的问题，谁先达到上限，谁就成为约束瓶颈

tutorial 3 中有例子

 多个 block 可以在 SM 上并发执行(驻留)，如果一个 block 在等待，可以启动另一个 block，当blocks 数量远大于 SM 数量，对未来的设备有很好的伸缩性



occupancy 占用率：一个 SM 里面激活 warp 与最大可容纳 warp 数目的比值

延迟隐藏计算，需要多少warp去做



数据预取可以继续优化 matrix multiply



循环展开（缺点是可扩展性不强，还可能出错），指令优化（1<<n 来表示 2**n，一些内置数学函数...）