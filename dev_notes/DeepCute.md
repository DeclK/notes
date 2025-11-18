# DeepCute

notes when building kernels

## sm80 gemm

- `SM80_CP_ASYNC_CACHEGLOBAL_ZFILL` 和 `SM80_CP_ASYNC_CACHEGLOBAL` 的区别

  zfill 代表 zero fill，是更常见的用法，如果访问越界元素会自动填充为0，从 DeepSeek 和 Kimi 的回答来看，速度上甚至还更快，所以除非有特别的理由，请优先使用 zfill 版本

- When to use syncthreads?

- What kinds of checks need to be done?

  目前只做了只做了 cta tile 的检查

- `size<0>(CTATile)` 还是 `shape<0>(CTATile)`

  size 得到 int 大小，shape 不是我要的

- 为什么要使用 `void* __restrict__` 作为数据传入类型

  CUDA 驱动层只传 `void*`，写成具体指针类型不会带来任何好处，通过 `void*` 指针配合模板特质类 `GemmTraits`，可以在**编译时**决定实际的数据类型。如果使用具体类型指针，需要为每种数据类型组合都定义不同的 kernel。
  
- 为什么要在 `get_launch_config` 中使用 static

  为了在外面使用 `::` 解析符号来调用，否则这只能通过生成 object 来调用

- lambda 函数可以传入 auto 参数，这是和普通函数的一大区别

- cutlass host tensor 应该如何使用？

- epilouge 中的 pipe 还是有一定作用的，增加 pipe 也许增加了带宽的利用率

- 测试了我自己设置的 `swizzle<1, 3, 3>`，会产生 bank conflict，这是因为在 g2s 阶段写入是按照 32x32 进行写入的，此时会产生 2-way bank conflict，所以设计 smem 的时候需要考虑最大宽度读取情况

- permutation mnk 是以 `make_tile` 定义，而不是使用 `make_layout`

- 似乎现在不需要传入 mma traits & mma atom 给 make tiled mma 了，直接传 mma op 就行，具体特化 traits 会自动生成，我们可以去 `cute/atom` 当中找到对应的 traits，因为也不太好利用 vscode 自动跳转跳过去

## sm90 gemm

- 如何实现 [zhihu]() 中所提到的，处理 scheduler 与 cluster size 不对齐的情况

- 在 sm90 中出现了新的 level: cluster，是否需要重新更新我的 tile centric cuda programming 理论？我想我应该先实现 cooperative gemm，先不去考虑 pingpong，毕竟 deepgemm 本身也没有实现 pingpong

- 在 sm90 当中 cta tile mnk 的设置似乎可以相当灵活，如何设置最优的 mnk 是一个需要弄清楚的问题，在 deepgemm `common.hpp` 当中是有答案的，不过我觉得使用固定的 cta tile mnk 会适合我目前的情况，先不做过多的扩展

- 实际上在定义 atom 的时候，tiled mma & tiled copy 基本就已经被完全定义好了，全部都是高性能的不可分割的统一指令，不需要像 sm80 的 universal copy 一样还需要定义 thr & val layout

  为什么跳转不到正确的 traits 定义上去

  ```cpp
  using mma_op = GMMA::MMA_64x128x16_F16F16F16_SS<ABMajor, ABMajor>;
  using mma_traits = MMA_Traits<mma_op>;
  ```
  
- epilogue 中 shared memory 没有这么大，应该小于 CTATile，应该如何理解？
