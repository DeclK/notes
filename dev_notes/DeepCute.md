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

## sm90 gemm

