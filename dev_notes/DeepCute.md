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
