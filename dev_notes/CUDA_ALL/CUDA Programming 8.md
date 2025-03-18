# CUDA Programming 8

之前的 CUDA Programming 7 已经很长了，本来想就在 7 就终结。但是没办法，需要弄清楚的事情仍然很多！这篇笔记要整理的是 cutlass cute 当中的基础概念

感觉现在即使是 DeepSeek 也对 cutlass 没有太多的头绪，我只能够不断地从 zhihu 博客中学习，配合以 DeepSeek 进行一些简单的推理来构建清晰的 cutlass 如何抽象 GPU 的模型

- ldmatrix

  how a thread carries data, this needs allocate registers for each thread

  为什么使用 LDS 需要使用 4 条指令加载，而使用 ldmatrix 只需要一条指令就能够加载完成从 shared memory -> register 的过程

  通过各个线程提供的寄存器可以完成warp level的矩阵表示和存储，利用Tensor Core则可以完成高效的存储在寄存器上的矩阵计算。就数据从共享内存到寄存器的加载方面而言，可以通过SIMT意义下的LDS（load shared）来完成，但是由于数据是分布在不同的线程的寄存器上连续性方面不友好。

  如何理解上面这个句子中的不连续性？

  [tensorcore中ldmatrix指令的优势是什么？](https://www.zhihu.com/question/600927104/answer/3029266372)

  每一个 thread 可以最多操作 128 bit 的 register/shared memory 数据吗？如何让这个说法更加准确

- mma

  为什么 mma 的 register 数据分布和 shared memory 数据分布不一样，是 scatter 的，又如何保证计算的正确性？连续的数据被打散了

- tiledmma

  print latex to see data and thread

  在扩展时是否有限制，显然不能无限制地进行扩展，Threads 数量和 Vectorized Register 数量都是有限制的

  如何去推理整个 permute layout 的数据分布？ 

  为什么在 reed 的 gemm-simple 中 permutemnk 是 (1, 2, 1) 而不是具体的 shape，其中是否有什么隐藏等价变换？

- vectorized register 是可以变化的 [nv-doc](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-fragments-for-mma-m16n8k16-with-floating-point-type)

  对于 16x8x16 的 mma 来说，每一个 thread 持有一个 vectorized register，该 vectorized register 包含了 4 个 `.f16x2` 的 registers，也就是说每一个 register 涵盖了 2 个 fp16/bf16 的元素。再换句话说，每一个 thread 将会持有 8 个 fp16 元素，按照 `[a0, a1, ..., a7]` 排列，在 `print_latex` 中展示为 `[V0, V1, ..., V7]`

  为什么需要 vectorized register 这样的形式？

## TiledCopy

- 为什么在修改 `static constexpr int kMmaEURepeatN = 1;` 过后 multi-stage gemm 结果出现了变化？

- `cp_async_fence & cp_async_wait`

- `make_tiled_copy`

  global -> shared memory & shared memory -> register 使用的是不同的 atom
  
  next: understand shared memory to register copying