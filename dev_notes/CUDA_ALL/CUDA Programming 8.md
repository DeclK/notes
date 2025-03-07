# CUDA Programming 8

之前的 CUDA Programming 7 已经很长了，本来想就在 7 就终结。但是没办法，需要弄清楚的事情仍然很多！这篇笔记要整理的是 cutlass cute 当中的基础概念

感觉现在即使是 DeepSeek 也对 cutlass 没有太多的头绪，我只能够不断地从 zhihu 博客中学习，配合以 DeepSeek 进行一些简单的推理来构建清晰的 cutlass 如何抽象 GPU 的模型

- ldmatrix

  how a thread carries data, this needs allocate registers for each thread

  为什么使用 LDS 需要使用 4 条指令加载，而使用 ldmatrix 只需要一条指令就能够加载完成从 shared memory -> register 的过程

  [tensorcore中ldmatrix指令的优势是什么？](https://www.zhihu.com/question/600927104/answer/3029266372)

  每一个 thread 可以最多操作 128 bit 的 register/shared memory 数据吗？如何让这个说法更加准确

- mma

  为什么 mma 的 register 数据分布和 shared memory 数据分布不一样，是 scatter 的，又如何保证计算的正确性？连续的数据被打散了

- tiledmma

  print latex to see data and thread

  在扩展时是否有限制，显然不能无限制地进行扩展，Threads 数量和 Vectorized Register 数量都是有限制的

  如何去推理整个 permute layout 的数据分布？ 
  
  为什么在 reed 的 gemm-simple 中 permutemnk 是 (1, 2, 1) 而不是具体的 shape，其中是否有什么隐藏等价变换？