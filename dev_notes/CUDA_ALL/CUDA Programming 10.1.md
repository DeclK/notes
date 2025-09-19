# CUDA Programming 10.1

由于 CUDA Programming 10 的笔记太杂乱了，创建一个 10.1 版本，对其中的内容进行重新梳理。核心目标是梳理 Hopper GPU 编程

reference: [DeepGemm](https://github.com/deepseek-ai/DeepGEMM/tree/main) [CalebDu/Awesome-Cute](https://github.com/CalebDu/Awesome-Cute/tree/main)

## TMA

[Deep Dive on the Hopper TMA Unit for FP8 GEMMs – PyTorch](https://pytorch.org/blog/hopper-tma-unit/) [tma tutorial](https://research.colfax-intl.com/tutorial-hopper-tma/)

TMA 其实是 Hopper 架构中新引入的硬件单元，其功能是在 global memory 和 shared memory 中传输数据。有了 TMA 过后，有几个优势：

1. 能够节省数据传输中的 register（这些 register 通常用于寻址计算），所以 register 可以更多分配给 gemm，以计算更大的矩阵。同时还能再传输数据过程中直接完成 swizzle 计算和简单的 reduction 工作

   <img src="CUDA Programming 10/fg2-3.png" alt="A100-style data movement vs H100 with TMA.  TMA hardware eliminates the need for a large amount of threads and registers participating in bulk data transfers." style="zoom:50%;" />

   也许随着 GPU 的发展，之后的 register 只会参与 CUDA core 的计算，对于 tensor core 的计算数据将不会使用 register

2. 能够以单线程发起传输，简化了线程对数据的划分问题，同时也节省了线程资源、register 资源（`pipeline_states` 中用于同步的 phase, stage idx）

### tma descriptor

也叫做 `CUtensorMap`。如前面所述，tma 的功能是 global mem 和 shared mem 之间的数据传输。从 global mem -> shared mem 的传输就是 tma load；反之就是 tma store。r不管是 tma load or tma store，都是由 tma descriptor 发起 

在 cute 中使用 `make_tma_copy_tiled` 的方式来构建 tma descriptor（实际上是一个 tiled copy 对象）



### tma load



### tma store



sync

## Cluster

sync

## wgmma

## Warp Specialization

register 动态分配（producer 少，consumer 多）

## mbarrier & pipeline

能否用相同的 pipeline 思想，在 sm80 中实现 

sync

除此之外还有一个 named barrier 不知道有什么作用？

## fence & visibility

在之前的很多小节里都触及了 fence sync，这里再统一总结一下

## Scheduler

persistant warp

## Coorperative & PingPong

## GEMM 实践

如何构建 producer & consumer 完成高效的 gemm

## Question

1. 即使是 cute 也有很重的包装/抽象和 layout 计算，但实际上很多包装和计算都是没用的。而 DeepGemm 的包装非常非常少，基本就是 PTX 包了一层，然后使用了一些 cutlass 中的小组件。但是 DeepGemm 的优点也是缺点：极少的包装导致算法的可读性 maybe 差了一点

   可能这也是现在出现各个 DSL 的原因：构建自己熟悉的抽象，然后构建高效的算法。而如果要解决这两个问题就必须要学会阅读 PTX，构建自己的功能抽象；然后熟悉各个流水线算法的流程，以清晰的代码、文档逻辑进行展现

   阅读 PTX 可能是更难的，我现在能做的是熟悉 cute 当中的 PTX 抽象，理解其功能以方便构建高效算法，好消息是 cute 可能在可读性和可用性上也在努力（python DSL），这也是 Tri Dao 最近在 GPU mode 上推荐的，[link](https://zhuanlan.zhihu.com/p/1951029558313714196)。虽然 DeepGemm 的封装很少，但是要理解其用法也没那么容易，因为 CUDA 的 doc 并不好看。总之这个领域对新手从来都不是友好的