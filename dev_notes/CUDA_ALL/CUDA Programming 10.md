# CUDA Programming 10

Dive into [DeepGemm](https://github.com/deepseek-ai/DeepGEMM/tree/main)

- fp8 量化，fp8 scaling 计算
- hopper & blackwell gemm 优化技术
- 算子融合

DeepGemm 有趋势要取代所有的 Gemm 实现，in a clean & efficient way。摆脱了复杂的模板构建，对于开发者来说是更加友好的形式

首先我需要知道对比 Ampere，Hopper 架构出现了哪些新的优化

1. Tensor Memory Acceleration
2. Warp Specialization and Persistant Scheduling
3. Cluster Level Programming

对比 Hopper，Blackwell 架构的特性：

1. 第五代 Tensor Core，支持 fp4，速度是 Hopper 的 2x~4x [link](https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html)
2. 原生支持 block scaling
3. Tensor Memory 用于累加器
4. CTA Pair 能够横框两个 SM 协同工作

我需要对里面的 topic 逐个熟悉，然后再来对 DeepGemm 代码进行及学习可能会更简单一些

[CalebDu/Awesome-Cute](https://github.com/CalebDu/Awesome-Cute/tree/main) 参考 DeepGemm 实现了 cute fp16 的 GEMM，而 DeepGEMM 本身似乎更倾向于使用 cuda 更底层的命令。可以对比一下二者的表现差异

## TMA

首先明确一点：TMA 优化的是数据从 global memory 到 shared memory 这个过程，没有优化从 shared memory 到 register。

single thread can kick off a tma transfer, from [Deep Dive on the Hopper TMA Unit for FP8 GEMMs – PyTorch](https://pytorch.org/blog/hopper-tma-unit/)

在上面的 blog 当中首先介绍了 TMA 的几大优势

1. TMA is very lightweight as only a single thread is needed to kick off a TMA transfer.

   这会减少 register 使用。为什么需要使用 register？因为 global mem <-> shared mem 之间的运输需要使用 register 来保存二者的地址

   <img src="CUDA Programming 10/fg2-3.png" alt="A100-style data movement vs H100 with TMA.  TMA hardware eliminates the need for a large amount of threads and registers participating in bulk data transfers." style="zoom:50%;" />

   ```cpp
   cute::copy(gmem_t, smem_t) // both of them need register to save
   ```

   所减少的 register 就可以用于存放更多的数据用于 mma 计算

   > Further, within threadblock clusters, producers can lower their max register requirements since they are only issuing TMA calls, and effectively transfer additional registers to MMA consumers, which helps to alleviate register pressure for consumers.

2. A single thread can issue large data movement instructions, allowing the majority of a given thread block to continue working on other instructions while data is in-flight.

   这减少了线程的使用。多余的线程可以进行更多的操作。确保了大部分的线程都是用于计算，配合 async 可以掩藏掉数据运输时间

   > This lightweight invocation for data movement enables the creation of warp-group specialized kernels, where warp-groups take on different roles, namely producers and consumers. Producers elect a leader thread that fires off TMA requests, which are then asynchronously coordinated with the consumer (MMA) warp-groups via an arrival barrier. Consumers then process the data using warp-group MMA, and signal back to the producers when they have finished reading from the SMEM buffer and the cycle repeats.

3. TMA handles the address computation for the shared memory destination where the data requested should be placed. This is why calling threads (producers) can be so lightweight.

   TMA 会进行地址计算，尤其对于 swizzle layout 来说重要

tma 的基本用法

1. 在 host 上构建 TmaDescriptor (i.e. cuTensorMap) 对象，传入到 kernel 当中。并且需要以 `const __grid_constant__ CUtensorMap tensor_map_a` 进行修饰

2. 只需要一个线程进行 tma 操作。需要确认 `cute::block(7)` load 了哪一个 block

3. mbarrier 作为 tma load 的同步工具，lives in shared memory

   arrival count

   transaction bytes

   phase: used for wait, arrive count is hit => phase flip if not the first arrival; else pahse = 0

   ```python
   if arrive_acount is hit:
       if first_arrival:
           phase = 0;
       else:
           phase = flip phase (0 -> 1, 1 -> 0)
   ```

4. fence 作为 tma store 的同步工具

5. 可以使用 `SM90_TMA_REDUCE_ADD` 来在 store 的时候进行 reduce

   ```python
   for cta_idx in range(number_of_ctas):
     gmem_dst[cta_idx] += smem_src[cta_idx]
     # or this:
     gmem_dst[cta_idx] = max(gmem_dst[cta_idx], smem_src[cta_idx])
     # or this:
     gmem_dst[cta_idx] = min(gmem_dst[cta_idx], smem_src[cta_idx])
   ```

6. tma 在 cluster 中可以广播 smem 数据以达到数据快速 loading (locality)

## Warp Specialization

## Cluster Programming

sync in clusers

1. `cute::cluster_sync()`

2. `cutlass::arch::fence_barrier_init()`

   make the barrier init visible to the cluster，just a rule needs to be followed...

**visibility**

这个概念会在描述中经常看到，这是 Grok 所给出的定义

> **visibility** ensures that memory operations (e.g., mbarrier initialization, TMA writes to SMEM) are observable by the appropriate threads, CTAs, or hardware components (e.g., the TMA unit) at the right time.
>
> **Invisibility** occurs when these operations are not yet propagated due to the lack of synchronization, leading to race conditions, data corruption, or kernel failures.

这样的解释还是比较清晰的。以向 smem 写入为例，对于 thread idx = 0 的线程来说，假设它已经把数据写入到了 smem 当中，其他线程或者 TMA 是如何知道它已经完成写入了呢？所以需要有一个过程，来让“写入完成”这一结果变得对其他单元（线程、cluster、tma）可见，这就是 visibility。而这个过程通常就是一个 fense or barrier (sync)

**fence & barrier & sync**

这三个的作用有一点混淆，需要弄清楚！

barrier 更为具象：所有的线程必须要执行到这一行代码，或者某个条件满足过后，才能够继续执行下一个代码

fence 更为抽象：通常有其“配合的命令”，保证了在“配合的命令”运行之前，fence 之前的各个线程的代码已经执行完整。和 barrier 最大的区别是不阻塞线程

二者都是用于线程同步的工具，保证线程任务都处于满足要求的状态，从而保证程序的正确性

我必须要把 tma 当作一个独立的硬件单元来看待，这样才能够说通所有的 fence 操作：为什么在 sync 过后还要使用一个 tma store fence？不都已经同步过了吗？数据明明都已经写入到了 smem 当中了！原因是 tma 是独立于 smem 的硬件单元，smem 完成了写入，但是 tma 并不知道 smem 完成了写入，我们需要 fence 来告知 tma：所有的线程都已经完成了 smem 写入，请你搬运这些写入的数据（这个过程也是让数据对 tma visible 的过程）。如果不告诉 tma 这些 smem 写入完成的话，tma 会搬运一些“幽灵数据”（上一次对 tma 可见的数据）。所以说个人认为这里的 fence 更像是一种“搬运/提交”，将 smem 的数据“搬运/提交”到 tma 上，再由 tma 进行搬运回 gmem。该效果从外部来看：确保了 smem 写入一定发生在 tma store 之前，同时并没有阻塞线程的执行

barrier 一定会阻塞线程的执行，例如 `syncthreads` 就是最常用的 barrier

对于 gmem -> smem 使用 mbarrier 来进行同步，smem -> gmem 使用 fence 来进行同步

**`get_slice` in cluster**



## Questions

1. Split-K or Stream-K 是否能加速 gemv or small batch decoding？

2. what is a qualifier

   在 PTX 当中 `xxx.yyy.zz` 中的 `xxx & yyy & zz` 就是 qualifier 