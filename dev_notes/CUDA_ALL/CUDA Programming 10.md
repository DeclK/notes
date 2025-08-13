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

## wgmma

参考 [blog1](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) [blog2](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)

第一篇博客要讨论的 topic 还挺广泛的：

1. wgmma
2. warp specialization & ping pong
3. persistent kernel & stream-K

> We hope that after going through this series, readers will become experts on the GEMM algorithm, and can utilize some of the beautiful ideas that go into this algorithm to design and implement other kernels in their own work.

希望读完这个系列过后就能成为 GEMM 大师！

A *warpgroup* consists of four contiguous warps, i.e., 128 contiguous threads

This operation typically follows one of these forms, where matrix `C` serves as the accumulator:

- `C = A * B + C`
- `C = A * B`, where the input from accumulator `C` is disabled.

A notable requirement of WGMMA is that operand `B` must always be stored in shared memory (SMEM). In contrast, operand `A` can be located in either SMEM or register memory (RMEM), and the accumulator `C` is always held in RMEM.

这里提了一个很重要的规则：B 矩阵一定是保存在 shared memory 当中的，而 A 矩阵既可以在 shared memory 也可以在 global memory。累加器 C 必须在 register memory

SM90 MMA atoms are then labeled as `SM90_MxNxK_XYZ_SS` or `SM90_MxNxK_XYZ_RS`

```cpp
TiledMMA tiled_mma = cute::make_tiled_mma(
  SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});
```

在 DeepGEMM 当中，所有的 mma 都是使用 `SS` atom，也就是说 Tensor core 都是直接在 shared memory 上去获得数据，而不是 register

- `X` and `Y` are the datatypes of the operands.
- `Z` is the datatype of the accumulator.
- `MxNxK` are the tile sizes that the `wgmma` instruction computes with — the “wgmma atom”. Not all values of `MxNxK` are possible. Here is the [list of allowed shapes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shape): `M` is always 64, `N` is a multiple of 8 from 8 to 256, and for 16-bit operand datatype, `K` is 16 (more generally, `K` is fixed to be 32 bytes).

wgmma 的构建和 mma 的构建是类似的，都有 `AtomLayoutMNK` and `PermutationMNK` 

```cpp
TiledMMA tiled_mma = make_tiled_mma(
 SM90_64x64x16_F16F16F16_SS{},
 Layout<Shape<_2,_1,_1>>{});
```



**smem layout requirements**

1. M N K 必须要能够被 mma tile shape 整除

2. 对 sA 和 sB 的 layout 根据 swizzle function 而定

   > However, as a practical matter, we can always construct layouts guaranteed to be compatible with `wgmma` using certain pre-defined layout atoms provided by CUTLASS, followed by the `cute::tile_to_shape` method.

   tile to shape 的实际用途。似乎必须使用 `GMMA:Layout_XX` 中的 layout 来构建 smem layout
   
   These layout atoms must then be passed into `tile_to_shape` with the SMEM shape for `sA` and `sB` given by `make_shape(bM,bK,bP)` or `make_shape(bN,bK,bP)`, with the modes of the shape given **in that order**, such that the tile sizes of the layout atoms divide into those of the larger SMEM shape.
   
   ```cpp
   GMMA::Layout_MN_INTER_Atom<T>
   GMMA::Layout_MN_SW32_Atom<T>
   GMMA::Layout_MN_SW64_Atom<T>
   GMMA::Layout_MN_SW128_Atom<T>
    
   GMMA::Layout_K_INTER_Atom<T>
   GMMA::Layout_K_SW32_Atom<T>
   GMMA::Layout_K_SW64_Atom<T>
   GMMA::Layout_K_SW128_Atom<T>
   ```
   
   这也省的我们自己去构建 swizzle 了，应该是件好事吧



The WGMMA-specific thing to notice here is that `tCsA` isn’t actually a thread-level slice of SMEM, but rather the entire SMEM tensor with a reorganized layout.

Next, printing the “fragments” `tCrA` and `tCrB` for any thread index shows:

```cpp
tCrA: GMMA::DescriptorIterator o (_1,_2,_4,_3):(_0,_64,_256,_1024)
tCrB: GMMA::DescriptorIterator o (_1,_2,_4,_3):(_0,_64,_256,_1024)
```

Internally, CUTLASS constructs a “[matrix descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor)“, which is 64-bit value held in registers that describes the SMEM in a way suitable for use by the `wgmma` instruction. For the programmer, the most important thing to bear in mind is that values of SMEM are **not** copied into RMEM; rather, accessing the values of `tCrA` and `tCrB` instead accesses these 64-bit descriptors. Moreover, these tensors being “iterators” means that only the single 64-bit descriptor used for a given `wgmma` instruction is held in registers at a time (e.g., as opposed to all 24 of them).

上面这一段话也非常重要：最终生成的是一个 descriptor (just like tma did.)，我认为也简化了我们对 layout 的操作，把注意力集中于对数据位置的描述，剩下的交给 cuda 去管理



**synchronization in wgmma**

```cpp
cute::warpgroup_arrive();
cute::gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
cute::warpgroup_commit_batch();
cute::warpgroup_wait<0>();
```

- `warpgroup_arrive` 其实也是一个 fence，其作用是保证 warpgroup 的执行一定在 memory 操作完成之后执行。

  > From Kimi
  >
  > **`warpgroup_arrive()` 是一道“护栏”，它告诉 GPU：本 warpgroup 所有线程对寄存器/共享内存的写操作已经完成，接下来可以安全地让 `wgmma.mma_async` 去读这些地址。**
  >
  > `wgmma.mma_async` 是异步的，硬件可能把它的**读操作提前**。
  > 如果你在它之前还有往共享内存或寄存器写 A/B 矩阵数据的指令，而**不写 fence**，就可能读到**旧值** → 结果错误。

  上面的解释也指向了 relaxed consistency model。之前所见到的是 `fence.proxy.async`，其涉及到 generic proxy 和 async proxy 之间的同步；而在 wgmma 当中的是 `wgmma.fence.sync.aligned`，这实际上是在 async proxy 内部的同步，不涉及到 generic proxy。这也说明了 relaxed consistency model 不仅存在在不同的 proxy 之间，也存在在 asycn proxy 内部。

- `warpgroup_commit_batch`

  这里的作用类似于 `cp_async_fence`，其实是一个 commit 命令，将当前的所有的 wgmma 命令打包提交，然后在之后使用 wait 命令等待具体的命令

- `warpgroup_wait`

  允许最新提交的任务中，有最多 N 个 wgmma 任务未完成。N = 0 说明等待所有的 wgmma 任务完成

**Just like [TMA operations](https://research.colfax-intl.com/tutorial-hopper-tma/), `wgmma.mma_async` is performed in the [async proxy](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-proxy).** 

In situations where the warpgroup has the opportunity to perform independent computation, flexibility with the parameter `N` comes in handy. For example, this comes into play with the GEMM-softmax overlapping strategy employed in the design of [FlashAttention-3](https://research.colfax-intl.com/flashattention-3-fast-and-accurate-attention-with-asynchrony-and-low-precision/).

**wgmma core matrices**

看上去对我们构建 kernel 没什么大用，感觉是对 wgmma 的一些底层原理介绍：为什么对 smem layout 有这样的要求。我不想深入探索这里的底层原理。

> From Kimi
>
> **Core matrix 就是 WGMMA 在 Shared Memory 里能“一次性吃进嘴里的最小数据块”；记住它的大小、排布方式和 swizzling 规则，就能把 SMEM 布局写对。**
>
> 你只需在 CUTLASS 里选 `Layout_MN_SW128_Atom<>` 这类原子布局，再 `tile_to_shape`，就能保证 LBO/SBO/swizzle 都合法，不必手算。

## Warp Specialization

在介绍 warp specialization 之前先简单介绍了一些背景

1. 现在的 tensor core 计算能力远强于数据的运输能力，所以一切的优化都围绕着如何打满 tensor core 的算力。这个过程叫做 "feading the beast"

2. 总体上有两种优化技巧

   1. 有效的 threadblock scheduling，以提升 L2 cache hits

      we refer curious readers to the techniques of [threadblock rasterization](https://github.com/NVIDIA/cutlass/blob/main/media/docs/efficient_gemm.md#threadblock-rasterization) and persistent kernels, for instance as implemented in CUTLASS.

   2. overlap copying with math computation，pipeline

讨论两种流水线：multi-stage 和 warp-specializatioin

With warp-specialization, some warps are dedicated to memory fetches (*producers*), while others are dedicated to compute (*consumers*), and named barriers are used for synchronization between them. The idea is that the warp schedulers can then more easily hide the latency of copy operations within compute

The fastest Ampere GEMM kernels, as well as the famous FlashAttention-2, use the multistage kernel design.

It is not trivial to implement pipelines correctly and efficiently. Programmers must handle the multiple buffers as well as asynchronous load calls across multiple threads. In the next section, we show how to implement pipelining via a CUTLASS abstraction: the `Pipeline` class.

可以使用 cutlass 中定义的 pipeline class 快速完成流水线搭建，因为流水线搭建真的不是一件简单的事情

buffer: shared memory with N stages

**Barriers.** To synchronize the buffer stages across the producer and the consumer, a Pipeline adheres to the standard *acquire and release model* that uses locks to manage accesses to the buffers. To this end, let `full_barrier` and `empty_barrier` be two arrays of *barrier objects*, both of size `N`. These barrier objects possess a *phase bit* value which is initialized to 0 and flips between 0 and 1.

定义了 barriers 来进行管理这些 buffers，什么时候 lock 什么时候 release

Concretely, these barrier objects will be [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier) objects resident in SMEM. An mbarrier object is initialized both with the aforementioned phase bit as well as an *arrival count*. It then supports arrive-on and wait operations and flips its phase based on reaching the arrival count threshold. Importantly, the values of these barrier objects can and should be visible to all threads.

有了这些概念过后再去看 tma 的一些代码就会跟清楚

首先定义了 pipeline state，其有两个核心成员 index & phase。pipeline state 会重载算符 `++`，此时 index 会不断递增，直至增加到 stage number N，而 phase 则在 stage number 增加到 0 时，相位翻转

```cpp
    void operator++(int) {
      count += 1;
      if ((++stage_idx) == kStage) {
        phase ^= 1;
        stage_idx = 0;
      }
    }
```

那么这个 pipeline state 是如何同步 producer & consumer 的呢？

**Synchronization**. We now explain how the barrier objects and thread-local pipeline states are used to synchronize producers and consumers. To avoid confusion, let us distinguish the producer *action* from the producer thread(s) issuing that action, as these may potentially be decoupled (think of TMA). First, the producer action will flip the phase of `full_barrier[i]` to signal that it has filled the `i`th stage of the buffer, so that the consumer threads can now read from it. Similarly, the consumer threads will flip the phase of `empty_barrier[i]` to signal that they have finished consuming the `i`th stage of the buffer, so that the producer can now write to it.

这意味着我们有 N 个 `full_barrier & empty barrier`，每一个 barrier 都有一个自己的 pipeline state？arrival count 又在其中扮演什么角色？arrival count 和 stage 是相关的概念吗？

Finally, each thread, whether consumer or producer, keeps track of a phase to match against the phases of the barrier objects, and in fact threads taking on both consumer and producer roles will need to track *both* phases. These “internal” phases of the threads need to be flipped as well as the kernel proceeds through iterations of its mainloop.

整个过程描述下来还是比较抽象的，这是因为描述中缺少了对 mbarrier 和 pipeline state 之间的联系与区分：

1. Mbarrier，管理两个成员：arrival count & phase
2. PipelineState，管理两个成员：index & phase

可以看到二者都拥有各自的 phase，但是二者的 phase 是联系起来看待。

通过阅读 PTX doc 知道了各个命令的本质

1. mbarrier 实际上有4个成员：phase, arrive count, pending count, tx-count

   mbarrier 的[初始化](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-init)只传入一个 `count`，此时

   - Initializing the current phase to 0.
   - Initializing the expected arrival count to `count`.
   - Initializing the pending arrival count to `count`.
   - Initializing the *tx-count* to 0.

2. [arrive](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-arrive-on) 会 decreament pending count

3. [expect_tx](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx-operation)(bytes) 会增加 tx-count  += bytes

4. tma copy 会自动调用 [complete_tx](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)，会减少 tx-count -= bytes。查看 tma load 的 ptx 就可以看到其中有 complete_tx 的 qualifier 字段

5. 当 pending count = 0 以及 tx-count = 0 时，触发 [phase complete](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion) 条件，此时：

   1. phase flip: phase ^= 1
   2. pending count 还原会 `count`

根据以上机制，就可以顺利推理整个流水线的同步过程。另外根据 [zhihu](https://zhuanlan.zhihu.com/p/1905383022901059783) 的说法：mbarrier.wait 只检查current phase 的完成，即 phase = 1, barrier.wait(phase)，若 barrier 内置 phase 为 0，则此 wait 不会等待。这也是为什么一开始要把 producer pipeline_state 的 phase 初始化为 1。因为初始化时不必等待 consumer 完成计算，直接发起 tma load

Phase 的物理含义是什么？如何理解同一个 phase 的各个 barrier & pipeline states？

对于这个问题，Kimi & DeepSeek 都没回答到我能 get 到的点上，不过他们的回答都具有启发性

> From DeepSeek
>
> 在您提供的代码中，`phase` 是一个关键概念，用于管理生产者（producer）和消费者（consumer）之间的同步。它的物理意义可以理解为 **数据缓冲区（或同步屏障）的当前有效状态标识**，用于区分不同阶段的数据版本，确保生产者和消费者不会同时操作同一份数据。
>
> From Kimi
>
> `phase` 本质上就是 **“这次是第奇数次到达 barrier” 还是 “第偶数次到达”** 的 1-bit 计数器

我的理解： phase 的物理意义其实是一个不进位的计数器，用于记录这是第 phase 个数据

而 wait phase 的物理含义为：等待第 phase 批数据处理完成。这样理解起来就顺了：我们将 producer pipeline states phase 初始化为 1，consumer pipeline states phase 初始化为 0。其实这里的 1 看似比 0 要大，在我这里看来其实 1 < 0，换句话说，这里的 1 指代的是上一批的数据，0 是当前批的数据。那么这两个 wait 就变得相当形象了

```cpp
// wait for previous data to be consumed
empty_barrier.wait(phase=1)
  
// wait for current data to be loaded
full_barrier.wait(phase=0)
```

empty barrier 是在等待上一批的数据完成，而 full barrier 则是在等待这一批的数据加载。而我们把 empty barrier & full barrier 的 phase 都初始化为 0 也很好理解

```cpp
// dealing/consuming current data
empty_barrier.phase = 0
  
// dealing/producing current data
full_barrier.phase = 0
```

empyt barrier phase 初始化为 0 意味着 consumer 正在消耗第 0 批数据（虽然初始化时，tensor core 并不是真的就在运行了！不过从 empty barrier 的视角来看，就是在消耗第 0 批数据的状态）

full barrier phase 初始化为 0 意味着 producer 正在生产第 0 批数据 

由于 cutlass doc 当中的代码并没有被 DeepGemm 中采用，而且我所学习的 cute ws 代码也是参考 DeepGemm 来构建的，之后的学习全面针对 awesome cute 当中的代码学习


为什么设置了两个不一样的 arrival count？

```cpp
static constexpr int ConsumerArvCnt = size(TiledMma{}) * size(ClusterShape{}) / WarpSize;
static constexpr int ProducerArvCnt = 1;

for (int i = 0; i < Stage; i++) {
    shared_storage.pipelines.mainloop_full_bar[i].init(ProducerArvCnt);
    shared_storage.pipelines.mainloop_empty_bar[i].init(ConsumerArvCnt);
}
```

这是因为对于 full barrier，由于 tma 操作只需要一个线程进行发起即可。而对于 empty barrer 来说由于 simt 的原因，可能每一个线程都会发起一个 arrival signal。所以在具体的代码里有一个 predicate，让每一个 warp group 只由一个（或多个）thread 发起 arrival

```cpp
uint32_t lane_idx = threadIdx.x & 31;
uint32_t target_cta = lane_idx;
uint32_t pred_arrive = lane_idx < size(ClusterShape{}); // lane_id thread notify cluster_id cta barrier
// notify producer
shared_storage.pipelines.mainloop_empty_bar[pipeline_states.stage_idx].arrive(target_cta, pred_arrive);
```

可以看到，当 cluser size = 1 的时候，其实只有每一个 warp group 的 0 号线程发起了 arrival signal。那么就正好符合 `ConsumerArvCnt` 的需求

提出疑问：为什么不把 consumer arrival count 也设置为一，让 thread id = 0 的线程去发起就好

> From Grok
>
> 如果将 ConsumerArvCnt 设为 1，只让 threadIdx.x == 0 的线程执行 arrive，就会出现以下问题：
>
> **无法保证所有线程完成**：threadIdx.x == 0 的线程可能在自己的计算完成后立即调用 arrive，但其他线程（例如其他 warp）可能尚未完成 MMA 操作。这样，empty barrier 的 phase 会过早翻转，producer 可能开始加载新数据，覆盖 SMEM 中仍在被其他 warp 使用的内容，导致数据竞争和错误结果。

我有一个疑问：在之前的 SIMT 编程思想下，写 if-else 分支是效率比较低的行为。为什么在 warp specialization 就可以被允许了

> From KIMI
>
> 经典 SIMT 以 32 线程的 warp 为最小调度单位，同一个 warp 里的线程只要条件不同就会顺序执行 if 和 else 两段指令，造成浪费。
> Hopper 的 warp specialization 把粒度拉大到「整个 warp-group」（通常是 128 线程甚至 4-warp-group 的 512 线程）。也就是说，只要一个 warp-group 里的所有线程都走同一条路径，就不会出现传统意义上的 divergence。

这说明我之前对 SIMT 的理解有错误，其范围被限制在了 warp 大小内，而不是对整个 block 都需要进行 SIMT，在这样的意义下，就可以让 producer 和 consumer 进行独立运行，达到 warp specialization 的功能

除此之外，Hopper 还对寄存器有着动态分配的机制，这也是为了 warp specialization 服务

> From DeepSeek
>
> 1. **传统限制**：在Hopper之前，GPU kernel启动时**固定**每个线程的寄存器数量（编译时指定），整个kernel执行期间无法改变
> 2. **Hopper创新**：
>    - 每个线程块(CTA)维护一个**共享寄存器池**
>    - Warp Groups可以在运行时**动态申请/释放寄存器**
>    - 通过`setmaxnreg`指令实现（PTX 8.0+）
>
> `warpgroup_reg_alloc<232>()` 实际执行 PTX 指令 `setmaxnreg.inc.sync.aligned.u32 232;`
>
> - **作用**：从CTA寄存器池**申请更多寄存器**
> - **行为**：
>   1. 将当前warp group内每个线程的寄存器上限**提升到232个**
>   2. 如果池中寄存器不足，**阻塞等待**直到其他warp释放寄存器
>   3. 新增的寄存器内容**未初始化**（需程序显式初始化）
> - **使用场景**：Consumer需要大量寄存器进行MMA计算
>
> `warpgroup_reg_dealloc<40>()` 实际执行 PTX 指令 `setmaxnreg.dec.sync.aligned.u32 40;`
>
> - **作用**：向CTA寄存器池**释放多余寄存器**
> - **行为**：
>   1. 将当前warp group内每个线程的寄存器上限**降至40个**
>   2. 释放的寄存器**立即归还**到CTA共享池
>   3. 原寄存器内容**被丢弃**
> - **使用场景**：Producer只需少量寄存器管理TMA加载

我按照上限计算了一下每个 cta 所需要的寄存器个数为 64512 =(40x128 + 232x128x2)，会略低于一个 cta 的寄存器上限 65536，这应该也是为了性能考量，留一点寄存器作为余量

我发现把 PTX 文档中的内容直接丢给 GPT 让他们去整理总结其意义会比自己去读要容易得多

对于 warp 级别的操作，首先要考虑的就是洗牌函数，在代码中一开始就用了洗牌函数来确定 warp group 相关的 index

```cpp
    auto thread_idx = threadIdx.x;
    auto block_idx = blockIdx.x;
    auto lane_idx = thread_idx & 31;
    auto warp_idx = __shfl_sync(0xffffffff, thread_idx / WarpSize, 0);
    auto warp_idx_in_group = __shfl_sync(0xffffffff, warp_idx % 4, 0);
		auto warp_group_idx =  __shfl_sync(0xffffffff, thread_idx / WarpGroupSize, 0);
```

warp specialization 的代码其实很简单，就是一个 if-else 分支

```cpp
   	// WASP: consumer wg0, producer wg1
    // WASP_COOP: consumer wg0 wg1, producer wg2
    // WASP_PIPO: consumer wg0 wg1, producer wg2
    if (warp_group_idx == WarpGroupCnt - 1) {
      // producer
      // alloc 40 register for tma load
      cutlass::arch::warpgroup_reg_dealloc<40>();
      // elect 1 thread issue tma load
      if (warp_idx_in_group == 0 && elect_one_sync()) {
        producer(param, shared_storage, block_rank_in_cluster);
      }
    } else {
      // consumer
      // alloc 232 register for mma compute
      cutlass::arch::warpgroup_reg_alloc<232>();

      if constexpr (kernel_tag == KernelTag::WASP ||
                    kernel_tag == KernelTag::WASP_COOP) {
        ws_consumer(param, shared_storage);
      } else if constexpr (kernel_tag == KernelTag::WASP_PIPO) {
        ws_pipo_consumer(param, shared_storage);
      }
    }
```

同样由于 SIMT 思想的疑惑：对于 producer 来说，只有一个 thread 在进行操作，那其他的 thread 是不是就没有工作了？在此情况下还会给他们一起分配寄存器之类的资源吗（根据 SIMT 编程原则）？

对于此问题，DeepSeek 和 Kimi 的回答都是：会给不工作的线程也分配资源，但是这些线程的确不会工作

> From Kimi
>
> 是的，**整个 warp group 都会进入 producer 分支**，但**真正干活的只有 warp group 里被 `elect_one_sync()` 选出来的那一个 thread**，其余 127 个 thread 在这条代码路径上就是“空转”
>
> 不过，**它们仍然会被一起分配寄存器**，因为：
>
> 1. **warp 是调度/资源分配的基本单位**。
>    只要一个 warp 里有一条 thread 需要执行某条指令，整个 warp 都会被保留，寄存器也会按 warp 粒度整体预留。
> 2. **SIMT 执行模型**决定了 inactive lane 不会真的执行指令，但**资源（寄存器、调度槽）仍然被占用**。
>    所以，即使 127 个 thread 在 `if (elect_one_sync())` 之后什么也不做，它们也必须保留寄存器上下文，不能被别的 warp 复用。

另外我还询问了：对于 producer warp group 是 128 个线程都会分配寄存器吗？还是给单个 warp 分配寄存器就行。答案是给 128 个线程都会分配

> From Kimi
>
> 在 Hopper 架构（SM9x）的 **WGMMA（warpgroup matrix multiply accumulate）指令集**里，寄存器调度/分配的**最小粒度是“warp group”**（4 个 warp，128 lanes），而**不是单个 warp（32 lanes）**
>
> From DeepSeek
>
> - **设计合理性**：虽有效率牺牲，但通过寄存器资源再分配 (dealloc & alloc) 和简化同步，整体收益为正。这在内存受限的 GEMM 内核中尤为关键。

## Main Functions

有几个关键的组成部分

### Template & helper function

1. Param & Args

   - Args 是主机端输入，由用户在 host 端构造，只包含 **原始指针** 和 **矩阵尺寸**，没有任何设备端布局或调度信息

   - Param 是设备端执行参数，包含了 **TMA descriptor object、scheduler param、problem size** ，是 kernel 真正需要的东西。problem size 是 args 和 param 都没改变的东西

   另外 Param 还需要一些信息：`ClusterShape` & `CTATile` & `TMA` 等等，这些信息来自于 struct 当中的模板，即元编程所推导出来的模板类

   - Initialize_param

2. Shared storage

   所有放在 shared memory 相关的数据都在这里：matrix ABC & mbarrier。另外，他们都使用了 aligned array or alignas 来让这些数据在内存地址中进行了对齐

   ```cpp
     struct alignas(128) SharedStorage {
       struct alignas(128) TensorStorage {
         cute::array_aligned<ABtype, cute::cosize_v<SmemLayoutA>> smem_A;
         ...
       } tensors;
       struct alignas(16) PipelineStorage {
         // mainloop pipeline barrier
         // 2stage consumer pingpong barrier
         ...
       } pipelines;
     };
   ```

   我尝试了一下去掉这个 aignas 仍然能够成功运行，可能不需要太注意这个细节

3. pipeline state

4. All kinds of tempalte meta programming

   感觉这里才是最麻烦的地方，~120 行，希望我能够整理出一个清晰的逻辑以及结构，这样才能在之后的编写中有一个思路可循

   0. 模板元编程参数：cta shape, stage number, cluster shape，想要生成特化代码的参数。这些特化代码在真正编译的时候都会被编译，再从 host 端进 if-else 进行选择

   1. ABCType and layout，这应该是 args 所提供的最基础的信息

   2. **mma atom**

   3. **copy atom**

      这才是模版中占比最多的 atom，不仅需要定义各个存储之间的 copy atom (gmem <-> smem <-> rmem)，还要定义各个输入矩阵都单独定义一个 copy atom (Tensor A B C)。定义 copy atom 也不可避免地需要对 memory layout 进行定义，所以整个的代码行数就是大几十行。接下来就是一一进行分析

   4. 同时为了 kernel 的合法性，会进行一些 `static_assert` 检查：例如 cluster shape & cta tile 的合法性

   其实不用把一大堆的 static constexpr int 写在 struct 的最前面，我觉得在使用的时候再进行一些计算，可能会更好读一些，不然这些定义距离使用的地方太远了，写的时候不方便。我看 reed 中的 gemm-multistage 就是这么干的，这些 constexpr int 应该会被处理为编译期常量，而不会占用寄存器资源（Maybe\

5. Tile Scheduler

### TiledMMA

对于 Hopper 的 TiledMMA 构建 cute 有一个合适的工具函数来帮助我们 `cute::GMMA::ss_op_selector`

只需要传入 ABC matrix type & CTATile & ABMajor

```cpp
  // ws_cooperative use 2 warp group
  using AtomLayoutMNK =
      std::conditional_t<kernel_tag == KernelTag::WASP_COOP,
                         Layout<Shape<_2, _1, _1>>, Layout<Shape<_1, _1, _1>>>;
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<ABtype, ABtype, Acctype, CtaTile, GmmaMajorA,
                                 GmmaMajorB>(),
      AtomLayoutMNK{}));
```

GMMA 其实就是 group mma，也就是 warp group mma，和 wgmma 其实是一个东西

对于 cooperative kernel，会利用 `AtomLayoutMNK` 在 M 方向上进行扩展，以达到利用两个 warpgroup 完成一个 mma 的功能。（我之前还在思考，怎么将 M 维度 split 成两份，原来是反过来思考的，更简单了）

### TiledCopy

1. G2S copy

   使用 tma copy AB matrix

   1. 构建 copy atom

      copy atom 根据 cluster shape 来决定是否使用 tma multi-cast 指令。在 cute 当中有工具函数直接获得 atom

      ```cpp
        // tma g2s load
        using TmaG2STiledCopyA =
            decltype(sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape{})));
        using TmaG2STiledCopyB =
            decltype(sm90_cluster_shape_to_tma_atom(shape<0>(ClusterShape{})));
      ```

      这个函数名字很长，实际上就是一个 if-else 选择。如果 cluster shape = 2 就会启用 multi-cast。一个细节：对于 A 矩阵是看 `ClusterShape[1]` 的大小，因为是同一行的 CTA 才会共享 A 矩阵数据

   2. 构建 tma copy descriptor

      回想一下在 Ampere 架构中如何构建 G2S TiledCopy：

      如何构建 tma TiledCopy，其中的参数各自起到什么作用？如何使用 tma copy？对比 Ampere G2S 的 copy 过程有什么区别？

      他们二者的共同点：都需要对 smem 的 atom layout 进行定义（主要是定义 swizzle），并使用 `tile_to_shape` 完成完整 smem layout 构建 `(M or N, K, num_stages)`

      1. tma tiled copy 核心参数

         1. copy op: with or without mulit-cast
         2. Gtensor: 完整的 matrix 矩阵
         3. Slayout: smem layout，由 `tile_to_shape` 提供
         4. cluster size: 1 or 2

         实际上还有一个 cta tile 的参数，通常由 slayout shape 直接确定了

         ```cpp
         tma_desc = make_tma_copy(SM90_TMA_LOAD{}, 
                                  gmem_tensor, 
                                  smem_layout, 
                                  [cluster size])
         ```
      
         该 atom 应该能够一次对整个 `smem_layout` 大小的 tensor 进行 copy
      
      2. 如何使用 tma descriptor
      
         1. 获得 tma tensor。这区别于 Ampere 架构直接使用 gmem tensor，tma 需要调用 `get_tma_tensor` 来获得 tma tensor
         
            ```cpp
            Tensor A = tma_desc.get_tma_tensor(make_shape(m, k))
            ```
         
         2. 使用 cta id in cluster 获得当前的数据 slice，该 slice 也充当 `thr_copy` 的角色
         
            ```cpp
            g2s_tma_a = tma_desc.get_slice(cluster_idx.y)
            ```
         
            这里的 cluster 的作用就发挥出来了，参考 [tma tutorial](https://research.colfax-intl.com/tutorial-hopper-tma/) 当中的内容：对于 cluster size = 2 (通常最大也就是 2 了) 的情况，一个 cluster 内部的两个 cta 会对数据进行划分，各自搬运一半的数据，然后进行通讯共享以拼凑得到完整的数据
         
         3. 利用 `thr_copy` 对数据进行线程划分，但这对 tma 来说其实是多余的，因为只有一个线程，不过不过为了 cute 的统一编程风格，这样做也可理解
         
            ```cpp
            // gA_mk (BLK_M, BLK_K, m, k)
            Tensor gA_mk = local_tile(A, CTATile{}, make_cood(_, _, _), Step<_1, _X, _1>)
            // gA (BLK_M, BLK_K, k)
            Tensor gA = gA_mk(_, _, tile_info.m_idx, _)
            // tAgA (COPY, COPY_M, COPY_K, k)?
            Tensor tAgA = g2s_tma_a.partition_S(gA)
            Tensor tAsA = g2s_tma_a.partition_D(sA)
            ```
         
            TODO: 我猜测 REST_M & REST_K 都是 1，并且 COPY = BLK_M * BLK_K，这正好对应了 cta tile 大小
         
         4. 使用 `cute::copy & tma_desc` 完成 copy 任务
         
            ```cpp
            copy(
              tma_desc.with(*full_barrier_ptr, multicast_mask_a),
              tAgA(_, _, _, k_idx),
              tAsA(_, _, _, k_idx)
            )
            ```
         
            TODO: 如何理解这里的 full_barrier_ptr & multcast_mask 的作用

### Compute Logic

1. producer

2. consumer

   1. coorperative
   2. pingpong

3. issue_mma

   完成一次 big K iteration，i.e. 完成一个 output tile 的累加计算 (maybe?

4. mma tail

5. issue epilogue

似乎整个 GEMM 就是这些关键功能的合作，我应该要把他们的基本功能和原理都弄清楚，再考虑与 deepgemm 的对比

在进行 gemm 学习时一个简单的假设会使得流水线的图示变简单：gemm 是 compute bound。该假设就会使得取数据的时间小于计算时间，在这样的假设下才能够在流水线中将算力打满，同时在这样的条件下我们才能够看到： epilogue & prologue 的时延被计算所掩藏

简易的证明 issue mma 是最高效的：

1. smem_1 抵达，mma_0 还没有计算完成

   此时为 compute bound，我们需要等待 mma_0 计算完成，才能开始 mma_1 的计算，此时 tensor core 没有空闲，算力打满

2. Smem_1 抵达，mma_0 的计算已经完成（一段时间了）

   此时为 memory bound，我们必须等待 smem_1 的抵达才能够开启 mma_1 的计算。需要注意的是，我们在 mma_0 计算完成的瞬间，就已经通知 empty barrier 到达信息，让 smem_0 处于可写状态。此时 memory 没有空闲，算力受限，但无法进一步提升

应该不需要使用 prologue mma (one mma in-flight) 的操作？像 DeepGemm 一样直接等 mma 计算完就完事儿了！这一点我需要自己实验一下才知道差距多大。唯一我能够想到的差距在于：第一次 mma 需要使用

```cpp
    // fisrt mma with no accumulation to avoid init zeros
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
```

来告诉 mma：直接用 `C = AB`，而不要使用 `C = AB + C`。这样能节省一次对 accumulator 的清零。但这和 in-flight 与否无关

在 [efficient gemm](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md) 文档中简要描述了 coorperative & ping pong kernel 设计：

1. Cooperative

   两个 warp group 完成一个 tile 的 mma，但是他们会在 M 维度进行对半分，各自完成一半 tile mma

2. Ping-Pong

   两个 warp group 完成两个 tile 的 mma，他们会通过配合掩藏 prologue & epilogue

在 [为什么Hopper架构上warp-specialization比multi-stage要好？-zhihu](https://www.zhihu.com/question/11261005710) 中有简易的图示 

## Sync function

在代码当中会遇到很多用于同步的语句，需要逐一理清他们的作用，然后和 function 功能配合起来，彻底让 pipeline 编程白盒

1. `cute::prefetch_tma_descriptor(param.tma_a.get_tma_descriptor());`

   > From Kimi
   >
   > 作用是**提前将 TMA（Tensor Memory Accelerator）描述符预取到 GPU 的 L2 cache 中**，以减少后续实际执行 TMA 加载/存储操作时的延迟。
   >
   > TMA 描述符本身也存储在 global memory 中。如果不预取，当 kernel 中第一次使用 TMA 加载/存储时，GPU 需要从 global memory 读取描述符，这会带来额外的延迟。
   >
   > 通过 `cute::prefetch_tma_descriptor()`，我们可以**在 kernel 启动的早期阶段**（比如还在做计算准备时），**将这些描述符提前加载到 L2 cache 中**，这样后续真正执行 TMA 操作时，描述符已经在 cache 里，延迟显著降低。

   我发现 kimi 的回答一直都非常简练，应该是在回答的字数上有所限制，而 Grok 的回答则会无比冗长，DeepSeek 介于二者之间

2. `cutlass::arch::fence_barrier_init()`

   在之前我们讨论了 visibility，其发生在了 generic proxy 和 async proxy 之间。实际上这种 visibility 也存在在 cta 和 cta 之间。由于 Hopper 架构引入了 cluster level，所以在 cluster 之间也需要同步与通信。当我们初始化了 barrier 过后，同一个 cluster 的 cta 之间其实是看不见各自 barrier 的初始化情况的，所以为了让 barrier 初始化情况在 cluster 之内 visible，就需要使用该命令 `cutlass::arch::fence_barrier_init()`

   `fence_barrier_init` 一般会和 `fence_view_async_shared` 一起使用

   ```cpp
   cutlass::arch::fence_view_async_shared();
   cutlass:arch::fence_barrier_init();
   ```

   前者是让 barrier 对 async proxy 可见（e.g. tma），而后者就是让 barrier 对（同一 cluster 内的）其他 cta 可见

   还有一个配合这两个命令的是 `cluster_sync` or `__syncthreads`

   ```cpp
   (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();
   ```

   用于让所有的线程进行同步，避免在之后的 warp specialization 操作中有的线程已经开始使用 barrier 了

   > From Kimi
   >
   > **这个 barrier 是为了确保所有线程都完成了共享 barrier 的初始化，避免 producer 和 consumer 在使用未就绪的同步原语时出现竞态或死锁。**

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

为什么在 sync 过后还要使用一个 tma store fence？不都已经同步过了吗？数据明明都已经写入到了 smem 当中了！所有的解释都是：smem 完成了写入，但是 tma 并不知道 smem 完成了写入，我们需要 fence 来告知 tma：所有的线程都已经完成了 smem 写入，请你搬运这些写入的数据（这个过程也是让数据对 tma visible 的过程）。

以上的描述过于形象化，而不够严谨。我拷打了 Grok 很久，它也只是在这几个名词之间绕来绕去。我的逻辑很简单：

1. Generic proxy 对 smem 进行了写入，并且使用 syncthreads 保证了写入的完成
2. 不添加 fence 使得 async proxy 无法看见这些写入操作（invisible），其看见的是 outdated 数据
3. async proxy 看见的 smem 和 generic 看见的 smem 不一样
4. smem 在同一时刻只能有一种状态

以上就推出了矛盾点。所以我认为用 visible 来描述这个过程是不利于我们对 GPU 模型的理解的。但是在拷打 Grok 的过程中，其不断地提出另一个概念 **relaxed memory consistency model**。对这个概念进行解释：

> From Kimi
>
> 在 GPU 内存中，**relaxed consistency model（放松一致性模型）** 是一种比顺序一致性（Sequential Consistency, SC）或 Total Store Order（TSO）更弱的内存一致性模型，其默认允许内存操作（如加载和存储）被重排序，除非程序员通过显式的同步机制（如 `FENCE` 或 `__threadfence()` 等）指定顺序 [ref link](https://www.sigarch.org/gpu-memory-consistency-specifications-testing-and-opportunities-for-performance-tooling/#:~:text=g.%2C%20device,block%28%29%7C)

如此一来就能解释通了：tma store 和 smem write 由两个不同的 proxy 执行，他们二者的在执行时并不保证严格的顺序，可能 tma store 在 smem write 的过程中就开始了，所以其看到的内容就是 outdated，所以必须要使用 fence 来保证，tma store 的发起在 smem 写入之后，而 tma store 怎么看不到 generic proxy 对 smem 的操作，所以首先必须要让 generic proxy 的这些操作对 async proxy 操作可见。在操作可见之后，方可完成判断：这些操作是否完成，从而控制 tma store 的顺序一定在 smem 之后

**fence 最核心的目的其实是用于保证操作按照期望的顺序执行**，而这也是由 relaxed consistency model 所产生的直接影响。

有一个很形象但也许不准确的说法：barrier 是等线程；而 fence 是等数据

barrier 一定会阻塞线程的执行，例如 `syncthreads` 就是最常用的 barrier

对于 gmem -> smem 使用 mbarrier 来进行同步，smem -> gmem 使用 fence 来进行同步

**`get_slice` in cluster**



## Questions

1. Split-K or Stream-K 是否能加速 gemv or small batch decoding？

2. what is a qualifier

   在 PTX 当中 `xxx.yyy.zz` 中的 `xxx & yyy & zz` 就是 qualifier 

3. 为什么说 Hopper 架构是第一代真正的异步 GPU？

   [为什么Hopper架构上warp-specialization比multi-stage要好-zhihu](https://www.zhihu.com/question/11261005710/answer/1925679279854851325) 这位佬的知乎也有很多干货

4. 什么是 async proxy？我在 wgmma 和 tma 当中都看见了这个概念

5. `cutlass::arch::fence_view_async_shared()` 这个命令在 DeeGEMM 当中有看到，功能未知

6. 如何利用 mbarrier 构建 sm80 pipeline？

7. 为什么 warp specialization 比 multi-stage 要好？

   [为什么Hopper架构上warp-specialization比multi-stage要好？-zhihu](https://www.zhihu.com/question/11261005710) 之前看到的回答：persistant warp specialization 会隐藏 prologue & epilogue 的时间。但是问题来了：什么是 persistant？

   在 [Nvidia Cute 实战-WarpSpecialization Gemm for Hopper[zhihu](https://zhuanlan.zhihu.com/p/1905383022901059783) 中有提到 persistant 的含义：

   > **Persistent Scheduler and CTA Swizzle**
   >
   > Persistent Scheduler 不同于传统的 data parallel：grid 固定launch CTA 数目=SM数目**（cluster size=2条件下最优的配置）**，保证每个 CTA 运行多个 Gemm Tile 从而可以从第二个 Tile 开始隐藏 prologue 的开销。

   之前总是把 persistant 和 warp specialization 一起出现，但是二者并没有本质上的联系。而 persistant 和 scheduler 联系起来才会显得逻辑更自然

   在 Ampere 架构当中，grid dimension 的划分就是根据 cta tile 来简单划分

   ```cpp
   dim3 gridDim = {ceil(M/BM), ceil(N/BN)}
   ```

   每一个 cta 只会处理自己的 tile。处理完过后上下文就会进行切换，交由其他的 cta 继续完成下一个 tile

   而对于 persistant scheduler，我们固定下来了 cta 的数量，每一个 cta 会处理多个 tile，这样就省略掉了上下文切换的时间，并且在处理连续的两个 tile 时，可以隐藏 tile 间的 prologue 时间

8. 对比 fp16 & fp8 的 gemm 实现当中，我发现 fp8 gemm 没有 mma prologue，也就是先启动一个 mma，然后用 mma tail 进行收尾。另外 fp8 deepgemm 使用了多个 accumulator，为什么要多个累加器？这不会造成寄存器紧张吗？

9. 如何构建流水线的性能模型，其中常见的方式是用流水线图来简单对比

10. issue epilogue 和 issue_mma 之间是否是 async？只有是 async 形式，才能够隐藏掉 tile 之间的 prologue 时间。同时必须要使用 ping-pong consumer 才能隐藏掉 epilogue 的时间

11. tma 会自动判断数据 out of bound 吗？具体的表现和使用形式是怎么样的？

12. 什么是 wave？

13. 如果不使用 one mma in-flight 会降低多少表现？

14. 为什么使用 struct 而不是使用 class 来进行 kernel 构建

    > From Kimi
    >
    > 在 C++ 中，`struct` 和 `class` 在语法上几乎等价，区别只在于默认访问权限；在 CUDA kernel 设计中用 `struct` 更多是出于**约定、简洁性和 POD 风格**的考虑，而不是技术限制。

    简洁性来自于 POD 风格

    > From Kimi & DeepSeek
    >
    > **POD 就是“长得像 C 语言里的结构体”——简单、没有隐藏行为；非 POD 就是“带 C++ 特性的类”——有构造、析构、虚函数、继承等“额外动作”。**
    >
    > 并且 POD 风格只有静态方法，没有成员变量，这样避免了实例化。同时也表征了：内核本身**不需要维护自身状态**（所有数据通过参数传递）

    另外 struct 的成员默认都是 public 属性，而 class 成员默认为 private 属性，自然代码会更加简单

    除此之外我还看到有的函数加了 inline 而有的函数没有加 inline，但实际上都可以不加，他们都是定义在结构体之内的，会被默认当做 inline 函数（隐式 inline）

    > From Kimi
    >
    > **只要函数体写在类（或结构体）的大括号里面，它就是隐式 inline；写在类外就必须自己加 inline，除非它是模板。**

15. constexpr

    让 Kimi 和 DeepSeek 分别对 constexpr 进行了解释

    > From Kimi
    >
    > 在 C++11 之前，想让编译器在编译期就把一个值算出来，只能靠「模板元编程」或「宏」之类的奇技淫巧，代码晦涩难读。
    > C++11 引入了 `constexpr`，让“写普通函数/变量就能在编译期完成计算”成为现实：只要告诉编译器“这东西可以在编译期求值”，编译器就会尽量在编译期把它算出来。
    >
    > 到底什么是 constexpr？
    >
    > 一句话：
    > **constexpr = “此实体（变量/函数/构造函数/析构函数 …）可以在编译期求值，并且满足一定约束”。**
    >
    > - 当它被当作“常量表达式”(constant expression) 使用时，编译器**必须**能在编译期算出结果；
    > - 当它被当作普通变量/函数使用时，仍然可以在运行期使用，**不会损失任何功能**。

    > From DeepSeek
    >
    > 这触及了 `constexpr` 函数的核心特性——**双重性**（编译期和运行时的双重可用性）
    >
    > ```cpp
    > constexpr int add(int n, int m) {
    >     return n + m;
    > }
    > 
    > // 情况1：编译时计算
    > constexpr int sum = add(3, 4);  // ✅ 编译时计算
    > 
    > // 情况2：运行时计算
    > int y = 10;
    > int runtime_sum = add(y, 5);    // 🔴 运行时计算
    > ```
    >
    > **`constexpr` 函数的双重性质**：
    >
    > - 它不是"只能在编译期运行"的函数
    > - 而是"满足条件时**可以**在编译期运行"的函数
    > - 当条件不满足时，自动退化为普通运行时函数

    在进行 kernel 模板编程的时候，一般需要搭配 `static`

    ```cpp
    template <KernelTag kernel_tag_, class CtaTile_, class ClusterShape_,
              int Stage_>
    struct GemmKernelSM90A {
    
      static constexpr int WarpSize = 32;
      static constexpr int WarpGroupSize = 128;
      ...
    }
    ```

    否则编译就会报错，这是因为 Non-`static` data members cannot be declared as `constexpr`. [StackOverflow](https://stackoverflow.com/questions/50332569/why-i-am-getting-this-error-constexpr-is-not-valid-here)

16. 选择哪些参数作为模板元编程的参数？

    模板元编程是在对代码编程，而不是编程本身。利用元编程可以控制在编译时代码的具体实现。有点类似于：不使用 if-else 来实现多种代码的手段。更准确的来说，模板元编程的核心目的在于

    > From DeepSeek
    >
    > **模板元编程是通过编译器对模板的递归实例化和特化机制，在编译期生成类型专属代码或完成计算的技术，本质是将运行时的逻辑判断转移到编译期，实现零开销抽象**

    这对于 if-else 不友好的 CUDA 编程来说是非常有用的。但其实 if-else 不会消失，而是从 kernel 端移动到了 host 端，原因在于：如果你想要运行这样的特化代码，就必须要进行编译，而你想要能够运行所有的特化代码，那就要对所有的特化代码进行编译。所以有两种选择：

    1. 通常使用 host 端的 if-else 来进行特化代码选择

       ```cpp
       // From sglang-kernel
       template <typename OutType>
       void sm90_fp8_dispatch_shape(
           torch::Tensor& out,
           const torch::Tensor& a,
           const torch::Tensor& b,
           const torch::Tensor& scales_a,
           const torch::Tensor& scales_b,
           const c10::optional<torch::Tensor>& bias) {
         uint32_t const m = a.size(0);
         using FastPingpongScheduler = cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum;
         using FastBasicScheduler = cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum;
         using PersistentTileScheduler = cutlass::gemm::PersistentScheduler;
         using BasicTileScheduler = void;
         if (m <= 1) {
           return sm90_fp8_dispatch_bias<
               OutType,
               Shape<_64, _64, _128>,
               Shape<_1, _8, _1>,
               FastBasicScheduler,
               BasicTileScheduler>(out, a, b, scales_a, scales_b, bias);
         }
         if (m <= 64) {
           // m in [1, 64]
           return sm90_fp8_dispatch_bias<
               OutType,
               Shape<_64, _64, _128>,
               Shape<_1, _4, _1>,
               FastPingpongScheduler,
               PersistentTileScheduler>(out, a, b, scales_a, scales_b, bias);
         } else if (m <= 256) {
           // m in (64, 256]
           ...
         } 
         ...
       ```

    2. 使用 jit (just in time) 的方式进行即时编译，从而获得动态的编译代码

    3. Awesome-cute 应该是有参考 cutlass 代码来写自己的 kernel，我需要找到他所参考的核心代码在哪里

    4. mma selector 只对 sm90 存在，在 Blackwell 架构中的 cute 并不存在，我需要知道选择 mma 的启发式规则，这应该得参考下 deepgemm 的实
    
    5. GMMA 和 UMMA
    
       > From Kimi
       >
       > GMMA 与 UMMA 分别是 Hopper 与 Blackwell 两代 GPU 架构提出的新一代 Tensor Core MMA（矩阵乘-累加）指令的统称。
       >
       > 1. GMMA（Hopper） • 在 Hopper 架构的 PTX 里写作 wgmma.mma_async，官方文档/社区也简称 GMMA（Group-level MMA）。
       >    - 以 warpgroup（32×4=128 线程）为执行粒度，异步完成大块矩阵乘法，结果累加在寄存器文件。
       >    - 需要程序员显式管理共享内存→寄存器的数据搬运，并配合 WGMMA 的 pipeline barrier 使用。
       > 2. UMMA（Blackwell） • Blackwell 废弃了 wgmma.mma_async，引入新的 PTX 指令 tcgen05.mma，官方/社区把它叫 UMMA（Universal MMA）。
       >    - 运算粒度更灵活：支持 FP4/FP6 等新精度，原生支持 block-scaling
       >    - 累加器不再占用通用寄存器，而是落到一块叫 Tensor Memory（TMEM）的专用 SRAM
       >    - 由单个线程即可发起，两个 CTA 还能跨 SM 成对协同，进一步降低寄存器压力。因而 CUTLASS 中把原来的 “warp-group” 概念替换为 “CTA-pair” 抽象。
    
    6. load shared 在 deepgemm 当中被使用，在 cute 中应当如何实现
    
    7. 在 warp group consumer 当中是不是不应该使用 `syncthreads` 这样会形死锁，应该会有 warp group 专属的 sync 命令