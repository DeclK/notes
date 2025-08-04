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

4. tma copy 会自动调用 [complete_tx](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx-operation)，会减少 tx-count -= bytes

5. 当 pending count = 0 以及 tx-count = 0 时，触发 [phase complete](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-phase-completion) 条件，此时：

   1. phase flip: phase ^= 1
   2. pending count 还原会 `count`

根据以上机制，就可以顺利推理整个流水线的同步过程。另外根据 [zhihu](https://zhuanlan.zhihu.com/p/1905383022901059783) 的说法：mbarrier.wait 只检查current phase 的完成，即 phase = 1, barrier.wait(phase)，若 barrier 内置 phase 为 0，则此 wait 不会等待。这也是为什么一开始要把 producer pipeline_state 的 phase 初始化为 1。因为初始化时不必等待 consumer 完成计算，直接发起 tma load



由于 cutlass doc 当中的代码并没有被 DeepGemm 中采用，而且我所学习的 cute ws 代码也是参考 DeepGemm 来构建的，之后的学习全面针对 awesome cute 当中的代码学习



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

   [zhihu](https://www.zhihu.com/question/11261005710/answer/1925679279854851325) 这位佬的知乎也有很多干货

4. 什么是 async proxy？我在 wgmma 和 tma 当中都看见了这个概念

5.  `cutlass::arch::fence_view_async_shared()` 这个命令在 DeeGEMM 当中有看到，功能未知

6. 如何利用 mbarrier 构建 sm80 pipeline？