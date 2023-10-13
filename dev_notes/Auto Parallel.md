# Auto Parallel

## Links

- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning](https://arxiv.org/abs/2201.12023)
- [Colossal-Auto: Unified Automation of Parallelization and Activation Checkpoint for Large-scale Models](https://arxiv.org/abs/2302.02599)
- [Auto-Parallelizing Large Models with Rhino: A Systematic Approach on Production AI Platform](https://arxiv.org/abs/2302.08141)
- https://github.com/ray-project/ray
- https://github.com/microsoft/DeepSpeed
- https://github.com/alpa-projects/alpa

## Alpa

### Concept

- Auto Parallel

  分成两个部分：自动和并行。并行就是让训练过程在多个设备上同时进行，从而加速训练。而自动就是解决这个并行过程中遇到的一些问题，例如如何让数据并行、如何让模型并行、如何在节点间进行通信等

- Intra-operator & Inter-operator parallelisms

  这是论文中的两个核心概念

  Intra-op 并行就是将算子沿着张量的一个轴或者多个轴（batch or non-batch）进行分割，然后利用分布式设备来同时进行计算（data/tensor/ZeRO parallelism）

  Inter-op 并行是将整个模型切分为几个部分（子图），然后利用分布式设备来进行流水线运作（pipeline parallelism）

  下图使用一个两层的 Linear 来直观展示

  ![image-20231007162157269](/home/lixiang/Projects/notes/dev_notes/Auto Parallel/image-20231007162157269.png)

  Intra-op para 由于需要对数据进行切分和整合，所以需要在设备之间进行大量的交流，这需要高带宽进行支撑。对比于 inter-op para，则不需要这么高带宽来进行设备间的交流

- Alpa

  本论文提出的方法，是一个模型编译器，用于生成执行计划

  > Alpa is a compiler that generates model-parallel execution plans by hierarchically optimizing the plan at two different levels: intra-op and inter-op parallelism

- Megatron-LM

  论文发表时的 sota training system，专门为 transformer 语言模型的并行训练而设计

- ZeRO

  权重分片更新，一种 data para 的优化技巧

- Compilation Pass

  > A compilation pass refers to a specific phase or step in the process of compiling source code into executable machine code. Each pass performs a particular set of tasks on the source code, transforming it in preparation for the next pass. 

  一个具体的例子就是编译 C++ 程序，编译过程需要3个 pass：

  1. Lexical Analysis，这个 pass 扫描源代码，并将其分为有意义的单元
  2. Syntax Analysis，这个 pass 分析语法，并生成 syntax tree
  3. Code Generation，这个 pass 使用 syntax tree 来生成机器码

- Passes in Alpa

  1. Inter-op Pass，将模型分割为子图（stages），分割结果由动态规划算法决定
  2. Intra-op Pass，优化子图在 devices 中的运行速度，执行计划由整数线性规划算法决定
  3. Runtime Orchestration，将上面的最优策略进行编译生成 pipeline 和执行指令，完成不同子图间的数据交流和 resharding

  ![image-20231007172514939](/home/lixiang/Projects/notes/dev_notes/Auto Parallel/image-20231007172514939.png)

- SPMD-style intra-op para

  scalable parallelization for ml computation graphs

  > which partitions operators evenly across devices and executes the same instructions on all devices

- Space of Intra & Inter op para

  这相当于是一个搜索空间，在这些空间中存在一个最优的方案，使得 Inter & Inter op para 获得最优速度

  在 Intra op para 中论文使用了 mat mul 作为例子，我们可以对矩阵乘法中的三个循环 i,j,k 进行组合和并行，来生成一个搜索空间

- Device mesh

  > 2-dimensional logical view of a set of physical devices
  >
  > We assume different groups of devices along the same mesh dimension have the same communication performance

- Sharding Spec 分片规范

  对一个 N 维张量，使用一个规范来表示其某个轴是否被划分。该规范为 $X_0X_1 ···X_{n−1}, where\  X_i ∈ \{S,R\}.$

  $X_i=S$ 表示第 i 个轴被切分，$X_i=R$ 表示第 i 个轴不被切分，会被完整复制

  除此之外还使用上标来表示某轴被分配到 device mesh 中的哪个维度。例如 $S^0$ 代表被切分到 mesh 的横轴，$S^01$ 代表会被同时切分到 mesh 的横轴和竖轴

  resharding 其实就是 reshape，即将数据重新进行分片

- ILP Formulation

  将计算图所花费的时间进行数学抽象

  计算图中的每一个节点代表一个算子，每一个算子需要花费计算时间以及通信时间，而节点与节点之间需要进行 resharding

  总时间就是节点的时间和边的时间
  $$
  \operatorname*{min}_{s}\sum_{\nu\in V}s_{\nu}^{T}(c_{\nu}+d_{\nu})+\sum_{(\nu,u)\in E}s_{\nu}^{T}R_{\nu u}s_{u},
  $$
  其中把一个 stage 仍然抽象为一个图 $(V,E)$，下标 v 代表一个节点（一个或多个算子，这里混用了节点和算子概念），s 则代表所选用的策略，为一个 one-hot 向量，c 为算子内通信时间，d 为算子计算时间，R 为节点之间的通信时间

- DP Formulation

  将计算图所花费的时间进行数学抽象，与 ILP Formulation 不同的是，这里的计算图是整个模型的计算图，而 ILP 则为一个固定的子图
  
  需要通过动态规划来获得最小的计算时间，但是求解时间太久，论文使用了两个技巧：1. early pruning 来提前停止搜索；2. operator clustering 来融合细碎算子
  $$
  T^{*}=\operatorname*{min}_{(n_{1},m_{1}),...,(n_{S},m_{S})}\left\{\sum_{i=1}^{S}t_{i}+(B-1)\cdot\operatorname*{max}_{1\le j\le S}\{t_{j}\}\right\}.
  $$
  其中 t 为某个 stage 所花费的时间，B 为 micro-batches 数量。下面为整个计算图的消耗时间 T 的示意图
  
  ![image-20231010151313826](/home/lixiang/Projects/notes/dev_notes/Auto Parallel/image-20231010151313826.png)
  
- XLA, Ray, Jax, NCCL

  上述四个工具用于论文的代码实现

  - https://github.com/openxla/xla
  - https://github.com/ray-project/ray
  - https://github.com/google/jax
  - https://github.com/NVIDIA/nccl

### Layouts



### Question

- ZeRO Optimizer 同时使用了 Data Para & Op Para? 具体是如何实现。同时为什么 ZeRO 被放在了 intra-op parallelism 当中
- 在 Op Para 中，为什么第二个 matmul 是 replicated，而没有被划分
- 如何将 op 进行分割，从而在多设备上运行
- 在 ILP Formulation 中，这些计算时间、通信时间是如何获得的
- 在 DP Formulation 中，动态规划的代码是如何实现的
- 为什么使用更多的 GPU 反而 out of memory？原因可能在于并行时额外的通信增加，导致显存增加
