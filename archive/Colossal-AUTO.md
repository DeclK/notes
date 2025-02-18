# Colossal-AUTO

[colossal-auto-github](https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/auto_parallel) [arxiv](https://arxiv.org/abs/2302.02599)

## Concept

- Meta Tensor

- d_tensor `ColossalAI/colossalai/tensor/d_tensor/README.md`

- Conversion between tensor sharding specs

  one-step transformation，one-step 应该是指使用一个转换操作。可使用的转换操作：all-gather, shard, all-to-all

  transform path，从 source specs 到 target specs 的所有操作

  heuristic function `dim_diff` 对每一个维度的 specs 差异进行简单评分

  conversion 只获得路径，不获得最终的 communication cost。communication cost 是 alpha-beta cost model [link](https://downey.io/notes/omscs/cse6220/distributed-memory-model-mpi-collectives/)
  $$
  T_{msg}(n) = \alpha +\beta n + (\tau)
  $$
  alpha 是 latency，即数据从网络中的一个点到另一个点所花费的时间

  beta 是带宽倒数，带宽指的是网络单位时间内所发出的数据。注意这和延迟是两个概念

  tau 是网络计算时间，一般不放在公式中

## Layout

### 论文

- 之前的方法都是分别优化 distributed execution plans & activation checkpointing scheduling，这篇论文将二者都使用了

- 论文提供了一个好用的 symbolic profiler，能够快速生成算子所使用的内存和计算耗时数据

- 目前 auto parallel 难点：

  1. 大量的 sharding 搜索空间：一个是对张量的划分，一个是对集群的划分。尤其是对张量的划分，还要考虑算子之间输入输出形状的转换
  2. 额外的优化技巧会对上述 sharding 产生影响。例如 activation recomputing, data offload 会让模型运行的时间变慢，增大优化难度

- 定义一个 pytorch 模型，调用 `auto_parallel` 接口转化为可在分布式系统上执行的模型，参考 `ColossalAI/examples/language/gpt/experiments/auto_parallel/auto_parallel_with_gpt.py`

  ```python 
  model = resnet50(num_classes=10)
  # convert original model to optimized model
  model = autoparallelize(model, input_sample)
  
  # a normal training loop
  for epoch in range ( NUM_EPOCHS ):
      for img , label in data_loader :
          optimizer.zero_grad()
          output = model(img)
          train_loss = criterion(output, label)
          train_loss. backward ( train_loss )
          optimizer.step()
  ```

  实际上主要使用的接口如下

  ```python
  from colossalai.auto_parallel.tensor_shard.initialize import initialize_model
  ```

- Analyzer 系统

  1. `symbolic_profiler`，对 computation graph 进行内存和时间分析。基于 pytorch FX 改进，提供了更好的的算子支持、控制流、meta tensor 支持
  2. `cluster_detector`，alpa 中假设一个节点的设备有相同的通信能力，但实际上会发生：在一个8卡机上，只用了4个 nvlink 进行连接，这样某些没有 nvlink 的 GPU 之间通信就会慢些。论文提供了更好的 `cluster_detector` 来获得更完整的通信信息
  3. `tensor_layout_manager`，对张量的分片规范转换进行记录

- 优化 ILP 策略

  将一些算子进行打包，以避免 ILP 要解决的问题过大。经过打包后的算子类别不超过20个，参考 `ColossalAI/colossalai/auto_parallel/tensor_shard/constants.py`

  ```python
  import torch
  CONV_FUNC_OP = [
      torch.conv1d,
      torch.conv2d,
      torch.conv3d,
      torch.conv_transpose1d,
      torch.conv_transpose2d,
      torch.conv_transpose3d,
  ]
  EMBEDDING_MODULE_OP = [torch.nn.modules.sparse.Embedding]
  LINEAR_MODULE_OP = [torch.nn.Linear]
  LINEAR_FUNC_OP = [torch.nn.functional.linear, torch.matmul, torch.bmm]
  ```

  可以看到把 linear 和 bmm 放到了一个类别当中，这是合理的，它们都将采用相同的 parallel strategy

  只计算前向的耗时

  融合 computational-trival nodes: getitem node, element-wise node, slice node

  移除一些不包含张量的节点，例如一个节点只对两个 scaler 进行相加

- activation checkpointing 使用的是 Rotor algorithm

  这个算法需要将网络进行线性化 linearization，也就是网络一个层的输出只作为下一个层的输入，而不能成为其他层的输入。更具体来说就是用 `nn.Sequential` 来表达网络

- 论文将 intra-op 和 activation checkpointing solver 分开求解，合在一起求解都不知道该咋弄。但是二者都是为了减少 memory 的使用，所以需要给二者进行 memory 分配。到底是给 intra-op 更多的内存，还是让 activation checkpoint 去解决内存。论文的方法是尝试多种分配方案，选择最优方案

### AlphaBetaProfiler

- `alpha_beta_dict`

  存储了一个字典

  ```python
  {(0, 1): (1.9641e-05, 4.7404e-12), ...}
  ```

  key 是一个 tuple，代表了两个 device 的 global rank

  value 也是一个 tuple，分别代表了 alpha 和 beta

  这个字典应该是存储了两两 GPU 之间的通信参数。但实际上代码所计算的结果并不是准确的 alpha & beta，直接用 beta 就行了？
  $$
  beta = \frac{nbytes}{total\ time}
  $$

- `detect_homogeneous_device`

  返回一个字典 `homogeneous_device_dict` 

  ```python
  {4.7404e-12: [(0, 1), (0, 2), (4, 5)],
   beta: [...], ... ...}
  ```

  key 是一个基础的 beta 值

  value 是一个列表，列表中的元素为一个 process group

  这个字典代表的是，这些 group 的通信代价都在对应的 beta 值附近。这说明在这些节点之间进行通信，代价都是相同的

  如果找不到一个 homogeneous group 能够包含所有的节点，那就会采用一个默认的方阵 mesh

- `search_best_logical_mesh`

  搜索最优的 logical mesh，这是对所有的 device 而言的，而没有对其进行划分

  在该 2D logical mesh (Y, X) 角度下，X 轴的通信速度是最快的

- `extract_alpha_beta_for_device_mesh` 

  针对于所使用的 2D logical mesh 进行 profile，查看 X 轴和 Y 轴的 alpha beta 参数

### DeviceMesh

- `convert_map`

  是一个字典，将 global rank 映射到 logical rank 当中

  所谓的 logical rank 其实是一个二维坐标 [0, 0], [0, 1]...

- `create_process_groups_for_logical_mesh`

  一个 process group 就是在一个轴上的 devices

  举一个例子，一个 logical mesh 为 2x3 的集群

  ```txt
  [[0, 1, 2],
   [3, 4, 5]]
  ```

  其 process group 有 2 + 3 = 5 个

  ```txt
  [0, 1, 2]
  [3, 4, 5]
  [0, 3]
  [1, 4]
  [2, 5]
  ```

- 并且将 mesh alpha 和 mesh beta 作为属性放在 DeviceMesh 中

### initialize_model

所有的重头戏应该都包含在这个函数当中了！

- `ColoTracer` & `ColoGraphModule`

  基于 Fx 的图

  具有哪些性质？

- `StrategiesConstructor`

  有三个初始化参数：`graph, device mesh, solver_options`

  其中 solver options 可看做一个结构体，仅包含三个元素

  1. solver preference：standard, DP, TP，不清楚 standard 表示什么 solver，猜测 DP 和 TP 分别表示 data & tensor
  2. dataloader option: replicate, distributed
  3. shard option: standard, shard, shard last axis, full shard

  估计 standard 是不作任何限制


- dataclass decorator in python [zhihu](https://zhuanlan.zhihu.com/p/59657729)

## Question

- 什么是 meta tensor

- 如果不在具体的设备上进行运行，怎么知道模型的运行时间？如何测试 symoblic profiler 与真实 profiling 的时间

- alpa 中的 specs conversion path 是如何计算的

- 修改了 Device mesh 中的一个方法

  ```python
      def _global_rank_to_logical_rank_map(self, tensor, index_list):
          '''
          This method is a helper function to build convert_map recursively.
          '''
          ### chk mark, if tensor only has 1 element, has to process individually
          if tensor.numel() == 1:
              assert tensor.item() == 0, 'this situation happens if you only have one GPU, but give wrong index'
              self.convert_map[0] = [0, 0]
  ```

  不太清楚为什么要使用一个 flatten mesh
  
- ✅alpha beta profile 中

  beta 的计算是对 1GB 的数据计算 all reduce 的时间，此时 latency 时间忽略不计

  alpha 的计算是发送极少量的数据来测量 all reduce 的时间，此时 beta 时间忽略不计

- 基于 Fx 的图具有哪些性质？