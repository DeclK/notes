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

## Question

- 什么是 meta tensor

- 如果不在具体的设备上进行运行，怎么知道模型的运行时间？

- 如何测试 symoblic profiler 与真实 profiling 的时间

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