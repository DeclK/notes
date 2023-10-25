# Scaling Distritubed Machine Learning with the Parameter Server

[bilibili](https://www.bilibili.com/video/BV1YA4y197G8)

## Layout

- 问题

  1. 在获得参数时需要大的带宽

  2. 有很多顺序任务（sequential），导致产生一些 barrier，使得同步效率低，延迟大

     gradient accumulation

  3. 大规模的机器中少部分机器出错的可能性非常大，会导致任务停滞，这需要系统有 fault torlerance

     解决方法是做一个实时的备份，或者每隔一段时间做一个备份

# Gpipe

## Layout

- 问题

  模型太大，一个 GPU 放不下

  1. 流水线并行
  2. activation checkpoint，会额外增加 20%~30% 计算开销

  模型并行的致命问题：bubble!!!但是只要你的块数足够多，bubble 的占用比例会越小

  ![mp-pp](./ML System/parallelism-gpipe-bubble.png)

  批量归一化会计算得不一样些，因为要算均值和方差

  衡量 activation checkpoint 的最大存储空间 $O(N+\frac{L}{K}·\frac{N}{M})$ 其中 N 就是数据的 batch size，L 就是模型的总长度，K 是将模型切分成为 K 个部分，你可以看做 K 个 GPU，而 M 就是将 batch size 分成 M 份，每一份为一个 micro batch

  而没有使用 activation 的最大存储空间为 $O(N·L)$，这里其实只是用一个乘积的方式来表示我们将一个 N 大小的批量输入到 L 长度的网络中所需要的存储，而真实的存储关系不是严谨的乘积，单独的 N 应该表示为模型的输入

  而流水线的时间则表示为 $O(\frac{K-1}{M+K-1})$，其中 K 仍然表示模型将切分为 K 个部分，M 仍表示为有 M 个 micro batch。可以看到 micro batch 越大时间越低，因为并行度越高

# Megatron-LM

## Layout

- 张量并行 horizontal parallelism

  实现简单，不需要编译器，Pytorch 里面添加通信的操作就行

  系统简单，牺牲的是通用性，只针对 transformer-based 模型

  下面是 transformer block 中的 MLP 层的 tensor 切分方式，一般来说 X 输入没那么大，但是模型参数量很大，所以这里只讨论了参数的切分，MLP 就是两层的 Linear，对应的参数就是 A 和 B，A 被竖着切分成了2份，B 则被横着切分成了2份，这两份就可以分到2块 GPU 上。这样就能做到通信最少，不信的话，你横着切 A，那么输入 X 就得要被竖着切掉，之后就要进行整合通信。（切分搜索空间）

  ![parallel shard processing](./ML System/parallelism-tp-parallel_shard_processing.png)

  下面是 transformer block 中的 attention 的 tensor 切分方式，其就是将 Head 作为切分代为，在不同的 GPU 上算不同的 Head 就行了

  ![parallel self-attention](./ML System/parallelism-tp-parallel_self_attention.png)

  在 embedding 层也进行了 GPU 的划分，所以也是有并行优化的

  tensor parallelism 的牺牲点就是在每一个层出来过后，都需要做一个 all-reduce 的 collective 操作，这个操作是一个 sequential 操作，无法并行。另外的一个缺点就是，你的层得能切！（又是一个切分搜索空间）

  通讯量比较：

  1. tensor parallel $O(B·L·K·N)$，其中 B 为 batch size，L 为序列长度，K 为隐藏层大小，N 为 transformer block 层数
  2. data parallel $O(K^2N)$，此时只有模型的梯度更新才需要通信

# ZeRO

