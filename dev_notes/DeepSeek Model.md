# DeepSeek Model

本文目标是理清 DeepSeek 模型结构，并理解其中的动机：为什么从 llama-like transformer 发展到了现在的 deepseek-like transformer。从 [DeepSeek V3 Technical Report](https://arxiv.org/pdf/2412.19437v1) 中可以看到，模型结构目前是深度学习中最简单（但需要大量的实验试错）的一个环节，报告中仅用了5页篇幅。剩余大量的篇幅集中在 Infrastructure & Training algorithm

## MoE

随着对深度学习的理解越来越多，似乎有这样一个事实逐渐浮现出来：在 transformer 当中的 MLP 占据了模型的大部分参数，其功能是作为模型的数据库来进行查询。3b1b 做了一个科普视频来进行讲解 [How might LLMs store facts](https://www.bilibili.com/video/BV1aTxMehEjK) 对应的 [blog](https://www.3blue1brown.com/lessons/mlp)

基于此事实，每一次在经过 MLP 时都会对所有的“数据库”进行加载，这样就会导致资源的浪费，因为有的时候我们并不需要所有的信息。此时引入 Miture of Experts (MoE) 就会显得更加自然：我们可以把 MLP 拆分成多个部分，每一个部分被称为一个专家。每次经过 MLP 时只需要加载对应的专家知识，即可获得好的查询结果

基于以上理解 MoE 最大好处有两个：

1. 显著降低单个 token 的计算成本。从另外一个角度来说，在计算成本相同的情况下，模型的容量显著增加，能够存储更多的知识。从计算效率和模型能力的两个角度来说，都有很好的帮助
2. 更强的多模态能力。这一点是直接询问 DeepSeek 获得的🤔，其解释为不同的模态可以选择对应的专家组合，实现分治学习

## MLA

### RoPE

- 一个计算最大 sequence length 的简单方法
  $$
  2\pi · (\text{rope theta})
  $$
  同时通常取该值的一半，因为正弦余弦的对称性所导致

  > From DeepSeek
  >
  > 位置 $\theta$ 和 $2\pi - \theta$ 呈现镜像对称，在实际过程中，这两个位置的语义可能完全不同，用镜像的位置来表达并不合适





why shared k & v

query means different patterns, k & v means facts, normally the facts are the same, but the pattern can be various

positional embedding chages from rotation to adding bias, this would work if the added bias is good enough

## TODO

1. 负载均衡优化
2. 专家并行（Expert Parallel） & Grouped Gemm
3. MoBA & NSA: MoE in the Attention