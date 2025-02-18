# ZeRO Memory Optimization

[arxiv](https://arxiv.org/abs/1910.02054)

## Concept

- > ZeRO eliminates memory redundancies in data- and model-parallel training while retaining low communication volume and high computational granularity

- Model states

  对大模型来说，大部分显存占用都是由 optimizer states (momentums and variances), gradients, parameters 组成。

- Horizontal & Vertical Model Parallelism

  参考 [huggingface Model Parallelism](https://huggingface.co/docs/transformers/v4.15.0/parallelism) [ColossalAI](https://colossalai.org/docs/concepts/paradigms_of_parallelism/)

  Horizontal 就是 tensor parallelism

  Vertical 就是 pipeline parallelism

  这两种叫法可能还不一样，因为在论文里面 horizontal 指代的是 pipeline parallelism，所以还得随机应变。另外一个结论是，model parallelism 有时候会和 tensor parallelsim 混用，其在 Megatron-LM 中提出，也称为 intra-layer model parallel

- Residual states

  指代 activation, tempt buffers, unusable fragments 等内存占用。Model states 和 residual states 成为显存占用的大头

- ZeRO-DP three stages

  1. Optimizer state partition $P_{os}$
  2. Add gradient partition $P_{os+g}$
  3. Add parameter partition $P_{os+g+p}$

- ZeRO-R

  ZeRO 中优化 residual states 的方法，基本思路如下

  1. activation partition and offload
  2. 定义合理的 tempt buffer size
  3. 管理 tensor 的生命周期，以避免碎片内存

## Layout

- 问题提出

  > DP has good compute/communication efficiency but poor memory efficiency while MP can have poor compute/communication efficiency.

## Questions

- ZeRO-DP 如何保持通讯的高效？high computational granularity 指的是什么
- Batch Norm 在并行当中是怎么处理的？为什么说 batch nome 在 pipeline parallelism 很难实现
- ZeRO-R 是如何保持 activation checkpointing 高效的 GPU-CPU 通信