# Zen-Nas

[arxiv](https://arxiv.org/abs/2102.01063)

## Concept

- 计算网络的 accuracy 是一件很耗时的事情，论文设计了 Zen-score 作为一个 accuracy proxy，来代表网络表达能力。Zen-score 只需要几次前向计算即可获得

- 整个 nas 搜索方式是 data free 的，只需要 half GPU day

- 之前使用进化算法（Evolution Algorithm）以及强化学习算法（Reinforcement Learning），这两个方法均需要大量计算。为了减小计算提出了 predictor-based 方法，训练一个 predictor 来预测网络的 accuracy（将网络表示为一个向量输入到 predictor 当中）。One-shot 方法是训练一个大的 supernet（不理解）

  之前的方法无法在搜索速度和最终结果之间做好的平衡，在 ImageNet 上的表现也不 SOTA

- Search Space 参考 RegNet

  

## Layout



## Question

- Zen-score 的计算方式