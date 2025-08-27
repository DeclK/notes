# SVDQuant

[arxiv](https://arxiv.org/abs/2411.05007) [github](https://github.com/nunchaku-tech/deepcompressor)

## Introduction

所面临的问题：当前的主流方法  SmoothQuant (W8A8) 会将 activation outlier 转移到权重当中，但是对于 W4A4 的量化方法，这种 smoothing 方式也将受到更多限制，因为 4-bit 权重无法像 8-bit 权重一样对 outlier 有很好的精度保证。

解决思路：使用一个 low-cost branch 将这些 outlier 进行吸收。具体来说，论文先利用 smoothing 的方式将 activation 的 outlier 移动到 weight 上，然后将 weight 的 outlier 用两个低秩矩阵 $L_1L_2$ 进行吸收。具体来说 weight $W$ 将被分解为两个部分：
$$
W = R + L_1L_2
$$
最终得到的 residual $R$ 会是一个更好量化的矩阵。如此 activation & weight 都能够进行很好的 4-bit 量化

<img src="SVDQuant/image-20250827171801449.png" alt="image-20250827171801449" style="zoom: 67%;" />

论文在 related work 中也提到了其他方法也使用了 low-rank 的方式来做量化，不过他们的缺陷在于没办法做出加速效果，只专注于权重压缩效果。实际上把量化模型进行加速并不简单，这就是写算子的魅力时刻🫡

## Method

矩阵惩罚量化误差的定义
$$
E(\boldsymbol{X},\boldsymbol{W})=\|\boldsymbol{X}\boldsymbol{W}-Q(\boldsymbol{X})Q(\boldsymbol{W})\|_{F}
$$
Frobenius 范数定义
$$
\|A\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij}|^2} = \sqrt{\operatorname{tr}(A^H A)}
$$
论文利用缩放得到了量化误差的一个上界，这个上界会更好分析和优化
$$
E(\boldsymbol{X},\boldsymbol{W}) \leq \|\boldsymbol{X}\|_{F} \|\boldsymbol{W} - Q(\boldsymbol{W})\|_{F} + \|\boldsymbol{X} - Q(\boldsymbol{X})\|_{F} \left( \|\boldsymbol{W}\|_{F} + \|\boldsymbol{W} - Q(\boldsymbol{W})\|_{F} \right)
$$
证明过程如下
$$
\begin{align*}
&\|\boldsymbol{X}\boldsymbol{W} - Q(\boldsymbol{X})Q(\boldsymbol{W})\|_F \\
&= \|\boldsymbol{X}\boldsymbol{W} - \boldsymbol{X}Q(\boldsymbol{W}) + \boldsymbol{X}Q(\boldsymbol{W}) - Q(\boldsymbol{X})Q(\boldsymbol{W})\|_F \\
&\leq \|\boldsymbol{X}(\boldsymbol{W} - Q(\boldsymbol{W}))\|_F + \|(\boldsymbol{X} - Q(\boldsymbol{X}))Q(\boldsymbol{W})\|_F \\
&\leq \|\boldsymbol{X}\|_F \|\boldsymbol{W} - Q(\boldsymbol{W})\|_F + \|\boldsymbol{X} - Q(\boldsymbol{X})\|_F \|Q(\boldsymbol{W})\|_F \\
&= \|\boldsymbol{X}\|_F \|\boldsymbol{W} - Q(\boldsymbol{W})\|_F + \|\boldsymbol{X} - Q(\boldsymbol{X})\|_F \|\boldsymbol{W} - (\boldsymbol{W} - Q(\boldsymbol{W}))\|_F \\
&\leq \|\boldsymbol{X}\|_F \|\boldsymbol{W} - Q(\boldsymbol{W})\|_F + \|\boldsymbol{X} - Q(\boldsymbol{X})\|_F \left(\|\boldsymbol{W}\|_F + \|\boldsymbol{W} - Q(\boldsymbol{W})\|_F\right).
\end{align*}
$$
本质上利用了三角不等式和柯西不等式。这个上界由4个关键因素限制

1. activation & weight 的 F-范数大小
2. activation & weight 的量化误差大小

如果是做 w4a16 量化的话，第二项直接为零，所以直接缩小 activation 的 F-范数将变得非常有收益，这也就是 AWQ 方法的直接体现

## Question

- 什么是 Theorem & Proposition & Lemma & Corollary？

  > From DeepSeek
  >
  > 它们都是指“被证明为真的陈述”，但根据其重要性、功能和在逻辑链条中的位置，被赋予了不同的名称。你的混淆感非常正常，因为它们的区别有时确实有些主观。
  >
  > | 术语            | 中文     | 角色                     | 重要性                           | 依赖性                   |
  > | :-------------- | :------- | :----------------------- | :------------------------------- | :----------------------- |
  > | **Theorem**     | **定理** | 主要成果，理论核心       | **非常高**                       | 相对独立                 |
  > | **Proposition** | **命题** | 次要成果，有用的独立事实 | **中等**                         | 相对独立                 |
  > | **Lemma**       | **引理** | 辅助工具，证明的中间步骤 | **较低**（作为工具）             | **高**（为证明定理服务） |
  > | **Corollary**   | **推论** | 主要结果的直接推论或特例 | **可变**（但低于它所依赖的定理） | **极高**（直接源于定理） |

  