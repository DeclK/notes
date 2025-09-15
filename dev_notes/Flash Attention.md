# Flash Attention

Reference link

[From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf) 对 FA 的数学分解得特别清楚

[FlashAttention核心逻辑以及V1 V2差异总结](https://zhuanlan.zhihu.com/p/665170554) 对 FA 的 CUDA 加速细节有描述，并且伴随了大量的其他参考链接，作者有全面且深厚的功底

[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://crfm.stanford.edu/2023/07/17/flash2.html) FA 作者的简要解读，其中有一个对 FA 计算流程的图解很清晰。如果不考虑 safe softmax 的话，online softmax 会异常简单，只需要对之前的结果进行缩放，再加入当前的结果，即可获得等价计算
$$
x_{i} \leftarrow Q[k,:] K^{T}[:,i]\\
d_{i}^{\prime} \leftarrow d_{i-1}^{\prime} +e^{x_{i}}\\
\boldsymbol{o}_{i}^{\prime} \leftarrow \boldsymbol{o}_{i-1}^{\prime} \frac{d_{i-1}^{\prime}}{d_{i}^{\prime}}+\frac{e^{x_{i}}}{d_{i}^{\prime}} V[i,:]
$$
其中的 $d_i^\prime$ 就是前 $i$ 个数据的 exponential sum

## Question

- 为什么 softmax 无法使用 one-pass 完成，而 attention 却可以使用 one-pass 完成？这样的 iterative 性质能够扩展到其他 op 当中吗？

- FA2 的核心优化点，以及他们如何使用 block 划分问题