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

- 全网 RoPE 唯一指定学习材料 [Transformer升级之路：2、博采众长的旋转式位置编码](https://spaces.ac.cn/archives/8265)

  经过苏神的一顿推理，我们最终找到了符合以下要求的位置编码
  $$
  \operatorname{Re}\left[\boldsymbol{f}(\boldsymbol{q}, m) \boldsymbol{f}^{*}(\boldsymbol{k}, n)\right]=g(\boldsymbol{q}, \boldsymbol{k}, m-n)
  $$
  该要求使得两个向量之间具有了相对位置信息。最后得到的位置编码的形式为
  $$
  f(\boldsymbol{q}, m)=R_{f}(\boldsymbol{q}, m)e^{\mathrm{i}\Theta_{f}(\boldsymbol{q}, m)}=\|\boldsymbol{q}\|e^{\mathrm{i}(\Theta(\boldsymbol{q})+m\theta)}=\boldsymbol{q} e^{\mathrm{i}m\theta}
  $$
  这个结果非常的 clean！就是用复数的形式来表示向量的旋转，故称为旋转位置编码

  旋转位置编码的二维形式是最好理解的
  $$
  \boldsymbol{f}(\boldsymbol{q}, m)=\left(\begin{array}{cc}
  \cos m\theta & -\sin m\theta \\
  \sin m\theta & \cos m\theta
  \end{array}\right)\left(\begin{array}{l}
  q_{0} \\
  q_{1}
  \end{array}\right)
  $$
  对于多维（偶数维）向量的旋转位置编码，利用内积的线性叠加性，可以将二维的形式进行重复的拼接
  $$
  R_m \boldsymbol{q} = 
  \begin{bmatrix}
  \cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
  \sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
  0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\
  0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\
  \vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
  0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\
  0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1}
  \end{bmatrix}
  \begin{pmatrix}
  q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d-2} \\ q_{d-1}
  \end{pmatrix}
  $$
  以上矩阵将仍然满足相对位置编码的要求
  $$
  \left(\mathcal{R}_{m} \boldsymbol{q}\right)^{\top}\left(\mathcal{R}_{n} \boldsymbol{k}\right)=\boldsymbol{q}^{\top} \mathcal{R}_{m}^{\top} \mathcal{R}_{n} \boldsymbol{k}=\boldsymbol{q}^{\top} \mathcal{R}_{n-m} \boldsymbol{k}
  $$
  以矩阵形式来进行计算会浪费算力，因为 $R_m$ 非常稀疏，所以建议把计算分解为 element wise 乘法来进行计算
  $$
  \begin{pmatrix}
  q_0 \\
  q_1 \\
  q_2 \\
  q_3 \\
  \vdots \\
  q_{d-2} \\
  q_{d-1}
  \end{pmatrix}
  \otimes
  \begin{pmatrix}
  \cos m\theta_0 \\
  \cos m\theta_0 \\
  \cos m\theta_1 \\
  \cos m\theta_1 \\
  \vdots \\
  \cos m\theta_{d/2-1} \\
  \cos m\theta_{d/2-1}
  \end{pmatrix}
  +
  \begin{pmatrix}
  -q_1 \\
  q_0 \\
  -q_3 \\
  q_2 \\
  \vdots \\
  -q_{d-1} \\
  q_{d-2}
  \end{pmatrix}
  \otimes
  \begin{pmatrix}
  \sin m\theta_0 \\
  \sin m\theta_0 \\
  \sin m\theta_1 \\
  \sin m\theta_1 \\
  \vdots \\
  \sin m\theta_{d/2-1} \\
  \sin m\theta_{d/2-1}
  \end{pmatrix}
  $$
  第二项会对于输入向量 $\boldsymbol{q}$ 进行重排，这个排列方式比较琐碎，奇偶和正负都在交替。在 huggingface transformers 的实现中，会采用不同的排列方式来实现 RoPE，具体来说旋转位置编码的计算形式变为了如下

  ```python
  def apply_rotary_pos_emb(q, k, cos, sin):
      """
      q & k: (B, H, N, C)
      cos & sin: (B, N, C) 
      """
      rotate_half = lambda x: torch.cat((-x[..., x.shape[-1] // 2 :], 
                                         x[..., : x.shape[-1] // 2]), dim=-1)
      sin = sin.unsqueeze(1)
      cos = cos.unsqueeze(1)
      q_embed = (q * cos) + (rotate_half(q) * sin)    # (B, H, N, C)
      k_embed = (k * cos) + (rotate_half(k) * sin)
      return q_embed, k_embed
  ```

  计算形式发生了比较大的变化，其中 `cos` 会在 C 维度上进行重复即
  $$
  (\cos{m\theta_0}, \cos{m\theta_1},...\cos{m\theta_{d/2-1}},\cos{m\theta_0}, \cos{m\theta_1},...\cos{m\theta_{d/2-1}})
  $$
  和原 RoPE 相比发生了两大变化：

  1. 重新排列了旋转矩阵，把偶数维度放在前半部分，把奇数维度放在后半部分。虽然我们只更改了旋转矩阵，但是 element wise 形式中的 $\mathcal{q}_i$ 也需要随之更改，整体效果是：旋转矩阵和向量都进行了重排，这样消除了正负交替和奇偶交替
     $$
     \begin{pmatrix}
     q_0 \\
     q_2 \\
     q_4 \\
     \vdots \\
     q_{d-5} \\
     q_{d-3} \\
     q_{d-1}
     \end{pmatrix}
     \otimes
     \begin{pmatrix}
     \cos m\theta_0 \\
     \cos m\theta_1 \\
     \cos m\theta_2 \\
     \vdots \\
     \cos m\theta_{d/2-3} \\
     \cos m\theta_{d/2-2} \\
     \cos m\theta_{d/2-1}
     \end{pmatrix}
     +
     \begin{pmatrix}
     -q_1 \\
     -q_3 \\
     -q_5 \\
     \vdots \\
     q_{d-6} \\
     q_{d-4} \\
     q_{d-2}
     \end{pmatrix}
     \otimes
     \begin{pmatrix}
     \sin m\theta_0 \\
     \sin m\theta_1 \\
     \sin m\theta_2 \\
     \vdots \\
     \sin m\theta_{d/2-3} \\
     \sin m\theta_{d/2-2} \\
     \sin m\theta_{d/2-1}
     \end{pmatrix}
     $$

  2. 重新排列了向量，该排列产生了一个映射关系，映射关系如下

     ````python
     ###### even		###### odd
     0	-> 0		1	-> (d/2)
     2	-> 1		3	-> (d/2)+1
     4	-> 2		5	-> (d/2)+2
     ... 			...
     d-2	-> (d/2)-1	d-1 -> d-1
     ````

     这样其实就把第一项的所有 $\mathcal{q}_i$ 变为正常顺序了，而第二项的 $q_i$ 是一个 rotate half 的形式
     $$
     \begin{pmatrix}
     q_0 \\
     q_1 \\
     q_2 \\
     \vdots \\
     q_{d-3} \\
     q_{d-2} \\
     q_{d-1}
     \end{pmatrix}
     \otimes
     \begin{pmatrix}
     \cos m\theta_0 \\
     \cos m\theta_1 \\
     \cos m\theta_2 \\
     \vdots \\
     \cos m\theta_{d/2-3} \\
     \cos m\theta_{d/2-2} \\
     \cos m\theta_{d/2-1}
     \end{pmatrix}
     +
     \begin{pmatrix}
     -q_{d/2} \\
     -q_{d/2+1} \\
     -q_{d/2+2} \\
     \vdots \\
     q_{d/2-3} \\
     q_{d/2-2} \\
     q_{d/2-1}
     \end{pmatrix}
     \otimes
     \begin{pmatrix}
     \sin m\theta_0 \\
     \sin m\theta_1 \\
     \sin m\theta_2 \\
     \vdots \\
     \sin m\theta_{d/2-3} \\
     \sin m\theta_{d/2-2} \\
     \sin m\theta_{d/2-1}
     \end{pmatrix}
     $$

  为什么这样的变化是被允许的？实际上二者相当于引入了两个常量的重排矩阵 $P_1, P_2$，对 $f, g$ 函数进行重新的定义
  $$
  \boldsymbol{f}(\boldsymbol{q}, m)=P_1\mathcal{R}_mP_2\boldsymbol{q}
  $$
  其中 $R_1$ 就是对旋转矩阵的重排，而 $R_2$ 就是对向量的重排。最后的恒等式变为了
  $$
  \left(P_1\mathcal{R}_mP_2 \boldsymbol{q}\right)^{\top}\left(P_1\mathcal{R}_nP_2 \boldsymbol{k}\right)=\boldsymbol{q}^{\top} P_2^{\top}\mathcal{R}_{m}^{\top}P_1^{\top}P_1 \mathcal{R}_{n}P_2 \boldsymbol{k}=\boldsymbol{q}^{\top}P_2^{\top} \mathcal{R}_{n-m} P_2\boldsymbol{k}
  $$
  此时 $g$ 的形式中就多了重排矩阵 $P_2$，不过仍然遵守我们一开始提出的要求👀
  $$
  \operatorname{Re}\left[\boldsymbol{f}(\boldsymbol{q}, m) \boldsymbol{f}^{*}(\boldsymbol{k}, n)\right]=g(\boldsymbol{q}, \boldsymbol{k}, m-n)
  $$

why shared k & v

query means different patterns, k & v means facts, normally the facts are the same, but the pattern can be various

positional embedding chages from rotation to adding bias, this would work if the added bias is good enough

## TODO

1. 负载均衡优化
2. 专家并行（Expert Parallel） & Grouped Gemm
3. MoBA & NSA: MoE in the Attention