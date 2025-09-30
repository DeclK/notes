# Linear Algebra: PCA & SVD

From Linear Algebra to the Essence of Eigen

最近在研究一些量化问题，最后回到了对 SVD 的本质理解上。在学习过程中对 SVD 所展现的一些性质能够理解，但是对特征向量的本质仍然无法理解：

1. 什么是特征向量？
2. 为什么会存在特征向量？
3. 特征向量与基之间有什么关系？
4. 特征向量在实际应用中会有什么用处？

希望在这篇笔记过后，能够有一些 intuition 的建立：SVD 与降维之间有着显著的关联，特征向量可以是数据的主成分

## 线性代数的本质

在 3b1b 的视频中，对特征值和特征向量的介绍放在了非常靠后的章节。其开头就说了：如果你不理解特征值和特征向量，估计就是对线性代数中的基本概念不理解。所以，我应该需要把他之前章节中的核心进行掌握，再来看特征值和特征向量的本质

> "There is hardly any theory wnich is more elementary than linear algebra, in spite of the fact that generations of professors and textbook writers have obscured its simplicity by preposterous calculations with matrices."	- Jean Dieudonné

对于这种成熟的概念以及总结性质的问题，交给 AI 来做再好不过了，而且好消息是这些教程不仅仅以视频的形式存在，还被整理成为了网站，我只需要把网站交给 Kimi 就好了！

### Vectors, what even are they?

**向量的三种视角**

| 视角               | 定义                             | 特点                                                |
| ------------------ | -------------------------------- | --------------------------------------------------- |
| **物理视角**       | 空间中的箭头                     | 由长度和方向定义，可平移，二维或三维                |
| **计算机科学视角** | 有序的数字列表                   | 每个位置代表一个维度，顺序很重要，如 `[面积, 价格]` |
| **数学视角**       | 任何可以进行“加法”和“数乘”的对象 | 抽象，强调运算规则而非具体形式                      |

线性代数的核心在于**向量加法**和**数乘**这两种操作。无论是将向量看作空间中的箭头，还是数字列表，真正重要的是**在这两种视角之间自由切换**的能力：

- 对数据分析者：将高维数据可视化，发现模式。
- 对物理学家或图形程序员：用数字精确描述空间与变换。

另外注意到：在教程中的向量都是竖着表示的

### Linear combination, span and basis vectors

1. **基向量（Basis Vectors）**

   - **标准基**：二维空间中，单位向量 î（x 方向）和 ĵ（y 方向）构成标准基。

   - **广义的基**：任何两个**不共线**的向量都可以作为一组基，构建新的坐标系。

   - **基的用途**：通过缩放基向量再相加，可以表示平面中的任意向量。

2. **线性组合（Linear Combination）**

   - 定义：对两个向量 **v** 和 **w**，任意实数 a 和 b，表达式 **a·v + b·w** 就是一个线性组合。

   - 几何意义：线性组合是通过缩放两个向量再相加，得到新的向量。

3. **张成空间（Span）**

   - 定义：一组向量的**所有可能线性组合**构成的集合，称为这些向量的**张成空间**。

4. **线性相关与线性无关**

   - **线性相关**：一组向量中，至少有一个向量可以被其他向量的线性组合表示（即“多余”）。
   - **线性无关**：每个向量都贡献了一个新的维度，移除任何一个都会减少张成空间的维度。

5. **空间的基向量**

   一组向量构成一个空间的**基**，当且仅当：

   1. 它们是**线性无关**的；
   2. 它们的**张成空间是整个空间**。

### Linear transformations and matrices

Space is fixed, it is the ultimate playground, what can be changed is the basis and vector

Keep in mind, it is what we do to the vector, not what we do to the space

1. **什么是线性变换？**

   - **变换**本质上是函数，输入向量，输出另一个向量。

   - **线性变换**在视觉上满足两个条件：
     1. 所有直线变换后仍是直线（不弯曲）；
     2. 原点保持不动（即零向量映射到零向量）。

2. **矩阵与线性变换的关系**

   **矩阵的列**就是基向量变换后的坐标

$$
\begin{bmatrix}
  a & b \\
  c & d
  \end{bmatrix}
  \begin{bmatrix}
  x \\
  y
  \end{bmatrix}
  = x \begin{bmatrix}
  a \\
  c
  \end{bmatrix} + y \begin{bmatrix}
  b \\
  d
  \end{bmatrix}
  = \begin{bmatrix}
  ax + by \\
  cx + dy
  \end{bmatrix}
$$

3. **线性变换的数学性质**

   **线性变换**满足两个数学性质： 

   1. 可加性：$ T(u + v) = T(u) + T(v) $
   2. 齐次性：$ T(ku) = kT(u) $

### Matrix multiplication as composition

1. 矩阵乘法的几何意义：变换的复合

   连续施加两个线性变换（例如先旋转再剪切）的结果，仍然是一个线性变换。

2. 重要性质

   **非交换性**：矩阵乘法不满足交换律，$AB \neq BA$，因为变换的顺序会影响最终结果。

   **结合性**：矩阵乘法满足结合律，$(AB)C = A(BC)$，从变换的角度看是显然的：无论怎么加括号，变换的顺序始终是 C → B → A。

### The determinant

1. **行列式的几何意义：面积/体积的缩放因子**

   行列式衡量一个线性变换对空间**面积（2D）或体积（3D）**的缩放程度。我认为这里的“空间”应当理解为“有限空间”，而不是理解为整个 space。举一个例子，有限空间可以是 2D space 中的一个单位正方形，其面积为1，我们将该单位正方形中的所有向量（点）进行线性变换 $A$ 过后，会形成一个新的区域。该区域的面积就是行列式的大小，即其面积变为 $\det(A)$

2. **行列式为 0 的含义**

   如果行列式为 0，说明整个空间被压缩到更低维度（如 2D 压缩成一条线，或 3D 压缩成一个平面甚至点）。这也意味着矩阵的列向量是**线性相关**的，同时也说明该矩阵是不可逆的。除此之外，**Ax = 0 有非零解** 当且仅当 **det(A) = 0**。从反正法来说明：如果行列式不为零，则矩阵可逆，则空间中的向量映射是一一对应的，任意非零向量，对会被映射到唯一对应的非零向量，此时无法找到 $Ax=0$ 的非零解。所以如果要零空间有解，则行列式必定为0

   **负行列式的含义：空间方向被翻转**

3. **矩阵乘积的行列式**

   两个矩阵相乘后，其行列式等于各自行列式的乘积：
   $$
   \det(M_1 M_2) = \det(M_1) \cdot \det(M_2)
   $$
   这是因为行列式本质上是变换对空间的**缩放因子**，连续变换的缩放因子等于各步缩放因子的乘积。

### Inverse matrices, column space, and null space

1. **线性方程组的几何意义**

    一个线性方程组可以表示为矩阵与向量的乘法：  
   $$
   Ax=v
   $$
   其中 A 是系数矩阵，x 是未知向量，v 是常数向量。**几何上，这相当于寻找一个向量 x，使得线性变换 A 将其映射到向量 v。**

2. **逆矩阵（Inverse Matrix）**

   当变换 A 没有“压缩空间”（即行列式不为零），A 是可逆的，存在逆矩阵 A⁻¹。解方程组的方法：**x = A⁻¹ v**，即“倒带”变换，找到原向量 x。

3. **不可逆的情况（行列式为零）**

   如果 A 的行列式为零，意味着它将空间压缩到更低维度（如平面压缩成线）。此时 A 没有逆矩阵，因为无法从压缩后的结果恢复原始空间。 但方程组仍可能有解，只要向量 v 落在 A 的“输出空间”中。

4. **列空间（Column Space）与秩（rank）**

   列空间是矩阵 A 的所有可能输出向量构成的空间。它的维度称为“秩（rank）”，表示变换后空间的维度。如果 v 不在列空间中，方程组无解。（个人理解：列空间其实就代表了解空间）

5. **零空间（Null Space 或 Kernel）**

   零空间是所有被 A 映射到零向量的输入向量构成的集合。用线性方程组来表示，零空间的向量满足
   $$
   Ax=0
   $$
   如果 A 是“压缩”变换，零空间是非零的，表示多个输入对应同一个输出。反之，如果 A 是一个满秩矩阵，那么仅有全零的情况满足

附注：Nonsquare matrices as transformations between dimensions，非方阵用于维度之间的转换（2D <-> 3D）

### Dot Product

在这一小节中探讨了点积与投机之间的关系，从对偶性和可视化的角度描述了二者之间的等价。不过我觉得这还是不够本质。我在思考的过程中，最终需要得到结论：投影是一种线性变换，这样才能够比较通畅地理解。所以对于这一节的整理就不按照 3b1b 的视频来了

要理解点积和投影之间的关系，首先我们需要对二者进行定义

1. 什么是点积

   点积的定义非常简单，就是两个向量的 element wise 乘积之和
   $$
   a·b = \sum_i a_ib_i
   $$

2. 什么是投影

   对于投影的定义相对来说比较难，最终会需要对面积进行理解？从比较简单的代数角度，投影代表了一个向量 $a$ 在另一个向量 $b$ 上的分量，并按照 $b$ 的模进行缩放
   $$
   proj(a,b)=\lVert a\rVert \lVert b \rVert \cos{\theta}
   $$

3. 投影与点积

   在这里我们似乎还没有将投影与向量之间显示地联系起来，其中的关键在于**模的定义**。模的定义（在此情形下）就是将几何与代数进行联系的桥梁
   $$
   \lVert a\rVert=\sqrt{\sum_i a_i^2}
   $$
   这个公式异常的平常，以至于我忽略了其重要性。通过对模的定义，我们完全可以推到出投影的向量化形式

   根据平面几何的余弦定理
   $$
   c^2 = a^2 + b^2 − 2ab \cos θ
   $$
   余弦定理的证明由 [Euclid's Element](https://en.wikipedia.org/wiki/Law_of_cosines) 欧几里得几何原本中已经得到了证明，没有使用任何向量化语言。

   在向量的语言中用 $a,b$ 来表示向量 $c$ 非常简单（三角形法则），其模的计算用向量化形式表示为
   $$
   \rVert c \lVert = \lVert a-b\rVert = \sqrt{\sum_i (a_i-b_i)^2}=\sqrt{a^2+b^2-2a·b}
   $$
   我用了点积的定义来简化了上述式子的表达。把式子带入到余弦定理当中就可得到
   $$
   a·b =\lVert a\rVert \lVert b \rVert \cos{\theta}
   $$
   虽然我用一些简单定理将投影与点积的等价关系进行了证明，但是这些简单定理的证明也显得没有那么普通了。例如，为什么向量 $c = a - b$ 为什么成立？为什么向量的长度就是几何平均？不过由于这些结论足够平常，我们可以将其作为公理进行看待（虽然他们并不是），最后会或许深入到线性空间的定义

4. **投影是一种线性变换**

   现在更进一步，用线性变换的角度来看待投影。我们可以证明投影是一种线性变换，对向量 $x$ 投影到向量 $a$ 上
   $$
   proj_a(x) = x·a
   $$
   可以很容易证明该变换的可加性和齐次性
   $$
   proj_a(x_1+x_2)=proj_a(x_1)+proj_a(x_2)\\
   proj_a(k·x)=k·proj_a(x)
   $$
   既然投影是一种线性变换，而矩阵即可描述一个线性变换，我们就可以从线性变换的角度来重新审视投影。此时投影矩阵 $a$ 是一个“倒下”的向量，即一个一维矩阵。该矩阵其实是一个降维变换，把向量全部压缩到一维空间当中。最终的结果是 $a$ 中各个值的线性组合，我把其看做在各个坐标分量的线性组合
   $$
   a = (a_1,a_2) = (a_1, 0)+(0, a_2)\\
   x·a = x·(a_1,0)+x·(0,a_2) = x_1a_1 + x_2a_2
   $$
   相当于**将投影分解成为了在各个轴上的分量之和**，以线性组合的角度重新结构了投影

### Change of Basis

其实早在第二节的时候就引入过基向量和标准基的概念了，在这一节讨论了基的变换，并且简单引出了相似变换的本质意义

1. 问题引出

   对于同一个向量，使用不同的基向量表示也不相同。例如，使用标准基来表示 $(1, 0)$ 向量，需要只需要 1 个 $\hat i$ 向量，其坐标也是 $(1, 0)$；使用基向量 $(1, 1)$ & $(1, -1)$ 来表示，则需要两个基向量的一半，其坐标表示为 $(0.5, 0.5)$

2. 基变换矩阵

   基变换矩阵 $A$ 是一个矩阵，**其列向量代表新坐标系的基向量在当前坐标系中的坐标。**（个人觉得）不失一般性当前坐标系可认为是标准基构成的坐标系

   **用途**：用于将向量从**新坐标系**转换到**当前坐标系**。反之，则需要基变换矩阵的逆矩阵

3. 基变换矩阵与相似变换

   考虑如何在新坐标系下描述一个旋转变换。在新坐标系下的向量 $x$，首先将其转变为标准基下的向量表示，然后再使用旋转矩阵，最后再用逆矩阵把向量还原为新坐标系下的向量表示
   $$
   \text{Rotation in New Basis}=A^{-1}MA
   $$
   中间的矩阵 $M$ 表示在当前坐标系中的旋转变换，其可以延伸为任意的线性变换。此时理解相似变换就变得更加容易了：即为线性变换 $M$ 在不同坐标系下的矩阵表示，所以称为相似

   > An expression like $A^{-1}MA$ suggests a mathmatical sort of empathy
   >
   > 表达式 $A^{-1}MA$ 暗示着一种数学上的转移作用

### Eigenvectors and eigenvalues

虽然特征向量本身的定义是非常直观的，但是无法从该定义出发，推导出特征值和特征向量更深层的含义。我在 SVD 中观察到，特征向量的求解，并不是针对于任意矩阵本身，而是针对于一个对称矩阵 $M^TM$，其中的 $M$ 可以是任意的矩阵。在这样的意义下，特征值和特征向量在其定义之上延伸出了更具体的性质，使得特征值和特征向量变得不再空虚

所以在这一小节我只对特征值和特征向量的基本定义进行整理，具体的性质讨论会在 SVD 部分进行深入

- **特征向量与特征值**

  在线性变换中，某些向量在变换后仍然保持在原来的方向上，这些向量被称为特征向量。特征值即为特征向量在变换中对应的缩放因子

- 特征值和特征向量的计算方法

  核心思路是求解
  $$
  A\eta=\lambda \eta
  $$
  延伸成为求解
  $$
  (A-\lambda I)\eta=0
  $$
  线性组方程的零空间有非零解，说明其行列式为零。通过行列式为零，可以获得特征值
  $$
  |A-\lambda I|=0
  $$
  找到了特征值过后，通过高斯消元法可以求得 $(A-\lambda I)\eta=0$ 的解，即为特征向量

- 可对角化

  可对角化的意义在此处也不是很明确，我仅介绍概念

  如果一组基向量都是特征向量，那么线性变换在这组基下的矩阵表示是对角矩阵，对角线上的元素即为对应的特征值，此时我们也说该矩阵是可对角化的

  可对角化在谱定理处才会显示出真正的作用，而谱定理的直接推论就是：**实对称矩阵一定是可对角化的**

### Abstract vector spaces

这一章节是线性代数的向量再探讨：**“向量到底是什么？”**，最终对向量进行了高度且准确的抽象：向量是**任何可以相加和数乘的对象**。更严谨的来说，任何东西只要满足八条公理，就可以是向量，并且这些向量构成了一个抽象的向量空间（也叫**线性空间**）

- **函数也可以是向量**

  因为函数可以像向量一样**相加**和**数乘**：
  $$
  (f+g)(x)=f(x)+g(x)\\
  (c⋅f)(x)=c⋅f(x)
  $$
  导数操作可以写成一个**无限维矩阵**，主对角线上方偏移一位填充整数

- **抽象向量空间的定义**

  八条公理，满足就是线性空间：

  - 加法封闭、交换、结合
  - 零向量存在
  - 加法逆元存在
  - 数乘封闭
  - 数乘对向量加法的分配律
  - 数乘对标量加法的分配律
  - 数乘与标量乘法兼容
  - 1 乘向量不变

  > Basically it's a checklist to be sure the notions of vector addition and scalar multiplication are reasonable.
  >
  > These axioms are not so much fundamental rules of nature as they are an interface between you, the mathematician discovering results, and other people who may want to apply those results to new sorts of vector spaces.

  至于这八条公理是怎么产生的？公理的产生是数学家从无数例子当中总结出来的不言自明的规则，并且这八条公理并不是完全独立的，他们也是冗余的，有人说只需要其中的六条即可 [Is every axiom in the definition of a vector space necessary?](https://math.stackexchange.com/questions/1412899/is-every-axiom-in-the-definition-of-a-vector-space-necessary/2385192#2385192)

## SVD

[低秩近似之路（二）：SVD](https://www.spaces.ac.cn/archives/10407)

[SVD分解(一)：自编码器与人工智能](https://www.spaces.ac.cn/archives/4208) 

[白板机器学习-降维](https://www.bilibili.com/video/BV1aE411o7qd)

### 谱定理

对于任意实对称矩阵 $ \boldsymbol M \in \mathbb{R}^{n \times n}$ 都存在谱分解（也称特征值分解）
$$
\boldsymbol{M} = \boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U}^\top
$$
 其中 $\boldsymbol U,\boldsymbol{\Lambda} \in \mathbb{R}^{n \times n}$，并且 $\boldsymbol{U}$ 是正交矩阵 $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \cdots, \lambda_n)$ 是对角矩阵

谱定理的几何特性：实对称矩阵代表的线性变换会将单位圆缩放为椭圆，椭圆的缩放方向为特征向量，在 [zhihu](https://zhuanlan.zhihu.com/p/1914024366347388318) 中有二维的可视化结果。这说明实对称矩阵的变换没有旋转（rotation），也没有剪切（shear），只有单纯的缩放，因为任何的旋转和剪切效应都会破坏实对称性质

证明谱定理最简单的方式是数学归纳法。假设对于 $n-1$ 维的实对称矩阵能够被对角化，证明 $n$ 维的实对称矩阵能够被对角化。对于一维的矩阵，显然成立，第一块积木推倒。现在对于实对称矩阵 $ \boldsymbol M \in \mathbb{R}^{n \times n}$，设 $\lambda_1$ 是其一个非零特征值（可以证明实对称矩阵至少有一个特征值），对应的特征向量为 $\boldsymbol{u}_1$。将 $\boldsymbol{u}_1$ 扩展成为一组正交基，其余的基向量为 $\boldsymbol{Q} = (\boldsymbol{q}_2, \ldots, \boldsymbol{q}_n)$

现在将矩阵 $(\boldsymbol u_1, \boldsymbol Q)^\top \boldsymbol M (\boldsymbol u_1,\boldsymbol Q)$ 表示为分块矩阵：
$$
(\boldsymbol u_1, \boldsymbol Q)^\top \boldsymbol M (\boldsymbol u_1,\boldsymbol Q) = \begin{pmatrix}
     \lambda_1 & \boldsymbol{0} \\
     \boldsymbol{0} & \boldsymbol{B}
     \end{pmatrix}
$$
其中 $\boldsymbol B=\boldsymbol Q^\top \boldsymbol M \boldsymbol Q$，是一个 $n-1$ 维的实对称矩阵。根据数学归纳假设，该矩阵可以被对角化，所以将其表示为 $\boldsymbol{B} = \boldsymbol{V} \boldsymbol{\Lambda}_1 \boldsymbol{V}^\top$，其中 $\boldsymbol V$ 是 $n-1$ 维的正交矩阵，通过将 $\boldsymbol B$ 展开，然后把 $\boldsymbol V$ 移到左侧可得到
$$
(\boldsymbol Q \boldsymbol V)^\top \boldsymbol M (\boldsymbol Q \boldsymbol V) = \boldsymbol{\Lambda}_1
$$
于是乎，我们可以得到
$$
(\boldsymbol u_1, \boldsymbol Q \boldsymbol V)^\top \boldsymbol M (\boldsymbol u_1,\boldsymbol Q \boldsymbol V) = \begin{pmatrix}
     \lambda_1 & \boldsymbol{0} \\
     \boldsymbol{0} & \boldsymbol{\Lambda_1}
     \end{pmatrix}
$$
此时我们就证明了 n 维实对称矩阵也是可以被对角化的

### SVD 的存在性证明

有了谱定理过后能够比较轻松地证明 SVD 存在性。为了更符合机器学习的习惯，我这里用 $\boldsymbol M \in \mathbb{R}^{n \times k}$ 来表示，其中可以理解为 $n$ 数据的数量，而 $k$ 则表示数据的维度，这个 notation 在之后讨论 PCA 的时候也会使用。现在我们先拿出 SVD 的结论：

对于任意矩阵 $\boldsymbol M \in \mathbb{R}^{n \times k}$，都可以找到如下形式的奇异值分解（SVD，Singular Value Decomposition）
$$
\boldsymbol M=\boldsymbol U \boldsymbol \Sigma \boldsymbol V^\top
$$
其中 $\boldsymbol U \in \mathbb R^{n\times n}, \boldsymbol V\in \mathbb R^{k\times k}$ 都是正交矩阵，$\boldsymbol\Sigma \in \mathbb R^{n\times k}$ 是非负对角阵
$$
\boldsymbol{\Sigma}_{i, j}=\left\{\begin{array}{ll}
\sigma_{i}, & i=j \\
0, & i \neq j
\end{array}\right.
$$
并且对角线元素从大到小排布：$\sigma_1\ge \sigma_2 \ge \sigma_3\ge...\ge0$，这些对角线元素就称为奇异值

在证明 SVD 的存在性之前，可以直观推倒：如果矩阵是一个实对称矩阵，那么 SVD 结论就退化成为谱定理，即：谱定理是 SVD 中的特例情况。但事实还要更有趣一点，谱定理和 SVD 可以有更复杂的互动：我们可以简单地构造实对称矩阵 $\boldsymbol M^\top \boldsymbol M $ 或者 $\boldsymbol M \boldsymbol M^\top$，这样的矩阵就符合谱定理。而如果我们将 SVD 的结论带入到上面构造的对称矩阵当中
$$
\boldsymbol{M}\boldsymbol{M}^{\top}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}\boldsymbol{V}\boldsymbol{\Sigma}^{\top}\boldsymbol{U}^{\top}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{\top}\boldsymbol{U}^{\top}\\
\boldsymbol{M}^{\top}\boldsymbol{M}=\boldsymbol{V}\boldsymbol{\Sigma}^{\top}\boldsymbol{U}^{\top}\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}=\boldsymbol{V}\boldsymbol{\Sigma}^{\top}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}
$$
有趣的事情发生了，我们直接得到了 $\boldsymbol{M}\boldsymbol{M}^{\top}$ 和 $\boldsymbol{M}^{\top}\boldsymbol{M}$ 的谱分解结果！换句话说：SVD 分解中的 $\boldsymbol U, \boldsymbol V, \boldsymbol \Sigma$ 其实就是谱分解中对应的特征向量、特征值的平方根

其实以上互动也给我们证明 SVD 的存在性提供了很好的方向：从某一个实对称矩阵的谱分解出发（e.g. $\boldsymbol{M}^{\top}\boldsymbol{M}$），获得其特征向量（e.g. $\boldsymbol V$），利用 SVD 的形式直接构造出另外一组向量 $\boldsymbol U$，我们只需要证明这一组向量是相互正交的，即可证明 SVD 的存在性

我们设 $\boldsymbol{M}^{\top}\boldsymbol{M}$ 的谱分解为 $\boldsymbol{M}^{\top}\boldsymbol{M} =\boldsymbol V \boldsymbol \Lambda\boldsymbol V^\top$，并且不失一般性设 $\boldsymbol M$ 的 rank 为 $r$，$\boldsymbol \Lambda$ 的特征值排列为降序排列，并且由于 $\boldsymbol{M}^{\top}\boldsymbol{M}$ 是半正定矩阵，其特征值都为非负数

OK，一切准备就绪，开始构造 $\boldsymbol U$
$$
\boldsymbol{\Sigma}_{[:r,:r]}=(\boldsymbol{\Lambda}_{[:r,:r]})^{1/2},\quad \boldsymbol U_{[:n,:r]}=\boldsymbol M\boldsymbol V_{[:k,:r]}\boldsymbol{\Sigma}_{[:r,:r]}^{-1}
$$
由于 rank 的限制，只构造了 $r$ 个向量。现在我们来证明这 r 个向量是相互正交的
$$
\begin{aligned}
\boldsymbol{U}_{[:n,:r]}^{\top}\boldsymbol{U}_{[:n,:r]} & =\boldsymbol{\Sigma}_{[:r,:r]}^{-1}\boldsymbol{V}_{[:k,:r]}^{\top}\boldsymbol{M}^{\top}\boldsymbol{M}\boldsymbol{V}_{[:k,:r]}\boldsymbol{\Sigma}_{[:r,:r]}^{-1} \\
& =\boldsymbol{\Sigma}_{[:r,:r]}^{-1}\boldsymbol{V}_{[:k,:r]}^{\top}\boldsymbol{V}\boldsymbol{\Lambda}\boldsymbol{V}^{\top}\boldsymbol{V}_{[:k,:r]}\boldsymbol{\Sigma}_{[:r,:r]}^{-1} \\
& =\boldsymbol{\Sigma}_{[:r,:r]}^{-1}\boldsymbol{I}_{[:r,:k]}\boldsymbol{\Lambda}\boldsymbol{I}_{[:k,:r]}\boldsymbol{\Sigma}_{[:r,:r]}^{-1} \\
& =\boldsymbol{\Sigma}_{[:r,:r]}^{-1}\boldsymbol{\Lambda}_{[:r,:r]}\boldsymbol{\Sigma}_{[:r,:r]}^{-1} \\
& =\boldsymbol{I}_{r}
\end{aligned}
$$
可以看到，基于谱分解的结论，很快就消除了各个部分，得到了正交结论。此时可以直接把 $\boldsymbol U_{[:n,:r]}$ 扩充成为完整的正交基即可得到完成的 $\boldsymbol U$ 矩阵。但是我们其实还没有直接证明 SVD 的存在，因为不管是我们构造的 $\boldsymbol U_{[:n,:r]}$ 还是扩充的 $\boldsymbol U$ 都没有直接地以 SVD 的形式给出，$\boldsymbol M$ 始终在等式的右侧并且与其他矩阵相乘。不过我们现在手上的材料足够多，要直接证明 SVD 的形式已经不难了

我们首先从未扩充版本的 SVD 形式开始推，直接带入我们之前的 $\boldsymbol U_{[:n,:r]}$ 结论
$$
\boldsymbol{U}_{[:n,:r]}\boldsymbol{\Sigma}_{[:r,:r]}\boldsymbol{V}_{[:k,:r]}^{\top}=\boldsymbol{M}\boldsymbol{V}_{[:k,:r]}\boldsymbol{\Sigma}_{[:r,:r]}^{-1}\boldsymbol{\Sigma}_{[:r,:r]}\boldsymbol{V}_{[:k,:r]}^{\top}=\boldsymbol{M}\boldsymbol{V}_{[:k,:r]}\boldsymbol{V}_{[:k,:r]}^{\top}
$$
现在只需要证明 $\boldsymbol{M}\boldsymbol{V}_{[:k,:r]}\boldsymbol{V}_{[:k,:r]}^{\top} = \boldsymbol M$ 然后扩充我们的结论为完整的 SVD 即可
$$
\begin{aligned}
\boldsymbol{M} & = \boldsymbol{M} \boldsymbol{V} \boldsymbol{V}^{\top} \\
  & = \begin{pmatrix} \boldsymbol{M} \boldsymbol{V}_{[:k,:r]} & \boldsymbol{M} \boldsymbol{V}_{[:k, r:]} \end{pmatrix} \begin{pmatrix} \boldsymbol{V}_{[:k,:r]}^{\top} \\ \boldsymbol{V}_{[:k, r:]}^{\top} \end{pmatrix} \\
  & = \begin{pmatrix} \boldsymbol{M} \boldsymbol{V}_{[:k,:r]} & \boldsymbol{0}_{k \times (k-r)} \end{pmatrix} \begin{pmatrix} \boldsymbol{V}_{[:k,:r]}^{\top} \\ \boldsymbol{V}_{[:k, r:]}^{\top} \end{pmatrix} \\
  & = \boldsymbol{M} \boldsymbol{V}_{[:k,:r]} \boldsymbol{V}_{[:k,:r]}^{\top}
\end{aligned}
$$
上述证明的关键就是在第三个等式，利用了特征值为0的特征向量的性质：$\boldsymbol M \boldsymbol v_i=0$

最后将 $\boldsymbol{U}_{[:n,:r]}\boldsymbol{\Sigma}_{[:r,:r]}\boldsymbol{V}_{[:k,:r]}^{\top}=\boldsymbol M$ 进行扩充，即可得到完整的 SVD 形式
$$
\boldsymbol M=\boldsymbol U \boldsymbol \Sigma \boldsymbol V^\top
$$

### 降维

废了不少劲证明了 SVD 的存在性，但是还是没体会到其妙处。现在我们通过机器学习中的降维为例子，来一探 SVD 的妙处。

参考 [zhihu-白板机器学习-降维](https://zhuanlan.zhihu.com/p/326074168)

首先我们可能希望了解：我们为什么需要降维？我们从一个高维的反直觉思维开始：在高维当中单位球体的体积趋近于 0，而单位立方体的体积永远都是1，高维球体体积的计算公式如下：
$$
V_n = \frac{\pi^{n/2}}{\Gamma(n/2+1)}
$$
分母伽马函数的增长速度远超分子，所以随着维度的增加，体积迅速减少。这体现了高维空间的数据稀疏性。稀疏性的本质来源于维度灾难，这里有另一个直观的解释

> From DeepSeek
>
> 1. **一维空间（一条线）**：
>
>    假设我们有一条长度为1米的线段。我们在这条线上均匀地撒上100个点。这些点会非常密集，相邻点之间的距离大约是1厘米。空间感觉很“满”。
>
> 2. **二维空间（一个正方形）**：
>
>    现在，我们扩展到一个1米 x 1米的正方形平面。为了保持和线上同样的“密度”（即单位面积内的点数），我们需要多少点？我们需要 10,000 个点！因为现在点不仅要覆盖长度，还要覆盖宽度。如果还是只有100个点，它们在这个平面上就会显得非常稀疏、孤零零的。
>
> 3. **三维空间（一个立方体）**：
>
>    再进一步，到一个1米 x 1米 x 1米的立方体。要保持同样的密度，我们需 1,000,000 个点！如果还是只有100个点，那么这些点就像宇宙中寥寥无几的星星，彼此之间的距离非常遥远。

球壳上的体积变得异常的大，你可以想象将多维球壳向外扩展 10%，将会带来体积的爆炸性增长，这就是因为每一个维度都要向前扩展 10% 形成了维度灾难，此时增加的球壳体积会远远超过球体的体积

维度灾难带来的部分挑战：

1. 距离度量失效：高维空间中的各个点的距离都差不多（e.g. 数据都集中在球壳上，距离原点的距离都是一样的），导致了以距离为基础的机器学习方法直接失效（KNN、聚类）
2. 采样困难：在高维空间中的采样随着维度增加指数上升

所以降维能够带来计算和度量上的便利，并且我们常常希望找到事物中的核心影响因素，保留最重要的维度，所以降维的好处是不言而喻的。不过我觉得仍要从两面性来看待维度灾难：

1. 希望简化计算复杂度时，使用降维能够极其有效地带来收益
2. 希望增加模型能力/容量时，利用维度的上升也可带来模型容量的快速提升

#### 最大投影方差

最大投影方差的思路是：寻找一个方向，数据点在该方向上的投影方差最大（i.e. 该方向是最能区分数据之间差别的方向）。接下来做一些 Notation：

1. 数据样本 $X \in \mathbb R^{n\times k}$，数据样本个数为 $n$，维度为 $k$

2. 中心矩阵 $H \in \mathbb R^{n\times n}$，其作用是将数据进行中心化，等价于 `X - torch.mean(X, dim=0, keep_dim=True)`
   $$
   H = I_n - \frac{1}{n}1_n1_n^\top
   $$
   其中 $1_n$ 为一个全 1 的向量 $1_n \in \mathbb R^{n\times1}$。中心化过后的数据样本即为 $HX = X-\overline X$。中心矩阵有两个常用的性质 $H^T=H$，$H^2=H$，其中第二个性质也比较好理解：对中心化过后的数据再做中心化是不变的

3. 数据的方差矩阵 $S \in \mathbb R^{k\times k}$表示
   $$
   S=(X-\overline X)^\top(X-\overline X)=(HX)^\top(HX)=X^\top H X
   $$
   真正的方差就是求对 $S$ 求和，然后除以样本个数

4. 

接下来可以直接根据思路写出优化目标：投影的方差最小。我们先写出投影的方差矩阵
$$
L=(Xu)^TH(Xu)=u^\top X^\top H Xu=u^\top Su
$$
接着可以写出优化目标，找到最优的方向 $u$ 使得方差最大
$$
L=\operatorname*{argmax}_{u} u^{T}Su \\ \text{s.t.} \ u^{T}u=1
$$
解这个方法直接用 lagrange 乘子法
$$
Lagrange(u, \lambda) = u^{T} S u + \lambda(1 - u^{T} u) \\
\frac{\partial L}{\partial u} = 2 S u - 2 \lambda u = 0 \\
S u = \lambda u
$$
**最终发现，我们要找到的 $u$ 就是矩阵 $S$ 的特征向量，$\lambda$ 就是对应的特征值！此时最优的 $L=\lambda$，那么我们就选择特征值最大的特征向量即可**。此时我们可以根据特征值的大小，选择最重要的 topk 个特征向量作为投影方向

#### 最小重构代价

我认为最小重构代价更为直观一点，更能贴合我们降维的目的：维度降低，但是信息量减少最少。在 [zhihu-白板机器学习-降维](https://zhuanlan.zhihu.com/p/326074168) 中使用了构造一个正交基的方法来证明最小重构代价，最后只需要放弃特征值最小的特征向量即可。这里我整理我自己的思路，最终推导出和最大投影方差相同的结论

我同样聚焦于一个向量：把（已中心化）数据投影到向量上 $u$，计算得到投影过后的新数据坐标，要求新数据与原始数据的 F-范数最小
$$
L=\operatorname*{argmin}_u ||X-(Xu)u^\top||_F^2
$$
接下来的证明求助了 DeepSeek，我自己是没证出来的，其中运用了 trace 的性质

> From DeepSeek
>
> 由于Frobenius范数的平方等于矩阵迹的形式，有：
> $$
> J(\mathbf{u}) = \operatorname{tr} \left( (\mathbf{X} - \mathbf{X} \mathbf{u} \mathbf{u}^T)^T (\mathbf{X} - \mathbf{X} \mathbf{u} \mathbf{u}^T) \right)
> $$
> 展开括号内的表达式
> $$
> (\mathbf{X} - \mathbf{X} \mathbf{u} \mathbf{u}^T)^T (\mathbf{X} - \mathbf{X} \mathbf{u} \mathbf{u}^T) = \mathbf{X}^T \mathbf{X} - \mathbf{X}^T \mathbf{X} \mathbf{u} \mathbf{u}^T - \mathbf{u} \mathbf{u}^T \mathbf{X}^T \mathbf{X} + \mathbf{u} \mathbf{u}^T \mathbf{X}^T \mathbf{X} \mathbf{u} \mathbf{u}^T
> $$
> 令 $\mathbf{S} = \mathbf{X}^T \mathbf{X}$，带入上式得到
> $$
> J(\mathbf{u}) = \operatorname{tr} \left( \mathbf{S} - \mathbf{S} \mathbf{u} \mathbf{u}^T - \mathbf{u} \mathbf{u}^T \mathbf{S} + \mathbf{u} \mathbf{u}^T \mathbf{S} \mathbf{u} \mathbf{u}^T \right)
> $$
> 由 trace 性质可以将其展开 [wiki-trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra)) 为四项
>
> - $\operatorname{tr}(\mathbf{S})$ 是常数，与 $\mathbf{u}$ 无关。
> - $\operatorname{tr}(\mathbf{S} \mathbf{u} \mathbf{u}^T) = \operatorname{tr}(\mathbf{u}^T \mathbf{S} \mathbf{u}) = \mathbf{u}^T \mathbf{S} \mathbf{u}$（因为 $\mathbf{u}^T \mathbf{S} \mathbf{u}$ 是标量）。参考 [wiki-trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra)) 有性质：$\operatorname tr(AB)=\operatorname tr(BA)$
> - $\operatorname{tr}(\mathbf{u} \mathbf{u}^T \mathbf{S}) = \operatorname{tr}(\mathbf{u}^T \mathbf{S} \mathbf{u}) = \mathbf{u}^T \mathbf{S} \mathbf{u}$。
> - $\operatorname{tr}(\mathbf{u} \mathbf{u}^T \mathbf{S} \mathbf{u} \mathbf{u}^T)$：注意 $\mathbf{u}^T \mathbf{S} \mathbf{u}$ 是标量，记为 $c$，则 $\mathbf{u} \mathbf{u}^T \mathbf{S} \mathbf{u} \mathbf{u}^T = c \mathbf{u} \mathbf{u}^T$，所以 $\operatorname{tr}(c \mathbf{u} \mathbf{u}^T) = c \operatorname{tr}(\mathbf{u} \mathbf{u}^T) = c \mathbf{u}^T \mathbf{u} = c$（因为 $\mathbf{u}^T \mathbf{u} = 1$）。因此，这一项也是 $\mathbf{u}^T \mathbf{S} \mathbf{u}$.
>
> 所以整个优化方程化简为：
> $$
> J(\mathbf{u}) = \operatorname{tr}(\mathbf{S}) - \mathbf{u}^T \mathbf{S} \mathbf{u} - \mathbf{u}^T \mathbf{S} \mathbf{u} + \mathbf{u}^T \mathbf{S} \mathbf{u} = \operatorname{tr}(\mathbf{S}) - \mathbf{u}^T \mathbf{S} \mathbf{u}
> $$
> 由于第一项为常数，所以最终的优化目标变为
> $$
> J(\mathbf{u}) = \operatorname*{argmin}_u- \mathbf{u}^T \mathbf{S} \mathbf{u}
> $$

把优化目标再一转换，马上就得到了和最大投影方差相同的结论。我们只要选择特征值最大的特征向量，就能够最好地减少重构前后矩阵的误差

#### SVD 与降维

OK，又绕了一大圈，讲了降维。那么 SVD 与降维之间有什么关系？现在提出 intuition：**对数据做 SVD 就是在对其进行降维**。可以看到 SVD 当中的特征向量矩阵 $\boldsymbol V$，其实就是 $\boldsymbol{M}^{\top}\boldsymbol{M}$ 的特征向量矩阵，而如果我们把 $\boldsymbol M \in \mathbb R^{n\times k}$ 看做中心化过后的数据 $X \in \mathbb R^{n\times k}$，**此时 $\boldsymbol V$ 就是 PCA 中一直寻找的主成分！**而 $\boldsymbol U \boldsymbol \Sigma$ 就是投影过后的坐标，因为 $\boldsymbol M \boldsymbol V =\boldsymbol U \boldsymbol \Sigma$。另外由于矩阵 $\boldsymbol U$ 的正交性，其坐标向量是相互正交的

#### SVD 与低秩近似

虽然通过以上降维的论证，我们知道如何计算主成分，但是低秩近似的证明与降维当中的优化目标并不一样，不过神奇的是二者的答案都指向了 SVD。**这更一步加深了 SVD 与低秩相关的 intuition是**

定义低秩近似的优化问题：
$$
\underset{A,B}{argmin}\|AB-M\|_{F}^{2}
$$
其中 $A\in\mathbb{R}^{n\times r},B\in\mathbb{R}^{r\times m},M\in\mathbb{R}^{n\times m},r<\min(n,m)$，说白了，这就是要寻找矩阵 $ M $的“最优 $ r $ 秩近似”。这里直接给出结论，最优解和最小值分别为
$$
A=U\Sigma,B=V^T\\
\min_{A,B}|AB-M\|_{F}^{2}=\sum_{i=1}^r{\sigma_i^2}
$$
其中 $U,\Sigma,V,\sigma$ 就是 $M$ 的奇异值分解中对应的各个矩阵和奇异值，为了满足 rank 的要求，我们只取前 $r$ 个特征值/特征向量即可，即：$U\in \mathbb R^{n\times r}, \Sigma \in \mathbb R^{r\times r}, V \in \mathbb R^{m\times r}$。注意：该解为最优解之一，不保证唯一性，可能存在其他矩阵也能达到最小值

这一节本质就是对 **Eckart-Young-Mirsky定理** 的在F-范数下的证明。在这个过程中也自然地引出了伪逆的定义。证明过程还是比较复杂，这里我只简单记录两个知识点

1. Moore-Penrose 伪逆的定义，[低秩近似之路（一）：伪逆](https://spaces.ac.cn/archives/10366)

   伪逆定义的出发点其实是寻找最优解
   $$
   \underset{B}{argmin}\|AB-I\|_{F}^{2}
   $$
   通过矩阵求导的方式最终我们可以推理得到
   $$
   A^\dagger = \lim_{\epsilon\to0} (A^\top A + \epsilon I_r)^{-1} A^\top
   $$
   如果 $A^\top A$ 可逆，可以证明伪逆等价于 $A^{-1}$。不过这个形式有点难看，因为有极限的存在，该极限保证了可逆操作的合法性。实际上我们可以利用谱分解写一个更好看的形式，首先我们有谱分解
   $$
   A^TA=U\Lambda U^\top
   $$
   带入到伪逆公式当中
   $$
   \begin{aligned} \left(\boldsymbol{A}^{\top} \boldsymbol{A}+\epsilon \boldsymbol{I}_{r}\right)^{-1} \boldsymbol{A}^{\top} & =\left(\boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U}^{\top}+\epsilon \boldsymbol{I}_{r}\right)^{-1} \boldsymbol{A}^{\top} \\ & =\left[\boldsymbol{U}\left(\boldsymbol{\Lambda}+\epsilon \boldsymbol{I}_{r}\right) \boldsymbol{U}^{\top}\right]^{-1} \boldsymbol{A}^{\top} \\ & =\boldsymbol{U}\left(\boldsymbol{\Lambda}+\epsilon \boldsymbol{I}_{r}\right)^{-1} \boldsymbol{U}^{\top} \boldsymbol{A}^{\top} \end{aligned}
   $$
   可以看到，这里的可逆符号转移仅作用于对角矩阵，不过由于 $UA$ 矩阵在 rank > r 过后的向量也全部为零（可由 $(UA)^T(UA)=\Lambda$ 证明），所以把可逆矩阵中的零分之一的情况完全抵消（零乘以任何数都是零），所以直接把极限给拿掉得到形式
   $$
   A^\dagger =  U\Lambda^{-1} U^\top A^\top
   $$
   现在有了 SVD 分解，我们可以把 $UA$ 也简化表达。注意上面讨论的所有 $U$ 其实是 SVD 分解中的 $V$，而我们下面所写的式子按照 $A=U\Sigma V^\top$ 来写伪逆形式：
   $$
   A^\dagger =  V\Sigma^\dagger U^\top
   $$
   其中 $\Sigma^\dagger$ 就是 $\Sigma$ 中的非零元素取倒数然后转置

2. 单位正交矩阵不改变矩阵范数

   这很好理解：旋转不改变长度。也可以通过 trace 性质轻松证明

同时在 [SVD分解(一)：自编码器与人工智能](https://www.spaces.ac.cn/archives/4208)  提到了一种观点：SVD 与自编码器的定价性。优化一个没有激活函数的3层 MLP，等价于 SVD 的求解
$$
M_{m\times n}≈M_{m\times n}C_{n\times r}D_{r\times n}
$$
而进行 SVD 分解可以得到，此时的最优性是有保证的
$$
M_{m\times n}≈U_{m\times r} \Sigma_{r\times r} V_{n\times r}^T
$$
如果我们利用反向传播去优化矩阵 $C_{n\times r}$ 和 $D_{r\times n}$ 最终的估计误差一定会收敛于 SVD 分解结果的估计误差，在此可以“近似”地认为：$M_{m\times n}C_{n\times r}=U_{m\times r} \Sigma_{r\times r}$，以及 $D_{r\times n}= V_{n\times r}^T$

**Review Eigen & Eigen Value**

从以上的分析中，我们其实并没有刻意地去寻找特征向量，而特征值和特征向量的形式，非常自然地从我们的推导之中出现了。这意味着特征值和特征向量在最优化中是具有显著的意义的。而“特征”一词的意义，在其中显得更加明显：如果我们选择这些特征值大的特征向量，对矩阵进行重构，那么重构前后矩阵的信息获得了最优的保留，也就是说：我们保留了矩阵的“特征”

## Question

- 为什么实对称矩阵一定可以被对角化？这其中有没有什么深层次的原因？

  询问了 Kimi & DeepSeek，二者都给出了两个过程：

  1. 证明实对称矩阵可对角化的三个步骤

  2. 引出更一般的谱定理（[Spectral Theorem](https://en.wikipedia.org/wiki/Spectral_theorem)）

     **谱定理**大致指出：**自伴算子（或矩阵）可以在某个标准正交基下被对角化**。所以，实对称矩阵可对角化是谱定理的一个直接推论。谱定理是泛函分析、量子力学等领域的基石，它将算子（或矩阵）的“结构”完全由其“谱”（即特征值的集合）来描述。

  这确实很神奇🧐
  
- 在学习 SVD 的过程中还在科学空间中看到了激活函数以及 EM 算法的相关解读，包含了 silu 为什么会 work，EM 算法与梯度下降算法之间的联系

  [浅谈神经网络中激活函数的设计](https://spaces.ac.cn/archives/4647)

  [梯度下降和EM算法](https://www.spaces.ac.cn/archives/4277)
  
- 矩阵分块与矩阵求导的基础

  [矩阵求导术-上](https://zhuanlan.zhihu.com/p/24709748) 我应该在研究生时期就看过这一篇，不过当时完全没看明白。当我接触了过一些 trace 相关的技巧后，再来看感觉轻松很多，很多思想都非常值得学习，而且配合了不少例子，绝对是一篇不可多得的高质量博客。**这里仅讨论标量对矩阵的求导，不讨论向量/矩阵对矩阵的求导。**矩阵对矩阵的求导遵循另外的求导法则
  
  向量对矩阵求导从形式上来说非常简单，就是对每一个元素逐个求导，最后形成矩阵
  $$
  \frac{\partial f}{\partial X}=\left[\frac{\partial f}{\partial X_{ij}}\right]
  $$
  然而，这个定义在计算中并不好用，实用上的原因是对函数较复杂的情形难以逐元素求导；哲理上的原因是逐元素求导破坏了**整体性**。所以我们希望在求导时不拆开矩阵，而是直接通过矩阵运算，从整体出发推导出结果。首先我们复习一下一元微分和多元微分
  $$
  df = f'(x)dx\\
  df = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} dx_i = \frac{\partial f}{\partial x}^T d\boldsymbol{x}
  $$
  多元微分中，我们可以把其看待成为导数与微分之间的内积。现在我们来看矩阵微分
  $$
  df = \sum_{i=1}^{m} \sum_{j=1}^{n} \frac{\partial f}{\partial X_{ij}} dX_{ij} = \operatorname{tr} \left( \frac{\partial f}{\partial X}^T dX \right)
  $$
  我们把矩阵微分也用矩阵乘法 + trace 的形式表达出来了。其中使用了一个非常重要的 trace trick：$\sum_{i,j} A_{ij}B_{ij}=\operatorname {tr}(A^TB)$，可以看到 trace 在矩阵求导的过程中是非常常见的形式，所以会频繁使用到一些 trace trick，所以掌握常用的 trace trick 是非常有必要的（好在这些 trick 都不是很复杂，也很好理解）
  
  在进行矩阵求导之前，回顾我们是如何对一元函数求导的：我们首先建立了初等函数的求导结果、推导出求导的四则运算、建立复合函数的求导法则（i.e. 链式求导法则），通过利用链式求导法则和四则运算法则将函数求导进行逐渐的拆分，直到只对初等函数进行求导，最终将结果整合起来，获得最终的导数表达。**所以，现在我们建立了矩阵求导的矩阵形式，我们还需要建立矩阵微分的运算法则来拆解复杂的矩阵函数**
  
  1. 基础运算
     - 加减法：$d(X\pm Y)=dX\pm dY$
     - 矩阵乘法：$d(XY)=(dX)Y+XdY$
     - 转置：$d(X^{T})=(dX)^{T}$
     - 迹：$d\operatorname{tr}(X)=\operatorname{tr}(dX)$
  
  2. 逆
     - $dX^{-1}=-X^{-1}dXX^{-1}$。此式可在$XX^{-1}=I$两侧求微分来证明。
  
  3. 行列式
     - $d|X|=\operatorname{tr}(X^{\#}dX)$，其中 $X^{\#}$ 表示 $X$ 的伴随矩阵，在X可逆时又可以写作 $d|X|=|X|\operatorname{tr}(X^{-1}dX)$。此式可用 Laplace 展开来证明，详见张贤达《矩阵分析与应用》第279页。
  
  4. 逐元素乘法
     - $d(X\odot Y)=dX\odot Y+X\odot dY$，$\odot$ 表示尺寸相同的矩阵逐元素相乘。
  
  5. 逐元素函数
  
     - $d\sigma(X)=\sigma^{\prime}(X)\odot dX$，$\sigma(X)=[\sigma(X_{ij})]$ 是逐元素标量函数运算，$\sigma^{\prime}(X)=[\sigma^{\prime}(X_{ij})]$ 是逐元素求导数
  
  6. 复合函数
  
     无法直接套用一元的复合函数公式，因为我们没有矩阵对矩阵求导的定义。所以唯一的方法是将符合函数中的微分形式进行带入
     $$
     df=\operatorname{tr}(\frac{\partial f}{\partial Y}^TdY)=\operatorname{tr}(\frac{\partial f}{\partial Y}^Tdg(X))
     $$
  
  法则看上去很多，但是基本都符合我们之前的一元情况，除了逆和行列式。也如之前所说，**要完成矩阵求导，我们还需要一些 trace trick**
  
  1. 矩阵乘法的 trace 展开：$\operatorname{tr}(A^\top B)=\operatorname{tr}(B^\top A)=\Sigma_{i,j}A_{ij}B_{ij}$
  
  2. 标量套上迹：$a = \operatorname{tr}(a)$
  
  3. trace 转置性质：$\operatorname{tr}(A^{T}) = \operatorname{tr}(A)$
  
  4. trace 线性性质：$\operatorname{tr}(A \pm B) = \operatorname{tr}(A) \pm \operatorname{tr}(B)$
  
  5. trace 中矩阵乘法可交换：$\operatorname{tr}(AB) = \operatorname{tr}(BA)$，需要保证 $AB, BA$ 是可行的矩阵乘法。该性质还可拓展为 trace 循环
     $$
     \operatorname{tr}(ABC)=\operatorname{tr}(BCA)=\operatorname{tr}(CAB)
     $$
  
  6. trace 中矩阵乘法/逐元素乘法可交换：$\operatorname{tr}(A^{T}(B \odot C)) = \operatorname{tr}((A \odot B)^{T} C)$ 两侧都等于 $\sum_{i,j} A_{ij} B_{ij} C_{ij}$
  
     左侧：$\operatorname{tr}(A^T(B \odot C)) = \sum_{i,j} A_{ij} (B \odot C)_{ij} = \sum_{i,j} A_{ij} B_{ij} C_{ij}$
  
     右侧：$\operatorname{tr}((A \odot B)^T C) = \sum_{i,j} (A \odot B)_{ij} C_{ij} = \sum_{i,j} A_{ij} B_{ij} C_{ij}$
  
  由于 trace 的引入，我们可以对矩阵的乘法顺序方便地进行交换，从而将 $dX$ 放到最右侧，所以我们通常用的技巧是：把标量导数套上 trace，通过 trace trick 和微分符号与 trace 可互换的性质，调整 $dX$ 到所需位置，最后对比 $\operatorname{tr}(\frac{\partial f}{\partial X}^TdX)$ 和 $\operatorname{tr}(g(X)dX)$，我们即可确定 $g(X)=\frac{\partial f}{\partial X}$
  
  我们以最常见的两个例子来说明
  
  1. **Example 1:** $f=a^TXb$，其中 $a\in \mathbb R^{m\times1}, b\in \mathbb R^{n\times 1}, X\in \mathbb R^{m\times n}$
  
     思路：先利用标量套上 trace 不变的性质，变为 trace 形式，然后利用 trace 矩阵乘法可交换性质把 $dX$ 换到最右侧
     $$
     df=\operatorname{tr}(a^{T}dXb)=\operatorname{tr}(ba^{T}dX)=\operatorname{tr}((ab^{T})^{T}dX)
     $$
     按照上述思路与 $\operatorname{tr}(\frac{\partial f}{\partial X}^TdX)$ 形式对比，我们可以立刻得到导数为
     $$
     \frac{\partial f}{\partial X}=ab^{T}
     $$
  
  2. **Example 2:** $f = \operatorname{tr}(W^T X^TX W)$，其中 $X \in \mathbb R^{m\times k}, W \in \mathbb R^{k\times n}$，这其实就是求解 F-范数平方的导数 $f=||XW||_F^2$，其中我们关注的是 $W$ 的导数
  
     思路：既然已经套上 trace 了，直接利用四则运算中的矩阵乘法微分进行拆分，然后逐个求解。为了简便我们用 $M=X^TX$ 来表示一个对称矩阵
     $$
     df = \operatorname{tr}((dW)^T M W) + \operatorname{tr}(W^T M dW) = \operatorname{tr}(W^T M^T dW) + \operatorname{tr}(W^T M dW)
     $$
     上述第二个等号利用了 trace 的转置性质，现在结果已经明了
     $$
     \frac{\partial f}{\partial W}=(M^T+M)W=2MW=2X^TXW
     $$
