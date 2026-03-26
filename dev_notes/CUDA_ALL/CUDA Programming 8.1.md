# CUDA Programming 8.1

我需要将之前的 cutlass 笔记 (CUDA Programming 7 & 8) 进行一个调理清晰的整理，所以将该笔记命名为 8.1

目前的一个 GPU 编程趋势：以 Tile 视角进行编程。最近有一个 [TileLang](https://github.com/tile-ai/tilelang) 项目也很火，不过我了解不多😂

一个 Tile 即一个 Block 所能处理的数据块，一个 Tile 需要做到承上启下的作用：

1. Tile 中的数据向下分配到 thread level (layout tv parition)
2. Tile 进行重复，处理完整的 problem size (tiler zipped divide)

通过 tiler 作为解决问题的粒度，能够更清晰地构建出 kernel pipeline，这也是 triton 的优势之一：以 block 作为编程粒度，开发者不用去考虑 thread level 的问题。在 cutlass 当中，通过 tile 承上启下的功能，完成具体的 thread level 代码。

## 核心抽象

### Layout Algebra

这是整个 cute 的核心，并且 cute 本身文档很难读，而且网上没有太多的学习资料，所以就算是 GPT 也很难给出好的回答。我的学习资料主要来源于三个部分：1. Reed zhihu 2. [Lei Mao's blog](https://leimao.github.io/article/CuTe-Layout-Algebra/) 3. [A note on the algebra of CuTe Layouts](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/)

我想以四个部分来介绍，目的是为了形成对 layout algebra 的清晰理解，使得我在阅读代码的时候能够进行逻辑推理

1. layout 基本概念
2. layout algebra 基本运算
3. layout algebra 组合运算
4. layout algebra 直观总结

#### 基本概念

layout 概念非常简单，就是由两部分组成：shape & stride，二者共同构建出一个整数到整数的映射：$\mathbb{N} \rarr \mathbb{N}$
$$
shape=(s_0,s_1,...,s_{n-1})\\ stride=(d_0,d_1,...,d_{n-1})\\
$$
为了完成这个映射，我还需要引入一个概念：整数与多维坐标的同构性 ([isomorphism](https://en.wikipedia.org/wiki/Isomorphism))。在数学上，两个东西同构意味着二者本质上是一个东西，二者可以通过一个映射进行可逆的转换。现在我们来构建整数与多维坐标的转换，就能够证明二者的同构性。我们定义多维坐标是 shape 空间中的一个点 $(x_0,x_1,...,x_{n-1})$，通过点积我们就能完成多维坐标到整数的转换
$$
x=f(x_0,x_1,...,x_{n-1}) = x_0·1+x_1s_0+...+x_{n-1}\prod_{i=0}^{n-2}s_{i}
$$
而整数到多维坐标的转换则是通过取余完成
$$
f'(x)=\left( x \bmod s_0,\ \left\lfloor \frac{x}{s_0} \right\rfloor \bmod s_1,\ \ldots,\ \left\lfloor \frac{x}{s_0 \times \cdots \times s_{n-2}} \right\rfloor \bmod s_{n-1} \right)
$$
实际上这就是列优先的顺序排列方式

```python
# shape (2, 3) with its int
(0,0)<->0      (0,1)<->2      (0,2)<->4
(1,0)<->1      (1,1)<->3      (1,2)<->5
```

有了以上的转换过后就可以定义 layout function 映射了，定义为如下
$$
Layout(x) = g(f'(x))
$$
其中 $f'(·)$ 即为将整数转换为坐标的映射，而 $g(·)$ 为将坐标转换为整数的映射，其本质是坐标 coord 与步长 stride 的点积
$$
g(x_0,x_1,...,x_{n-1})=coord ·stride=s_0d_0+s_1d_1+...+s_{n-1}d_{n-1}\\
$$
如此一来我们就完成了从整数到整数的映射：我们从整数 $x$ 出发，寻找其对应的坐标点，然后通过步长进行新的映射

此时你可能发现了，将 $f$ 与 $g$ 其实非常相似，都是将坐标映射到整数。在之前我也提到了，$f$ 本身就是 row-major 的排列方式，其可用一个特殊的 layout 来表示，该 layout 我们称之为 layout left (or natural layout)
$$
shape=(s_0,s_1,...,s_{n-1})\\ stride=(d_0,d_1,...,d_{n-1})\\
d_i=\prod_{j=0}^{i-1}s_j=d_{i-1}s_{i-1},d_0=1
$$
举一个例子，一个 shape 为 `(2, 3, 4)` 的 natural layout 为

```python
Layout(shape=[2, 3], stride=[1, 2])
     0      2      4
     1      3      5
```

有了 layout left，那就有 layout right，也就是行主序排列

```python
Layout(shape=[2, 3], stride=[3, 1])
     0      1      2
     3      4      5
```

Layout 其中一个作用就是用来描述坐标与内存位置。这是很自然的事情，因为物理内存永远都是以一维的形式来表达，**所以在 cutlass cute 中就是用一个指针 + 一个 layout 来描述一个 tensor，并且在 cutlass 中以 `shape:stride ` 的形式 print layout**

```c++
Tensor(Ptr const& ptr, Layout const& layout)
Layout(shape=[2, 3], stride=[3, 1])	// (2, 3):(3, 1)
```

而实际上 Layout 可以用来描述更多的事情，例如：如何将一个 $(M,N)$ 形状 tensor 分配到 $(T,V)$ 形状当中。其中 $T$ 就是 threads 数量，$V$ 是每一个 thread 拥有的 values，这将在基本运算小节中进行介绍

#### 基本运算

layout algebra 最抽象的部分在于其基本运算，尤其是以下两个基本运算：

1. complement，补运算
2. compose，复合运算

当然还有其他的运算，例如 concat, coalecse，我用代码来简单解释

```python
"""
@dataclass
class Layout:
    shape: List[int]
    stride: List[int]
"""

A = Layout([2, 3], [1, 2])
B = Layout([4], [10])
coalesce(A)	# Layout(shape=[6], stride=[1])
concat(A, B)# Layout(shape=[2, 3, 4], stride=[1, 2, 10])
```

concat 就是将 shape & stride 分别连接，而 coalecse 则是合并 shape & stride，以更少维度呈现

##### Complement

补运算需要两个元素，整数 $M$ 和 layout 本身。我先用一个一维的例子来说明补运算的作用，这也是 reed zhihu 中所使用到的例子

```python
A = Layout([4], [2])
B = complement(8, A)    # Layout(shape=[2], stride=[1])
```

在 reed zhihu 中说到

> 当codomain存在不连续时，则存在空洞的位置，如图4所示，这时候我们可以构造一个Layout2能够填充上codomain的空洞位置，此时我们构造的Layout则为原Layout的补集

我认为 complement 的作用是计算出了 layout 所需要重复的“次数”以填满整个 $M$ 空间。用上面的例子来说

```python
0 2 4 6
0 1 2 3 4 5 6 7
```

A 还需要重复两次才能够填满 0~8 的整个空间，而后面的 stride 则描述了重复空间之间的间隔，在这里间隔是 1。实际上只需要将 A 和 A 的补 concat 起来就会发现，二者组成了一个连续的空间

```python
C = concat(A, B)	# Layout([4, 2], [2, 1])
```

在这个 case 中 concat 过后的结果是一个 layout right 排布

再举一个二维的例子

```python
A = Layout([2, 3], [2, 4])
B = complement(24, A)	# Layout(shape=[2, 2], stride=[1, 12])

# Layout A
#     0      4      8
#     2      6     10

# Layout([4, 6], [1, 4])
#     0      4      8     12     16     20
#     1      5      9     13     17     21
#     2      6     10     14     18     22
#     3      7     11     15     19     23
```

可以看到 A 需要在两个维度（在 cutlass 中习惯把一个维度称之为一个 mode）的方向上都分别重复两次。在第一个 mode 上重复空间的间隔是 1，而在第二个 mode 重复空间的间隔是 12。我们仍然可以将 A 和 B 进行对应 mode 的 concat

```python
A = Layout([2, 3], [2, 4])
B = Layout([2, 2], [1, 12])
C = Layout([(2, 2), (3, 2)], [(2, 1), (4, 12)])
```

concat 之后的 Layout C 实际上可以看做一个合并的 `Layout([4, 6], [1, 4])`

现在我们再来看 complement 的公式就会发现其中的奥秘：
$$
\operatorname{complement}(A, M) = \left( d_{0},\ \frac{d_{1}}{s_{0}d_{0}},\ \frac{d_{2}}{s_{1}d_{1}},\ \cdots,\ \frac{M}{s_{a}d_{a}} \right) : \left( 1,\ s_0 d_{0},\ s_1 d_{1},\ \cdots,\ s_a d_{a} \right)
$$
其本质就是在计算每一个 mode 还需要重复多少次才能够填满整个空间，重复空间的间隔即为子空间大小 $s_id_i$

##### Compose

既然是映射（函数），那么将两个函数进行复合是再正常不过的想法了。从直观上来说将两个 layout 进行 compose 非常简单，毕竟都是整数到整数的映射：
$$
g_3=g_1(g_2(x))
$$
但是需要考虑的问题是，如何将新的 compose 结果 $g_3$ 描述为一个合法的 layout 结构 `(shape, stride)`。而这个描述其实还是要化不少笔墨介绍的，这里省略，可参考 Definition 2.13 from  [A note on the algebra of CuTe Layouts](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/)

<img src="CUDA Programming 8.1/image-20250525152931437.png" alt="image-20250525152931437" style="zoom:67%;" />

NOTE: 其实 layout algebra 对于输入其实都是有要求的，并不是任意两个 layout 进行 compose 都是可行的，其对于整除性还是有不少要求。好消息是如果数值都是以 $2^n$ 存在，整除性质就会得到很好的保障，而这正是在 GPU 编程中常用的数值。我在笔记 CUDA Programming 8 中有简要说明

虽然说需要严谨的数学来保证 compose admissibility，但这不妨碍其本质就是上述所说的复合函数，即：从一个 domain 映射到另一个 domain。我将以一个非常具体的例子帮助理解这个 compose 过程

```python
TV2MN = Layout([4, 2, 2], [2, 1, 8])
MN2Memory = Layout([4, 4], [4, 1])
```

首先我定义了两个 layout，第一个 `TV2MN` 描述了 thread values 所对应的 MN 映射。第二个 `MN2Memory` 描述了 MN 到内存的映射。更具体来说

1. `TV2MN` 描述了 4 个线程，每一个线程拥有有 (2, 2) 个 values，这些 values 将映射到一个 shape 为 (M, N) 的 tensor 上。该 layout 也将描述 tensor 是如何被分配到线程当中的
2. `MN2Memory` 描述了 tensor 中各个坐标的 value 在内存当中的位置。在例子当中是一个 layout right 的排布，也就 tensor 在内存中是行优先排列

通过 compose 我们可以直接获得 `TV2Memory` 这样的映射，该映射即代表了内存中的数据如何被分配到线程当中

```python
TV2Memory = compose(MN2Memory, TV2MN) # Layout(shape=[2, 2, 4], stride=[1, 8, 2])
```

我们将这个例子打印出来，通过 step by step 的方式看下整个 compose 的过程：

```python
TV2MN: Layout(shape=[4, 2, 2], stride=[2, 1, 8])
     0|     1|     8|     9|
     2      3     10     11
     4      5     12     13
     6      7     14     15
MN natural: Layout(shape=[4, 4], stride=[1, 4])
     0|     4      8|    12
     1|     5      9|    13
     2      6     10     14
     3      7     11     15
MN2Memory: Layout(shape=[4, 4], stride=[4, 1])
     0|     1      2|     3
     4|     5      6|     7
     8      9     10     11
    12     13     14     15
TV2Memory: Layout(shape=[2, 2, 2, 2], stride=[8, 1, 4, 2])
     0|     4|     2|     6|
     8     12     10     14
     1      5      3      7
     9     13     11     15
```

以 thread 0 为例：

1. 其对应的 MN index 为 `(0, 1, 8, 9)`
2. 通过 MN index 可以找到 `(0, 1, 8, 9)` 分别对应坐标 `(0,0), (1,0), (0,2), (1,2)`
3. 通过对应坐标找到 `MN2Memory` 所对应的值为 `(0, 4, 2, 6)`
4. 所以 thread 0 的 4 个 values 将会寻找内存中第 0, 4, 2, 6 个元素

由此我们就完成了一个映射，其从 TV domain 出发，映射到了 Memory domain。这也引出了 compose 的一个直观性质：不改变 source domain，即输入的 layout “形状”是不会改变的

```python
TV2MN: Layout(shape=[4, 2, 2], stride=[2, 1, 8])
TV2Memory: Layout(shape=[(2, 2) 2, 2], stride=[8, 1, 4, 2])
```

##### Inverse

同样的，在函数中也存在逆函数。在 layout algebra 中的逆函数定义可参考 [reed-zhihu](https://zhuanlan.zhihu.com/p/662089556) 中的 two line notation 表示形式。所谓的 two line 就是：input domain 为一个 line，output domain 为一个 line，下面举一个例子

```python
# Layout(shape=[2, 3], stride=[3, 1])
# [0, 1, 2]
# [3, 4, 5]

coord: [0, 1, 2, 3, 4, 5]
value: [0, 3, 1, 4, 2, 5]

# sort the pair according to value
coord: [0, 2, 4, 1, 3, 5]
value: [0, 1, 2, 3, 4, 5]

# switch coord and value as new layout
coord: [0, 1, 2, 3, 4, 5]
value: [0, 2, 4, 1, 3, 5]
```

上述 two line notation 用于理解 inverse 是比较直观的，但是对于理解 inverse 过后 layout 形式是怎么样的，没有太大帮助。具体来说，他们的 shape & stride 应该如何得到？在 [Lei Mao's blog](https://leimao.github.io/blog/CuTe-Inverse-Layout/) 当中证明了 compact layout inverse 过后的 shape & stride 应当如何计算，不过 blog 当中的叙述顺序对我来说略显晦涩，我这里用我自己的思考逻辑来整理

Conditions:

- Layout function: $f_L(x)$

- shape & stride 为 $S=(s_0,s_1,...,s_n),D=(d_0,d_1,...d_n)$

- natural layout funciton 将多维坐标 $(x_0, x_1, ...,x_n)$ 映射为 $x$
  $$
  x=x_0+x_1·s_0+...+x_n·\prod_0^{n-1}s_i
  $$

Target:

- 找到 inverse layout: $f_{L'}(x)$ 使得满足
  $$
  f_{L'}(f_L(x)) = x
  $$

- inverse layout $L'$ shape & stride 为 $S'=(s_0',s_1',...,s_n'),D'=(d_0',d_1',...d_n')$

现在开始正式推导。对于输入 $x$ 对应的 $L$ 坐标为 $(x_0, x_1, ..., x_n)$，我们设其输出为 $x'$
$$
f_L(x)=x'
$$
输出 $x'$ 所对应的 $L^{-1}$ 坐标为 $(x_1',x_2',...,x_n')$，由 $L'$ shape 的 natural layout function 完成映射。由等式条件得
$$
\begin{aligned}
f_{L'}(f_L(x)) &= f_{L'}(x') \\
               &= f_{L'}(x_0',x_1',...,x_n') \\
               &= x_0' \cdot d_0' + x_1' \cdot d_1' + \cdots + x_n' \cdot d_n' \\
               &= x \\
               &= x_0 + x_1 \cdot s_0 + \cdots + x_n \cdot \prod_{i=0}^{n-1} s_i
\end{aligned}
$$
其中最重要的等式为
$$
x_0' \cdot d_0' + x_1' \cdot d_1' + \cdots + x_n' \cdot d_n' =x_0 + x_1 \cdot s_0 + \cdots + x_n \cdot \prod_{i=0}^{n-1} s_i
$$
下面的证明思路为：如果我们能够找到一个 permutation $I=\{i_0,i_1,...,i_n\}$，使得 $x_{i_0}'=x_0,x_{i_1}'=x_1,...,x_{i_n}'=x_n$，那么我们就能对应多项式的每一项，直接算出每一个 $d'$ 的值。现在我们来考察 $(x_0,x_1,...,x_n)$ 与 $(x_0',x_1',...,x_n')$ 之前的联系是什么，是否存在这样的 permutation

他们之间的关系非常清晰
$$
(x_0,x_1,\ldots,x_n) \xleftrightarrow{L} x' \xleftrightarrow{N} (x_0',x_1',\ldots,x_n')
$$
这里的 $N$ 就是 inverse layout 的 natural function。现在问题转换为：对于一组  $(x_0,x_1,...,x_n)$ 与 $(x_0',x_1',...,x_n')$，他们彼此都是对方的 permutation，我们需要找到合适的 natural layout function 即可。其实对于第一个要求非常好满足（忽略 natural layout 限制），我们可以直接对 $L$ 中的 shape & stride 进行 permute 即可。以简单的 `Layout(shape=[2,3], stride=[3,1])` 为例子，当 permute shape & stride 时，坐标也随之 permute
$$
(x_0,x_1) \xleftrightarrow{(2,3):(3,1)} x' \xleftrightarrow{(3,2):(1,3)} (x_1,x_0)
$$
现在只需要考虑 natural layout 的限制即可，而答案也就随之浮出水面：只需要将 $L$ 的 shape & stride permute 成为一个 natural layout (left layout) 即可。更具体来说，根据 stride 的大小，从小到大进行排列，由于 layout 有 compact 保证，没有任何空洞，所以排列出来的 layout 必定也是 natural layout。所以此 permutation 存在且唯一，确定了 inverse layout 的 shape，其对应的 stride 也可由下面的式子进行计算
$$
d_{i_0}'=1,\\
d_{i_1}'=s_0,\\
...,\\
d_{i_n}'=\prod_{i=0}^{n-1} s_i,\\
$$
那么根据上述结论，我们就找到了 $L'$ 的 shape & stride 了！**其中 shape 的结论会很 clean，就是将 $L$ 进行 sort 过后的 shape。从定性来说：原始 stride 小的 shape 在 inverse 过后会靠前；反之则会靠后**

而在 [写给大家看的 CuTe 教程：Layout compose & Inverse](https://zhuanlan.zhihu.com/p/1962625273636845008) 中提到，通常 inverse 过后还会使用 `with_shape` 来构建我们期望的 layout shape，我们必须要了解 inverse 的输出形状到底是什么，才能正确地使用 `with_shape`。具体的例子在 retile 部分中，计算 `(t, v) -> (m, n)` layout 进行展示，其精妙地展示了 inverse 的一个核心作用：domain 的交换。如果我们获得了 `(m, n) -> (t, v)` 的映射，直接使用 inverse 就可以获得 `(t, v) -> (m, n)` 映射

#### 组合运算

有了 layout algebra 所定义的基础运算就可以定义一些更复杂更有用的运算：divide & product

##### divide

divide 是划分数据中最常用的方法，尤其是 zipped divide。我先介绍 logical divide 的一维运算公式（B 是维度为1的 layout，A 没有限制）

```python
def logical_divide(A, B):
    M = A.size()
    c_B = complement(M, B)
    concatenated = concat(B, c_B)
    return compose(A, concatenated)
```

可以看到，其先计算了 B 补集，然后与 B 进行了 concat，最后用 concat 过后的 layout 与 A 进行了 compose。通常我们称 layout B 就是一个 **Tiler**，以 Tiler 为粒度对 A 进行了划分。在实际应用过程中都是对一个 layout 进行逐维度 divide (by-mode divide)

```c++
Layout Shape : (M, N, L, ...)
Tiler Shape  : <TileM, TileN>

logical_divide : ((TileM,RestM), (TileN,RestN), L, ...)
zipped_divide  : ((TileM,TileN), (RestM,RestN,L,...))
```

<img src="CUDA Programming 8.1/divide1.png" alt="divide1.png" style="zoom:33%;" />

在上面的例子中 Tiler 是不连续的，而我们更常会遇到的 Tiler 是最简单的 stride 为 1 的 Tiler。如 `B = Layout([4], [1])`，这样就会以 4 为单位切分该轴。zipped divide 会将 Tiler 维度直接提到最前面来，以方便我们进行索引操作，通常这个维度可以是 thread，这样通过索引就获得具体某个线程所对应的数据

通常我们遇到的情况都是：A & B 都是 1-dim，如果 A 为多维 layout，那么就需要谨慎看待，最后的结果一般不是我们想要的。举个例子

```python
l1 = Layout([5, 4], [1, 30])
l2 = Layout([4], [1])
# logical_divide(l1, l2) won't work
A size: 20
complement of B: Layout(shape=[5], stride=[4])
concated (B, c_B): Layout(shape=[4, 5], stride=[1, 4])
```

原因在于 concated layout 无法和 A 进行 compose。不过好消息是在进行数据 divide 时，通常是对 MN shape 进行 divide，这是一个非常规整的 domain，满足我们在 by-mode divide 时各个 mode dim 都是 1 的需求

##### product

这里有个割裂感：我们说 product 为 divide 的逆运算，但实际上我发现二者并不能进行可逆操作。例如 `C != A.product(B).div(B)`。但是这个定义并不符合我们的直觉，严谨的数学定义在 [Lei Mao's blog](https://leimao.github.io/article/CuTe-Layout-Algebra/) 中有所阐述。这里以一个 [2D exmaple](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/02_layout_algebra.html#logical-product-2-d-example) 作为说明

<img src="CUDA Programming 8.1/image-20251027162320196.png" alt="image-20251027162320196" style="zoom: 67%;" />

这个 product 的结果非常直观：把 `(2, 5): (5, 1)` 进行重复，重复维度为 `(3, 4)`。在我的期望中，直接使用 tiler `<3:1, 4:1>` 就能完成上述功能，但实际上用的 tiler 为 `<3:5, 4:6>`，这就是因为 product 的定义并不是我们想象中的直观，仍然是根据 complement & compose 来定义的。为了让 product 功能与我们的编程直觉相符，cute 直接构建了几种常见的 api 方便调用，参考 [reed zhihu](https://zhuanlan.zhihu.com/p/662089556)

| 乘法模式 | 乘积的shape      |
| -------- | ---------------- |
| logical  | ((x, y), (z, w)) |
| zipped   | ((x, y), (z, w)) |
| tiled    | ((x, y), z, w)   |
| blocked  | ((x, z), (y, w)) |
| raked    | ((z, x), (w, y)) |

上面只列举了 shape，对于 stride 而言，**相同 dimension 的 stride 也是一样的**：即任意乘法模式中所有 x 对应的 stride 都一样。需要注意的是，这些操作是 layout x layout，而不是 layout x tiler。所以他们都是 rank sensitive 的，即两个 layout 的维度必须一致。同时和 divide 一样，通常使用在相对规整的 domain，即 layout 的 size 和 cosize 一致。否则存在空洞的话，product 也可能无法进行，举一个例子

```cpp
auto l1 = make_layout(make_shape(_4{}, _5{}), make_stride(Int<30>{}, _1{}));
auto l2 = make_layout(make_shape(_2{}, _4{}));
// can't do logical_product(l1, l2)
```

这里点出一个 product 和 divide 的重要差异：divide 习惯使用 layout divide tiler，而 product 习惯使用 layout product layout。另外一个实验是，product 的顺序是会改变结果的

```cpp
auto base_layout = make_layout(make_shape(_4{}, _3{}), make_stride(_4{}, _1{}));
auto layout_x2 = blocked_product(base_layout, make_layout(make_shape(_1{}, _2{})));
auto layout_x2_x2 = blocked_product(layout_x2, make_layout(make_shape(_2{}, _1{})));
auto layout_x4 = blocked_product(base_layout, make_layout(make_shape(_2{}, _2{})));

// Product order test
// ((_4,_1),(_3,_2)):((_4,_0),(_1,_16))
// (((_4,_1),_2),((_3,_2),_1)):(((_4,_0),_32),((_1,_16),_0))
// ((_4,_2),(_3,_2)):((_4,_16),(_1,_32))
```

我先对 base layout 在第二个 dim 进行扩张，然后再对第一个维度进行扩张，其结果和同时扩张两个维度是不一致的。在之后的内容当中，我们可以使用组合运算和基础运算来获得所需的 layout 排布，在实践中学习

### MMA

#### MMA Atom

mma atom 可以大致认为由两个部分组成：mma op & mma traits

1. MMA op 用于描述所使用的 PTX 命令，以及该命令所需要的寄存器

2. MMA traits 用于描述需要完成一个 MMA 所缺失的部分：包含数据类型、数据形状，线程数据排布 tv layouts

以 mma op `SM80_16x8x16_F16F16F16F16_TN` 为例来说明

```c++
// MMA 16x8x16 TN
struct SM80_16x8x16_F16F16F16F16_TN
{
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];

  CUTE_HOST_DEVICE static void
  fma(uint32_t      & d0, uint32_t      & d1,
      uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
      uint32_t const& b0, uint32_t const& b1,
      uint32_t const& c0, uint32_t const& c1)
  {
#if defined(CUTE_ARCH_MMA_SM80_ENABLED)
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,  %1},"
      "{%2,  %3,  %4,  %5},"
      "{%6,  %7},"
      "{%8,  %9};\n"
      : "=r"(d0), "=r"(d1)
      :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
         "r"(b0),  "r"(b1),
         "r"(c0),  "r"(c1));
#else
    CUTE_RUNTIME_ASSERT("Attempting to use SM80_16x8x16_F16F16F16F16_TN without CUTE_ARCH_MMA_SM80_ENABLED");
#endif
  }
};
```

该 mma op 就是用来封装 PTX 接口的，给出所使用的命令以及该命令需要的寄存器。该 PTX 命令是一个 16x8x16 的矩阵乘，对应的数据类型都是浮点，而 `TN` 代表的是 transposed & normal，分别代表 row-major & col-major。需要强调两点：

1. 是 `TN` 并不是代表矩阵 A & B 他们的数据排布就是 row-major & col-major，这其实只是 PTX 遵循 BLAS 当中的语言约定。而真实的 A & B 数据排布，参考 [TN & NT & TT & NN](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0x_gemm_tutorial.md#aside-m-major-n-major-k-major)，`TN` 其实都是 row-major。并且输出的 C 也是 row-major
2. PTX 命令名字虽然包含了矩阵形状以及数据类型，但是只是名字，实际上在 mma op 中并不具体包含这些信息，所以仍需要 mma traits 提供

接下来看该 mma op 对应的 mma traits

```c++
template <>
struct MMA_Traits<SM80_16x8x16_F16F16F16F16_TN>
{
  using ValTypeD = half_t;
  using ValTypeA = half_t;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using Shape_MNK = Shape<_16,_8,_16>;
  using ThrID   = Layout<_32>;
  using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2,  _2>>,
                         Stride<Stride<_32,_1>,Stride<_16,_8,_128>>>;
  using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_2, _2>>,
                         Stride<Stride<_16,_1>,Stride<_8,_64>>>;
  using CLayout = SM80_16x8_Row;
};

```

正如我之前所说，mma traits 提供了：数据类型 (val type)、数据形状 (shape mnk)、线程数据排布 (thread id, ABC layout)

线程排布其实就是 tv layouts，描述的 (threads, values) -> MK 的映射关系，在 reed zhihu 中用更详细的注释说明：

```c++
using ALayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat MK-coord
using BLayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat NK-coord
using CLayout =      // (Logical thread id (tid), Logical value id (vid)) -> Flat MN-coord
```

#### TiledMMA

mma atom 提供了一个 warp 所能完成的矩阵乘大小，通常我们会在一个 block 中使用更多的 threads，将多个 mma atom 组成一个 tiled mma。该组合通过参数 `AtomLayoutMNK` 来定义 atom 在 MNK 方向上重复的次数。

```c++
  static constexpr int kMmaEURepeatM = 2;
  static constexpr int kMmaEURepeatN = 2;
  static constexpr int kMmaEURepeatK = 1;
  using MMA_EU_RepeatT = decltype(make_layout(make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{});
```

上述代码在 MN 方向上重复了两次，于是从原来的 `16x8x16` 变为了 `32x16x16` 的矩阵乘

NOTE：绝大多数的情况下，都是在 MN 方向上重复 mma atom，几乎从来不会在 K 方向上重复 mma atom [[QST] TiledMMA with `>1` Atoms in K dimension --- how to reduce?](https://github.com/NVIDIA/cutlass/issues/1391#issuecomment-1987272892)。这其实是合理的，在 MN 方向上的重复可以通过简单的 atom 重复完成，而 K 方向上的重复需要进行额外的累加：即需要将多个重复的 mma atom 结果进行累加。通常在 K 方向的累加是通过 main loop 完成

另外还有一个参数 `PermutationMNK`，该参数是比较迷惑的，对于该参数的解释最终都会回到 [[QST] What is PermutationMNK in TiledMMA in CUTLASS 3.4 changes?](https://github.com/NVIDIA/cutlass/discussions/1345)。其中对 `PermuationMNK` 最本质的介绍是：

> The easiest way to think about it is that the `Permutation` parameter is a **Tiler** for the MNK modes of the MMA.

我先举一个实际例子说明其功能，再总结一下其影响

```c++
  // mma atom shape is 16x8x16
  using mma_atom_shape = mma_traits::Shape_MNK;
  static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
  static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
  static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});
  using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
  using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
```

这里 `MMA_P_T` 就是 `PermutationMNK`，在例子中的具体值为 `M=(16x2), N=(8x2x2), K=(16)`，即 `32x32x16`。由此就形成了一个 `32x32x16` 的 Tiler，会将输入数据按照这个 Tiler 形状进行分割。可以看到我们在 `AtomLayoutMNK` 重复的基础上，再对 N 方向又扩大了一倍

该参数有两个功能：

1. 对数据进行 permute，影响 data partition 结果（现在基本不使用该功能）

   如果 tiler 中某一个维度使用了特殊的 layout 例如 `Layout<Shape <_2,_4,_4>, Stride<_1,_8,_2>>`，这将会对数据进行重新的排布。但并不会影响最终的矩阵乘结果，因为 permutation 不改变 reduction 结果，并且最后数据在 copy 的过程中也会回到 permutation 之前的位置

2. **影响 `get_layoutA/B/C_TV` & `tile_size`。不影响 data partition 结果**

   该功能用于扩大 tiler size 以增加 A/B/C tv layouts 中的 v size，从而满足 tiled copy 对 v size 的要求（这一句话高度抽象，一定要配合之后对 tiled copy 的学习）。简单来说，有的 mma atom tv layouts 中，size of v 为 4，即每一个线程分配 4 个 values；而 ldmatrix copy atom 会要求 size of v 至少为 8。在此情形下，直接使用 mma tv layouts 将不会满足要求，而需要增加 v size，该需求就是利用 `PermutationMNK` 扩大 MN shape 而满足的

#### ThrMMA

thread mma 的作用是根据 tiled mma 中所定义的 block tv layouts & mnk shape 对 tensor 进行划分（这里我忽略 `permuationMNK` 所带来的数据排布影响），获得每一个线程所需要的数据。对于一个 tensor shape `(M, N)`，使用 thread mma 按照 matrix A 的 tv layouts & mn shape 对 tensor 划分过后得到每个线程的 tensor shape 为：
$$
(\text{num}_V, \text{num}_M, \text{num}_N)=(V, \frac{M}{m},\frac{N}{n})
$$
第一个维度 `num_v` 代表了 block tv layouts 当中每一个 thread 控制的 values 数量，而 `num_M` 和 `num_N` 则代表 tensor 中的的 M & N 在各自维度上包含了多少个 atom。以上述 tiled mma 为例子，matrix B block tv layouts 中每一个 thread 有 4 个 values，nk shape 为 `(16, 16) = (8x2, 16)`，所以如果我们给定一个 tensor shape 为 `(128, 32)` 的话，得到的 thread tensor shape 为 `(4, 8, 2) = (4, 128/16, 32/16)`

**ThrMMA 的作用仅限于划分，最终传入 `cute::gemm` 方法的仍然是 TiledMMA**

### Copy

copy 其实是比 mma 更加灵活更加复杂的操作。因为其要考虑到不同的硬件结构 (global memory, shared memory, register)，以及 source & destination 对于数据排布不同的要求。GPU 编程的魅力之一就在于如何搬运大量数据以增加数据吞吐量

#### Copy Atom

copy atom 我认为由三个部分组成：copy op, copy traits, copy type。

1. copy op 用于描述 PTX 指令以及所需的寄存器
2. copy traits 用于描述 src & dst tv layouts，以及线程数量。这里的 tv layouts 区别于 mma atom，其映射的 domain 不是矩阵的 shape，而是 bits，在实际使用过程中实际上是提供的数据的**逻辑位置**。这在之后的 ldmatrix/tiled copy 小节中将具体表现
3. copy type 表示数据类型

相比于 mma traits，copy traits 不一定是以 warp 单位来定义，即 tv layouts 中的 t 大小不一定是 32。我对此有一些疑问：难道 GPU 不都应该以 warp 为单位来执行吗？看来我将执行单元和内存操作的最小单位混淆了，二者应当区分看待

> From DeepSeek
>
> Warp 是执行单元，但不是内存操作的最小单位。确实，warp（32线程）是 GPU 的基本执行单元，但内存操作的最小单位不一定与 warp 对齐。这些指令可以由单个线程发起（虽然通常整个 warp 会协同工作）支持各种大小和模式

下面就是一个具体的 copy atom 及其对应 copy traits 在实际代码中的使用

```c++
  using T = cute::half_t;
  using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
  using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
```

这里创建了一个 global to shared memory 的 copy atom，每一个 copy atom 可以完成一个 128bit 的数据搬运，由于我们使用的数据类型为半精度 16bit，所以一次将搬运 8 个数据元素

#### TiledCopy

同样的，和 tiled mma 一样，我们在一个 block 当中通常会有多个 threads，我们仍然需要对 copy atom 进行排布，组成一个更大的 tiled copy。下面就是一个创建 tiled copy 的例子

```c++
  // Each Tile will copy 32x32 half_t elements
  using G2SCopyA = decltype(make_tiled_copy(g2s_copy_atom{},
                                            make_layout(make_shape(Int<32>{}, Int<4>{}),
                                                        make_stride(Int<4>{}, Int<1>{})),
                                            make_layout(make_shape(Int<1>{}, Int<8>{}))));
  using G2SCopyB = G2SCopyA;
```

该 tiled copy 负责将 A & B 矩阵从 global memory 复制到 shared memory，每一次 copy 的 mn shape 为 `(32, 32)`。我想从 `make_tiled_copy` 的具体实现来看下传入参数的含义，我认为非常巧妙

```c++
make_tiled_copy(Copy_Atom<Args...> const& copy_atom,
                ThrLayout          const& thr_layout = {},     // (m,n) -> thr_idx
                ValLayout          const& val_layout = {})     // (m,n) -> val_idx
{
  // Take the raked_products to compute the Layout_MN
  // (M,N) -> (thr_idx, val_idx)
  auto layout_mn = raked_product(thr_layout, val_layout);
  // (thr_idx, val_idx) -> (M,N)
  auto layout_tv = right_inverse(layout_mn).with_shape(make_shape(size(thr_layout), size(val_layout)));
  // Tiler for extracting relevant elements
  // (M,N) -> tensor coord
  auto tiler = product_each(shape(layout_mn));
  return make_tiled_copy_impl(copy_atom, layout_tv, tiler);
}
```

可以看到在构造 tiled copy 中我们传入了两个 layout，一个是 `thr_layout`，另一个是 `val_layout`，我在一开始看到这两个 layout 的时候，只是单纯地觉得这就是在描述 thread 和 values 的排布，然后把这两个 layout 乘起来就获得了一个 `(32, 32)` 的 layout，正好就是 tiled copy 所覆盖的 tensor 区域，并且我错误地认为了这是一个 tv -> mn 的映射。而实际上这两个 layout 在描述 `(m=32, n=4) -> tid` 和 `(m=1, n=8) -> vid` 的映射，通过 raked product 进行了 interleaved 重复获得了 `(m, n) -> (tid, vid)` 的映射。所谓 interleaved 重复即为：在第二个维度是将 8 重复 4 次，而不是将 4 重复 8 次。这在实际的映射中表现为，在 n 方向会先看到同一个 thread 所拥有的连续 values，而不是同一个 value 的连续 thread。最后通过 right inverse 将映射返回成为 `(tid, vid) -> (m, n)`

```c++
auto l = Layout<Shape<_32, _4>, Stride<_4, _1>>{};
auto tiler = Layout<Shape<_2, _8>, Stride<_8, _1>>{};
auto lxtiler = logical_product(l, tiler);
auto lxtiler_rake = raked_product(l, tiler);

((_32,_4),(_2,_8)):((_4,_1),(_1024,_128))
((_2,_32),(_8,_4)):((_1024,_4),(_128,_1))
```

可以看到 `make_tiled_copy` 中还有一个 `make_tiled_copy_impl`，这个函数接受了两个参数 `layout_tv` 以及其对应的 `tiler`，他们二者就共同描述了 tiled copy 如何去划分一个 tiler 大小的数据，然后进行 copy。在实践过程中这个 `layout_tv` 通常可以是 tiled mma 中的 `get_layoutA/B/C_TV`，而 tiler 大小就是 `PermutationMNK` 所设置的 tiler size 大小

在上述例子当中只需要一个 block 进行一次 copy 就能够完成 `(32, 32)` 大小的 copy 任务。还有一种情况，**一个 tiled copy 需要一个 block 进行多次来完成 `(32, 32)` 大小的 copy 任务**，例如将上述例子中的 copy atom 换为 `Copy_Atom<UniversalCopy<cute::uint32_t>, T>`，一个线程只会复制两个 fp16 元素，此时 128 个线程只能够复制 256 个 fp16 元素，很明显并不能够一次完成 `(32, 32)` 大小的 copy 任务。所以一个 tiled copy 会执行多次来完成该 copy 任务

#### ThrCopy

利用 tiled copy 当中的 tiled tv layout & mn shape 对 tensor `(M, N)` 进行划分，得到每一个线程所拥有的 tensor，表达公式其实和 ThrMMA 是一样的
$$
(\text{num}_V, \text{num}_M, \text{num}_N)=(V, \frac{M}{m},\frac{N}{n})
$$
但不一样的是 `num_V` 不一定就是 copy atom 中的 values 数量，还可能是由于 tiled copy 会重复多次执行 copy atom 所导致的 `num_V` 的增加

**ThrCopy 的作用仅限于划分， 最终传入 `cute::copy` 方法的仍然是 TiledCopy**

#### ldmatrix

ldmatrix 是为了满足 mma atom 的特殊排布应运而生，ldmatrix 能够将自己线程的数据发送到其他线程当中，这在常规的 CUDA 编程中是做不到的，因为在 SIMT 编程下我们认为寄存器是线程私有的。

<img src="CUDA Programming 8.1/v2-c1031c4aa65e40d119c601740b9afd1c_1440w.jpg" alt="img" style="zoom:50%;" />

第一张图描述了 ldmatrix 的高效性：一个 thread 将搬运 8 个元素，并分配到不同的线程当中。在一般的 LDS 命令下，一个 thread 只能搬运 2 个元素，所以要进行 4 次搬运，效率大大降低。

<img src="CUDA Programming 8.1/v2-5a2257c2bea9b2f6652cfe579444f3bb_720w.webp" alt="img" style="zoom:67%;" />

第二张图则需要对应我们的 copy traits 一起食用。该图其实就是 ldmatrix 的 warp 版本。其搬运了一个 `(16, 16)` 大小的 half 矩阵。需要注意的是数据排布顺序要按照图示中的箭头来看

```c++
template <>
struct Copy_Traits<SM75_U32x4_LDSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape < _32,_128>,
                           Stride<_128,  _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape <_32,Shape <_32,   _4>>,
                           Stride<_32,Stride< _1,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
};
```

我们把 src layout 和 dst layout 都打出来看，由于所使用的 data type 为 half，所以 src layout 和 dst layout 转化为 `(t, v) -> logical mem id` 映射

<img src="CUDA Programming 8.1/image-20250811162318861.png" alt="image-20250811162318861" style="zoom:50%;" />

上面的打印中相同的数字代表了相同的 logical mem id，即他们代表了统一个元素。可以看到在 src 当中的 t0 拥有数据 0~7，他们分别分配到了 dst 当中的 t0~t3 中的前两个 values 当中。而对于 dst 当中的 t0 数据则来自于 t0, t8, t16, t24 的前两个 values

为什么我始终强调逻辑位置 logical mem id，这是因为这些元素在内存中的位置与逻辑位置并不一致。最重要的是：**根据 logical memory id 我们可以构建一个 src tv -> dst tv 的映射关系，从而能够轻松获得 src tv 中的元素在 dst tv 当中的位置**

#### How to build?

构建 tiled copy 的双核心逻辑

1. 对于使用 universal copy 的场景，针对所需的 mn shape，计算得到 thr & val layout（二者的 layout product 即为 mn shape）。由 `make_tiled_copy` 通过 thr & val layout 自己推导得到 tv layouts & mn shape
2. 对于 tv layouts 有特殊要求的 copy 场景（i.e. mma），通常我们都是通过 tiled mma 中的 `get_layoutA/B/C_TV` 直接获得 tv layouts & mn shape，而不是通过 copy atom 去自行推导这个 tv layouts & mn shape。此时需要考虑的是 tv layouts 与 copy atom 之间的合法性问题，即 copy atom 的整除要求（size of v 需要至少为 8）。此时一个 cta block 的 copy 能力是 mma atom mn shape 的重复，可通过 permutation mnk 参数进行调整

**CopyAtom 本身并不包含对 MN Tiler 的描述，其只描述了 src & dst tv 之间的映射关系。真正包含 Tiler shape 信息的 atom 是 MMA Atom，CopyAtom 服务于 MMA Atom，帮助其完成 Tile 数据的搬运。而 TiledCopy 恰好是必须要 MN shape 信息的 (which is shared by src & dst tv mapping)，这就会和我们理解 CopyAtom 造成违和感，但理解这一点是至关重要的**。另外一个重要的经验结论：(在 s2r or r2s 场景下) TiledCopy 要么是完整的 mma mn shape，要么是 mma mn shape 的一个子区域，二者的区别是使用了不同的 tiled copy create function

```cpp 
using R2STiledCopy = decltype(make_tiled_copy_C(r2s_copy_atom{}, TiledMMA{}));	// complete mn shape
using R2STiledCopy = decltype(make_tiled_copy_C_atom(r2s_copy_atom{}, TiledMMA{}));	// atom mn shape, deduced internally
// atom mn shape meaning the mn shape under 1 copy atom operation
```

对于 smem -> rmem 这个环节当中，我们利用 mma atom mn shape 作为基础的 building block，为了配合 copy atom 合法性，我们对其 mnk tile 进行了相应的重复，最终**构建出实际使用的 mnk tile**，cta problem 将由这个 tile 进行切分解决

此时一个大的 picture 正在浮现开来：**tile centric CUDA programming**。核心问题：

1. **Is this CopyAtom legal?** 

   参考重要补充材料-Copy 连续性要求 

2. **What's your mn tile to partition a cta problem?**

   一旦 CopyAtom 合法，根据 dst tv -> mn 映射，我们可以通过 `make_tiled_copy_C_atom` 计算得到一个 atom mn shape，作为 copy 的基本单位去解决 cta problem shape 层面的 copy。当然我们也可以不从 atom mn shape 粒度出发，从更大的粒度出发也是可行的 `i.e. make_tiled_copy_C`。而在 tma copy 中，这个 atom mn shape 的定义变得更加简单，可以很自由且方便地去定义一个 copy box 作为我们的基本 copy 单位

### 重要补充材料

**补充（2025/09/17）：retile 到底要解决一个什么样的问题？结论：解决线程 register 的 layout 转换问题**

我们在思考 copy 的问题时，其实还是更容易从整体去思考，例如把一个 MN shape 的数据进行划分，每一个线程获得各自的数据，然而最后我们都是面向 thread 编程，各个线程的 register 数据都是各自独立（互不可见）的，我们必须要将自己的视角进行转换。以下有三个划分视角：

对于一个 MN shape 数据

1. 我们可以使用 mma atom 的 layout 对 MN shape 的数据进行划分，每一个线程的数据 `tCrC_0`

   假设 mma atom layout 的 mn shape 为 `(m, n)`，每一个 thread 有 4 个 values，那么 `tCrC_0.shape = (4, M//m, N//n)`

2. 我们可以使用 s2r copy atom 的 layout 对 MN shape 的数据进行划分，每一个线程的数据 `tCrC_1`

   假设 s2r copy atom 的 mn shape 为 `(2m, n)`，每一个 thread 有 8 个 values，那么 `tCrC_1.shape = (8, M//2m, N//n)` 

3. 我们可以使用 r2s copy atom 的 layout 对 MN shape 的数据进行划分，每一个线程的数据为 `tCrC_2`

   假设 r2s copy atom 的 mn shape 为 `(m, 2n)`，每一个 thread 有 8 个 values，那么 `tCrC_1.shape = (8, M//m, N//2n)`

以上三种划分，最终得到了三种数据 `tCrC_0/1/2`，而这**三种数据实际上包含了相同的数据内容**，更具体来说，这三个 tensor 的 `tensor.data()`，指向的是同一片内存，但是他们的排布 `tensor.layout()` 完全不同。实际上 retile 干的事情就是这样，把相同拥有相同 data 的 tensor 转换为所需要的 layout，本质上就是做了这么一件事

```cpp
// retile A tensor to B tensor's layout
A_retiled = make_tensor(A.data(), B.layout())
```

但是这个 B 的 layout 计算有时候并不是那么明显的，所以 retile 将 B layout 计算都隐藏起来了。拥有了 retile 过后，就能够在各个形态进行丝滑转换，我们无论是在进行 mma 计算，还是在进行数据 copy，就可以构建同一份 register 数据的不同排布，以确保在 `cute::copy & cute::gemm` 在进行坐标 index 的时候获得了正确的数据

我之前对于 retile & tiled copy 没有那么熟，所以认为要用更多的概念来进行区分。实际上从始至终，我们都是在 block level 上进行编程，更多由重复所带来的功能，都可以由 `cute::gemm & cute::copy` 进行完成。而由于 copy & mma block 之间，对数据的划分各有不同，所以产生了对数据 layout 的操作转换，这带来了极大的学习困难

**补充（2025/10/28）：retile solved by compose & inverse**

[写给大家看的 CuTe 教程：Layout compose & Inverse](https://zhuanlan.zhihu.com/p/1962625273636845008) 受到其中的例子启发，我又重新审视了一下 retile，并且更深入地对 product/divide 和 inverse 进行了练习，获得了一些不错的经验。现在对 retile 问题进行更具体的阐述：

Condition：对于一个 gmem tensor x，使用了两种 partition 方式（e.g. 不一样大小的 tiler），`partition_A` & `partition_C`，划分过后每个线程所获得的数据分别为 `gA` 和 `gC`，并且已经申请了 register `rA = make_fragment_like<AType>(gA)` 用于 copy `gA`

Target：以最小代价构建 `rC`

有三个不一样的思路（包含错误思路），我都来分析一下：

1. 直接使用 `gC` 的 shape 和 `rA` 的数据

   ```cpp 
   rC = make_tensor(rA.data(), make_layout(gC.shape()))
   ```

   这显然是行不通的，`gC` shape 所生成的 layout 是一个 natural layout，其 stride 和真正的 `rC` 是不一样的

2. 使用 `make_fragment_like` 构建 `rC`

   ```cpp
   rC = make_fragment_like<AType>(gC)
   ```

   该方法的确能够获得正确的 `rC` layout，但是会额外申请寄存器，造成资源浪费。如果我们知道 `make_fragment_like` 计算 `rC` layout 的方法也是可行的

3. 构建 `gC coord -> gA coord` 的映射，利用 compose 获得 `rC coord -> offset` 映射，该映射即为正确的 `rC` layout

   首先我们来看几个 tensor layout 所代表的映射

   - `gA` layout 是 `gA coord -> gmem offset`，即 tensor coordinate 到 gmem offset 的映射
   - `gC` layout 是 `gC coord -> gmem offset`，类似 `gA`
   - `rA` layout 是 `rA coord -> rmem offset`，即 tensor coordinate 到 register offset 的映射，其中 `rA` 的 shape 和 `gA` 是一致的
   - `rC` layout 是 `rC coord -> rmem offset`，类似 `rA`

   我们构建 `gC coord -> gA coord` 的桥梁就是：`gA & gC` 有着相同的 gmem offset domain，即他们的数据是一样的，此时我们可以通过 inverse + compose 构建映射

   ```cpp
   // gmem offset -> gA coord
   inv_gA = left_inverse(gA)
   // gC coord -> gA coord 
   gC_to_gA = inv_gA.compose(gC) // gC -> gmem -> gA
   ```

   有了 `gC -> gA` 的映射过后，直接利用 compose `gA -> rmem offset` 的映射即可完成 `gC -> rmem offset` layout 的构建，因为 `gC` 和 `rC` 有相同的 shape，所以得到的就是 `rC` 的 layout

   ```cpp
   // rA & gA has the same shape
   // gC -> (gA = rA) -> rmem offset
   rC = rA.compose(gC_to_gA)
   ```

**补充（2025/10/31）：mma tv layout solved by product & inverse**

以上例子都需要有一个前提：不同的 partition 过后，thread 所获得的数据都是相同的。这个前提如何确保满足？我开始对 mma layout 进行了更多的研究，我发现 mma layout 只不过是同一种模式的复制粘贴：不断地重复一个 8x8 的 tile，其 tv layout 可写作

```python
# tv -> mn
mma_basic_layout = Layout(
    shape=[4, 8, 2],
    stride=[16, 1, 8]
)
```

<img src="CUDA Programming 8.1/image-20251104210802954.png" alt="image-20251104210802954" style="zoom:50%;" />

我们可以模仿 `make_tiled_copy` 中的方式，推导出这个 tv -> mn layout

```cpp
// (m1, n1) -> tid
auto mn2tid = make_layout(make_shape(_8{}, _4{}), make_stride(_4{}, _1{}));
// (m2, n2) -> vid
auto mn2vid = make_layout(make_shape(_1{}, _2{}), make_stride(_0{}, _1{}));

// ((m2, m1), (n2, n1)) -> (tid, vid)
// raked product to make v comes first
// ((_1,_8),(_2,_4)):((_0,_4),(_32,_1))
auto mn2tv = raked_product(mn2tid, mn2vid); 

// inverse & with shape
// (tid, vid) -> (m, n)
auto tv2mn = left_inverse(mn2tv).with_shape(make_shape(_32{}, _2{}));
```

其中 inverse 过后，如何确保 `with_shape` 一定是正确的？万一 inverse 过后的 shape 是 `(vid, tid)` 呢？不会，一定会是 `(tid, vid)`，这是由于 product & inverse 的性质所决定的：

1. product 中，mn2vid 中的维度所对应的 stride 一定是被 multiply 的一方，这就决定了 vid 对应的 stride 会是最大的
2. inverse 过后 stride 最大的 shape 会在最后（请回看 inverse 的推导过程）

两个性质决定了 inverse 过后一定会是 `(tid, vid)` 的排列顺序，所以我们用 `with_shape` 能够很方便进行 reshape

现在得到了 mma 中的 basic tv -> mn layout，那么上图中重复 4 次的 tv -> mn layout 如何得到？很简单，我们在其中使用一个 blocked product 重复 4 次即可

```cpp 
// repeat (2, 2) mn -> tv
auto mn2tv_4x = blocked_product(mn2tv, make_layout(make_shape(_2{}, _2{})));
// inverse to get (t, v, 2, 2) -> (m, n)
// give all the repeat to v
// ((_4,_8),(_2,_2,_2)):((_32,_1),(_16,_8,_128))
auto tv2mn_2x = left_inverse(mn2tv_2x).with_shape(make_shape(_32{}, _8{}));
```

正如 product 和 inverse 的性质导致，重复的 mode 会在 inverse 之后的 shape 排在最后。我们有一个 `(2, 2)` 的 blocked product，不过我们到底是重复 4 次 t，还是重复 4 次 v，还是 tv 各自重复两次？这就需要根据需求进行 permute & reshape，在此情形下，是将 v 重复 4 次，所以直接用 with shape 即可，最后得到的 layout 和 mma traits 中的 layout 一模一样👏

除了上述重复方法外，还有一个方法，参考自 `mma_atom.hpp` 当中的 `thrfrg_A`：从扩张过后的 MN -> MN tensor 开始，利用 zipped divide 获得 tensor `(AtomM, AtomN), (RestM, RestN)`，然后利用 compose atom tv layouts 获得 `(t, v), (RestM, RestN)` layout，最后通过简单的 flatten & group 也可获得正确的 layout

`with_shape` 的实现本质是一个 compose，这也指导我们，reshape 可以使用 compose 直接完成，尤其是对某一个 mode 做 reshape 的时候可以用 `compose(_, layout, ...)` 来跳过其他 mode。注意当 `layout.compose()` 传入多个 layout 的时候会自动使用 `make_tile(layouts)` 进行 by mode compose。所以对于 nested layout 中的某一个 mode 进行 reshape 时，也应当使用 `make_tile`

然而对于 permute 没有优雅的方法，只有老老实实构建新的 tensor 了

- `_` 在 product, divide, compose 当中的作用

  在 compose 当中其实就是跳过某个 mode，另外没有 `make_layout(_ ,)`

  divide，只有 `logical_divide(_, shape, ...)` 是跳过某一个 mode，其他的 divide 都很难成功，`zipped_divide` 只有针对两个 shape 的时候才会成功

  product 无法使用 `_` 进行跳过，不然 `_` 会直接进入到 shape 当中，可以使用乘 1 的方式来跳过，最后使用 with shape 进行整合

**Copy 连续性要求**

我们通常不会考虑 copy 的连续性要求，因为由于 copy 与使用场景的强绑定性，连续性要求都是会被满足的，不过在此我仍然以 ldmatrix 为例子，看下该要求的基本形式。ldmatrix 其实是要求 src tv 中每一个 thread 所拥有的 8 个 values 在 shared memory 中是连续的。这种约束也存在在 universal copy 当中

```c++
using R2SCopyAtomC = Copy_Atom<UniversalCopy<cute::uint16_t>, T>; // 16-bit contiguous
using R2SCopyAtomC = Copy_Atom<UniversalCopy<cute::uint32_t>, T>; // 32-bit contiguous
using R2SCopyAtomC = Copy_Atom<UniversalCopy<cute::uint64_t>, T>; // 64-bit contiguous
```

可以从 ldmatrix 中的 src tv 与 dst tv 之间的映射找到如下关系

```python
DST						SRC		 
----------------------------
T0~T3    V0~V1 <=> T0  V0~V7
T4~T7    V0~V1 <=> T1  V0~V7
...
T28~T31  V0~V1 <=> T7  V0~V7
----------------------------
T0~T3    V2~V3 <=> T8  V0~V7
T4~T7    V2~V3 <=> T9  V0~V7
...
T28~T31  V2~V3 <=> T15 V0~V7
----------------------------
```

用语言描述一下第一行：dst T0~T3 线程的 V0~V1 数据，对应了 src T0 线程的 V0~V7 数据。对于 ldmatrix 而言，其要求 src thread 中的 V0~V7 在内存中是连续的。OK，现在我们就用 mma atom 的 tv layout 来实际看一下，其 src thread 中的 V0~V7 是否真的连续。以 `SM80_16x8x16_F16F16F16F16_TN` 中的 matrix A 的 (dst) tv layout 为例，用 `print_latex` 打出来得到如下排布

<img src="CUDA Programming 8.1/image-20250811163804449.png" alt="image-20250811163804449" style="zoom: 33%;" />

我们可以发现 T0~T3 的 V0~V1 数据，正好是横向连续的 MK 坐标，这也说明了 T0 线程的 V0~V7 就是连续的 MK 坐标，但是为了保证内存的连续，MK -> Memory 的映射必须是 LayoutRight 即 row-major 排布内存，否则这些横向连续的 MK 坐标所对应的数据在内存仍然不连续

综上，在所给的 ldmatrix + mma layout + tensor layout 的条件下，copy 的连续性得到了满足。这也凸显出了三者的高度定制性：ldmatrix 必须和匹配的 mma layout 以及匹配的 tensor layout 进行使用，否则将会报错

**Async Copy**

在进行 copy 的时候经常会使用异步的 copy，即发出命令过后不会等待 copy 完成而是会继续执行后面的代码。但是我们也需要一些等待指令，以保证在计算时数据的确已经 copy 完成了。cutlass 提供了两个结构 `cp_async_fence & cp_async_wait` 用于完成这样的操作，在之后的 hgemm 实践中会有具体表现，这里先仅二者的功能

`cp_async_fence`

- 这是一个内存屏障（fence）操作，用于标记当前所有已提交的异步拷贝（`cp.async`）任务的完成点。
- 它的作用是确保在该 `fence` 之前的所有 `cp.async` 操作（即从全局内存到共享内存的异步拷贝）被视为一个批次，后续的 `cp.async_wait` 可以对这些批次进行同步。
- 它并不阻塞线程，只是标记一个任务提交的边界。

`cp_async_wait`

- 这是一个同步操作，用于等待之前提交的异步拷贝任务完成。
- 参数 `N` 表示“等待除了最新的 `N` 个批次之外的所有批次完成”。例如：
  - `cp_async_wait<0>`：等待所有之前提交的异步拷贝完成。
  - `cp_async_wait<1>`：允许最多 1 个批次的异步拷贝未完成（即等待除最新提交的 1 个批次外的其他所有批次完成）。
- 通常用于实现流水线的同步，确保数据在计算之前已经加载到共享内存。

## 核心优化

### 多级流水线 (Double Buffer)

多级流水线在 [cute 之 GEMM流水线](https://zhuanlan.zhihu.com/p/665082713) 中已经介绍地比较完善了，我这里将其中译中一下

<img src="CUDA Programming 8.1/v2-f9c13c984a5d8364e2d67e592cf7ddbf_1440w.jpg" alt="img" style="zoom:67%;" />

解释图中各个模块的含义：

1. 浅绿色长方形代表：全局内存到共享内存的数据搬运 $G^i \rarr S^i$ ，上标 $i$ 代表的是第 $i$ 个 Tile 的数据（我称之为大 k 循环）

2. 浅橙色长方形代表：共享内存到寄存器的数据搬运 $S_j \rarr R_j$，下标 $j$ 代表的是第 $j$ 个小 k 循环（Tile 内循环）

3. 深绿色的长方形代表：TiledMMA 利用寄存器上的数据进行矩阵计算

4. 黑色实线之间代表：完成一个 Tile 的矩阵运算（完整的小 k 循环）。并且黑色实线上方使用了曲线虚线进行了连接，代表完成了一个 Tile 计算之后继续计算下一个 Tile

5. 黑色虚线代表：进行 `cp_async_wait`，等待 shared memory 搬运完毕

整个流水线的关键步骤：

1. 首先将 `Stage - 1` 个全局内存到共享内存的加载任务异步地发布出去（发布过后不进行等待，直接执行之后的任务）

2. 等待 $S^0$ 的数据完成加载

3. 在进入小 k 循环之前，首先从 $S^0$ 中取出第一个小 k 循环所需要的数据，将其发送到寄存器上 $S_0\rarr R_0$

4. 此时正式进入到小 k 循环，可以分为 4 个要点：

   1. 发射异步读取新 Tile 的任务请求，即图中的 $G^3 \rarr S^3$
   2. 从共享内存中异步读取下一个小 k 循环所需要的数据 $S_j\rarr R_j$
   3. 执行第一个小 k 循环矩阵运算
   4. 重复步骤 2~3 直到当前小 k 循环完成

   需要注意的是，在做最后一个小 k 循环时，我们需要读取下一个 Tile 中的第一个小 k 循环数据，该操作需要使用 `cp_async_wait ` 来保证下一 Tile 的数据已经完全加载到 shared memory 当中。这也是图中的虚线所表达的含义

我们也经常听说 double buffer 这个词，其实就是多级流水线的一个特例，即流水线的级数等于 2，级数数量就等于 buffer 数量。在上图所示的流水线中，shared memory 流水线级数为 4，register memory 流水线级数为 5

### Swizzle

[cute 之 Swizzle](https://zhuanlan.zhihu.com/p/671419093) 已经将 swizzle 将得特别清楚了。这段话极其本质

> 回顾之前的介绍我们知道描述逻辑空间我们可以使用 [Layout（本质是函数）](https://zhuanlan.zhihu.com/p/661182311)，而为了避免 bank 冲突，cute 中定义了 swizzle 抽象，swizzle 的本质也是函数，swizzle 作用在 layout 上，即函数作用在函数上，复合函数复合的定义。Layout 的作用是给定坐标返回 offset，而 swizzle 的作用则是给定 offset 返回 bank conflict free 的 offset。即
> $$
> offset_{\text{no-conflict}}=Swizzle(Layout(coord))
> $$

通过 swizzle 获得了新的 layout，将 (M, N) -> offset 的位置进行改变。所以当在进行 read & write 时，会将数据读写到 swizzled position 从而避免 bank conflict

并且 swizzle (晃动/摇动) 这个名字特别的形象，想象你正在向 tensor `x` 的某个 coord `(m, n)` 写入数据

```c++
x(m, n) = 1.0
```

它本来该在 `layout(coord)` 位置写入该数据，结果 swizzle 了一下，写到了 `swizzle(layout(coord))` 位置。物理位置对于读和写其实是无感的，因为读和写操作的是 tensor coord `(m, n)`

```c++
print(x(m, n))	// 1.0
```

swizzle 不同于普通的 layout algebra，没办法用之前的 composition 来统一表达，但其本质仍然是函数映射。通过 M, B, S 三个参数来完全表示。最小单元为 $2^M$，而这个单元就是从 layout offset 顺序进行 group 和排序

swizzle 似乎给我上面的连续性分析带来了矛盾：swizzle 会打乱数据的连续性，但如果以 $2^M$ 为单位的话，基本的连续性还是有保障的。例如 $2^3$ 为单位的话，那么连续 8 个数据则都会是连续的，这就能满足 ldmatrix 的连续性要求

Swizzle 具体的计算过程在这里下不整理，在之后用 Swizzle 解决 bank conflict 处再详细说明，理解其意义，并且知道如何用 swizzle 来解决不同情况的 bank conflict

#### Bank Conflict

首先定义两个概念：

1. shared memory bank

   共享内存被划分为多个独立的、等宽的存储单元，称为 **Bank**。每个 Bank 的宽度：**4 bytes（32-bit）**（所有现代 NVIDIA GPU 均如此）。Bank 总数：**32 个**（对应一个 Warp 的 32 个线程）

   每个 Bank 可以独立读写，因此 **32 个线程可以同时访问 32 个不同的 Bank**（无冲突）。如果多个线程访问同一个 Bank 的不同地址，则发生 **Bank Conflict**，导致访问串行化

2. phase

   **1 个 Phase** = 硬件一次性完成的 **128B 数据传输**（32 Banks × 4B）

   **线程参与 Phase 的方式**：

   | 每个线程的请求位宽 | 填满 128B 所需的线程数 | 是否典型优化   |
   | :----------------- | :--------------------- | :------------- |
   | 4B（32-bit）       | 32 线程                | 否（低效）     |
   | 8B（64-bit）       | 16 线程                | 部分场景       |
   | 16B（128-bit）     | 8 线程                 | **是**（最优） |

   **为什么 8 线程 × 16B 是最优的？**

   - 减少指令数（1 条 `LDG.128` 代替 4 条 `LDG.32`）
   - 最大化带宽利用率（单次 Phase 完成更多数据搬运）

   bank conflict 考虑范围的是一个 phase 内，不会考虑两个 phase 或更多，因为同时考虑两个 phase 一定会产生 bank conflict，因为一个 phase 就把 bank 宽度填满了，两个 phase 中必定有不同线程指向相同的 bank

   正如本文之前所示的 ldmatrix 示意图，一个黑色方框 (8x8 half matrix) 就是一次 phase 读取

   <img src="CUDA Programming 8.1/v2-5a2257c2bea9b2f6652cfe579444f3bb_720w.webp" alt="img" style="zoom:67%;" />
   
   update 2025/07/19 补充一下 `LDG.128` 与合并访问之间的关系
   
   > From Kimi
   >
   > **LDG128 是向量化加载指令，天然利于合并访存**。在 CUDA 中，**一个 warp（32线程）如果使用 LDG.128 连续访问内存地址**，则：
   >
   > - 每个线程请求 16 Byte；
   > - 整个 warp 请求 32 × 16 = **512 Byte**；
   > - 如果地址对齐且连续，这 512 Byte 可以合并为 **4 次 128 Byte 的事务**（512/128 = 4）。
   >
   > 这**极大提高了合并度（coalescing degree）**，减少 memory transaction 数量，提升带宽利用率。
   
   使用4次 `LDG.32` 仍然可能仅使用在 4 次 128 Byte 的内存事务完成，但是相比 `LDG.128` 会使用更多的指令，这也会消耗更多的时间。所以尽可能使用 `LDG.128` 指令

在 reed zhihu 中有一个分析 bank conflict 的思路

> 完整的512byte需要4个phase才能完成访问。**这种情况也可以看作是：shared memory基本单元为16byte，总bank数为8，冲突与否的分析不在是32线程，而变成4个phase中的不同线程。如果采用64bit的访问形式，则相应的基本单元可以看作是8byte，总bank数目为16，冲突与否的条件变成两个phase内的线程是否冲突。**整体上shared memory空间可以看作二维存储空间，其中列方向表示bank情况，行方向表示自由定义的大小。

我们可以从不同的粒度来构建简化过后的 shared memory 模型，方便我们分析。用这个模型来分析一个 16x16 or 16x64 size 的矩阵读写

**所以Bank Conflict数量其实可以等价的理解为，在一个Phase内需要额外多少访存次数**。From [zhihu](https://www.zhihu.com/question/667972067/answer/43935974172)

理解 swizzle 以及其使用需要对多个概念进行熟悉。网络上的教程每一个都有自己对 swizzle 的定义和理解，我结合了三篇 blog 总结出自己对 swizzle 的理解：

1. [LeiMao-CuTe Swizzle](https://leimao.github.io/blog/CuTe-Swizzle/)，最为严谨的 blog，给出了准确概念，并且有实际例子与计算过程，能够推导出一般 swizzle 参数的计算公式
2. [Swizzle 本质思考](https://zhuanlan.zhihu.com/p/32954684694)，给出了逻辑行列和物理行列的思考模式
3. [实用 Swizzle 教程系列](https://zhuanlan.zhihu.com/p/20579515046)，是第二篇 blog 的参考，我也列在这里

我将按照用五个部分来叙述 swizzle 概念以及其使用方法，并在最后给出解决 bank conflict 的一般思路

1. Swizzle Arguments，介绍 swizzle 概念

2. Introduce Examples，用例子来熟悉 swizzle 概念

3. Logical & Physical view，介绍逻辑 & 物理的不同视角来看到 swizzle bits

4. Common Examples，利用逻辑 & 物理 offset 分析一些常见例子

5. General Methods，给出一般解决思路

#### Swizzle in Bits

cutlass swizzle 其实是按地址的 bit 来解释的，其注释写得其实很清楚，但很容易被其迷惑的排版给迷惑了

```c++
// A generic Swizzle functor
/* 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
 *                               ^--^ MBase is the number of least-sig bits to keep constant
 *                  ^-^       ^-^     BBits is the number of bits in the mask
 *                    ^---------^     SShift is the distance to shift the YYY mask
 *                                       (pos shifts YYY to the right, neg shifts YYY to the left)
 *
 * e.g. Given
 * 0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
 * the result is
 * 0bxxxxxxxxxxxxxxxYYYxxxxxxxAAAxxxx where AAA = ZZZ xor YYY
 */
```

swizzle 一共有3个参数：M, S, B。在 reed 的教程中分别解释为：基本单元包含的元素，一行包含的单位单元，有多少行。这当然是最直观的解释，不过现在我们要将这些参数用一般的 address 来看待。这里一个 address 是一个 32bit 的数据（可以数一下，上面注释一个地址包含了 32 个字母），下面是英译中

1. `M or MBase`，保持不变的位数量

   例子中用了 4bit，代表着一个基本单元包含 16 个元素。其中的字母都是一个 bit 其值为 0 or 1

2. `B or BBits`，mask 当中的位数量

   例子中用了 3bit，我们可以将其直观解释为“行号”

3. `S or SShift`，需要位移的位数量

   例子中用了 10 bit，我们可以将其直观解释为“列号”

例子中的直观解释：一个基本单元包含 16 个元素，一行包含了 1024 个基本单元，一共有 8 行。在进行 swizzle 计算时，其实就是用行号 `YYY` 和列号 `ZZZ` 进行了异或操作，获得了新的列号 `AAA`。这里有一个隐藏的限制：`S <= B`，否则无法有足够的位完成异或操作

异或操作由于其封闭性和双射性会将数据进行完美的重排，即不会有多个数据排到相同位置，也不会有数据排布到规定范围之外。下面用一些基本的例子来看如何利用 swizzle 将数据进行重排，从而避免 bank conflict

#### Introduce Examples

**Example 1**

读取一个 fp32 matrix 中的一列，matrix layout 为 `Layout(shape=[32, 128], stride=[128, 1])`

shared memory bank 一行能够装下 1024bit 的数据，矩阵的一行有 128 个 32bit 元素，会填满 4 行的 bank。假设我们读取第一列的数据，各个数据的 offset 根据 layout algebra 的运算为

```python
128 * 0
128 * 1
128 * 2
...
128 * 31
```

由于 `offset % 32` 的结果都是 0，所以这些数据都会落在 bank0 的位置，会引起非常严重的 32-way bank conflict。读取其他列也是类似的情况

不过我们可以通过 swizzle 来解决这一个问题：

1. `M = 0`，一个基本单位包含 1 个 fp32 元素
2. `S = 7`，一行包含 128 个基本单位
3. `B = 5`，一共有 32 行

我们的 swizzle bit version表示如下
$$
\underline{xxxxx}\ yy\underline {yyyyy}
$$
第一列的列号为 `00000000`，32行的行号为 `00000~11111`，通过异或操作对应的 5bit 得到新的列号（公式中加下划线的部分）

```python
00000 xor 00000 = 0
00000 xor 00001 = 1
...
00000 xor 11111 = 31
```

此时第一列的所有数据通过 swizzle 被分配到了 32 个不同的 bank，彻底解决了 bank conflict。其他列同理可证

**Example 2**

在 Example 1 的基础上，使用向量化内存读取（Vectoriezed Memory Access），让单个线程一次性读取或写入连续的多个数据元素。既然一个线程读取的数据变多了，那么一个 phase 所包含的线程数量就会减少。所以我们讨论的范围变为：用 8 个线程，每一个线程读取 4 个 fp32，即读取 matrix 当中的一个 (8, 4) 区域

如果未经过 swizzle，那么就会产生 8-way bank conflict，每一个线程的起始地址都在相同的 bank 当中

直接计算 swizzle 中的参数，就可以将这些在相同 bank 的地址，重排到其他地址当中 $\underline{xxx}\ yy\underline {yyy}\ zz$

1. `M = 2` 一个基本单位包含 4 个 fp32 元素
2. `S = 5` 一行包含 32=(128/4) 个基本单位
3. `B = 3` 一共有 8 行

另外再强调一个“显而易见”的事情：通常产生 bank conflict 的情况都是在访问“列”方向上，而不会出现在访问“行”方向上。因为一行中的数据本身就放在了不同的 bank 当中，并且我们讨论的范围还是一个 phase，即 32 个 bank 的总宽度，那么在访问连续的“行”数据时，一般是不会发生冲突的

在以上两个例子中你会发现他们的 M + B 都等于 5，他们的 M + S 都等于 7，这并不是巧合，而是我们推导 Swizzle 所遵循的公式

#### Logical & Physical view

这一节我将通过例子来介绍如何从逻辑视角转移到物理视角来解释如何计算 swizzle bits

**Example 1**

以一个 fp16 的 matrix 为例，其 matrix layout 为 `Layout(shape=[16, 16], stride=[16, 1])`，线程读取方式仍然是老图的左侧所示

<img src="CUDA Programming 8.1/v2-5a2257c2bea9b2f6652cfe579444f3bb_720w.webp" alt="img" style="zoom:67%;" />

我们先写一个其逻辑上的 swizzle bits

1. `M = 3` 一个基本单位包含 8 个 fp16 元素，这里我们仍然假设是使用 128bit 向量化读取
2. `S = 1` 一行包含 2 个基本单位
3. `B = 4` 一共有 16 行

用 swizzle bits 的方式来看
$$
xxxx\ y\ zzz
$$
但这样来看我们很难看出和 bank conflict 之间的关系。此时我们要以物理上的 swizzle bits 来看待。memory bank 一行有 1024bit 将包含 8 个基本元素，即 `S = 3`，再回到 Bank Conflict 小节的末尾，就能明白 reed 对于 bank 的一种逻辑抽象：此时我们可以认为一共有 8 个逻辑 bank

我们将这个 swizzle bits 修改为如下：`B=2, S=3, M=3`，相当于从 B 挪了两个 bit 到 S 当中
$$
xx\ xxy\ zzz
$$
此时我们可以看到，$xyy$ 这 3bit 就对应了 bank 的一整行，即 8 个逻辑 bank。当 $xxy=000$ 时就代表了逻辑 bank 0，此时对于前面两个 bit $xx$ 的任意值，他们都属于同一个逻辑 bank，所以会产生 bank conflict！再从逻辑视角来看，每 4 行会占据一整行的 bank 宽度，第 0，4，8，12 行的数据都会落在同一个 bank 当中

现在我们需要考虑线程读取的方式了，因为我们只考虑一个 phase 的读取，在本例当中，一个 phase 读取 (8, 8) 区域的矩阵，按照 swizzle bits 来算的话是 (8, 1) 个单位，即原来的 $xxxx$ 4bit 表示 16 行，我们现在只考虑 8 行 $xxx$
$$
\cancel xx\ xxy\ zzz
$$
现在可以看到目前是第 0，4 行就会产生 2-way bank conflict，我们直接在这个位置上进行 xor 操作，把 bank conflict 解决
$$
\underline x \ xx\underline y\ zzz
$$
此时我们的 swizzle 表示为 `Swizzle<B=1, S=3, M=3>` 就可以把这些冲突给解开。不过如果我们并不是使用 ldmatrix 的读取方式，仍有可能读取不连续的 8 行，所以此时设置 `Swizzle<B=2, S=3, M=3>` 才能解决掉所有冲突
$$
\underline{xx} \ x\underline {xy}\ zzz
$$
需要注意的是，ldmatrix 也可以使用 `Swizzle<B=2, S=3, M=3>` 来解决冲突，本质上该 swizzle 解决冲突的能力更强。按此推理，我们继续增加 B，使用 `Swizzle<B=3, S=3, M=3>` 也能够完全解决冲突。然而 B 的增加并不是没有上限的，其受限于逻辑 bank 的总数。在本例当中逻辑 bank 一共有 8 个，如果 B 超过 3 则没有足够的逻辑 bank 用于分配。B 也没有必要超过逻辑 bank 的总数，因为一个 phase 的大小就是逻辑 bank 的总大小，我们只需要考虑一个 phase 内可能产生冲突的情况

**Example 2**

在 **Example 1** 当中我们读取的是一个 (16, 16) 的矩阵，那么如果我们读取的是一个 (16, 32) 大小的矩阵，也是一个 phase 读取 (8, 8) 区域大小的数据，应该采用怎样的 swizzle 呢？

按照上面的分析我直接把这个 swizzle bits 写出
$$
\cancel x \underline{xx}\ x\underline{yy}\ zzz
$$
此时我们的 swizzle 表示为 `Swizzle<B=2, S=3, M=3>`，相比上一个例子多了一位的 mask bit，因为矩阵的一行会占一半的 bank，我们这样的读取方式会产生 4-way bank conflict，需要分配到 4 个不同的 bank 当中，所以 mask bit 需要为 2

同样的 `Swizzle<B=3, S=3, M=3>` 也能够解决上述冲突

#### General Methods

接下来我将给出 Swizzle 的通用公式，modified from [LeiMao-CuTe Swizzle](https://leimao.github.io/blog/CuTe-Swizzle/)

Consitions：一个 phase 为 1024 bit，每个数据为 `k` bit，一行有 `X` 个，向量化读取一次读取 `V` 个元素

Target：读取不同的列时不产生 bank conflict

1. `M` 是最好计算的参数，根据向量化读取的情况决定
   $$
   M =\log_2{V}
   $$
   
2. `B` 按照解决冲突能力最强的 swizzle 来计算，即访问一个 phase 所有的 bank 都在同一个 logic bank 当中的情况
   $$
   B=\log_2{\frac{1024}{k}} - M
   $$
   超过一个 phase 的情况则不在考虑范围内，因为不同 phase 之间不产生冲突

3. `S` 的计算需要分情况讨论，这是因为 swizzle 要求 `S >= B`

   1. 一行数据 `X` 未占满 bank：`S` 和 `B` 相等
      $$
      S=\log_2{\frac{1024}{k}} - M
      $$
      此情况没有被 [LeiMao-CuTe Swizzle](https://leimao.github.io/blog/CuTe-Swizzle/) 所考虑，但是是必要的。其对应于上面例子中把 $x$ 移动到 $y$ bit 部分，不会产生额外的 bank conflict，并满足 `S >= B` 要求
      
   2. 一行数据 `X` 已占满 bank：`S` 将计算一行元素会包含多少基本单元
      $$
      S=\log_2{X}-M
      $$
      
   3. 

   所以两个公式合成一个公式
   $$
   S=\log_2{\max{(\frac{1024}{k},X)}} -M
   $$

4. 

**该公式能够完美解决读取列数据产生 bank conflict 问题**

不过还有一点我想要指出，以 fp16 的数据类型为例：如果一行数据很多，即 `X` 很多，那就需要大的 `S`，这意味着 $y$ bit 位数增加
$$
\underline{xxx} \ yy\underline {yyy}\ zzz
$$
访问的 $y$ bit 位置为 `00xxx or 01xxx or 10xxx or 11xxx` 时就会产生 bank conflict，他们都属于同一个逻辑 bank `xxx`。**这是由于我们尝试一次读取不连续的行元素**。如果我们总是读取连续的行元素，那么这种情况将不会发生，因为如果我们在读取连续的行元素时，如果出现了 bank conflict 的情况，说明这一行元素已经占满了完整的 bank 长度，也就是说会超过一个 phase 大小，从而避免 bank conflict

### Epilogue

在计算完成后，我们需要将累加器（寄存器）中的结果，全部都运输到 global memory 当中存储起来。但直接完成这件事并不是最优选项，因为会造成不连续的数据写入（如下图），这样会导致存储时需要更多的内存事务，而不能使用向量化存储指令（STG.128）

<img src="CUDA Programming 8.1/v2-ddece7971d1161bbf7c7fa8022859993_1440w.jpg" alt="img" style="zoom: 50%;" />

针对这个问题，cute 中专门提供了 Epilogue 来通过共享内存作为中间媒介。先将寄存器数据存储到共享内存，然后再从共享内存中以更连续、更高位宽的形式存储到全局内存中去。对于 half 元素来说应该至少让一行有 8 个元素进行运输，这样就能用 128bit 的向量化存储指令了

## hgemm 实践

我在之前的笔记中提出了一个：tile centric CUDA programming 的思路，在这一小节中我将沿着这个核心思路，并进行更详细地拓展，利用这些思想解决高性能 hgemm kernel。这些思路也是借鉴了 tilelang 的 [demo](https://github.com/tile-ai/tilelang?tab=readme-ov-file#gemm-example-with-annotations-layout-l2-cache-swizzling-and-pipelining-etc)

在此我提出一个 2-level tile 的概念：

1. first-level: CTA Tile。作为最高 level 的 tile，该 level 非常方便我们设计宏观的 pipeline，e.g.: multi-stage or producer-consumer pipeline
2. second-level tile 会有许多种，其核心是具体解决 CTA tile 的各阶段问题，包含：各个阶段的 cta tile copy；计算 cta tile mma

tilelang 将专注于 first-level tile programming，把 pipeline 和 second level tile 问题都自动解决了，这给我们设计 kernel 带来了极大的便利，这必定是以后的大趋势。不过在此我们仍然要讨论清楚这些细节

- 可以从不同的 level 来设计流水线：from cta tile level to second-tile level，pipeline inside of a pipeline

### Define tile

我们以 tile 为 centric 作为构建模块，而 tile 的核心参考就是 mma shape。以 `SM80_16x8x16_F16F16F16F16_TN` 作为 mma op，其 mnk shape 为 `(16, 8, 16)`，我们以此为基础推理出合理的 tile 设置。为了方便讨论，我们把条件设置更具体一些：使用 4 个 warps，以 `(2, 2)` 的 layout 进行排列

1. mma mnk tile 的大小将从单个 warp 的形状 `(16, 8, 16)` 扩展为 4 个 warp 的形状 `(32, 16, 16)`
2. g2s tile，一定要使用向量化读写，每一个 thread 将对应 128-bit 数据（i.e. 8 个 fp16），128 个线程则能够复制 1024 个 fp16 数据，我们可以构建一个 `(32, 32)` 的 tile
3. s2r tile，需要满足 mma 的特殊 tv 要求，同时满足 ldsm 命令的合法性（size of v 必须为 8），我们需要在 mma shape 的 N 维度上进行扩展，构建出 `(32, 32, 16)` 的 tile，为什么要扩展两倍，请参考 TiledMMA & ldmatrix 小节
4. r2s tile，可以使用 `(32, 32)` 的 tile，注意由于 register 的特殊排布，无法使用 128-bit 的向量化读写
5. s2g tile，可以使用 `(32, 32)` 的 tile，使用高效的向量化读写

以上是 second-level tile 的设置，对于 cta mnk tile 的设置我们可以设置为 `(128, 128, 32)`，其中有两个参考理由：

1. 我们需要较大的 cta tile size 来增加计算时间，从而掩藏 copy 时间
2. 需要使用 double buffer，所以扩大了 k 方向大小

### Define smem

在 gemm 算法中定义 shared memory 主要从 3 个方面来考量：

1. 定义一个 block 需要处理的 Tiler MN shape（区别于 tiled mma mn shape）
2. 定义 shared memory 流水线 stages
3. 定义 register 流水线 stages

在 hgemm 实践中我们定义为如下：

1. 一个 block 需要处理 `(128, 128)` 区域的 MN 矩阵乘法（Matrix C view）
2. shared memory 流水线为 3 级
3. register 流水线为 2 级

根据以上定义我们可以计算得到所需要的 shared memory 大小以及 swizzle

1. matrix A & B 各需要 `(128, 32, 3)` 大小的 shared memory，其中 `32 = 16 * 2` 代表了 register 的两级流水线，会在小 k 循环中进行 2 次。最后一个维度 `3` 则代表了 shared memory 的 3 级流水线
2. matrix C 并不需要全部存储到 shared memory 当中，shared memory 只是作为一个中转站以方便进行向量化读取，所以需要 `(32, 32)` 大小即可，在 reed 所给代码中使用了 `(32, 32, 2)` 的大小，相当于申请了更大的 shared memory 作为中转，但在我的实验过程中发现加速效果不明显
3. 根据之前的 swizzle 计算思路，我们只讨论一个 phase 当中的 shared memory 读取，也就是 `(8, 32)` 大小的 shared memory 读取。那么利用公式可以得到 `Swizzle<B=2, S=3, M=3>`，而在 reed 所给代码中则使用了 `Swizzle<B=3, S=3, M=3>` 其能够处理更大范围的 bank conflict

### Pipelines

在之前我只是对 reed multi-stage pipeline 进行了简单的描述。可是要自己构建一个流水线应该如何做到？其实流水线的核心非常简单，就是任务重叠，具体到 GPU model 当中就是数据搬运与计算的重叠。最简单有效的 pipeline 就是 double buffer pipeline，可以用下图表示，横向为时间

<img src="CUDA Programming 8.1/image-20251110150608284.png" alt="image-20251110150608284" style="zoom:67%;" />

在重叠二者的编程中，有2个关键的要素：

1. **在计算当前 data MMA 时，同时预取下一个 data**
2. **在计算当前 data MMA 时，必须确保当前 data 填充完毕**

在具体实现时还有一些细节，例如计算 buffer index，以及在循环正式开始之前需要做的前置操作（e.g. 0-data load）等等。如果把 double buffer 进行扩展，有多个 buffer（也被称为 multi-stage），可以用下图表示

<img src="CUDA Programming 8.1/image-20251110170235632.png" alt="image-20251110170235632" style="zoom:67%;" />

以上展示了一个 4 buffer 的流水线过程。我们先预先发起 3 个 buffer 的数据搬运，在真实计算 MMA 0 的时候发起最后一个 buffer 的数据预取，这能够让我们预取更多的数据。我认为这并不是典型的 producer-consumer model，因为 producer 并不是持续地在进行搬运数据，而是在当前 MMA 计算时，同时去预取了下一个 data

在上图中 MMA 直接从 shared memory 中获得数据开始计算了，实际上在 sm80 架构上 MMA 需要从 register 获得数据进行计算。所以有一个 smem -> register 的数据搬运过程。这个过程也可以用 double buffer 的思路进行流水线并行，所以两个流水线构成了 pipeline in the pipeline。两个 pipeline 会有数据上的依赖性，具体来说 register pipeline 中要求对应的 shared memory 必须完成 copy，这一点需要在编程中显示确认。下图展示了一个 double register buffer 的 pipeline 示意图，每两个 register 将消耗一批 smem buffer，每两个 MMA 计算完成一批数据

<img src="CUDA Programming 8.1/image-20251110173829549.png" alt="image-20251110173829549" style="zoom:67%;" />

在实现过程中，我们可以逐层地实现 pipeline，把第 0 批的数据先预取好，然后直接开启 pipeline 循环。对于 epilogue 似乎没有使用 pipeline，可以直接按照常规的方案逐 tile 进行 regsiter -> smem -> gmem 搬运

### Pseudo code

问题定义与上述相同：解决 `MNK = (4096, 4096, 1024)` 矩阵乘，CTA Tile 为 `(128, 128, 32)`，CTA threads 为 128，warp layout `(2, 2)`，smem 有 3 个 stages

```cpp
// CTATile_MNK = (128, 128, 32)
// gA (CTATile_M, CTATile_K, num_k)
// gB (CTATile_N, CTATile_K, num_k)
// gC (CTATile_M, CTATile_N)
// sA (CTATile_M, CTATile_K, stages)
// sB (CTATile_N, CTATile_K, stages)

// tiled_mma (32, 16, 16)
// tiled_g2s (32, 32)
// tiled_s2r (32, 32, 16)
// tiled_r2s (32, 32)
// tiled_s2g (32, 32)

// register allocation
int idx = threadIdx.x;
auto thr_mma = tiled_mma.get_slice(idx); 
t_rA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (8, 128/32, 32/16)
t_rB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (4, 128/16, 32/16)
t_rC = thr_mma.partition_fragment_C(gC(_, _)); // (8, 128/32, 128/16)
clear(t_rC);

// g2s copy partition
t_g2s_gA = tiled_g2s.partition_S(gA); // (8, 128/32, 32/32, num_k)
t_g2s_sA = tiled_g2s.partition_D(sA); // (8, 4, 1, stages)
t_g2s_gB = tiled_g2s.partition_S(gB); // (8, 4, 1, num_k)
t_g2s_sA = tiled_g2s.partition_D(sB); // (8, 4, 1, stages)

// s2r copy partition
t_s2r_sA = tiled_s2r_A.partition(sA); // (8, 4, 2, stages)
t_s2r_rA = tiled_s2r_A.retile_D(t_rA); // (8, 4, 2)
t_s2r_sB = tiled_s2r_B.partition(sB); // (8, 4, 2, stages)
t_s2r_rB = tiled_s2r_B.retile_D(t_rB); // (8, 4, 2)

// prepare before mainloop
// 1. launch the stages - 1 copy
// 2. launch s2r first small iter k copy
int load_tile_idx = 0;
int mma_tile_idx = 0;
for (int istage=0; istage < stages - 1; istage++){
    copy(tiled_g2s, t_g2s_gA(_, _, _, istage), t_g2s_sA(_, _, _, istage));
    copy(tiled_g2s, t_g2s_gB(_, _, _, istage), t_g2s_sB(_, _, _, istage));
    cp_async_fence(); // commit
    load_tile_idx++;
}
cp_async_wait<stages - 2>();
__syncthreads();
copy(tiled_s2r_A, t_s2r_sA(_, _, 0, 0), t_s2r_rA(_, _, 0));
copy(tiled_s2r_B, t_s2r_sB(_, _, 0, 0), t_s2r_rB(_, _, 0));

// mainloop
int num_k = size<3>(t_g2s_gA);
int num_k_inner = size<2>(t_s2r_rA);
int buffer_idx = 0;
for (int itile = 0; itile < num_big_k; itile++) {
    // load next k tile
    if (load_tile_idx < num_k) {
        buffer_idx = load_tile_idx % stages;
        copy(tiled_g2s, t_g2s_gA(_, _, _, load_tile_idx), t_g2s_sA(_, _, _, buffer_idx));
        copy(tiled_g2s, t_g2s_gB(_, _, _, load_tile_idx), t_g2s_sB(_, _, _, buffer_idx));
        load_tile_idx++;
    }
    cp_async_fence();
    
    // small k iteration
    for (int ik = 0; ik < num_k_inner; ik++) {
        // load next small k tile
        if (ik == num_k_inner - 1){
            // make sure the next k tile complete
            cp_async_wait<stages - 2>();
            __syncthreads();
        }
        int ik_next = (ik + 1) % num_k_inner
        // calculate read tile
        int read_stage = (ik == num_k_iker - 1) ? itle % stages : (itile + 1) % stages
        copy(tiled_s2r_A, t_s2r_sA(_, _, ik_next, read_stage), t_s2r_rA(_, _, ik_next));
        copy(tiled_s2r_B, t_s2r_sB(_, _, ik_next, read_stage), t_s2r_rB(_, _, ik_next));
        
        // gemm
        gemm(tiled_mma, t_rC, t_rA(_, _, ik), t_rB(_, _, ik), t_rC);
    }
}
```

可以看到大量代码其实不是在 mainloop 当中，而是在资源申请和数据切分，本身流水线还是非常清晰！

## 总结

如何学习一个陌生且没有那么多资料的领域？

一些描述对于我来说或许非常抽象：数学公式，C++...但实际上这些都是非常清晰的描述，如果转换成为 python 或者我熟悉的语言描述我就能很好地理解。而这个过程恰好是 GPT 比较擅长的：因为 GPT 对这些语言都非常熟悉，将一个语言翻译为另外一种语言基本上不在话下，只要所提供的描述是准确且基础的，通过切入到我所熟悉的语言，那么理解起来就事半功倍了。但是如果所问的问题是一个没有太多资料的复杂领域：例如 layout algebra，如果不提供基础的数学证明材料，很难获得一个让我满意的回答，我也无法完成对问题的解决

在学习 cutlass 的路上 Grok & DeepSeek 给与了很大的帮助，可以具体看下其解决了哪些疑问

1. Layout Algebra python scripts

   利用原始数学证明材料写出了 layout algebra 各个基础运算的 python 代码。通过利用代码交互，能够更快地发现 layout algebra 中的一些性质

2. Compose first impression: fit spots to memory，但不够本质

   对于 compose 的最终顿悟来源于对 right inverse 的理解，彻底理解了 compose 是“映射”，赋予映射的 source domain & target domain 以含义具有重要意义

3. Cutlass recasting

   利用清晰的 C++ 代码得出了 recast 的算法过程

4. Swizzle Parameters

   利用 Lei Mao's blog 的清晰描述与定义，给出了 swizzle 例子的中间推导过程，理解 swizzle in bits 形式

**重大的突破其实来源于清晰的学习目标以及选择优秀的学习材料**。我需要学习材料包含足够多的上下文以支持我去完成所指定的目标。上下文主要包含几点：1. 清晰的文档结构与教程；2. 足够简洁的原理代码；3. 准确的公式推导（与第一点有所重叠）

三点钟任意满足一点就是不错的材料，满足两点就是非常优秀的材料。因为有了 GPT 的存在，对于不熟悉的领域可以“翻译”成为你所熟悉的语言，方便你进行理解：例如 c++ -> python or math -> python，并且可以通过构建最小例子来完成特例到通用的抽象化理解。所以拥有了好的学习材料，很大程度上就能保证学习的成功

