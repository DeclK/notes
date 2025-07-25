# CUDA Programming 9

本文档想要对 cutlass cute 进行更多的实践

1. 在写 kernel 中常用的 helper：如何构建 reference impl & test scripts
2. 各个原子操作的实现：copy & gemm & 数值基本操作

最终的目标有两个：

1. 实现一个 sota w8a8 gemm
2. 实现一个 sota rms norm

这两个的 kernel 实现其实都有的，不过使用 cute 来实现网上应该是找不到的

拓展：将自己写的 kernel 像 DeepGemm 一样构建，能够处理各种 shape 的情况，并且能集成到 python 当中

## Project Building

使用 cmake 来构建项目。先上 cmake 三件套

1. minimum version
2. project language
3. cuda & cxx common settings

## Create Tensor

在 claude code kimi 的帮助下，很快就构建出了 host 上的 tensor，无法使用 print values 查看，会出 segmentation fautl。说明所构建的 tensor 只是一个空壳

observations:

1. cute tensor 区别于 torch tensor，更倾向于对 layout 的变换，底层的数据基本不变 [reed zhihu](https://zhuanlan.zhihu.com/p/663093816)

   在 reed zhihu 当中还提到了栈上和堆上，我的理解就是栈上 tensor 就是 register，其余的都是堆上的。在 cute doc 提到的相似的概念是 owning 和 non-owning tensor，其中 owning tensor 就是 register 上的 tensor

   既然 tensor 不承担计算相关的抽象，其成员方法也就很有限了，主要就是：data, shape, stride, size, rank, depth

   同时 cute tensor 也给了简单的 index & slice 操作，但无法完成 pytorch 中的花式索引或者 bool 索引，使用 `operator()/operator[]` 都可以

2. cute tensor 也可以在 cpu 和 gpu 上进行创建

3. cute tensor 在 gpu 上还要根据存储位置继续分情况：

   1. global memory tensor
   2. shared memory tensor
   3. register tensor, has to be static

non-owning tensor 占据大部分的使用时间，先来看下 non-owning tensor 的创建，只需要两个参数：

1. data pointer
2. Layout

data pointer 可以是 nullptr，此时 tensor 只是一个 layout representation

cpu 不在乎使用什么 pointer，即使用了 gpu pointer (gmem_ptr or smem_ptr) 在底层都会转换成普通的 cpu pointer

## Manipualte Tensor

操作 tensor 的方法同样有限，都是对 mode/axis/shape 进行操作，input x 可以是 tensor/layout/shape

1. `take<begin, end>(x)`

   说是 take，实际上是 slice，取 begin & end

2. `select<modes...,>(x)`

   真正的 take，不过返回的还是一个 tuple-like object

3. `flatten(x)`

   没有办法选取 modes 进行展开，只能一次性全部展开

4. `group<begin, end>(x)`

   对 modes 进行打包

5. `get<idx>(x)`

   可以认为是弱化版本的 select，同时自动解包

   ```cpp
   // x.shape() = (256, 128)
   get<0>(x.shape())		 // 256
   select<0>(x.shape()) // (256,)
   ```

以上操作其实都是完成 pytorch 当中的 view & permute & slice 操作。但对于 squeeze 这样的操作似乎没有特别好的方法？

> From Kimi & Claude Code
>
>   Key Files
>
>   - tensor_impl.hpp - Core tensor operations
>   - layout.hpp - Layout manipulation
>   - layout_composed.hpp - Advanced layout operations
>
>   These operations work by changing the layout (how data is accessed) without moving actual
>   data, which is efficient for GPU computations.

实际上更灵活的操作是直接对 layout 进行操作，这需要使用 `tuple_algorithm.hpp`，其中就有 append, prepend, insert 等等。这也间接表示了 stride & shape 都是 tuple，可以对他们进行操作

> From Kimi
>
>   Yes, you can directly set the layout and add singleton dimensions. Here are the key ways to
>   unsqueeze tensors:
>
>   Direct Layout Setting
>
> ```cpp
>   // Create tensor with new layout containing singletons
>   auto tensor = make_tensor<float>(make_layout(Int<4>{}, Int<8>{}));
> 
>   // Set new layout with singleton dimension added
>   auto unsqueezed = make_tensor(tensor.data(),
>                                 append(tensor.layout(),
>                                        make_layout(1, 0)));  // Add at end
> 	// (4, 1):(8, 0)
> ```

出了对 layout 进行 append 之外，还可以直接对 shape & stride 进行 append，目前 cute 对 python dsl api 做了比较好的文档 [link](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/cute.html#)，可以根据这个文档来倒退 c++ 中的 api 功能

或者数据类型转换

1. Recast

   因为在 w8a8 的过程中，直接 ouput 会是 fp32 精度，应该需要进行转换成为 fp16 精度

   我在 cpu 上进行 `recast<float>(half_tensor)`，会得到错误的结果，其会将两个 half 数据拼接成为一个 float 数据，所以最终这个 tensor 中的数据数量减半了，所以最好不要使用 `recast` 来进行 tensor 数据的类型转换

或者简单的赋值

1. fill
2. clear

剩下的略微复杂的操作，例如 local_partition axpby 放到具体的方法当中进行整理

## Partition Tensor

1. tensor index & slice

   sliced tensor 的数据并不会自动进行 copy，和原 tensor 的数据是共用的。这在 cute 和 torch 当中的行为表现一致

   但是 cute tensor slice 的语法和 torch 有很大差别，例如：对于一个 tensor `shape(3, 4)`，如果使用 `tensor[0]`，对于 torch 来说会返回一个 `shape(4)` 的张量，而对于 cute 来说只会返回第一个元素。显然 cute 把整个 shape 做了 flatten，才会有这样的情况。所以对于 cute tensor 来说，最好对每一个 mode 都进行显式的 slice，才能得到 torch 当中的效果 `tensor[0, _]`

2. tensor reshape & permute

   这些操作其实都是对 layout 的操作，目前我的想法是之间创建一个新的 tensor `make_tensor`，使用 `tensor.data()` 指针以保留数据，然后传入新的 `layout` 即可

## Simple Test Pipeline

在测试 kernel 过程中需要有两个方面

1. kernel 正确性 `TestCorrectness`

   为了验证正确性，首先需要一个 golden 作为检查标准，所以需要实现一个 reference impl.

2. kernel 速度 `TestPerformance`

确定一种写 kernel 代码的简单模板：

1. 写一个 struct config 来使用 cute 当中的 atom，作为元编程的体现

2. 写一个 kernel function，由于使用了 config 简化了模板传入，所以在 kernel 当中需要对 config 当中的各个类进行 unpack

3. 写一个 void run_kernel<config>(input_params) 作为 host api，其中还需要计算 block & grid 数量，一般根据 problem size & config 即可完成计算

   在进行 grid 计算实际上是在算需要多少 blocks，并且为了保证运算结果正确，会对 blocks 数量进行上取整（下取整显然会少处理一些数据，导致结果错误）

在 main 函数当中会需要对测试数据进行构造，此时不太能像 torch 一样方便构造测试数据，需要经历几个步骤：

1. 构造 data pointer cpu & gpu，申请空间
2. 使用 data pointer cpu & gpu 构造 tensor
3. 给 cpu tensor 赋值，进行初始化
4. 将 cpu tensor 空间 memcopy 到 gpu tensor

而这四个步骤在 torch 中通常只有一个 `torch.randn(shape).cuda()`，除了在创建 tensor 有好处之外，在进行精度验证的过程也会更方便。但问题在于，我尝试了使用 torch package，整个程序编译异常缓慢...这对于快速开发来说非常不友好，编译的时间足够我来构建这些模板 pipeline 了

### Tips

在进行 cuda 编程时，想让 vscode intellisense 发挥作用，需要配置好 `c_cpp_properties.json`，尤其是 compiler path & include path

在实现 cpu reference implementation 的时候需要使用 float 格式，因为 cpu 不支持半精度计算，全部都要在 float 上进行计算

在进行 benchmark 的时候必须要进行 warm up，或者 iteration 的次数要 > 10000，显然 warp up 代价小很多

## Practice Problems

### silu_and_mul

silu and multiply, ref with flashinfer

input tensor: (B, N, C)

整个过程是以 float 精度计算的，所以在计算之前需要使用类型转换

当使用 GPU 并行来加速 kernel 时，可以想象的是：整个程序有多个 blocks 在同时运行，这是 block 级别的并行。而在 block 内部存在着 thread 级别的并行，通常来说还需要 thread 进行循环以处理多个数据

```cpp
__global__ void function(){
  // inside of a block
  while task is not done {
    threads keep doing jobs;
  }
}
```

并行化逻辑：

1. 每一个 block 处理一个 token，thread 数量根据 dimension 数量决定，最大取 1024 个线程，每一个线程使用向量化存储，会一次性搬运 128-bit 数据
2. grid 并行化处理全部 token

对比一般的 cuda 编程，由于需要构建 cute tensor，所以我还传入了 shape `(B, N, C)`

首先我需要将每一个 block 所需要的数据进行划分，然后对每一个线程的数据进行划分，最后将每个线程的数据使用 `cute::copy` api 进行直接拷贝

1. 对 block 所需数据进行划分 `local_tile`

   `local_tile` 本质就是 `zipped_divide` + coord chosing

   > From cute doc
   >
   > This `local_tile` is simply shorthand for
   >
   > 1. apply the tiler via [`zipped_divide`](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/02_layout_algebra.html#zipped-tiled-flat-divides)
   >
   > ```cpp
   > // ((BLK_M,BLK_K),(m,k))
   > Tensor gA_mk = zipped_divide(mA, select<0,2>(cta_tiler));
   > ```
   >
   > 1. apply the coord to the second mode, the “Rest” mode, to extract out the correct tiles for this CTA.
   >
   > ```cpp
   > // (BLK_M,BLK_K,k)
   > Tensor gA = gA_mk(make_coord(_,_), select<0,2>(cta_coord));
   > ```

   coord slice 针对的是 rest mode, instead of tile mode

   由于我们的问题太简单了，不需要使用 `local_tile` 直接进行 tensor index 即可

2. 对 thread 所需数据进行划分 `local_partition`。

   `local_partition` 的本质和 `local_tile` 的本质是非常相似的，也是 `zipped_divide` + coord chosing，只是 coord chosing 针对的是 tile mode, instead of rest mode
   
   在本问题当中，我也没有使用 local partition 来获得每一个线程所需要的数据，而是直接使用 reshape 的方式（define a new tensor）直接获得了每一个 thread 对应的数据

在本例当中我也没有考虑边界条件

Tips：

- 在进行 local tile 的时候 tile 请选择最简单的 shape tiler，而不要选择复杂的 tiler。这一般就是对 MN 进行划分，简单的 shape tiler layout 其输入和输出的 mapping 是在同一个 domain，即：输入经过 layout function 过后不变
- 如果要使用复杂的 tiler，请在上一步过后单独使用 compose 完成。如果是在使用 local partition，直接使用 `ThreadLayout` 也可以一步到位
- 似乎不需要对所有的数据都分配寄存器，可以一边算一边存，这样也是很高效的，这表明了：load & store & compute 都是异步的
- 需要加入 `-use_fast_math` 才能获得最优性能，我检查了所有的可能，包括 convert & load & pragma & cute operations & num input params... 最后通过实现 flashinfer 自身的 kernel 在我自己的环境下对比，发现二者其实速度一样。所以怀疑是编译 flag 导致了速度问题
- 发现 kernel 不 launch，运行时间为0，尝试各种编译 flag，都没什么用。最后由 claude code + kimi 大师找出来原来是 kernel 内部的代码写错了。但是由于没有 cuda check 所以这个错误没有报出来，这告诉我们在开发过程中一定要加入 check

### softmax

softmax, ref with leet cuda

### rmsnorm

Rmsnorm, ref with flashinfer or pytorch or vllm

并行化逻辑：

1. 每一个 block 处理一个 token。先计算 sum 统计值，然后再 norm & weight scale
2. grid 并行化处理所有 token

这个逻辑和 silu and mul 其实是一样的。不过相比 silu and mul 有了 reduce 过程，归约的时候涉及到 thread 之间的通信，所以需要 shared memory，更加复杂一点

使用 `__shfl_xor_sync` 先进行 warp 内部的 reduce，可显著加速（未测试过）。由于这个操作的存在，我们在构建 tensor 的时候就会多一个 warp 维度

对线程数量的安排也比较讲究：类似于 silu and mul，尽量安排与数据量对应的线程数，知道达到一个 block 线程数量的上限。

Tips

似乎经过了  `__shfl_xor_sync` 过后，每一个线程所得到的 reduction sum 是一样的

相比于 if 条件 `condition ? a : b` 似乎会是更快的选择

sync 通常在两种地方使用：

1. 使用 shared memory 时
2. 对部分线程进行操作时

如何进行循环 (round and stride)，这很重要。对于非对齐的 thread，循环是否会进行等待呢？

### unaligned cases

Predication

make identity tensor 会构建出一个比较特殊的 tensor，tensor 中的每一个元素不是一个 value，而是对应的 coordinate [make identity layout doc](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_api/cute.html#cutlass.cute.make_identity_layout)

```python
# Create a simple 1D coord tensor
tensor = make_identity_tensor(6)  # [0,1,2,3,4,5]

# Create a 2D coord tensor
tensor = make_identity_tensor((3,2))  # [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]

# Create hierarchical coord tensor
tensor = make_identity_tensor(((2,1),3))
# [((0,0),0),((1,0),0),((0,0),1),((1,0),1),((0,0),2),((1,0),2)]
```

在 cpp 当中每一个元素其实是一个 nested tuple (a shape)，tuple 有两个成员 `first_ & rest_`，通过递归的方式可以访问其中的元素

这个 tensor 经常用作 predication，从而对边界条件进行判断。predication 

logical divide 会比其他 divide 限制更少一些。我尝试了以下代码只有 logical divide 能够通过

```cpp
auto x = make_layout(make_shape(128, 128), LayoutRight{});
auto tiler = make_tiler(_, make_shape(8));
auto out = logical_divide(x, tiler); // (128, (8, 16)) : (128, (1, 8))
```

另外 logical divide 还会进行自动的 padding 然后再进行 divide

```cpp
auto x = make_layout(make_shape(1000, 128), LayoutRight{});
auto tiler = make_tiler(make_shape(128));
auto out = logical_divide(x, tiler); // ((128, 8), 128) : ((8, 1024), 1)
```

利用这个性质加上 predication 就可以很好地处理 unaligned situations [predication doc](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0y_predication.html)

predication 的本质就是用 bool tensor 来控制线程的运行

### utils

cuda check

build input & output on host & device

reference code & verify

## Simple Copy

又出现一篇新的 cute 教程，师承 reed，主要讲了下 copy [写给大家看的 CuTe 教程：tiled copy](https://zhuanlan.zhihu.com/p/1930389542784964333)

## Simple Gemm

## W8A8 Gemm

compare with cutlass sota results

## DataType Convertion

1. dequant (4-bit to 16/32-bit)
2. fp32 to fp16 bit
3. from int4/8 to int32? so we can see the actual numbers

## Questions

1. Constexpr 的作用？这似乎是在模板元编程中需要的

2. half2

   > From DeepSeek
   >
   > The `half2` type is a CUDA-specific vector data type that combines two `half` (16-bit floating-point) values into a single 32-bit register. Its usage in this kernel serves two key optimization purposes:
   >
   > 1. **Memory Access Efficiency**
   >
   > - **Problem**: Individual `half` elements (16 bits) are too small for efficient global memory access. GPUs achieve peak bandwidth when accessing 128-bit aligned chunks.
   > - **Solution**:
   >   - `half2` allows processing two `half` elements at once, matching the 32-bit register size naturally.
   >
   > 2. **Compute Throughput Optimization**
   >
   > - **Problem**: Modern GPUs (Pascal+) have special hardware for packed math operations.
   > - **Solution**:
   >   - Operations on `half2` leverage `HFMA2` instructions (Fused Multiply-Add for two `half` values in one cycle).
   >   - Without `half2`, each element would require separate instructions, cutting throughput in half.

   实际测试下来使用与否差别不大，即使在 10240 x 10240 的规模之下也没有显著差距。但这仍然告诉我们，需要用最快的硬件指令

3. 如何进行数据类型转换？

   目前还没有尝试过实际应用，不过似乎可以使用 `cutlass/numeric_conversion.h` 当中的 converter

   ```cpp
   // Convert using NumericConverter
   cutlass::NumericConverter<half_t, float> converter;
   for (int i = 0; i < size(tensor_float); ++i) {
       tensor_half(i) = converter(tensor_float(i));
   }
   ```

   对于 half 转 float 可以直接用强制类型转换 `(float)`，我也可以使用 cuda inline function `__half2float` 完成类型转换

4. sizeof 是测量什么？

   以 byte 为单位，返回一个类型所占据的字节数，`sizeof(float)=4`

5. 如何处理尾部未对齐数据？

   尤其是在矩阵乘法当中。在简单的情况下，我已知的解法是：使用边界循环，也可以使用填充。我还没见过矩阵乘法的边界处理循环代码，可能需要从 deepgemm 中寻找答案

   在 cute doc 当中也提供了一个解决方案，之后可以尝试下 [predication](https://docs.nvidia.com/cutlass/media/docs/cpp/cute/0y_predication.html)

6. 为什么在 warp reduce 过程中使用 `shfl_xor_sync` 而不是使用 `shfl_up_sync`

   因为它是最快的，而且不需要考虑线程 id 越界问题