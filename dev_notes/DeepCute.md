# DeepCute

notes when building kernels

## sm80 gemm

- `SM80_CP_ASYNC_CACHEGLOBAL_ZFILL` 和 `SM80_CP_ASYNC_CACHEGLOBAL` 的区别

  zfill 代表 zero fill，是更常见的用法，如果访问越界元素会自动填充为0，从 DeepSeek 和 Kimi 的回答来看，速度上甚至还更快，所以除非有特别的理由，请优先使用 zfill 版本

- When to use syncthreads?

- What kinds of checks need to be done?

  目前只做了只做了 cta tile 的检查

- `size<0>(CTATile)` 还是 `shape<0>(CTATile)`

  size 得到 int 大小，shape 不是我要的

- 为什么要使用 `void* __restrict__` 作为数据传入类型

  CUDA 驱动层只传 `void*`，写成具体指针类型不会带来任何好处，通过 `void*` 指针配合模板特质类 `GemmTraits`，可以在**编译时**决定实际的数据类型。如果使用具体类型指针，需要为每种数据类型组合都定义不同的 kernel。
  
- 为什么要在 `get_launch_config` 中使用 static

  为了在外面使用 `::` 解析符号来调用，否则这只能通过生成 object 来调用

- lambda 函数可以传入 auto 参数，这是和普通函数的一大区别

- cutlass host tensor 应该如何使用？

- epilouge 中的 pipe 还是有一定作用的，增加 pipe 也许增加了带宽的利用率

- 测试了我自己设置的 `swizzle<1, 3, 3>`，会产生 bank conflict，这是因为在 g2s 阶段写入是按照 32x32 进行写入的，此时会产生 2-way bank conflict，所以设计 smem 的时候需要考虑最大宽度读取情况

- permutation mnk 是以 `make_tile` 定义，而不是使用 `make_layout`

- 似乎现在不需要传入 mma traits & mma atom 给 make tiled mma 了，直接传 mma op 就行，具体特化 traits 会自动生成，我们可以去 `cute/atom` 当中找到对应的 traits，因为也不太好利用 vscode 自动跳转跳过去

## sm90 gemm

- 如何实现 [zhihu]() 中所提到的，处理 scheduler 与 cluster size 不对齐的情况

- 在 sm90 中出现了新的 level: cluster，是否需要重新更新我的 tile centric cuda programming 理论？我想我应该先实现 cooperative gemm，先不去考虑 pingpong，毕竟 deepgemm 本身也没有实现 pingpong

- 在 sm90 当中 cta tile mnk 的设置似乎可以相当灵活，如何设置最优的 mnk 是一个需要弄清楚的问题，在 deepgemm `common.hpp` 当中是有答案的，不过我觉得使用固定的 cta tile mnk 会适合我目前的情况，先不做过多的扩展

- 实际上在定义 atom 的时候，tiled mma & tiled copy 基本就已经被完全定义好了，全部都是高性能的不可分割的统一指令，不需要像 sm80 的 universal copy 一样还需要定义 thr & val layout

  为什么跳转不到正确的 traits 定义上去

  ```cpp
  using mma_op = GMMA::MMA_64x128x16_F16F16F16_SS<ABMajor, ABMajor>;
  using mma_traits = MMA_Traits<mma_op>;
  ```
  
- epilogue 中 shared memory 没有这么大，应该小于 CTATile，应该如何理解？

  似乎在绝大部分场景中都无需去定义 ctatile，既然如此，为了简洁性且不发生歧义，我直接使用 `make_tma_copy` 并且放弃 CTATile 参数

- 在 DeepGemm 当中使用了 align 1024 Btypes，这是为什么？

  因为我们使用了 swizzle 128 layout，而这个 layout 就是以 8x1024 bit 为单位的

- 在 Awesome cute 当中使用了 struct 作为 shared memory 的分配工具，这其实还挺方便的，不需要再手动去计算 address 了。并且由于 persistant warp scheduler 的存在，C tensor smem 也无法复用之前的 shared memory，使得 struct 显得更为合适

- arrival count 是否可以进行调整？为什么 cluster size > 1 过后就需要多个 thread 同时 arrive

  这可能需要我自己验证，初步猜想是，会影响效率，不会影响正确结果。因为我们希望一个 cluster 内部的两个 cta 是同步的，这样他们可以高效地利用共享的 matrix 数据搬运
  
- 下面的注释解释了，为什么需要使用 shfl sync，其实不使用 shfl sync 获得的结果也是一样的

  ```cpp
  // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
  const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
  ```

- 因为是用了 warp persistant scheduler，所以不需要在 local tile 的时候根据 block index 获得对应的 tile，而是根据 scheduler 的结果获得数据

- tma tensor 是以 coordinate 作为基础的，而不是以普通的 tensor  = ptr + layout

  参考 [NVIDIA TMA 全面分析](https://zhuanlan.zhihu.com/p/1945136522455122713)，tma 在搬运 tensor 的时候是根据首坐标 + box dim 来确定搬运数据的范围。即以 bounding box 为单位来搬运数据，bounding box 通常就是我们定义的 smem 大小。开发者只需定义“搬什么”，而无需关心“怎么搬”，避免与复杂的物理地址接触
  
- 当我是用 warp specialization 的时候，并且producer wg 为 wg0，我在进行 `tiled_mma.get_slice(x)` 是否应该也跳过 wg0 的 thread number?

  不应该跳过

- Epilogue 当中，make tiled copy 实际上的 MN shape 应该是 `(128, 128)`，但是 smem C 的大小没有这么大，只有 `(128, 64)`，这为什么不会出错呢？

  原因在于其使用的 tiled copy 是 `make_tiled_copy_C_atom`

  ```cpp
  // returns the smallest tiled copy that can retile LayoutC_TV
  // for use with pipelined epilogues with subtiled stores
  make_tiled_copy_C_atom(Copy_Atom<CArgs...> const& copy_atom,
                         TiledMMA<MArgs...>  const& mma)
  ```

  这也和我最初的猜想一样，需要构造一个最小的 TilerMN，其既能够被 copy atom 整除，又能够整除 mma C TilerMN。我的思路和这个函数的最终目的其实是一样的，但是结果确有差别，所以需要在之后验证我的 layout tv 和 tiler mn 是否正确

  对于 mma C TV layout  而言其构建映射 `(T, V) -> (M, N)`，为了方便讨论我这里用具体的值来替代 `(256, 64) -> (128, 128)`，其是一个 128x128 的输出矩阵

  对于 copy atom 而言，其构建的映射为 `(t, v) -> (m, n)`，其中 `t` 应当和 mma C TV layout 中的 `T` 是一样的，其区别在于 `v=8`，和 `V=64` 相差8倍。可以暂时将其 layout 映射写作 `(256, 8) -> (m, n)`。如果直觉强烈的同学可能已经猜到，最终可以设置 `(m, n) = (128, 16)`，即把 N 维度切分成了8份，这正是因为 `V / v = 8`

  不过为了更准确的计算出最终的 tv layout 和 tiler mn，还是需要把准确的推导过程拿出来。我们完全可以利用 `(t, v)` zipped divide `(T, V)`，然后取 `rest_V` 中的 0-idx，即可获得一个完整的 layout tv，不过更简单的做法是直接做 compose，这也算是 compose 用于 reshape 的妙用

  ```cpp
  layout_tv = compose(mma_C_TV, make_layout(make_shape(size<0>(mma_C_TV), // size<0>(mma_C_TV) = 256
                                                       copy_v)));		    // copy_v = 8
  // ((_4,_8,_8),(_2,_2,_2)):((_256,_1,_16),(_128,_8,_1024))
  ```

  如果把 tv -> mn 的图像画出来可以发现 TilerMN 就是 `(128, 16)`，但是如何计算得到呢？这里利用的妙计是 filter，我们分别将 M & N mode 的 stride 设置为0，然后用这样的 MN layout compose with layout tv，最后将 stride = 0 和 shape = 1 的 mode 过滤掉，最后就能够得到 `(t, v) -> M` 和 `(t, v) -> N` 的映射，这个映射应当是 compact 的，所以直接可得 `M = size<0>(tiler), N = size<1>(tiler)`

  ```cpp
  // Tiler -- Find the active elements in the MMA tensor and generate a tiler to extract them
  // Convert to the awkward by-mode tiler to preserve the modes of the tiled MMA
  auto mma_tiler = make_shape(tile_size<0>(mma),tile_size<1>(mma));
  auto mma_zeros = repeat_like(mma_tiler, Int<0>{});
  
  auto tiler = transform(make_seq<rank(mma_tiler)>{}, [&](auto i) {
  return filter(composition(make_layout(mma_tiler, replace<i>(mma_zeros, Int<1>{})), layout_TV));
  });
  // tiler((_8,_8,_2):(_1,_16,_8),(_4,_2,_2):(_2,_1,_8))
  // M = size(tiler(0)) = 128
  // N = size(tiler(1)) = 16
  ```
