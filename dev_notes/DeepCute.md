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

  如果我们 align 出现了错误是否会出现 memory issue？待定，目前问题没有定位到

  另外 struct alignas 和 shared memory `__align__` 有什么区别？前者是 c++ 语法，后者是 nvcc 特定语法，单位为 Byte，前者通用性更强。他们的功能其实都是一样的：将内存地址进行对齐

  > From Kimi
  >
  > `alignas` 只决定“**该对象本身的起始地址**”必须是对齐值的倍数，**不会**让“前一个对象”为了它而额外填充到 128 B。真正发生的填充只有两处：
  >
  > 1. 结构体**开头**：保证整个 `SharedStorage` 的首地址 ≡ 0 (mod 128)。
  > 2. 结构体**末尾**：如果当前总大小不是 128 的倍数，就补到 128 的倍数，以便放进数组或做下一次 128 B 对齐分配。

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
  
  以此为基础，我们可以很轻松地理解 tma out of bound 数据搬运是如何处理的：因为我们记录了原 tensor 的 shape，在利用 tma 进行搬运的过程中，可以很自然计算出每一个 bounding box 中 out of bound 的部分。tma 应该有好几种处理 oob 的模式，我猜测默认模式就是填充0
  
- 当我是用 warp specialization 的时候，并且producer wg 为 wg0，我在进行 `tiled_mma.get_slice(x)` 是否应该也跳过 wg0 的 thread number?

  不应该跳过。并且对于 wgmma 来说，每一个 thread 经过 `get_slice` 得到的是 matrix descriptor 而不是真正的数据。并且同一个 warp group 的 matrix descriptor 是相同的，所以同一个 warp group `get_slice(x)` 中的 `x` 可以是一样的

  ```cpp
  int warp_group_id_in_consumer = __shfl_sync(0xffffffff, threadIdx.x / 128 - 1, 0);
  // auto thr_mma = tiled_mma.get_slice(threadIdx.x - 128); // this also works
  auto thr_mma = tiled_mma.get_slice(warp_group_id_in_consumer * 128);
  ```

  其中 `threadIdx.x - 128` 是因为在我的代码当中 consumer 是从 thread 128 开始而不是从 thread 0 开始

- Epilogue 当中，make tiled copy 实际上的 MN shape 应该是 `(128, 128)`，但是 smem C 的大小没有这么大，只有 `(128, 64)`，这为什么不会出错呢？不过在 deepgemm 当中就是直接用一个 cta mn tile 的 smem 来做 epilogue 

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
  
- proper qualified

  我直接用模板当中的 int，在编译过程中报错

  ```cpp
  template <typename CTATile, bool MultiCast, int Stages>
  struct GemmFp16SM90 {
          using SmemLayoutA = decltype(tile_to_shape(SmemABAtom{}, make_shape(size<0>(CTATile{}), size<2>(CTATile{}), Int<Stages>))); 
  ```

- 可以使用 `static_cast<Dtype*>(nullptr)` 来构建一个空指针，从而构建一个空的 tensor。但是这里不能使用 `reinterpret_cast` 这是为什么？

  可以按照如下的方式来理解二者的使用方法：

  1. `reinterpret_cast` 用于除了 nullptr 意外的其他任何指针之间的转换

     ```cpp
     reinterpret_cast<uint64_t*>(&full_barrier);
     static_cast<half_t *>(nullptr);
     ```

  2. 其他的类型转换都使用 `static_cast`，虽然 `static_cast` 也可以用于转换指针，但基本只用于子类和父类之间的指针转换

  以上两个方法很难从 DeepSeek or Kimi 的回答中找到，二者都是给了一大段非常抽象的回复，很难理解。以上结论对我来说很清晰，其主要来自于一些传统资料(B站的一些讲解视频)和一些我遇到的情景总结得到

- retile 的功能不是我理解当中的功能，其功能比我想象中的 retile 更差一点

  对于 tiled mma size of v 大，而 tiled copy size of v 小的情况，需要单独重新讨论。而且正是这个问题，需要我使用 repeat 对数据进行重新展开。另外如何构建自己的 smallest copy atom 也是一个有趣的话题

- `__grid_constant__` 必须要搭配 `const` 修饰符

- dangling ptr 问题导致了 launch config 损坏，这个问题对我来说非常隐秘，但是被 Kimi 发现了，太强了

  这让我重新审视了 static 关键字，有点类似于 python 的 classmethod，或者 `self.xxx` 把变量编程一个成员

- 对于 get launch config 函数来说，我能理解使用 static，但为什么要使用 inline？

- 通过设置 cudaFuncAttributeMaxDynamicSharedMemorySize 似乎解决了 Got bad cuda status: invalid argument at /root/Projects/DeepCute/deepcute/sm90/fp16_gemm/gemm_ws.cu line: 47 问题，现在遇到了 illegal memory access 的问题，感觉一个比一个棘手

  目前通过注释代码，发现了问题出现在 prefetch tma 的代码上。很有可能是我在构建 tma 的时候出现问题了。最后发现是因为参数在传递的时候没有加 const& 导致了值传，可是 CUDA 要求 tma descriptor 一定要是 grid constant，值传破坏了这一个要求，所以报错
  
- 现在遇到 gemm 卡死问题，应该是我的 producer consumer 同步出现了问题

  尝试了一些小的更改，似乎都没有影响：

  - both use ClusterTransactionBarrier
  - barrier ptr use like mine
  - use simple `make_tma_copy` creation
  - checked my own CTATile
  - checked tmaloadbytes

  还有一些尝试没有

  - 使用 struct 构建 input args & params

  我重新检查了 producer，我认为 producer 没有问题。最后我还是认为问题发生在 tma 的构建上，我又想起来我经常会使用复制粘贴，很有可能 tma atom 直接用错了。结果一查，果然是 conditional_t 用错了！

  ```cpp
  // WRONG!!!
  using g2s_copy_atom_b = std::conditional_t<MultiCast, SM90_TMA_LOAD, SM90_TMA_LOAD_MULTICAST>;
  ```

  终于体会到智子的感觉了

- [discussion-ScaleOut::Zero]([`GMMA::ScaleOut::Zero` Not Equivalent to `clear()` ? · NVIDIA/cutlass · Discussion #2284](https://github.com/NVIDIA/cutlass/discussions/2284#discussioncomment-13063125)) 我遇到了和这个 discussion 一模一样的问题，确实解释得非常正确，`cute::gemm` 封装了 wgmma，其实里面运行了多次 wgmma，而在我看来只运行了一次，这就是认识的错误点。为了代码的易读性，我选择了我原始的（DeepGemm 也是这样实现的）clear + accumulate 的方式

  如何保障 `clear(t_rC)` 一定是在 wgmma 之前完成的？这里有两个命令

  ```cpp
  warpgroup_fence_operand(t_rC);
  warpgroup_arrive();
  gemm(...);
  ```

  我认为真正其作用的是第二个命令 `warpgroup_arrive`，在 [CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on NVIDIA® Hopper™ GPUs – Colfax Research](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) 和 [PTX-doc](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence) 中均提到这个 PTX 会作为一个 fence，确保 register 的使用会在 wgmma 之前完成。所以我们可以安全地删除掉 `warpgroup_fence_operand` 操作
  
- 现在在尝试对齐 deepgemm 的 performance，目前我还差 5% 左右

  我现在把目光放在了 scheduler 上，我发现 deepgemm 的 scheduler 功能很全面，不过在 normal gemm 上和我的 scheduler 没有本质区别，但是代码更加简洁易懂

  利用 `export DG_PRINT_CONFIGS=1` 可以看到 deepgemm 的配置设置

  发现了几点不一致

  1. **wgmma shape 不一致：important**
  2. stage 数量不一致：Not the point
  3. thread block swizzle 数量不一致，Not the point
  4. **epilogue 实现不同，important**
  5. grid size 不同：Not the point

  我现在发现去除 epilogue 部分， 我们的 mainloop 部分都要比 deepgemm 要差一点

  ​					Med			Min

  DeepGemm: 40128.0     39360

  AwesomeC  52544.0     50560

  ​					44000.0     42432, CTATile (128, 256, 64)

  ​					43744.0     42272	Swizzle = 8

  DeepCute:   53760.0     52256

  ​					39552.0     37568, CTATile (128, 256, 64)

  升级了 epilogue 的同步点过后，就和 Deepgemm 打平了

  ​					Med			Min

  DeepGemm: 267233.0    266145

  DeepCute:    268994.0    266946

  ​					 264928.0    264513, removing the first sync, but this is unsafe

  对比我和 deepgemm epilogue 的实现，二者时间的核心差异来自于 sync 的使用时机

  我是在最后进行 wait，然后进行 sync

  ```cpp
  // tma copy
  if (tma_predicate) {
  for (int j = 0; j < pipe2; j++) {
      copy(tma_c, t_s2g_sC_group(_, j), t_s2g_gC_group(_, i + j, tile_info.m_idx, tile_info.n_idx));
      tma_store_arrive(); //  we can also use tma_desc_commit_group
  }
  tma_store_wait<0>();
  }
  cutlass::arch::NamedBarrier(128*2).sync();
  ```

  但是问题在于，仅仅这样 sync 会导致出错。而如果我们在 epilogue 开始前再增加一个 sync 就不会出错。这可能是因为：增加的 sync 让 copy 更同步了，不会出现 tma thread 提前发起，使得 copy 的数据不正确（是不是可以认为 copy 并不是 async 操作，发生在 generic proxy 当中）

  另外 deepgemm 把 `tma_store_wait` 提前到 epilogue 之前，这样可以 overlap epilogue 和 wgmma，不过代价就是需要一个同步。但是如此做了过后并没有我想象中的大幅提升~3%，我有一个猜想是：s2g 的操作非常快，这么做的 overlap 并不能节省多少时间，同时使用了 sync 操作增加了同步时间。不过这的确是我和 deepgemm 操作差异的核心原因

- DeepGemm 当中的 1d1d & 1d2d 分别代表什么？

  应该是代表了 scaling factors 的维度：

  - 1d1d，A & B 的 scaling factor 都是 1d
  - 1d2d，A 的 scaling factor 是 1d，B 的 scaling factor 是 2d，应该是对应了 blockwise scaling

  最初的 DeepGemm 实现的是 1d2d，1d1d 更多是为 Blackwell 架构实现的

- 模板编程的顺序问题

  有的模板中，其类型是可以通过输入参数推断的，而有的必须要显式地进行输入

  ```cpp
  template <class AClass, class BClass, class CClass>
  void func(AClass a, CClass c){
      BClass b;
  }
  ```

  应该如何构建？

## sm90 fp8 DeepGemm & Hpc-ops

- 什么是 grouped gemm？

  以 m grouped gemm 为例，其在 moe 的 inference 阶段被使用（在 backward path 中会使用 k grouped gemm），下面的代码展示了 group gemm 所需要的 tensor 以及对应形状

  ```python
  def generate_m_grouped_contiguous(num_groups: int, expected_m_per_group: int, n: int, k: int,
                                    major_a: MajorTypeAB, major_b: MajorTypeAB,
                                    use_ue8m0: bool = False, use_bf16: bool = False,
                                    use_psum_layout: bool = False,
                                    quant_config: Optional[QuantConfig] = None):
      actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
      aligned_ms = [align(actual_m, 128) for actual_m in actual_ms]
      m = sum(aligned_ms)
  
      # a is activation, b is weight
      a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
      b = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
      grouped_layout = torch.empty(m, device='cuda', dtype=torch.int32)
      d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
      ref_d = torch.randn((m, n), device='cuda', dtype=torch.bfloat16)
  
      start = 0
      for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
          actual_end = start + actual_m
          aligned_end = start + aligned_m
          if use_psum_layout:
              grouped_layout[i] = actual_end
          else:
              grouped_layout[start: actual_end] = i
              grouped_layout[actual_end: aligned_end] = -1
          a[actual_end: aligned_end] = 0
          ref_d[start: aligned_end] = a[start: aligned_end] @ b[i].t()
          start = aligned_end
  
      if use_bf16:
          b = b if major_b.is_k_major() else b.mT.contiguous().mT
          return m, a, b, grouped_layout, d, ref_d
  
      # quantize input
      quant_config = QuantConfig() if quant_config is None else quant_config
      a = cast_fp8_fp4_with_major(a, major_a, quant_config.gran_k_a, quant_config.is_fp4_a, use_ue8m0)
      b = grouped_cast_fp8_fp4_with_major(b, major_b, quant_config.gran_k_b, quant_config.is_fp4_b, use_ue8m0, use_block_cast_for_fp8=True)    
  
      return m, a, b, grouped_layout, d, ref_d
  ```

  可以用图示来表示 grouped gemm，清晰地看见哪两个矩阵进行相乘，最后仍然得到输出矩阵 `(M, N)`，其稀疏性也能很轻松地理解

  <img src="C:\Data\Projects\notes\dev_notes\DeepCute\image-20260203163405033.png" alt="image-20260203163405033" style="zoom:80%;" />

  dense gemm 其实可以是 grouped gemm 的特殊情况（num groups 为 1），所以二者可以统一实现。我们在进行 warp persistent 编程的时候，就需要额外注意当前 m tile 到底对应了哪一个 group 的 n tile
  
- 熙哥提到了 DeepGemm 节省了寄存器，这是什么原理？

- 目前 hpc-ops 是 H20 上的 sota 实现，并且基于 cute 实现，我需要完全掌握其实现原理，这应该是我 deepcute 的最终模板。我现在看其中的代码还是感到非常吃力，不过我相信能够解决

  hpc-ops 把 A 和 B 的乘法进行了交换，让 B 去乘以 A，这样构建了一个转置的效果，所以我在看代码的时候发现 `M` 和 `N` 有时候是交换的，最终通过 retile 的方式，转换成为 mn layout，可能这就是为什么其 decode 速度会这么快的原因，这里需要更进一步的分析

  blockwise fp8 gemm 的计算过程 & kernel 计算逻辑：
  
  1. 对于一次 CTATile wgmma，先把对应的 A & B & Ascale & Bscale 都 load 进来，其中 A scale 为 TileM 个，而 B scale 为 1 个。显然这里的 B scale 是针对权重共享了 N dim 上的 scale，不然应该和 A scale 类似，有 TileN 个
  2. 对于 loading B scale，tma copy box 设置为 (1, 4)，这样每一次会 copy 4 个 CTATile 的 scale，这可能是 tma copy 的限制所导致的。在进行读取的时候要注意 index
  
  由于其交换了 m 和 n 的乘法顺序，看代码还是不太习惯，各种变化（例如：swizzle, mma 选择），而且代码中的标记也有一些迷惑（例如 retile mn），不过最终我还是明白了 blockwise 的原理。视角需要切换到一个 CTATile 内部会更加好解释

- 似乎这里使用了多个 tma，每一个 group 使用一个 tma

  alignment 对于 GPU 来说非常重要。每一个 expert 接收到的 token 很大概率上不是 128 对齐的，deepgemm 通过 masked groupgemm 来完成。但是既然已经有了 tma 这样的神器，对数据的 oob 处理应该不在话下。在 hpc-ops 的解决方案中，直接使用了多个 tma 来对每个 expert 的 MK 矩阵进行搬运，不过需要再写一个 `update_grouped_tma` 计算每一个 tma 的首地址在哪里

  看来我对于 tma 的理解还需要增加，tma 到底是存储在哪里，如何对其进行更新需要清楚

- cub 的 block scan & block reduce 似乎是不错的小工具

- hpc 没有使用任何的 multicast 但是性能依旧 SOTA，我重新测试了之前我写的 multicast 代码，把 multicast 去掉过后性能仅有 0.5us 级别的下降，下降比例约为 0.15%

  如此微小的优化在后期可能就不会优先考虑了
