# CUDA Programming 10.2

这是在 Blackwell 和 Hopper 之间的一个笔记。其目的在于学习 [Tencent/hpc-ops: High Performance LLM Inference Operator Library](https://github.com/Tencent/hpc-ops) 对于 fp8 kernel 的高效实现

在之前我已经对其中的内容有了大致的了解。不过由于一些事情搁置了，现在又有时间进行完整的整理。本次整理我希望借助 AI agent like claude code 的力量，完成更清晰的理解

先拟出一个大纲，列出自己想要学习的内容：

1. fp8 gemm vs fp16 gemm，他们之间除了精度外，还是否有其他的重要区别
2. group gemm 中对 tma 的应用技巧
3. 如何理解在 gemm 当中把 AB 矩阵反过来的操作
4. 如何设计简单的 scheduler 来统一应对 group gemm

从 hpc-ops 的代码量上来看，kernel 的核心代码也非常精简，~300 lines of code。不过如果我们直接暴力从头去解读其中的代码，或许并不是一个好的事情。我们还是从 top to bottom 的角度，从大的算法图景开始，理解这个 group gemm algorithm 到底做了什么事情，然后根据这个 overall algorithm 进行展开，再去看代码中的细节实现，这样能够理解更轻松

## Group GEMM 详解

### What's A Group Gemm?

要理解 Group GEMM，我们先从普通的 GEMM（通用矩阵乘法）开始。

**普通 GEMM** 计算的是：
```
Y = X * W^T
```
其中：
- X 形状为 [M, K]（输入激活）
- W 形状为 [N, K]（权重）
- Y 形状为 [M, N]（输出）

**Group GEMM** 是 GEMM 的扩展，它在一次操作中执行多个独立的矩阵乘法。可以把它想象成"批量处理"不同组的矩阵乘法。

在 Group GEMM 中：
- 我们有 `num_group` 个独立的权重矩阵 `W_0, W_1, ..., W_{num_group-1}`
- 输入 X 被分割成 `num_group` 个连续的组 `X_0, X_1, ..., X_{num_group-1}`
- 每个 `X_i` 与对应的 `W_i` 相乘
- 所有结果拼接起来得到最终输出 `Y`

使用简介的代码语言来表示

```python
# X (M, K)
# W (num_group, N, K)
# Y (M, N)
# M is the length of all tokens from different groups
Y[start_i : end_i, :] = X[start_i : end_i, :] * W[i, :, :]^T

# A torch version
def naive_group_gemm_pertensor_fp8(x, w, seqlens, cu_seqlens, scale):
    # 步骤 1: 获取张量形状
    m, k = x.shape           # m = total_seq, k = hidden_size
    num_group, n, _ = w.shape  # n = output_dim

    # 步骤 2: 初始化输出张量
    y = torch.zeros((m, n), dtype=torch.bfloat16, device=x.device)

    # 步骤 3: 遍历每个组，执行独立的矩阵乘法
    start_idx = 0
    for i in range(num_group):
        # 获取当前组的起始和结束位置
        start_idx = int(cu_seqlens[i].item())
        end_idx = int(start_idx + seqlens[i].item())

        # 如果该组没有数据，跳过
        if seqlens[i].item() == 0:
            continue

        # 提取当前组的输入和权重
        x_group = x[start_idx:end_idx]  # 形状: [seqlens[i], k]
        w_group = w[i]                   # 形状: [n, k]

        # 执行矩阵乘法（使用缩放的 FP8 运算）
        y_group = torch._scaled_mm(
            x_group, w_group.t(),        # w_group.t() 形状: [k, n]
            scale_a=scale, scale_b=scale,
            bias=None, out_dtype=torch.bfloat16
        )

        # 将结果写入输出的对应位置
        y[start_idx:end_idx] = y_group

    return y
```

### Why Group Gemm?

Group GEMM 的主要应用场景是**混合专家模型（Mixture of Experts, MoE）**的推理。如果不使用 Group GEMM，我们需要把输入 X 按照专家分组拆开，对每个专家分别调用一次 GEMM，最后再把结果拼回去

这样做的问题：
- **多次 kernel launch 开销**：每个专家都需要一次独立的 kernel launch，launch 本身有固定开销
- **分散的内存访问**：每个小 kernel 独立访问内存，难以形成高效的流水线访问模式。例如：前一个 group 的最后一部分数据在进行计算时，就可以开始预取下一个 group 的数据了，但独立的 kernel launch 无法做到这一点。另外，GroupGemm 最后也不需要再对各个 group 的计算结果进行拼接，减少数据读写

这么看来 GroupGemm 的优势就明显了：一次 kernel launch + 高效的内存流水线访问模式

## GroupGemm in hpc-ops Overview

本次学习的 kernel 代码在 `src/group_gemm/kernels.cuh`，这部分代码包含了对 group gemm pertensor & blockwise 进行了优化实现。我们先从简单的 pertensor group gemm 开始，当我们吃透了这部分代码过后，再来看下 blockwise 的实现有什么细节上的更改

接下来是针对 pertensor group gemm 算法的精髓总结：

1. **Warp Specialization**: 384 线程 = 256 线程 (数学计算 warpgroup) + 128 线程 (数据加载 warpgroup)
2. **Producer-Consumer Model**: 使用 mbarrier 在阶段之间进行同步和流水线控制
3. **TMA for Groups**: 为每个 group 预配置独立的 TMA descriptor，实现高效数据搬运（而不是所有的 group 使用一个 tma descriptor）
4. **Adaptive Tile Sizing (自适应 Tile 大小)**: 根据每组的平均序列长度选择 kTileM (16/32/48/64)
5. **Dual Scheduling Modes (双调度模式)**: Horizontal 模式（小矩阵用线性扫描）vs Vertical 模式（大矩阵用二分查找）

前两个算法算是老生常谈了，是 GEMM 算法中的基本。后面三个优化就是 hpc-ops 中的核心。其中我想提前提下第 4 点，对于 Tile Size Configuration，kernel 会根据 `num_seq_per_group_avg`（每组平均序列长度）选择不同的 tile 大小：

| avg_seqlen | kTileM | kTileN | kTileK | kStage |
|------------|--------|--------|--------|--------|
| ≤16        | 16     | 128    | 128    | 8      |
| ≤32        | 32     | 128    | 128    | 8      |
| ≤48        | 48     | 128    | 128    | 8      |
| >48        | 64     | 128    | 128    | 8      |

这样可以确保在不同的序列长度分布下都有良好的硬件利用率。这里的 Tile 看上去很奇怪，一般来说 Tensor Core 都是固定矩阵乘当中的 M 维度，i.e. `kTileM = 128`，而在这里确是固定了 `kTileN = 128`，这是因为 hpc-ops 对于 mma 进行了转置处理，这是一个非常巧妙的用法。这样的转置一下子就让 M 维度的粒度变得非常细，对于小 M 的场景非常友好

### Pseudocode with Producer-Consumer Structure

下面是一个简洁的伪代码，帮助我们抓住整体的算法流程。其中隐藏了对 schduler & tma & mbarrier & mma 等模块的大量细节，但不妨碍我们理解这个 producer-consumer gemm 的核心思想

```cpp
__global__ void group_gemm_pertensor_fp8_kernel(...) {

  // ===== PRELOGUE =====
  int idx = threadIdx.x;
  bool is_producer = (idx >= 256);  // 128 threads for load
  bool is_consumer = (idx < 256);    // 256 threads for math

  extern __shared__ uint8_t shm_data[];
  auto* shm_a = (Tin*)shm_data;          // [kTileM, kTileK, kStage]
  auto* shm_b = shm_a + ...;              // [kTileN, kTileK, kStage]
  int* shm_tiles = (int*)(shm_data + ...); // For scheduler

  // Initialize mbarriers (producer-consumer sync)
  if (is_leader) {
    for (int s = 0; s < kStage; s++) {
      initialize_barrier(readable[s], 1);  // Consumer waits on this
      initialize_barrier(writable[s], 1);  // Producer waits on this
    }
  }
  // Load scheduler metadata to shared memory
  for (int i = idx; i < num_group; i += 384)
    shm_tiles[i] = tiles_ptr[i];
  __syncthreads();

  // ===== MAINLOOP: Producer-Consumer Pipeline =====
  int phase = 0;

  if (is_producer && is_leader_in_load) {
    // PRODUCER: Load data via TMA
    int s_write = 0;  // Current stage to write
    while (true) {
      // SCHEDULER: Get next tile (igroup, itile_m, itile_n)
      if (!get_next_tile(shm_tiles, iblock, ...)) break;

      for (int k = 0; k < ntile_k; k++) {
        // MBARRIER: Wait for consumer to release stage
        wait_barrier(writable[s_write], phase);

        // TMA load X and W tiles to shared memory
        tma_copy(shm_a[_, _, s_write], global_X[igroup, itile_m, k]);
        tma_copy(shm_b[_, _, s_write], global_W[igroup, itile_n, k]);

        // MBARRIER: Signal consumer data is ready
        set_barrier_transaction_bytes(readable[s_write], ...);

        // Circular stage buffer
        s_write = (s_write + 1) % kStage;
        if (s_write == 0) phase ^= 1;
      }
    }
  }

  if (is_consumer) {
    // CONSUMER: Compute via GMMA
    int s_read = 0;  // Current stage to read
    while (true) {
      // SCHEDULER: Same as producer, get next tile
      if (!get_next_tile(shm_tiles, iblock, ...)) break;

      for (int k = 0; k < ntile_k; k++) {
        // MBARRIER: Wait for producer to fill stage
        wait_barrier(readable[s_read], phase);

        // GMMA: Compute on shared memory data
        gemm(shm_a[_, _, s_read], shm_b[_, _, s_read], accum);

        // MBARRIER: Signal producer stage is consumed
        if (is_leader_in_warpgroup)
          arrive_barrier(writable[s_read]);

        // Circular stage buffer
        s_read = (s_read + 1) % kStage;
        if (s_read == 0) phase ^= 1;
      }

      // ===== EPILOGUE =====
      cast_to_bf16(accum, output, pertensor_scale);
      tma_store(output, global_Y[igroup, itile_m, itile_n]);
    }
  }
}
```

下面我们将对这些核心优化进行逐个分析，把他们的原理和实现解释清楚。

## TMA for Group Gemm

在 hpc-ops 的 Group GEMM 实现中，TMA（Tensor Memory Accelerator）的使用非常巧妙。不同于普通 GEMM 中使用单一 TMA descriptor，这里为每个 group 预配置了独立的 TMA descriptor。这一节我们来详细分析这个设计。

### 为什么需要为每个 group 配置独立的 TMA descriptor？

核心原因就是**每个 group 的数据位置不同**：X 张量在全局内存中是连续存储的 `[total_seq, k]`，第 `igroup` 个 group 的起始位置是 `x_ptr + cu_seqlens[igroup] * k`

如果我们只有一个 tma descriptor，则只能按照这个 tma 的 gmem coord + copy box offset 的方式进行 copy。虽然 gmem coord 可以是任意设置的，但是在实际使用中，我们习惯使用 TiledCopy partition 对 gmem tensor 进行划分，此时 partitioned tensor 是 copy box 对齐的。对于 group gemm 来说，每个 group 的数据起始位置不可能都正好在 copy box offset 中。因此我们有两个选项：1. 把原始数据 Padding 为 copy box aligned 结构，这样每一个 group 都能和 copy box offset 对齐；2. 给每一个 group 都配置一个独立的 tma descriptor，这样每个 group 的数据都能按照自己的起始位置进行 copy

**Kernel Launch 配置**

```cpp
constexpr int kGroupPerThread = 8;
constexpr int kThreadPerBlock = 32;
kernels::update_grouped_tma<...>
    <<<num_group + 1, kThreadPerBlock, 0, stream>>>(...);
```

- **Grid/Block 配置**：`num_group + 1` 个 block，每个 block 32 个线程
- **Block 分工**：
  - Block `0 ~ num_group-1`：每个 block 处理一个 group，更新该 group 的 X 和 Y 的 TMA descriptor
  - Block `num_group`：计算所有 group 的 tile 统计信息

**Kernel 参数详解**

| 参数 | 类型 | 说明 |
|------|------|------|
| `td_xy` | `vec_t<TmaDescriptor, 2>` | **模板 TMA descriptor**，在 Host 端预配置好，包含正确的 stride 等信息 |
| `tma_xy` | `TmaDescriptor*` | **输出数组**，大小 `num_group * 2`，`tma_xy[igroup*2+0]` 是 X 的 desc，`tma_xy[igroup*2+1]` 是 Y 的 desc |
| `x_ptr` / `y_ptr` | `const Tin*` / `const Tout*` | X 和 Y 张量的全局指针 |
| `seqlens_ptr` | `const int*` | 每个 group 的 seqlen，形状 `[num_group]` |
| `cu_seqlens_ptr` | `const int*` | 累积 seqlen，形状 `[num_group + 1]` |
| `tiles_ptr` | `int*` | **输出**：每个 group 的 **tile M** 数量，形状 `[num_group]` |
| `cu_tiles_ptr` | `int*` | **输出**：累积 **tile M** 数量，形状 `[num_group + 1]` |
| `num_group` / `m` / `n` / `k` | `int` | 问题维度 |

**重要提示**：这里的 `tiles_ptr` 和 `cu_tiles_ptr` 只统计了 **tile M 的数量**，不涉及 tile N！tile N 的数量 `num_tile_n = (n + kTileN - 1) / kTileN` 在主 kernel 中直接计算

**(补充) BlockScan 的使用**

`cub::BlockScan` 是一个并行前缀和计算原语。这里使用的是 **Exclusive Sum Scan**：

Exclusive Scan 是一种并行计算原语，对数组进行前缀和计算，但每个位置的结果是该位置之前所有元素的和。

```txt
示例：
输入:  [a, b, c, d]
输出:  [0, a, a+b, a+b+c]  ← exclusive sum
总和:  a+b+c+d               ← block_aggregate

对比 Inclusive Scan：
输入:  [a, b, c, d]
输出:  [a, a+b, a+b+c, a+b+c+d]  ← inclusive sum
```

可以从 hpc-ops 中的代码代表了 block scan 的一般用法

```cpp
// 第 88 行：定义 BlockScan 类型
using BlockScan = cub::BlockScan<int, kThreadPerBlock>;
// - 模板参数 1: int - 扫描的数据类型
// - 模板参数 2: kThreadPerBlock = 32 - block 中的线程数

// 第 89 行：分配共享内存
__shared__ typename BlockScan::TempStorage temp_storage;
// - TempStorage 是 cub 内部定义的结构体
// - 需要共享内存来协调线程间的通信
// - 大小由 cub 自动计算

// 第 90 行：用于返回总和
int block_aggregate;

// 第 91 行：执行 Exclusive Sum Scan
BlockScan(temp_storage).ExclusiveSum(tiles, tiles, block_aggregate);
// 参数说明：
// - tiles (输入): 每个线程贡献的数据数组
// - tiles (输出): 扫描后的结果（原地修改）
// - block_aggregate: 返回整个 block 的总和
```

### TMA Descriptor 更新

当 `blockIdx.x < num_group` 时，为该 group 更新 TMA descriptor。注意，我们**不是在 device 端从头创建 TMA descriptor**，而是：
1. **Host 端创建模板**：`td_xy` 包含了正确的 stride、tile size 等配置
2. **Device 端只更新必要字段**：
   - **全局内存地址**：指向该 group 数据的起始位置
   - **Shape**：根据该 group 的 seqlen 设置，例如对于输入 X (activation)，其第 i 个 group 的 tma gmem tensor shape 应设置为 `[seqlens_ptr[igroup], k]`

**为什么 stride 不需要更新？**

| 张量 | Shape | Stride | 是否变化 |
|------|-------|--------|----------|
| X | `[num_seq, k]` | `(k, 1)` | **k 固定**，所有 group 一样 |
| Y | `[n, num_seq]` | `(1, n)` | **n 固定**，所有 group 一样 |

- `k` 是隐藏层维度（hidden_size），对所有 group 相同
- `n` 是输出维度（output_dim），对所有 group 相同
- 只有 `num_seq` 变化（`seqlens_ptr[igroup]`）

**`update_tma_gtensor` 的作用** 

该 device function 是更新 tma descriptor 的核心。会从 gmem tensor 中提取 shape & stride & gmem ptr，然后把这些信息更新到 TMA descriptor 中。我一开始还有疑问：为什么一定要用 shared memory 创建 cuTensorMap？虽然我之前了解到 tma 存储的信息都是放在 smem 当中的，但是我们仍然可以把这些信息放到寄存器当中，然后修改，最后再存回 gmem 当中呀。后来 agent 了解到 `tma_descriptor_replace_shapes_in_shared_mem` 该 PTX 要求操作源必须在 shared memory 当中，所以必须使用 smem

**`tma_desc_commit_group` 的作用**

```cpp
if (cute::elect_one_sync()) {
  cute::tma_desc_commit_group();
  cute::tma_desc_wait_group();
}
```

在我们的代码中，**同一个 warp 中的不同线程在修改不同的 TMA descriptor**：
- 线程 0 更新 `smem_tma_desc[0]` (X)
- 线程 1 更新 `smem_tma_desc[1]` (Y)

这时候需要用 `tma_desc_commit_group` 来确保 warp 中所有线程对 TMA descriptor 的修改都完成并且可见。这里的 PTX 和 `tma_store_fence` 是一样的，我们之前使用 `tma_store_fence` 是为了确保 tma store 操作必须要在 smem 写入完成之后。在这里起到同样的作用，因为我们之后要把修改好的 smem 内容写回到 gmem 中存储的 cuTensorMap 当中，必须要保证所有的 smem 写入完成才发起该操作。

补充 20260427：我们在之前进行了 `update_tma_tensor` 的操作，其实这是一个在 async proxy 发起的对 smem 上的修改。所以我们必须要保证整个修改的完成，才能在之后进行 `tma_descriptor_cp_fence_release`，将 smem 当中的 descriptor copy 到 gmem 当中。此时就需要一个 fence 来保证顺序，而 commit & wait 就是不错的选择。这里仅使用了一个 thread 来进行 commit & wait，实际上由于每一个 CTA 只有一个 warp，这里单个 thread wait，其他 thread 并不能够偷跑到下一个代码进行整形，而是也在空转等待

**`tma_descriptor_cp_fence_release` 的作用**

这个函数做两件事（fused copy + fence）：
1. **Copy**：把 128 字节的 TMA descriptor 从 shared memory 拷贝到 global memory
2. **Fence**：带 `release` 语义的内存屏障。此屏障的作用是：确保之后使用 tma 的操作，都必须在该写入操作完成之后执行。可以想象为，这个 release fence 把之前的所有写代码都拦住了，编译器不可能把他们重排到这个 fence 之后。还有另一种带 `acquire` 语义的内存屏障，它会保证之后的所有读操作都必须在该读操作完成之后执行。这也是为什么 `acquire & release` 通常成对出现，我查阅了下 `tma_store_fence` 它到底属于 acquire 还是 release 呢？我认为答案应该是 both！我们既不能让 smem 写操作跨越该 fence，也不让 tma store 操作跨越该 fence

**与主 kernel 配对使用**：

Producer（update_grouped_tma）:
```cpp
tma_descriptor_cp_fence_release(tma_xy + i, smem_tma_desc[i]);
// "Release": 保证之前的所有写操作都可见
```

Consumer（主 kernel）:
```cpp
tma_descriptor_fence_acquire(td_xy + i);
// "Acquire": 保证之后的读操作能看到完整的 descriptor
```


### 在 Group Gemm 中使用 TMA

#### TMA for X (activation)

在 hpc-ops 当中，其使用 tma 的方式是

```cpp
copy(tma_origin.with(new_tma_descriptor, mbarrier), gmem_tensor, smem_tensor);
```

此时，可以认为 copy 所使用的 tma descriptor 就不是 `tma_origin` 中原来在 Host 端定义的 tma descriptor 了，而是我们的 `new_tma_descriptor`。其 gmem ptr 和 shape 都发生了改变，以适应 group gemm 当中不同 group 的 activation 数据搬运。由于 tma descriptor 的改变，我们的 gmem coord 也需要进行相应的适配。在 hpc-ops 当中的 copy 代码如下

```cpp
// itile_m is not a global idx, it is relative to the group
cute::copy(tma_a.with(td_x, readable[ismem_write]), tAg(_, itile_m, itile_k), tAs(_, 0, 0, ismem_write));
```

在代码中只有一个 partitioned tensor `tAg`，但是所有的 group 都使用这个 `tAg`，这是正确的吗？其实这是一个 coordinate tensor，我们只需要填入正确的 coordinate 即可。所以对于 `itile_m`，其 tma descriptor 更新为了 `td_x`，我们需要计算当前的 m coordinate 是相对于该 group 的首地址的偏移量，而不是全局的偏移量即可。这一点我在之后的 scheduler 笔记当中也会再此提到

#### TMA for W (weight)

对于权重来说，其维度是三维的 `(n, k, num_group)`。相应的，我们在定义 tiled copy 时所使用的 **weight gmem tensor 也是三维的，不过需要注意的是：所使用的 copy box 却是二维的**

```cpp
using SLayoutW = decltype(tile_to_shape(SLayoutWAtom{}, 
                          make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));
// w is 3-dim tensor, but copy box is 2-dim
auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, w, take<0, 2>(SLayoutW{}));
```

我之前的理解是：tma 在搬运 tensor 的时候是根据首坐标 + box dim 来确定搬运数据的范围。此时首坐标是 3D 的，box dim 是 2D 的，这似乎挑战了我之前的理解。不过回答也非常简单，现在的数据范围是根据 3D 中的前 2D 坐标 + box dim 来确定的。合理猜测，如果我们把 gmem 维度顺序变成 (num_group, n, k)，但 slayout 保持不变，那么：
  - slayout 第 0 维 (kTileN) → gmem 第 0 维 (num_group)                                                
  - slayout 第 1 维 (kTileK) → gmem 第 1 维 (n)                                                        
copy box 就会沿着 num_group 和 n 维度进行 copy，这显然不是我们想要的结果


## Transposed MMA

### 问题背景

在普通的 GEMM kernel 中，通常固定 kTileM=128（M 维度的 tile 大小），这是因为 Tensor Core 的 MMA 指令通常在 M 维度有较大的粒度。但在 Group GEMM 场景中，每个 group 的 seqlen 可能很小（小 M 场景），如果 kTileM 太大，会导致：
- 硬件利用率低（小矩阵无法填满 Tensor Core）
- 需要大量 padding，浪费计算和内存

---

### 核心概念：为什么转置是必要的？

首先理解 CuTe/CUTLASS 中 MMA atom 的标准约定：

```
标准 MMA: C = A @ B
  - A.shape = (M, K)   ← 通常是 Input X
  - B.shape = (N, K)   ← 通常是 Weight W
  - C.shape = (M, N)   ← 输出 Y
```

但 SM90 架构的 MMA 指令有一个特点：**M 维度的粒度通常较大（64/128），而 N 维度的粒度可以更小（16/32）**。

看 `config.h` 中的指令选择：
```cpp
SM90_64x16x32_F32E4M3E4M3_SS_TN  // M=64, N=16
SM90_64x32x32_F32E4M3E4M3_SS_TN  // M=64, N=32
SM90_64x64x32_F32E4M3E4M3_SS_TN  // M=64, N=64
```

注意：**M 固定是 64，而 N 可以是 16/32/64！**在 Group GEMM 中：

- **M 维度** = seqlen（每个 group 的 token 数），可能很小（如 4, 8, 16）
- **N 维度** = output_dim（输出维度），通常很大（如 7168, 14336）

如果用标准 MMA：
- kTileM 最小是 64，对于 seqlen=16 来说太大了
- 大量 padding 浪费计算和内存

**所以，hpc-ops 采取了一个巧妙的解决方案：把问题转置过来！**

---

### hpc-ops transposed MMA

```
Gemm  :  C[M, N] = A[M, K] @ B[K, N]

原始问题: Y[M, N] = X[M, K] @ W^T[N, K]

转置后:   Y[N, M] = W[N, K] @ X^T[M, K]
           ↑              ↑           ↑
         输出          Weight      Input
```

GEMM 算法只要求你传入 A, B 两个矩阵的数据，并不会要求你的 A 矩阵一定是输入 X，B 矩阵一定是权重 W。所以 hpc-ops 把问题转置过来，A 矩阵传入的其实是权重数据，而 B 矩阵传入输入 X 数据。这样 MMA atom 在 M 维度粒度就能够变小了，对于小 seqlen or deocde 的场景非常有用

具体代码在 `kernels.cuh` 第 313-317 行：

```cpp
// sA 是 X [M, K], sB 是 W [N, K]

auto tBs4r = thr_mma.partition_A(sB);  // ← sB (W) 作为 MMA 的 A
auto tAs4r = thr_mma.partition_B(sA);  // ← sA (X) 作为 MMA 的 B

auto tBr = thr_mma.make_fragment_A(tBs4r);  // fragment A ← W
auto tAr = thr_mma.make_fragment_B(tAs4r);  // fragment B ← X

// call gemm kernel
cute::gemm(tiled_mma, tBr(_, _, ik, ismem_read), tAr(_, _, ik, ismem_read), tCr(_, _, _));
//                    ↑                          ↑                          ↑
//                  fragment A               fragment B               fragment C
//                  (W数据)                   (X数据)                   (Y^T)
```

不过此时，数据矩阵 Y 的 Layout 会发生改变
```txt
原始问题：Y[M, N], Y is layout right: (M, N):(N, 1)
转置问题：Y[N, M], Y is layout right: (N, M):(M, 1)
```
此时两个 Y 的内存排布就完全不一样了，前者可以认为是 shape `(M, N)` 的 row major layout，而后者 `(N, M):(M, 1)` 可以认为是 column major layout `(M, N):(1, M)`，只需要我们把维度排布一下即可，元素在内存上的排布式完全一致的。但是我们的仍然希望我们的输出仍然是 row major 的，这就需要在 stsm (store shared memory, r2s copy) 的时候进行转置操作。所以我们可以看到 hpc-ops 使用了 `cute::SM90_U16x8_STSM_T` 作为 copy atom，这样就能在 copy 时顺便完成该转置操作。所以我们能够看到在 `config.h` 中使用了 N 维度连续的 smem layout：

```cpp
// SMEM ATom is in MN major, which means continuous in N dimension
using SLayoutYAtom = decltype(slayout_selector<kSwizzleY, Tout, false>());
using SLayoutY = decltype(tile_to_shape(SLayoutYAtom{}, make_shape(Int<kTileN>{}, Int<kTileM>{})));
```

注意：如果我们不选择 trans atom，那么这样的操作是不合法的。因为 `cute::SM90_U16x8_STSM_N` copy 完成过后，smem 的 layout 就会是 `(N, M):(M, 1)`，其将会在 M 方向上进行连续的写入，而我们定义的 smem atom 为 `(N, M):(1, N)`， 其在 M 方向上是不连续的，违反 copy 的连续性要求

另外在 epilogue 当中 hpc-ops 选择了每一个 warpgroup 各自读取一半的数据，然后再进行 store。我一开始以为是 tma copy box 大小本身的约束，实际上并不是，虽然 tma copy box 确实有大小约束，根据 [写给大家看的 CuTe 教程：TMA Copy](https://zhuanlan.zhihu.com/p/2003198909405763007) 中的描述，单个维度的元素数量最大为 256，跟据 [cute 之 Hopper TMA](https://zhuanlan.zhihu.com/p/1985678344352731952) 最小的 copy 单元为 16 bytes。我之前的做法是需要两个 warpgroup 进行同步，等 rmem -> smem 完成写入过后用单个 thread 发起，这样同步的消耗显然会大于单个 warpgroup 级别的同步。此时无论哪一个 warpgroup 完成 smem -> rmem 的读取过后，都可以直接发起 store，这样可以减少同步的开销

```cpp
syncwarpgroup(iwarpgroup);
cute::tma_store_fence();
// code ...
cute::copy(tma_d.with(td_y), tDs(_, iwarpgroup, Int<0>{}),
           tDg(_, itile_n * 2 + iwarpgroup, itile_m));
```

## Scheduler for Group Gemm

在之前学习 Hopper Gemm 的过程中，我们了解到 scheduler 的本质是把 iteration idx 映射到 `(m_idx, n_idx)` 的过程，由此决定该 tile 要计算的矩阵区域。而对于 Group Gemm 来说，我们还需要额外考虑一个 group 维度，i.e. 我们需要把 iteration idx 映射到 `(igroup, m_idx, n_idx)`

hpc-ops 提供了两种调度模式：**Horizontal 模式**（小矩阵用线性扫描）和 **Vertical 模式**（大矩阵用二分查找）。有趣的是，hpc-ops 完全没有考虑 thread block swizzle 模式，单纯就是横向迭代和纵向迭代。这可能仍然由于 H20 本身带宽很强，对于数据的访问模式要求不高，所以 scheduler 就从简设计了

### Horizontal 模式

**适用条件**：`k <= 1024 || n <= 1024`（hidden_size 或 output_dim 较小）

**核心思想**：将 block index `iblock` 扁平化为 `(itile_m_total, itile_n)`，然后从上次的位置开始线性扫描，找到 `itile_m_total` 落在哪个 group。

让我们详细看 `get_next_tile_horizon` 函数的每个参数：

```cpp
__device__ __forceinline__ void get_next_tile_horizon(
    const int *tiles_ptr,    // [in] 每个 group 的 tile 数
    int iblock,              // [in] 当前 iteration_idx
    int num_group,           // [in] group 总数
    int &igroup,             // [in,out] 输入：上次找到的 group；输出：本次找到的 group
    int &itile_m,            // [out] 在 group 内的 tile m 索引
    int &itile_n,            // [out] tile n 索引
    int &sum_tile_m,         // [in,out] 累积 tile 数（用于判断 group idx）
    cutlass::FastDivmod flat_divider)  // [in] 预计算的 fast divider
```
**代码逻辑详解**：

```cpp
// 步骤 1: 将 iblock 分解为 (itile_m_total, itile_n)
// flat_divider 做的是：
//   itile_m_total = iblock / num_tile_n
//   itile_n = iblock % num_tile_n
flat_divider(itile_m_total, itile_n, iblock);

// 步骤 2: 从上次的 igroup 位置开始线性扫描
for (int i = igroup; i < num_group; i++) {
  num_tile_m = tiles_ptr[i];      // 获取第 i 个 group 的 tile 数
  sum_tile_m += num_tile_m;        // 累积
  if (itile_m_total < sum_tile_m) {
    // 找到！itile_m_total 落在第 i 个 group
    igroup = i;
    sum_tile_m = sum_tile_m - num_tile_m;  // 回退到 group 开始前
    itile_m = itile_m_total - sum_tile_m;   // 计算在 group 内的索引
    return;
  }
}
igroup = -1;  // 没有更多 tile 了，结束
```

**💡需要注意的是，所有的 `itile_m` i.e. `m_idx` 都是计算的 group 内的索引，而不是相对于第 0 个 group 的全局索引。**这是合理的，因为我们本来就为每一个 group 分配了独立的 tma，我们要计算的就是其 group 内的偏移

**为什么小矩阵用线性扫描？**
- 小矩阵意味着 `num_group` 不大
- 线性扫描实现简单，指令数少
- **增量搜索**：从上次的 `igroup` 位置继续，实际复杂度接近 O(1)
- 缓存友好：`tiles_ptr` 是连续访问的

#### 补充：cutlass::FastDivmod

首先，让我们**明确 FastDivmod 在 hpc-ops 中实际做了什么数学运算**。在 Horizontal 模式 scheduler 中，我们有一个线性索引 `iblock`，需要把它**分解成二维坐标** `(itile_m_total, itile_n)`。假设我们有一个固定的除数 `b = num_tile_n`（tile N 的总数），对于任意输入 `a = iblock`，FastDivmod 计算：

```
q = a / b    （商，整数除法，向下取整）
r = a % b    （余数）
```

使得：
```
a = q * b + r,    其中 0 ≤ r < b
```

在 hpc-ops 中的具体命名：
```
itile_m_total = q = iblock / num_tile_n
itile_n     = r = iblock % num_tile_n
```

BTW, 由于GPU 上整数除法指令很慢（~20 cycles），而 FastDiv 使用了乘法 + 移位来代替除法。这里我们就不做整理了。事实上我看 DeepGemm 仍然使用了整除运算，i.e. 直接使用除法 `/` 用于两个整型符号之间，即可获得 floor division

### Vertical 模式

**适用条件**：大矩阵（`k > 1024 && n > 1024`）

**核心思想**：利用 `cu_tiles_ptr` 的累积索引结构，通过二分查找快速定位 `igroup`。

```cpp
__device__ __forceinline__ void get_next_tile_vert(
    const int *cu_tiles_ptr,  // [in] 累积 tile 索引
    int iblock,                // [in] 当前 block 索引，i.e. iteration_idx
    int num_group,             // [in] group 总数
    int &igroup,               // [out] 找到的 group
    int &itile_m,              // [out] 在 group 内的 tile m 索引
    int &itile_n,              // [out] tile n 索引
    int total_m)               // [in] 总 tile m 数 = cu_tiles_ptr[num_group]
```

**代码逻辑详解**：

```cpp
// 步骤 1: 分解 iblock（注意这里和 Horizontal 模式不同！）
int itile_m_total = iblock % total_m;
itile_n = iblock / total_m;

// 步骤 2: 二分查找找最大的 right 满足 cu_tiles_ptr[right] <= itile_m_total
int left = 0;
int right = num_group;
while (left <= right) {
  int mid = left + (right - left) / 2;
  if (cu_tiles_ptr[mid] > itile_m_total) {
    right = mid - 1;
  } else {
    left = mid + 1;
  }
}

// 步骤 3: 计算在 group 内的 tile m 索引
itile_m = itile_m_total - cu_tiles_ptr[right];
igroup = right;
```

### Shared Memory 缓存优化

在主 kernel 开始时，会把 `tiles_ptr` 或 `cu_tiles_ptr` 缓存到 shared memory 中：

```cpp
if constexpr (IsLoopH) {
  // Horizontal 模式：缓存 tiles_ptr
  for (int i = idx; i < num_group; i += blockDim.x) {
    shm_tiles[i] = tiles_ptr[i];
  }
} else {
  // Vertical 模式：缓存 cu_tiles_ptr
  for (int i = idx; i < (num_group + 1); i += blockDim.x) {
    shm_tiles[i] = cu_tiles_ptr[i];
  }
}
```

这样后续的 scheduler 调用可以访问 shared memory，减少 global memory 访问延迟。

### 调度模式选择

在 `group_gemm_pertensor_fp8.cu` 中：

```cpp
if (k <= 1024 || n <= 1024) {
  // Horizontal 模式：小矩阵，线性扫描
  group_gemm_pertensor_fp8_kernel<..., true>(...);
} else {
  // Vertical 模式：大矩阵，二分查找
  group_gemm_pertensor_fp8_kernel<..., false>(...);
}
```

我认为没有 threadblock swizzle 的 scheduler 很难做到 L2 cache 的优化，我对这里的划分原理也不是很清楚。只能大致理解为：对于 n 比较小的矩阵，我们沿水平方向(i.e. n 方向)进行遍历，可能有更好的 L2 cache 利用，因为此时处理的 tile 会在 m 方向上有所延展，此时能够有一些数据复用。反之，对于 n 比较大的矩阵，沿着水平方向遍历，大家都在同一横排上，数据复用效果差，所以沿着 m 方向遍历还更有机会一些

## Scale for DeQuantization

在 FP8 量化计算中，输入数据和权重都是 FP8 格式，计算过程中使用 FP32 累积以保持精度，最后需要乘以 scale 进行反量化。hpc-ops 提供了两种模式：Pertensor（全张量共享一个 scale）和 Blockwise（每个数据块有独立的 scale）。Pertensor 模式甚至都不需要使用 tma，直接从 kernel 的参数传进来就行，这里就不多介绍。我们还是重点学习 blockwise 的 scale 是如何参与计算的。


### Blockwise Quantization & Scale Layout

我们首先介绍 blockwise quantization 具体是怎么计算的，然后再整理其 scale layout 形式

```python
# 维度定义
M: int                  # 输入序列长度
K: int                  # 隐藏层维度
N: int                  # 输出维度
block_size: int = 128   # 量化块大小

# 输入与权重
X: Tensor = [M, K]      # 输入激活
W: Tensor = [K, N]      # 权重
Y: Tensor = [M, N]      # 输出结果

# X 量化：每 K/block_size 块一个 scale
X_q, X_s = quantize_X(X)
# X_q.shape = [M, K]
# X_s.shape = [M, K // block_size]

# W 量化：每 (K/block_size, N/block_size) 块一个 scale  
W_q, W_s = quantize_W(W)
# W_q.shape = [K, N]
# W_s.shape = [K // block_size, N // block_size]

def blockwise_gemm(X_q, X_s, W_q, W_s, Y):
    # 维度重排，方便分块计算
    X_q -> [M, K // block_size, block_size]
        -> [K // block_size, M, 1, block_size]

    W -> [K // block_size, block_size, N // block_size, block_size]
      -> [K // block_size, N // block_size, block_size, block_size]

    Y -> [M, N]
      -> [M, N // block_size, block_size]

    # 分块矩阵乘法
    for i, j, k in iteration(M, N // block_size, K // block_size):
        Y[i, j] += X_q @ W_q * X_s[k, i] * W_s[k, j]  # 反量化缩放

    # 结果 reshape
    Y -> [M, N]
    return Y
```
实际上我们的 TensorCore M 方向上并不是一个一个计算的，而是以 kTileM 为单位进行 mma 计算。这里只是为了方便逻辑表示，对每一个 M 进行了 iteration

以上是一个朴素的 blockwise gemm，在 hpc-ops 当中，我们使用的是 group gemm，所以对于 X 和 W 都要做相应的 group 划分。我们从其 scale 的定义来一窥其中的改变

**X scale** 的 Layout 为：
```cpp
(num_block_k, m_pad) : (m_pad, 1)
```
`num_block_k = k / 128`（K 维度每 128 个元素一个块），`m_pad` 是 padding 后的 M 维度大小。**为什么需要对 m 进行 pad？**原因仍然是在于 group gemm 有多个 group，我们需要针对每一个 group pad 到 CTATile 对齐的情况（128 in this case），pad scale 应该不是一个耗时的操作

**W scale** 的 layout 为：
```cpp
(num_block_n, num_block_k_pad4, num_group) : (num_block_k_pad4, 1, num_block_n * num_block_k_pad4)
```
`num_block_n = n / 128`， `num_block_k_pad4` 是 padding 到 4 倍数的 `num_block_k`。**为什么要对 k 进行 pad？**可能原因在于 hpc-ops 定义的 weight scale copy box 的大小是 `(1, 4)`，所以最好进行 4 对齐处理。不过 tma 应该能够处理 out of bound 的情况，不太清楚这个 pad 是否是必须的

### CTA 问题划分

在 blockwise gemm 当中，我们的 `block_size` 设定并不是任意的。我们会将 `block_size` 必须与 CTA Tile 结合起来看。因为我们的 mainloop 计算都是以 CTA Tile 为单位进行计算的，在这个过程中必须要使用相应的 scale 进行反量化。本质上是因为 blocksize 和 CTA Tile 都会对问题进行切分，我们需要他们二者的切分能够对齐，i.e. 整数倍。例如我们的 CTA Tile MNK 定义为 `(kTileM, kTileN=128, kTileK=128)`，那么我们的 `block_size` 定义为 128 会比较方便，这样一次 mainloop stage 我们只需要 load A scale `(kTileM, 1)` & B scale `(1, 1)`

在 hpc-ops 就选择了 CTA Tile 和 block size 相等的切分方式：i.e. `kTileN = kTileK = block_size = 128`。每一次 tma copy 会 copy 一个 stage 的 scale 用于 mainloop 计算。同时在分配 smem 资源的时候，hpc-ops 分配了足够的空间 (kTileS) 来存储每一个 stage 的 scale，而不是刚刚好的空间

```cpp
using CopyBoxXS = decltype(make_layout(make_shape(Int<1>{}, Int<kTileM>{}),
                                      make_stride(Int<kTileM>{}, Int<1>{})));
using CopyBoxWS = decltype(make_layout(make_shape(Int<1>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})));
// kTileS = 64, max kTileM = 64, enough to store scale for X
using SLayoutXS = decltype(make_layout(make_shape(Int<kStage>{}, Int<kTileS>{}),
                                       make_stride(Int<kTileS>{}, Int<1>{})));
using SLayoutWS = decltype(make_layout(make_shape(Int<kStage>{}, Int<kTileS>{}),
                                       make_stride(Int<kTileS>{}, Int<1>{})));
```
需要注意的是：对于 B scale 来说，每一个 stage 我们只需要一个 fp32 scale 用于计算，但是由于 tma 的最小 copy box 限制，hpc-ops 每次将会 copy 4 个 fp32 scale。这里还有一个技巧：我们在定义 tma copy 的时候，可以用 copy box 作为 slayout 参数传入
```cpp
auto tma_xs = make_tma_copy(SM90_TMA_LOAD{}, xs, CopyBoxXS{});
auto tma_ws = make_tma_copy(SM90_TMA_LOAD{}, ws, CopyBoxWS{});
```
在之前的使用方法中，我们通常会直接使用真实的 smem layout 来作为 copy box 例如对 x 和 w 的 copy
```cpp
auto tma_x = make_tma_copy(SM90_TMA_LOAD{}, x, take<0, 2>(SLayoutX{}));
auto tma_w = make_tma_copy(SM90_TMA_LOAD{}, w, take<0, 2>(SLayoutW{}));
```
所以我们可以不用传入真实的 smem layout，而是传入 copy box 作为虚拟的 smem layout 传入，此时可以灵活控制 copy 数据的区域。这样 copy box 就不能是简单的 Tiler/Shape，而是完整的 layout，在这里其排布为 row-major，对齐了 gmem & smem 当中的排布

#### thread 获得 scale

我们在 CTA + tma 视角下，我们能够以整体的视角来看待数据的搬运，例如：一整块的 x scale `(1, kTileM)` 从 gmem 读取到 smem 当中。不过我们真正在使用这些数据时，都是要降低到 thread 视角当中，即：每一个线程应该获得哪些 scale 以进行正确的 mainloop accumulation 计算

在完成 gemm 计算过后， C matrix 中的数据会被各个线程划分。就像之前在 sm80 上所看到的一样，每一个 thread 有自己的数据

<img src="CUDA Programming 10.2/image-20260423212339000.png" alt="image-20260423212339000" style="zoom:50%;" />

现在对于一个 thread 我们要获得其对应的 scale，这应该如何做到？对于 W scale 很简单，因为一个 cta tile 只有一个 scale，大家都是共享的。对于 X scale 则复杂一些，我们需要寻找这些元素所对应的 m index，即：他们是属于哪一行的。此时 layout algebra 就要大展身手了！hpc-ops 使用了 identity matrix + partition 来解决这个问题：

所谓 identity layout 就是将其输入映射到自身的 layout，这里的自身是一个 coordinate 而不是单纯的整数。这在之前介绍 tma 的时候有所介绍

```python
layout = make_identity_layout((4,4))     # stride=(1@0,1@1)
layout(0, 1) = (0, 1)	# a coordinate tuple
```

所以当我们对 index 有所需求的时候，考虑 identity matrix 都会是不错的选择。hpc-ops 对 C matrix 构建了 identity layout，然后用相同的 partition 方法对齐进行划分，这样就得到了每一个元素的 coordiante

```cpp
auto gI = make_identity_tensor(gC.shape());	
auto tI = thr_mma.partition_C(gI);			 // (V, M, N)
auto tCr = thr_mma.partition_fragment_C(gC); // (V, M, N)
```

如此一来，对于任意的  `tCr` 的坐标都会知晓，我们可以直接取 coord 的 0 index 即可

```cpp
get<0>tI(any_index) // m coord for any tCr
```

hpc-ops 还使用 `retile_fragment` 把线程中的数据 retile 成为我们熟悉的二维 mn 形状，然后再进行循环累加。其实有了第一点的铺垫，这个 trick 完全可以省略。不过我认为这个变换能更帮助我深入理解 partition & C layout，所以整理。在上面我们提到 partition 过后的 tensor layout 为 `(V, M, N)`，这是一个 cute 当中非常常见且极其重要的一个形式，当我们使用 C layout partition 一个 tensor 时：

1. V 代表了一种 Local view，即：一个 C layout 内部，单个 thread 所得到的 value
2. M 代表了在 M-dim 重复的数量，即： C layout 在 M 维度上有 M 个
3. N 代表了在 N-dim 重复的数量，即： C layout 在 N 维度上有 N 个

**这就是 partition 的本质 zipped divide + compose**：用一个 layout mn 去划分一个 tensor mn (zipped_divide)，并通过 layout `tv -> mn` 最终获得每一个 thread 获得的 value (compose)。

回到我们的问题当中，我们需要把 `tI` 或者 `tCr` 从 `(V, M, N)` 转换到 `(M, N)` 上来，应该怎么做？我们势必要对前面的 `V` 维度做进一步的拆解。实际上这里的 V 维度还可拆解为 `(frg_V, frg_M, frg_N)` 三个维度。为何？直接看 mma trait 当中的 C layout

```cpp
using CLayout_64xN   = Layout<Shape <Shape <  _4,_8, _4>,Shape < _2,_2,Int<N/8>>>,
                        Stride<Stride<_128,_1,_16>,Stride<_64,_8,   _512>>>;

using CLayout_64x8   = CLayout_64xN<  8>;
using CLayout_64x16  = CLayout_64xN< 16>;
...
using CLayout_64x256 = CLayout_64xN<256>;
```

C layout 的 V 维度就是 3 个维度 `(2, 2, N/8)`，其对应了 `(frg_V, frg_M, frg_N)`。时刻记住 C layout 代表了 `tv -> mn` 映射，所以 `frg_V` 维度一定是在 N-dim 上连续的，因为其 stride = 64，而 C layout 所对应 M shape 始终为 64。好了我们现在就可以做 retile (or reshape)

```cpp
(V, M, N) -> ((frg_V, frg_M, frg_N), M, N) -> ((frg_M, M), (frg_V, frg_N, N))
// (frg_M, M) -> contigous in M dim
// (frg_V, frg_N, N) -> contigous in N dim
```

最终形成了 `((frg_M, M), (frg_V, frg_N, N))` 的形式，我们把 M 和 N 维度连续的轴各自排到了一起，这就是 hpc-ops 当中的 `retile_fragment` 所做的事

**注意：我们在之前提到 hpc-ops 在进行 GEMM 运算时是将 X 和 W 进行交换的，所以以上的分析在真实的代码当中，都需要进行交换**

### mainloop 伪代码

我们对着伪代码代码简单走读一下，对整个过程的实现有一个理解。注意这里使用了 hpc-ops 当中交换 mma AB 矩阵的技巧

```cpp
// Shared memory 中的 scale 张量布局
// SLayoutXS: (kStage, kTileS) with stride (kTileS, 1)
// SLayoutWS: (kStage, kTileS) with stride (kTileS, 1)
auto sAS = make_tensor(make_smem_ptr(shm_as), SLayoutAS{});
auto sBS = make_tensor(make_smem_ptr(shm_bs), SLayoutBS{});

// K 维度累加循环
for (int itile_k = 0; itile_k < ntile_k; ++itile_k) {
  // 等待数据加载完成
  wait_barrier(readable[ismem_read], phase);

  // 步骤 1: 从 shared memory 读取 wscale
  // 注意: itile_k % 4 以获得所需要的 wscale index
  float wscale = sBS(ismem_read, itile_k % 4);

  // 步骤 2: 计算 xscale * wscale 得到中间 scale tCS
  float tCS[kN];
#pragma unroll
  for (int in = 0; in < kN; in++) {
    tCS[in] = sAS(ismem_read, get<1>(tI_mn(0, in))) * wscale;
  }

  // 步骤 3: GMMA 计算（不包含 scale）
  tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
  for (int ik = 0; ik < size<2>(tAr); ++ik) {
    cute::gemm(tiled_mma, tBr, tAr, tCr);
  }

  // 步骤 4: 使用 scale 进行反量化
  auto tDr_mn = retile_fragment(tDr);
#pragma unroll
  for (int in = 0; in < kN; in++) {
    float yscale = tCS[in];  // 每个 N 维度有独立的 scale
#pragma unroll
    for (int im = 0; im < kM; im++) {
      tDr_mn(im, in) = tCr_mn(im, in) * yscale + tDr_mn(im, in);
    }
  }
}
```

## Questions

1. hpc-ops 并没有使用 multicast 功能。由于该原因，hpc-ops 在 H100 上的性能就不如 DeepGemm。这在 [issue](https://github.com/Tencent/hpc-ops/issues/28) 当中有提到：For the H100, which has higher compute throughput but lower memory bandwidth, the pipeline places more emphasis on memory access patterns. 这说明在 roofline model 当中，H100 需要更大的 GEMM 计算来达到 compute bound，否则很容易就变得 memory bound。在 Thor 上更是如此，估计其 fp16 算力为 250TFLOPS，而其带宽只有 275GB/s，此使需要计算强度超过 930+ Flops/Byte 才能达到 compute bound。对于端侧来说，几乎大部分的 gemm 都达不到这个计算强度，i.e. 都是 memory bound kernel。而对于 H20 来说，其算力低，带宽大，计算强度拐点 37 Flops/Byte = (148 TFlops / 4000 GB/s) 几乎所有的算子都是 compute bound，所以打不打开 multicast 对性能没那么大影响

2. tma copy 时 smem layout 和 gmem layout (在 swizlle 之前) layout 应当保持一致 (都为 MN-major or K-major)，否则将不符合要求。这是由 copy atom 的连续性要求决定的（128 bit 连续）。另外一个补充结论：swizzle 只处理二维的情况，因为 shared memory 就是二维的，在决定好一个最小的读取模式后（要求填满一整行 bank，否则不会产生 bank conflict）使用 tile_to_shape 进行 product，tile_to_shape 默认 col major product

3. 我们在让 AI 进行整理的过程中，一定要给 AI 列好提纲，按照自己的思路来，否则 agent 自行生成的整理，看似尽然有序，实则无法触及原理核心，并且冗长的 AI 叙述会增加我们的负担。所以与 agent 合作来写笔记算是以失败告终吧，可以让 agent 负责一些解释性工作

4. DeepGemm 当中的 1d1d & 1d2d 分别代表什么？
   1d1d 和 1d2d 代表的是 scaling factor (缩放因子) 的粒度 (granularity) 布局

    - 1d1d，A & B 的 scaling factor 都是 1d
    - 1d2d，A 的 scaling factor 是 1d，B 的 scaling factor 是 2d，应该是对应了 blockwise scaling

    | 类型     | Recipe          | SFA 粒度          | SFB 粒度            | 说明                                |
    | -------- | --------------- | ----------------- | ------------------- | ----------------------------------- |
    | **1d1d** | `(1, 1, 128)`   | per-token (1x128) | per-token (1x128)   | 两个矩阵都用细粒度 scaling          |
    | **1d2d** | `(1, 128, 128)` | per-token (1x128) | per-block (128x128) | 矩阵 B 用粗粒度 (128列共享) scaling |

    最初的 DeepGemm 实现的是 1d2d，我们在本文中讨论的也是 1d2d 的情况。而 1d1d 更多是为 Blackwell 架构实现的，因为其在硬件上直接原生支持了 blockwise scaling gemm，即：我们不需要再在 CUDA core 上进行 dequantize 的计算，tensor core 直接帮我们算完了

5. 为什么在 producer 时，不使用 commit & wait 方式的同步方式，而是一定要使用 mbarrier？

   因为我们需要跨 cta 进行同步，cluster 内可能需要进行 tma broadcast 进行 smem 的数据共享。普通的 commit & wait 方式无法满足同步要求

6. 如果 cta 只有一个 warp，那么其中的 syncwarp 还是有必要的吗？在 SIMT 的意义下，不都是以一个 warp 为单位进行知道吗？

   这里仍然设计内存屏障和可见性的问题，以下回答来自 DeepSeek V4

   > 没有 syncwarp 的代码如下
   >
   > ```cpp
   > __global__ void warp_bug(int *out) {
   >     __shared__ int smem[32];
   >     int tid = threadIdx.x;
   > 
   >     // 只有 tid < 16 的线程进行写入
   >     if (tid < 16) {
   >         smem[tid] = tid;            // (A) 写共享内存
   >     }
   > 
   >     // 所有线程读取
   >     int val = smem[tid];            // (B) 读共享内存
   >     out[tid] = val;
   > }
   > ```
   >
   > 编译器可能分析出：线程的写操作 `smem[tid]` 与读操作 `val = smem[tid]` 之间**没有显式的依赖或屏障**。如果它认为共享内存延迟较大，它可能把某些线程的 Load 提前到 Store 之前，比如优化成：
   >
   > ```cpp
   > int val = smem[tid];   // Load 提前了
   > if (tid < 16) smem[tid] = tid;
   > ```
   >
   > 这显然会读到错误值。即使编译器没有做跨线程的激进重排，**在分支结构下，编译器也可能将后续的 Load 移动到分支内部或之前**，产生完全不符合你直觉的指令顺序。

   这就说明了 `__syncwarp` 是带有内存屏障的同步指令，即使在 warp 数量为 1 的情况下，仍然需要他来保障分支的正确性