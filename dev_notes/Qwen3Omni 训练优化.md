# Qwen3Omni 训练优化

## Image Rebalance

为什么需要 image rebalance？原因在于每一个 rank 所处理的图像数量不均衡，会导致处理更多图像的 rank 持续计算，而更少图像的 rank 处于空闲，从而导致 GPU/NPU 计算效率降低。为了更好的利用资源，我们需要将各个 rank 的图像进行均衡处理，这就是 image rebalance 策略的出发点

在进一步介绍算法之前可能还需要一些考虑：

1. 当我们重新分配图像并进行计算，是否和原始计算等价

   如果我们只是对图像进行重分配而没有其他的操作，答案是计算是不等价的。因为图像和文字 prompt 是匹配的，图像重新分配过后匹配的是不同 rank 的文字 prompt，所以我们需要把图像还原回重分配之前的情况再进行之后的 forward 计算

2. 在进行 all to all 通信过后是否需要考虑梯度问题

   all to all 通信是否是一个可微操作？更具体来说其 autograd function 是否在 pytorch 当中实现完成？在 veomni 当中其自己实现了 `all_to_all` 以及其对应的 autograd function，说明 `dist.all_to_all_single` 是不可微的。而这个 PR 也说明了 collectives 是不可微的，[Differentiability Support for Functional Collectives by vishal9-team · Pull Request #168140 · pytorch/pytorch](https://github.com/pytorch/pytorch/pull/168140) 

   直到 pytorch 2.11 release 过后才会正式支持这些的 collectives 可微 [Release PyTorch 2.11.0 Release · pytorch/pytorch](https://github.com/pytorch/pytorch/releases/tag/v2.11.0)

### 算法流程

伪算法语言描述：

1. 每一个 rank 有自己 images。我们先统计所有 rank 的图像（or 视频，后面统称图像） `(n, 3)`，其中 n 就是图像数量，3 代表了 `(t, h, w)`，这三个维度来衡量这些 vision tokens 的大小。所有统计所有 rank 的地方都需要通信，之后不再特别提及

2. 为每一个图像构建 id，`(rank_id, idx)` 由两个维度组成，代表该图像来自于哪个 rank，以及它是这个 rank 的第几张图。此时就可以构建一个 `(n, 2)` 的 `all_image_ids`

3. 构建每个图像的经验 cost，`cost = T * H * W * (H * W + 6000)`，`cost.shape = (n,)`

4. 初始化一个最小堆 `cost_heap`，heap 当中每一个元素为 tuple `(cost_value, rank_id, list_of_images)`，heap 就是为了记录当前最小 cost 的 rank 是哪个，该 rank 此时包含了哪些图像，方便我们做分配。我们初始化 heap 很简单，其就是一个 list，每一个元素为 `(0, rank_id, [])`，即当前 rank 的 cost 为 0，不包含任何图像

5. 贪心分配。对所有图像的 cost 进行从高到低的排序。然后进行循环，对每一个当前的图像，我们 pop 出当前最小堆当中的最小 cost rank，然后把该图像分配给该 rank

   除此之外，因为我们还需要使用 all to all 进行数据分发，所以每一个 rank 必须要记录，自己的图像要分发到哪一个 rank 当中。使用 `source_split_sizes` 记录

   当整个循环完成后，我们也能知晓每一个 rank 所获得的图像来自于哪里，这也就是 `out_split_sizes`

6. 准备分发数据以及接受 buffer

   各个 rank 需要把自己的数据按照发送 rank 进行整合，发送到同一个 rank 的数据需要在合并在一起。同时构建 `source_reorder_map`，用于 reverse 时恢复原始顺序。然后我们需要准备接受 buffer，也是根据 `out_split_sizes` 去计算会接受其他 rank 的图像数据量。然后调用接口 `dist.all_to_all_single` 完成数据分发

7. 返回图像，以及 `out_split_sizes`、`source_split_sizes`、`source_reorder_map` 等数据用于之后恢复各自 rank 的图像数据

总之 image rebalance 的作用就是：**将图像按照计算代价均衡分配到各个 DP rank 上，让每个 rank 的计算量大致相等**。一个完整的 python 伪代码：

```python
def image_embed_rebalance(pixel_values: Tensor, grid_thw: Tensor) -> Tensor, Tensor, dict | None:
    # ---- Gate ----
    if dp_size == 1 or total_images <= dp_size:
        return pixel_values, grid_thw, None

    # ---- Step 0 (SP): gather scattered sequence back to full sequence ----
    if sp_enabled:
        pixel_values = gather_seq_scatter_heads(pixel_values)

    # ---- Step 1: aggregate metadata across all DP ranks ----
    all_grid_thw  = all_reduce(grid_thw)           # (total_images, 3): (T, H, W) per image
    all_image_ids = all_reduce((rank, local_idx))  # (total_images, 2): (rank_id, idx)

    # ---- Step 2: compute per-image cost ----
    cost[i] = T_i * H_i * W_i * (H_i * W_i + 6000)  # (total_images,)

    # ---- Step 3: greedy assignment via min-heap ----
    cost_heap = [(0, rank, []) for rank in range(dp_size)]  # (cost_sum, rank_id, [images])
    heapify(cost_heap)

    images_sorted = sort by cost descending
    for (cost_val, global_idx) in images_sorted:
        src_rank, local_idx = all_image_ids[global_idx]
        cost_sum, dest_rank, image_list = heappop(cost_heap)
        image_list.append((src_rank, global_idx))
        if src_rank == my_rank:
            dests.append((dest_rank, local_idx, global_idx))
        heappush(cost_heap, (cost_sum + cost_val, dest_rank, image_list))

    # ---- Step 4: build communication buffers ----
    bufs_to_send       = [[] for _ in range(dp_size)]
    source_split_sizes = [0  for _ in range(dp_size)]  # (dp_size,): tokens sent to each rank
    source_reorder_map = [(-1, -1) for _ in range(my_image_count)]

    for (dest_rank, local_idx, global_idx) in sorted(dests):
        start, end = token_offsets[local_idx]
        bufs_to_send[dest_rank].append(pixel_values[start:end])
        source_reorder_map[local_idx] = (dest_rank, source_split_sizes[dest_rank],
                                         source_split_sizes[dest_rank] + end - start)
        source_split_sizes[dest_rank] += end - start

    input_flat = cat([cat(buf) for buf in bufs_to_send])

    # ---- Step 5: compute receive sizes & new_grid_thw ----
    my_images       = sorted(cost_heap[my_rank].image_list)
    out_split_sizes = [0 for _ in range(dp_size)]          # (dp_size,): tokens received from each rank
    for (src_rank, global_idx) in my_images:
        out_split_sizes[src_rank] += all_grid_thw_prod[global_idx]

    new_grid_thw = stack([all_grid_thw[gi] for (_, gi) in my_images])

    # ---- Step 6: all-to-all data redistribution ----
    output = empty(sum(out_split_sizes), dim)
    all_to_all_single(output, input_flat,
                      output_split_sizes=out_split_sizes,
                      input_split_sizes=source_split_sizes)

    # ---- Step 7 (SP): scatter back to SP ranks ----
    if sp_enabled:
        output = pad_if_needed(output, sp_size * pad_scale)
        output = gather_heads_scatter_seq(output)

    return output, new_grid_thw, {
        'out_split_sizes':     out_split_sizes,
        'source_split_sizes':  source_split_sizes,
        'source_reorder_map':  source_reorder_map,
        'pixel_pad_size':      pad_size,
        'pixel_length':        original_seq_length,
    }
```

将 `hidden_states` 恢复到 rebalance 之前的原始图像顺序

```python
def image_embed_rebalance_reverse(hidden_states: Tensor, reverse_map: dict | None) -> Tensor:
    if reverse_map is None:
        return hidden_states

    # ---- Step 1 (SP): gather scattered sequence, strip padding ----
    if sp_enabled:
        hidden_states = gather_seq_scatter_heads(hidden_states)
        if reverse_map['pixel_pad_size'] > 0:
            hidden_states = hidden_states[:-reverse_map['pixel_pad_size']]

    # ---- Step 2: all-to-all back to original ranks ----
    hidden_states = all_to_all(hidden_states,
                               output_split_sizes=reverse_map['source_split_sizes'],
                               input_split_sizes=reverse_map['out_split_sizes'])

    # ---- Step 3: reorder images back to original sequence ----
    chunks = hidden_states.split(reverse_map['source_split_sizes'])
    reordered = [chunks[r][s:e] for (r, s, e) in reverse_map['source_reorder_map']]
    hidden_states = cat(reordered)

    # ---- Step 4 (SP): re-pad and scatter back to SP ranks ----
    if sp_enabled:
        if original_length != current_length:
            hidden_states = pad(hidden_states, original_length)
        hidden_states = gather_heads_scatter_seq(hidden_states)

    return hidden_states
```

## all to all autograd

> From DeepSeek
>
> **`all_to_all_single` 的正向操作本质上是一个跨进程的置换（permutation），而置换矩阵的逆就是它自身的转置，同时这个置换矩阵恰好对称，所以反向传播就是再做一次完全相同的置换——也就是同一个 all-to-all 操作**
>
> - **`forward` 方法**：执行数据传输，通常是调用底层通信后端（如 NCCL）的 `alltoall` 原语。
> - **`backward` 方法**：简单地在同一通信组上运行另一个 `alltoall` 操作，将从输出端接收的梯度正确分发给所有输入端

首先我们介绍下 `all_to_all_single` 本身的功能是什么

`dist.all_to_all_single(output, input, output_split_sizes, input_split_sizes)` 是一个跨 rank 的数据重分布原语。它以 rank 为维度，将每个 rank 拥有的一段数据同时分发给所有 rank。

```python
dist.all_to_all_single(
    output,                               # 接收 buffer，size = sum(output_split_sizes)
    input,                                # 发送 buffer，size = sum(input_split_sizes)
    output_split_sizes=output_split_sizes,  # (world_size,): 从各 rank 接收多少行
    input_split_sizes=input_split_sizes,    # (world_size,): 向各 rank 发送多少行
)
```

- input 和 output 都是 **2D tensor**，切分发生在第 `dim=0`
- `input_split_sizes[r]` = 当前 rank 发送给 rank r 的数据量（行数）
- `output_split_sizes[r]` = 当前 rank 从 rank r 接收的数据量（行数）

参考 veomni 实现

```python
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = dist.get_world_size(group=group)

        if world_size == 1:
            return input

        input = input.contiguous()

        if output_split_sizes is None:
            output = torch.empty_like(input)
        else:
            output = torch.empty(size=(sum(output_split_sizes), input.size(1)), dtype=input.dtype, device=input.device)
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )
```

**注意：**all to all single 都是针对于单个 tensor 而言，并且 tensor 的形状应该都是 2D 的，且 all to all single 只会在第 `dim=0` 进行切分和合并

可以看到 forward 和 backward 各自使用了一次 all to all，他们只是交换了 input & output split sizes，这对应了上述所说的对称转置

另外我们看到在 backward 当中是直接使用了的 `_AllToAll.apply` 而不是使用的 `dist.all_to_all_single`，这可能是为了高阶微分所设计的，此时代表了这个梯度也会被加入到计算图当中，i.e. 该算子的梯度也是可微的，我们可以对梯度再做 backward。不过通常我们不太会接触到高阶微分，这里的设计或许是为了更通用

在 image rebalance 策略当中，我们在 rebalance 的过程中使用了 `@torch.no_grad` 的装饰器，让函数不追踪梯度，这是因为我们的输入图像是起始节点，有没有 grad 都无所谓。不过我们在 `image_rebalance_reverse` 就必须要使用可微函数，否则梯度将无法被正确传递