# VeOmni Data Flow

针对 qwenvl 系列的数据流进行整理。谨防由于对数据处理不熟悉，而导致之后的开发错误。首先由 DeepSeek 总结了一个文档。果然读 AI 写的文档并没办法让我深刻理解，写作还是得自己来。我应该让 AI 写一个大纲，我自己根据这个大纲来自行探索和学习

1. 配置中跟 Data 相关的参数

   ```yaml
   max_seq_len: 32768
   micro_batch_size: 1
   global_batch_size: 4
   ulysses_size: 2                    # SP = 2
   token_compression_kernel_size: 2   # 2×2 AvgPool → 1/4 visual tokens
   compression_method: average_pooling
   ```

2. raw sample 结构

   **实际数据格式** (通过 `process_sample_qwen3_vl` 中注入 print 获得，数据源: `FV_cocotext`)：

   ```python
   sample = {
         _global_sample_id: int
         _spanner_sample_id: int64 shape=() dtype=int64
         id: int
         images: ndarray shape=(1,) dtype=object   # images[0] is a png bytes object
         messages: ndarray shape=(3,) dtype=object
               {'role': 'system', 'content': 'You are a helpful AI assistant.'}
               {'role': 'user', 'content': 'Identify the text displayed in this image.<image>'}
               {'role': 'assistant', 'content': '5863 SNOOLAND Star Eagle KENT classic THE DISTRACT FKN863V people 71 SOU KENT'}
         meta: list len=5 elem_type=tuple # not important
         source: str len=11               # not important
         tag: str len=7                   # not important
   }
   ```

   `moderation_sft_preprocess` 核心功能是把 raw sample 转换成一个 list of list

   1. 先用正则，把 `<image> & <audio>` token 分离出来

      ```txt
      "描述<image>文字" → ["描述", "<image>", "文字"]
      ```

      还会把 `<think>` 也分离出来

      ```txt
      ['assistant', ('text', '<think>\n\n</think>\n\n'), ('text', 'words')]
      ```

   2. 构建结构化 tuple

      - 普通文本 → ("text", content)
      - `<image>` → ("image", None)（剥离尖括号，值为 None）
      - `<audio>` → ("audio", None)

   ```python
   Input messages =
     [{"role":"user", "content":"这张图片里有什么？<image>"},
      {"role":"assistant", "content":"图片中有一只猫坐在沙发上。"}]
   
   Output conversations (List[List]) = 
     [["user",   ("text", "这张图片里有什么？"), ("image", None)],
      ["assistant", ("text", "图片中有一只猫坐在沙发上。")]]
   ```

3. image processsing

   首先是使用 `smart_resize` 把图像进行 resize，主要是进行 32 pixel 对齐。原因是通常要做 merge，而 `merge_size = 2 & patch_size = 16`，对齐到 32 pixel 方便处理

   然后调用 `processsor.image_processor` 进行处理，最终应该会返回一个 dict

   ```python
   image_inputs = processor.image_processor(images=[image], return_tensors="pt", do_resize=False)
   # image_inputs["pixel_values"]: torch.size(N_tok, token_dim)
   # image_inputs["image_grid_thw"]: torch.Size([1, 3])
   ```

   这里的 `token_dim = 1536 = 2 x 3 x 16 x 16`，他们分别代表：patch T, rgb channel, patch H, patch W。需要注意的是 patch T 为 2，但是我们的图只有一张，所以是进行了复制。同理当，t = 7 时，也会对最后一帧图像进行复制，以进行 2 对齐。`N_tok` 的计算就相当自然了，直接除以各个维度的 patch size 大小即可，同理可得 `image_grid_thw` 的值。这些值都是未压缩的。**需要注意的是** `pixel_values` 在 `N_tok` dim 是被重新排列过的：

   ```python
   pixel_values = 
   (T, H, W, token_dim) -> # use merge size to split & permute
   (T, H / merge_size, W / merge_size, merge_size, merge_size, token_dim) -> # flatten
   (N_tok, token_dim)
   ```

   也就是说在 `N_tok` 维度是按照 merge block 进行迭代的，而不是简单的先在 W 方向持续延申。这造成了之后我们在对 vision token 做位置编码的时候，一定要按照这个迭代顺序来

   在之后会进行两次压缩，我们需要计算两次压缩过后最终的 image token 以及对应的 `image_grid_thw_for_llm`。这些压缩过后的 token 才是会真正进入到 llm 部分进行计算的

   1. 第一次压缩是通过 `merge_size` 进行压缩（qwen 官方）

      在完成 vit forward 过后使用 merger 对 `merge_size x merge_size` token 压缩为 1 个 token，当然 token dim 也会对齐到的 llm hidden dim 当中

      ```txt
      (t, h, w) -> (t, h // 2, w // 2)
      ```

   2. 第二次压缩是通过 token compression 模块进行压缩

      这里是直接使用 avg pool 2D 对 vision tokens 进行压缩，一般 kernel size  为 2，又可以降低大约 4 倍

      ```python
      (t, h, w) -> (t, ceil(h, 2), ceil(w, 2))
      ```

   update：我以为 `image_grid_thw_for_llm` 会是两次压缩过后的结果，结果并不是，其还要再乘以一个 `merge_size`，这是为了在之后计算 mRoPE 的 work around，在计算 mRoPE 的时候还会额外除以一个`merge_size`

   ```python
   # --- Patch: Apply token compression to image token count ---
   image_grid_thw_for_llm[:, 1] = (
       ((image_grid_thw[:, 1] // merge_size) + token_compression_kernel_size - 1)
       // token_compression_kernel_size
   ) * merge_size
   ```

   最后输出了 4 个参数

   ```python
   image_grid_thw			# (N, 3) used for 2D RoPE in vit
   image_grid_thw_for_llm	# (N, 3) used for mRoPE in llm
   visual_tokens	# int, number of tokens after compression (both merge & token compression?)
   pixel_values	# (N_tok, token_dim)
   ```

4. 构建 chat template

   根据 `conversations` 构建

   1. `input_ids`，shape 为 `(seq_len,)`

      特殊 token 的处理：e.g. 把 image 使用实际的视觉 token 序列

      ```txt
      <|vision_start|><|image_pad|><|image_pad|>...×N<|vision_end|>
      ```

      然后使用 tokenizer 转为 token id，最后把多模态的 token 进行重新映射，在之后好替换

      | 原始 token        | 替换为 | 含义     |
      | :---------------- | :----- | :------- |
      | `<\|image_pad\|>` | -200   | 输入图像 |
      | `<\|video_pad\|>` | -300   | 输入视频 |
      | `<\|AUDIO\|>`     | -400   | 输入音频 |

      借助此还可以方便构建 `image_mask & video_mask` 这在之后方便把 vision tokens 使用 `mask_scatter` 放到序列当中

   2. `labels`，shape 为 `(seq_len,)`

      其实是把 `input_ids` 内的内容进行 mask，把不需要贡献 loss 的 id 使用 `-100` 替换掉

   3. `attention_mask`，shape 为 `(seq_len,)`

      一个全 1 的 tensor

   4. 对于 qwen35 模型来说额外使用了 `add_empty_thinking=True` 模板

5. position ids 计算

   针对于 mRoPE 计算 3D 的 position ids `(3, 1, seq_len)`，算法核心是针对 image position id 单独计算，然后加上其在 sequence 当中的 offset 即可，最终的效果

   ```txt
   pos:   0   1   2   3   4   5   6   7   8   9
   tok:  t0  t1  t2  i0  i1  i2  i3  t3  t4  t5
   dim0:  0   1   2   3   3   3   3   5   6   7
   dim1:  0   1   2   3   3   4   4   5   6   7
   dim2:  0   1   2   3   4   3   4   5   6   7
   ```

   对于 image 部分，没有增加 offset 的视角

   ```txt
    grid: (T=1, H=2, W=2)
    t_index = [0, 0, 0, 0]
    h_index = [0, 0, 1, 1]
    w_index = [0, 1, 0, 1]
   ```

   因为 image token 在之前还有 3 个 text token，所以 offset 添加 3

   ```txt
    t_index = [3, 3, 3, 3]
    h_index = [3, 3, 4, 4]
    w_index = [3, 4, 3, 4]
   ```

   单条样本 `tokenized_example`

   | Key            | Shape             | Dtype | 说明                              |
   | -------------- | ----------------- | ----- | --------------------------------- |
   | input_ids      | [163]             | int64 | image token 位置 = 0              |
   | attention_mask | [163]             | int64 | 全 1                              |
   | labels         | [163]             | int64 | user part = -100, assistant = ids |
   | position_ids   | [3, 163]          | int64 | (temporal, height, width) 3D RoPE |
   | image_mask     | [163]             | bool  | 130 个 True                       |
   | video_mask     | [163]             | bool  | 全 False                          |
   | pixel_values   | [N_toks, token_d] | float | vision patches (原始, 未压缩)     |
   | image_grid_thw | [1, 3]            | int64 | [[1, 50, 38]] (原始 grid)         |

6. Dynamic batching

   按照 `micro_batch_size * max_seq_len` 作为最大值，打包多个 sample，使得总的 seq len 尽可能地接近该最大值，这就是 dynamic batching 最核心的思想。`global_batch_size` 其实是用来做 gradient accumulation 用的

   ```python
   # i.e. num accumulation steps
   num_micro_batch = global_batch_size // (micro_batch_size * dp_size)
   ```

   `dp_size` 传入的其实是 `world_size`，这里 DeepSeek 说了一个很有意思的思路： DP 是所有并行策略的基础，其他并行是从 DP 中"切分"出去的。以 run.sh demo 为例（2 GPU 纯 FSDP2）。这其实是不正确的，但如果我们只考虑 SP & EP 的话，我认为这大体是正确的。无论如何，这里 global batch size 就是控制 num accumulation steps 的参数，通常可以设置其为 GPU 数量，使得 steps = 1

7. Collator

   怎样把所有的 sample 打包成为一个呢？这是核心中的核心

   collator 似乎是通过一个 pipeline 来进行打包的，有3个 pipeline

   ```python
   MainCollator.__post_init__:
     pipeline = [
       PrecomputePositionIDsCollator(),            # Stage 1
       PackingCollator(collate_infos=...),         # Stage 2
       SequenceParallelCollator(collate_infos=...), # Stage 3 (sp_enabled=True)
     ]
   ```

   - **Stage 1 — `PrecomputePositionIDsCollator`**: sample 缺少 `position_ids` 时补全为 `[0, 1, ..., seq_len-1]`
   - **Stage 2 — `PackingCollator`**: 将多个 sample 拼接为一条 packed 序列
     - 对 `labels`：每个 sample 的首 token 置 `-100`，防止 boundary token 参与 loss
     - 按 `collate_info` 的 `pack_dim` 做 `torch.cat`（`-1` 用于 1D 序列并 `unsqueeze(0)` 补 batch 维；`0` 用于 patch 级数据）
     - 非 SP 模式下，直接调用 `add_flash_attention_kwargs_from_position_ids` 从完整的 position_ids 提取 `cu_seq_lens` 和 `max_seqlen`
   - **Stage 3 — `SequenceParallelCollator`** (仅 `sp_enabled=True`): 将 packed 序列切分到各 SP rank
     - `labels` 左移一位；SP padding 补到能被 `sp_size * sp_pad_scale` 整除；然后 `narrow` 切分（跳过 `position_ids`）
     - 切分后用**完整 position_ids** 计算 `cu_seq_lens` 和 `max_seqlen`（必须在切分前算，因为 `position_ids == 0` 标记了 sample 边界，切分后会丢失）
     - 最后切分 `position_ids`

   在 log 当中会有一个表格，对当前所使用的 collator 进行展示

   ```txt
   ┌──────────────────────┬──────────┬──────────┬──────────────┬──────────────┐
   │ Key                  │ pack_dim │ sp_slice │ sp_pad_value │ sp_pad_scale │
   ├──────────────────────┼──────────┼──────────┼──────────────┼──────────────┤
   │ input_ids            │ -1       │ True     │ 0            │ 1            │
   │ labels               │ -1       │ True     │ -100         │ 1            │
   │ attention_mask       │ -1       │ False    │ 1            │ 1            │
   │ position_ids         │ -1       │ False    │ 0            │ 1            │
   │ pixel_values         │ 0        │ True     │ 0            │ 4 (=2²)      │
   │ pixel_values_videos  │ 0        │ True     │ 0            │ 4            │
   │ input_features       │ 0        │ True     │ 0            │ 1            │
   │ image_mask           │ -1       │ False    │ 0            │ 1            │
   │ video_mask           │ -1       │ False    │ 0            │ 1            │
   │ image_grid_thw       │ 0        │ False    │ None         │ None         │
   │ video_grid_thw       │ 0        │ False    │ None         │ None         │
   │ audio_feature_lengths│ 0        │ False    │ None         │ None         │
   │ feature_attention_mask│-1       │ False    │ 1            │ 1            │
   │ audio_mask           │ -1       │ False    │ 0            │ 1            │
   └──────────────────────┴──────────┴──────────┴──────────────┴──────────────┘
   ```

   注意：表面上 `position_ids` 设置了 `sp_slice = False`，但实际上当 sp 启动时，该设置不起作用，一定会被 sp 分割的

   注意：虽然 `sp_slice = False` 但是 `sp_pad_value` 和 `sp_pad_scale` 仍然会起作用

   注意：`input_ids` 没有写在 collate info 当中，但是有自己的默认值
   
   ```python
   DEFAULT_DATA_COLLATE_INFO: Dict[str, DataCollateInfo] = {
       "input_ids": DataCollateInfo(-1, True, 0, 1),
       "labels": DataCollateInfo(-1, True, IGNORE_INDEX, 1),
       "attention_mask": DataCollateInfo(-1, False, 1, 1),
       "position_ids": DataCollateInfo(-1, False, 0, 1),
       "pixel_values": DataCollateInfo(0, True, 0, 4),
       "pixel_values_videos": DataCollateInfo(0, True, 0, 4),
       "image_mask": DataCollateInfo(-1, False, 0, 1),
       "video_mask": DataCollateInfo(-1, False, 0, 1),
       "image_grid_hw": DataCollateInfo(0, False, None, None),
       "image_grid_thw": DataCollateInfo(0, False, None, None),
       "video_grid_thw": DataCollateInfo(0, False, None, None),
   }
   ```
   
   注意：`sp_pad_scale` 对于 `pixel_values` 使用了 4，这是因为 merge size 的原因。我们希望切分过后的 `pixel_values` 对齐 merge size square，否则可能发生同一个 merge size block 被分配到两个 rank 当中。而我们需要使用 merger 对 vision tokens 进行融合，如果 vision tokens 由于切分导致 merge block 不完整，则会影响正确计算
   
   这里提示：当我们在考虑的计算会与其他 token 发生交互时，sp 的切分都需要有额外的考虑：
   
   1. attention 是最明显的地方。需要使用 all to all 构建完整的 sequence
   2. token merger 需要考虑保存 local feature 的完整
   
   最终我们获得的 shape 为
   
   | Tensor Name    | Shape                     |
   | -------------- | ------------------------- |
   | input_ids      | torch.Size([1, 32761])    |
   | attention_mask | torch.Size([1, 32761])    |
   | labels         | torch.Size([1, 32761])    |
   | position_ids   | torch.Size([1, 3, 32761]) |
   | image_mask     | torch.Size([1, 32761])    |
   | video_mask     | torch.Size([1, 32761])    |
   | pixel_values   | torch.Size([92212, 1536]) |
   | image_grid_thw | torch.Size([60, 3])       |
   | cu_seq_lens_q  | torch.Size([59])          |
   | cu_seq_lens_k  | torch.Size([59])          |
   
   可以看到，基本上 batch 维度固定为 1，我们是将样本直接在 sequence 维度进行拼接 

## Questions

- 为什么所有的数据，thw 中的 t(ime) 维度都是 1 呢？

  qwen3vl 延续 qwen2vl 使用 temporal patch size  = 2。如果我们只输入一张图像的话，会将该图像进行复制，强行构建成两帧

- qwen35vl 当中的 position embeddings

  对于 vision 部分：

  1. learned abs pos emb

     跟随训练得到 pos embedding，其 shape 为 `(30*30, hidden_dim)`，即一个 30x30 的位置编码。把新图像的 `(H, W)` 利用 bilinear interpolate 获得绝对位置编码即可 

  2. 2D RoPE

     llm RoPE 只关注 1D 的 positional 差距，而到了图像中使用 2D 的 positional 差距是很自然的事情。对于 1D rope，一半 head dim 分给 cos，另一半分给 sin，到了 2D 的情况下，我们还要继续拆分，一半分给 H，另一半分给 W，i.e. 占总 head dim 1/4

     以 head dim 128 为例子，对于 H or W 来说 freq dim 就只有 32

     ```python
     inv_freq = (theta_0, theta_1, ..., theta_31)
     theta_i = 1 / (base ** (i / 32))
     ```

     每一个 patch 都有自己的 H & W id，可以算出自己的 position theta，把 H & W 的 position theta concat 就能够得到 1D 的 position theta 情形。剩下的仍然是构建 cos & sin

     ```python
     freqs = torch.cat([H_id * inv_freq, W_id * inv_freq])
     emb = torch.cat((freqs, freqs))
     cos = emb.cos()
     sin = emb.sin()
     ```

  对于 llm 部分：

  构建了 interleaved mRoPE（我更喜欢称其为 full frequency mRoPE），其核心目的就是为了让 llm 区分出 vision token 的位置关系 ，并且保持 llm 原本的位置感知能力

  mRoPE 将 rope 的维度从 1D 扩展成为 3D (T, H, W)，这样就能够保持空间的位置关系。区别于 vision 当中的 2D RoPE，三者的维度数量并不相等。以 `head_dim = 128` 为例子，分给 inv freq 的维度就是 64 维。在 qwen3vl 给 T 分配 43 维，H 分配 11 维，W 分配 10 维，总共 64 维度

  我们当然可以按照 2D RoPE 的方式，给每一个维度分配不同的 inv freq dims，然后再连接到一块，但是这样会产生两个问题：

  1.  跨模态的位置编码将产生歧义，即：纯文本时，将无法定义 H & W 位置编码
  2. 各个维度对频谱的利用不充分。大家都集中在低频区域，对于局部变化敏感，而对全局变化不敏感，这对长序列建模比较重要。我认为第一点是核心原因，但 DeepSeek 认为第二点是核心原因

  现在使用 interleaved position embedding 就能解决以上两个问题。用比较粗略的图示如下：

  ```txt
  vanilla 3D MRoPE for 1 token:
  dim:	T T T T H H H H W W W W
  freq:	0 1 2 3 0 1 2 3 0 1 2 3 
  
  interleaved MRoPE for 1 token:
  dim:	T H W T H W T H W T H W
  freq:	0 1 2 3 4 5 6 7 8 9 10 11
  ```

  我们将 T H W 进行交叉排布，并且他们的 `inv_freq` 按照正常的 1D frequency 逐渐递增。这样一来：

  1. 对于纯文本而言，我们直接采用 T H W 都是相同的内容，e.g. `T = H = W = 10`，即代表了位置 10 的 token 位置编码
  2. 各个编码分散到了全频域，尤其是 T 维度，能够获得更大的高频区域，从而有利于在时间维度上的全局建模

  在 python 实现中，我们可以先获得完整的 THW 完整的 sequence 位置编码，然后 T 维度上进行调整

  ```python
  # freqs = thw * inv_freqs
  # (3, seq_len, 1) * (1, 1, C) -> (3, seq_len, C)
  
  def _interleaved_mrope_freqs(freqs, mrope_section):
      """
      freqs: (3, bs, seq_len, head_dim // 2)  — T/H/W independent full freqs
      	T [T0, T1, T2, ..., T63]
      	H [H0, H1, H2, ..., H63]
      	W [W0, W1, W2, ..., W63]
      mrope_section: (3,)                     — e.g. (43, 11, 10)
      returns: (bs, seq_len, head_dim // 2)   — interleaved freqs
      	[T0, H1, W2, T3, H4, H5, ...]
      """
      freqs_t = freqs[0]  # copy T as base, will overwrite H/W slots
      for dim, offset in enumerate((1, 2), start=1):  # H=dim1, W=dim2
          length = mrope_section[dim] * 3
          idx = slice(offset, length, 3)              # H: 1,4,7,...; W: 2,5,8,...
          freqs_t[..., idx] = freqs[dim, ..., idx]
      return freqs_t
  ```

- vit 部分的 cu seq len 是如何计算的？

