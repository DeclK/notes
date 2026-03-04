# Flashinfer

终于有机会深入学习一下 flashinfer 了，现在 flashinfer 也已经发展成为一个巨大的 kernel library。不过好在其保持了代码结构上的清晰性，仍然值得对其进行深入的学习而不至于失去方向。并且 flashinfer 本身对技术前沿还跟得特别紧，例如对 claude code skill & tvm ffi 等等，都是值得学习的

## Flashinfer Design

flashinfer 现在根本就没有 cmake 文件，整个 kernel 使用的逻辑已经改变了：从原始的 build cuda cpp，到现在的 jit 形式，整个 installation 变得非常简单

根据 `.claude/skills/add-cuda-kernel` 可以对 flashinfer 的算子开发流程有整体的认知。通过此流程可总结出 flashinfer 的设计精髓

1. JIT 优先

   Just in time compilation 使得 flashinfer 仅在使用 kernel 的时候对 kernel 进行编译。这对于开发来说会非常友好，不必要的 kernel 我们就不去编译了，是轻量且高效的选择

   另外的好处是我们可以完全不必写 Cmake 了！整个框架的配置和依赖都变得更简单

2. CUDA 内核与框架解耦

   所有的 kernel 全部实现在 `flashinfer/include` 中，以 header only 的形式存在。所有和框架相关的代码都放置在了 `flashinfer/csrc` 当中，通过 tvm ffi 实现了与框架（e.g. pytorch）的绑定。更具体来说通过 tvm ffi 的 `TensorView` 可以完成与各个框架之间的 Tensor 进行完美的接收与兼容，并以此为基础构建 kernel launcher。通过 `TVM_FFI_DLL_EXPORT_TYPED_FUNC` 导出 kernel launcher 为 `.so` 文件，使得我们的 kernel launcher 可被 python 调用

这里可以看到 flashinfer 的

## KV Cache Layout

以下讨论均讨论 `(N,H,C)` 形式的 layout，其中

- N 代表 seq len
- H 代表 num of heads
- C 代表 head dim

本小节将会展示 kv cache 的存储格式，至于如何管理 cache 的存储，将在 Paged Attention 章节讨论

### Ragged Layout

在 llm serving 上永远不会去使用这个 layout，永远会使用 paged layout。我认为该 layout 设计是为了服务非 llm 场景下的 batch attention，e.g. vision attention，这会比直接使用 single attention with kv cache 更快

ragged layout 表示，kv 的存储的形式为 `(N=max_seq_len, H, C)`

<img src="Flashinfer/ragged.png" alt="Data structure of Ragged KV-Cache." style="zoom:50%;" />

**需要注意的是：第一个维度 `N` 会把多个 batch 的 seq len 合并起来存储**。`indptr` 是一个 token cumsum，i.e. 把每个 sample 中的 token 数量做 cumsum。所以如果我们在使用 batch prefill with ragged kv 的时候 batch 维度就不会存在了。在使用时需要把多个 sample query 进行 concat，避免了 padding 操作

### Paged Layout

paged layout 表示，kv 的存储形式为 `(num_pages, page_size, H, C)`

<img src="Flashinfer/page_layout.png" alt="Data structure of Paged KV-Cache." style="zoom: 33%;" />

这个图示就表示了 3 个 request，不同的 request 用不同的颜色表示。每一个 request 的 kv cache 会存储在不同的 pages 当中，当一个 page 存储满了，就会规划一个新的 page 以存储 kv cahe。和 ragged layout 一样，这里仍然会将多个 request 当中的 token concat 成为一个来进行看待以提高效率

我们需要获取某一个 request 的 kv cache 时，只需要把这些 kv indices 找到，配合 last page len 即可。完整的 `kv_len` 可用下面的公式计算：

```python
kv_len = page_size * (len(page_indices[i]) - 1) + last_page_length[i]
```

问题在于这些 Request 的 page indices 到底是怎么计算得到的，如果来了新的 request 会发生什么？这需要我们对 paged kv cache 进行精细的管理，也是 paged attention 的精髓

## Major Api Usage

flashinfer 的核心 API 其实就根据 prefill/decode & ragged/paged 分为如下几种

|         | Ragged KV                                | Paged KV                                |
| ------- | ---------------------------------------- | --------------------------------------- |
| Prefill | **BatchPrefillWithRaggedKVCacheWrapper** | **BatchPrefillWithPagedKVCacheWrapper** |
| Decode  | N\A                                      | **BatchDecodeWithPagedKVCacheWrapper**  |

其中 batch decode 阶段没有实现 ragged kv 实现，个人认为原因在于：对 batch 场景下的 decode，ragged kv 会非常低效。想象以下，如果我们对 batch request 进行了 prefill，并存储了以下的 kv cache

<img src="Flashinfer/ragged.png" alt="Data structure of Ragged KV-Cache." style="zoom:50%;" />

接着我们要进行 decode 生成 first token，其 kv cache 需要存储到各个 request kv cache 的末尾，如此以来几乎需要对整个 ragged kv cache 进行重写以插入新的 kv cache。这将会是巨大的消耗，而 paged kv cache 则不存在这个问题。如果非要使用的话，可以直接调用 prefill，并设置 input len 为 1 即可

### BatchPrefillWithRaggedKVCacheWrapper

其实 flashinfer 本身的 [example](https://docs.flashinfer.ai/api/attention.html#flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper) 是很好的说明。但是在未理解其 kv cache layout 之前，看着像天书一样。我对其中的一些变量命名进行更改，并省略了 device & dtype 设置，加入一些自己的注释以帮助阅读理解

```python
import torch
import flashinfer
num_layers = 32
num_qo_heads = 64
num_kv_heads = 16
head_dim = 128
# allocate 128MB workspace buffer, don't know why use this value
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer, "NHD")

batch_size = 7	# 7 requset
cat_kv_len = 100	# total kv of all request
cat_qo_len = 100	# total q of all request
device = "cuda:0"

# cumsum of query nums, shape is [batch_size + 1,]
qo_indptr = torch.tensor([0, 33, 44, 55, 66, 77, 88, cat_kv_len], dtype=torch.int32)	
# cumsum of kv nums, shape is [batch_size + 1,],  we set it the same as query, self-attn
kv_indptr = qo_indptr.clone()

# create query and kv cahe
q_at_layer = torch.randn(num_layers, cat_qo_len, num_qo_heads, head_dim)
k_at_layer = torch.randn(num_layers, cat_kv_len, num_kv_heads, head_dim)
v_at_layer = torch.randn(num_layers, cat_kv_len, num_kv_heads, head_dim)

# plan, set attrs shared by all layers
prefill_wrapper.plan(
    qo_indptr,
    kv_indptr,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    causal=True,
    # custom_mask=mask	# bring your own mask
)

outputs = []
for i in range(num_layers):
    q = q_at_layer[i]
    k = k_at_layer[i]
    v = v_at_layer[i]
    #### run ####
    o = prefill_wrapper.run(q, k, v)
    outputs.append(o)
```

对于 ragged kv 来说，其使用方式和 torch SDPA 还是比较相似，核心区别就是两点：

1. 没有 batch 维度，多个 sequence 直接进行 concat。所以你可以看到 `batch_size` 从来没有出现在任何的 qkv tensor creation 当中
2. 需要使用 plan method，把必要的参数传入：例如 query 和 kv 的 comsum，custom attention mask 等等。这些参数其实都是计算中必要的参数，完全可以放在 `run` 时输入，类似 SDPA。在之后的 `run` 当中会持续使用这些参数。除了设置参数外，plan 还会进行一些调度上的计算，例如将 sequences 进行合理的切分使得 work load balance，这些我也不太了解，就不整理了

### BatchPrefillWithPagedKVCacheWrapper

paged kv cache 相比 ragged 就要复杂一些，在 plan 阶段需要更多的参数 e.g. kv cache indices

```python
import torch
import flashinfer
num_layers = 32
num_qo_heads = 64
num_kv_heads = 16
head_dim = 128
max_num_pages = 128
page_size = 16
# allocate 128MB workspace buffer
workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8)
prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace_buffer, "NHD")

batch_size = 7		# 7 request
concat_qo_len = 100	# total q of all request

# cumsum of query nums, shape is [batch_size + 1,]
qo_indptr = torch.tensor([0, 33, 44, 55, 66, 77, 88, concat_qo_len])
### KV Cache Related Params ###
# cumsum of kv pages nums, shape is [batch_size + 1,]
paged_kv_indptr = torch.tensor([0, 17, 29, 44, 48, 66, 100, 128])
# concat of all request's kv pages idx, shape is [paged_kv_indptr[-1],], 128 in this case
paged_kv_indices = torch.arange(max_num_pages)
# last page len of each request, shape is [batch_size,]
paged_kv_last_page_len = torch.tensor([1, 7, 14, 4, 3, 1, 16])

# create query and kv cache
q_at_layer = torch.randn(num_layers, nnz_qo, num_qo_heads, head_dim)
k_at_layer = torch.randn(num_layers, max_num_pages, page_size, num_kv_heads, head_dim)
v_at_layer = torch.randn(num_layers, max_num_pages, page_size, num_kv_heads, head_dim)

# plan, set attrs shared by all layers
prefill_wrapper.plan(
    qo_indptr,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    causal=True,
)
outputs = []
for i in range(num_layers):
    q = q_at_layer[i]
    kv_cache = (k_at_layer[i], v_at_layer[i])
    #### run ####
    o = prefill_wrapper.run(q, kv_cache)
    outputs.append(o)
```

与 ragged kv cache 的核心区别在于：

1. 在 plan 阶段，ragged kv cache 只需要设置 kv 的 cumsum (i.e. `kv_indptr`) 即可，但是 paged kv cache 就要设置更多了：

   1. `paged_kv_indptr`，是各个 request kv pages 的 cumsum
   2. `paged_kv_indices`，是各个 request 的 page indices concat
   3. `paged_kv_last_page_len`，各个 request 的最后一个 page 的占用长度

   这三个参数的含义如果仍然不清楚，建议可以回去看 paged kv layout 中的图示。这三个参数在上面的例子中是随机设置的，在实际 serving 的过程中，通常由 serving frame work (sglang or vllm) 计算给出，所以这些框架是才是管理 paged kv cache 的核心，并非 flashinfer 本身，这也是 flashinfer 的设计理念

   > **How do FlashInfer manages KV-Cache?**
   >
   > FlashInfer itself is not responsible for managing the page-table (pop and allocate new pages, etc.) and we leave the strategy to the user: different serving engines might have different strategies to manage the page-table. **FlashInfer is only responsible for computing the attention between queries and keys/values stored in KV-Cache.**

2. 在 run 阶段，不单独传入 kv，需要将二者打包为一个 tuple 作为一个 kv cache 整体传入

### BatchDecodeWithPagedKVCacheWrapper

decode with paged kv cache 在使用过程中和 prefill 几乎一样，除了在 plan 的时候不传入 `qo_indptr`，因为每一个 request 的 seq len 都是固定为1的，无需传入

```python
decode_wrapper.plan(
    # missing the `qo_indptr`
    kv_page_indptr,
    kv_page_indices,
    kv_last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    pos_encoding_mode="NONE",
    data_type=torch.float16
)
```

## Paged Attention

经过对各个 AI (Kimi, DeepSeek, GLM) 的严厉拷打，我算是整合除了一套我自己所理解的 paged attention 原理，可能和各个框架的实现有所出入，但我能以自己满意的方式理解其原理已经达到目的

在以上 flashinfer api 的整理中，对 paged kv cache 管理我们始终缺少 3 个关键参数的计算：`paged_kv_indptr` & `paged_kv_indices` & `paged_kv_last_page_len`

为了清楚地解释这些参数是如何在 serving 当中计算得到的，我必须要构建一个清晰的场景以及所需要的工具，使得这些计算能够自然地进行推导：

1. 一个 batch request。其中包含了 prefill & decode request
2. 一个 `free_pages` 队列。包含了哪些 pages 目前是可用的
3. 一个 `page_table` 张量，其形状为 `(max_requests, max_seq_len)`，其中包含的值为1维的 offset，通过这个 offset 可以直接定位到该 token 在 paged kv cache 中的位置
4. 每一个 request 会维护自己的 page indices & last page len

有了这些条件进行 paged attention 就不难理解了：

1. request 进行 prefill 时，其 page indices 和 last page len 都是没有 history 的，我们可以从 `free_pages` 当中 pop 出一些新的 pages 用于其填充。其计算也比较简单，根据 prefill seq len 我们能够很快的计算出需要多少 pages，并且 last page len 为多少也能计算得到



### Decode



### Prefix Cache

prefix caching 问题

如果匹配上了的话，需要新开一个 page，并且把最后一个 page 的 last len 内容重新写入

## Other Hpc

rmsnorm

## TVM FFI

兜兜转转还是没走出 tvm 这个圈，tvm ffi 已经被 CuteDSL & TileLang & Flashinfer 进行了应用。这说明该方式有着足够的易用性，并且功能保障性也很强，不然短时间内这些框架没办法将 tvm ffi 进行集成

我说为什么 tvm 没有出现在 3rd party 当中，但是却可以出现再 csrc 文件当中。因为通过 pip install 下载好了 tvm ffi 在进行编译的时候会直接从这里 include tvm ffi 的头文件

```shell
-isystem /cyq/Projects/flashinfer/.venv/lib/python3.12/site-packages/tvm_ffi/include
```

可以通过加 env `FLASHINFER_JIT_VERBOSE=1 FLASHINFER_JIT_DEBUG=0` 来看到编译的命令

## Questions

- 我在 sm110 上编译 backend = cutlass 虽然成功，但是 kernel 却没办法正常运行。不过我使用原生 cutlass example fmha 能够运行，同样的情况也发生在了 batched prefill paged kv cache

  目前来看 kernel launcher 本事是成功引发的，但是整个 kernel 没有被调用这是非常奇怪的

- 如何利用 tvm ffi & ninja 完成 cuda cpp -> python 的构建，其中的 3rdparty 依赖应该如何完成设置？

Conclusion:

- 如果不是 llm serving 完全可以不用考虑 flashinfer，使用直接的算子集成。可以看到 flashinfer 主推的仍然是 Paged attention，其需要结合 scheduler 对 Page 进行复杂的管理，这种场景和 llm serving 高度的绑定