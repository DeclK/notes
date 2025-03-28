## EAGLE Speculative Decoding

EAGLE 投机采样作为目前最强的投机采样方法，值得整理一番。我将从投机采样基础开始，逐步地讲解 EAGLE 投机采样的改进过程，并配合以实际代码帮助理解。在本文中将以 SD 代表 Speculative Decoding 的缩写

Reference [github](https://github.com/SafeAILab/EAGLE)

## 投机采样基础

- **LLM decode process & the challenge**

  LLM 在 decode 阶段采用的是自回归式 decode，这样的方式每次 input tokens 数目只能为1，然后获得 next token，以此循环往复直至遇到 end of text token。用简洁的伪代码可以表示为如下

  ```python
  # decode_token is a single integer (1,)
  while True:
      # llm forward
      hidden_states = transformer(decode_token_id)
      logits = lm_head(hidden_states)
      
      # sample next token
      next_token_id = sampler(logits)
      
      # check end
      if next_token == end_of_text_token:
          break
      
      # continue
      decode_token_id = next_token_id
  ```

  这样的方式有一个明显的问题：整个过程计算强度低，成为 memory bound 场景。一个更形象的例子来说明：input tokens = 8 和 input tokens = 1 的 decode 时间是非常接近的，因为此时计算基本上不耗时，耗时的是读取/写入权重和激活值

- **Intuitive of Speculative Decoding**

  投机采样的核心思想，就是用一个 draft model 去猜测（speculate）后面的 tokens 是什么，猜完过后用原模型（本文有时候也会称原模型为 base model）去进行验证，如果猜对了，那就正好接受；如果猜错了，就扔掉猜错的 token。加速效果来自于并行计算了多个 input tokens，猜对得越多，加速越明显。可以粗略认为你用一次 decode 的时间，输出了多个接受的 next tokens，所以获得了加速效果

  下面用一个具体的例子，来展示整个过程

  

## EAGLE-1 Chain SD

## EAGLE-1 Tree SD

## EAGLE-2 Tree SD

## EAGLE-3 Train

## Question

- 如何进行 batched speculative sampling？

  挑战在于每一个 sequence 都有自己的 accept length，从而导致无法进行有效地 batch decode

  方案一：整个 batch 同步为最小的 accept-length。这样就能够仍然同步整个 batch 仍然增加相同的 token，持续进行 batch decode

  方案二：也许借鉴 continous batching 才是最终的解决方案？既然 continous batching 能够解决不同 sequence length 的 batch decode，那么这种情况放到投机采样上应该是最优解：每一个 sequence 都各自接收自己正确的投机 tokens，然后再开始下一次 decode。下图参考自 [blog](https://friendli.ai/blog/llm-iteration-batching) & [zhihu](https://zhuanlan.zhihu.com/p/680123256)

  <img src="EAGLE Speculative Decoding/v2-8092ac7d9ffc1eea2d2782d9a946b79e_b.webp" alt="动图" style="zoom: 80%;" />

## Concept

- EAGLE Model

  EAGLE model 实际上就是一个单层的 LLM，single layer transformer，这意味着 EAGLE 有所有 LLAMA 模型的功能，包括 kv cache, rope 等等，唯一一点不同的是：为了支持 tree attention，在计算 attention 的时候加入了 tree mask

  除此之外 EAGLE 还有一个 linear layer 用于将 `2 * hidden_size` 降维到 `hidden_size`

- Forward path of EAGLE

  EAGLE 使用了两个特征：`hidden_states & input_ids` 来预测 next hidden states。其中 `input_ids` 和 `hidden_states` 有着一个 token 的偏移

  <img src="EAGLE Speculative Decoding/image-20240813111345556.png" alt="image-20240813111345556" style="zoom:50%;" />

  ```python
  def forward(self, 
              hidden_states, 
              input_ids,
              tree_mask,
              position_ids = None, 
              past_key_values = None)：
  	"""
  	Args:
  		- hidden_states: (B, N, C) N input tokens's hidden states
  		- input_ids: (B, N), N next token ids
  		- tree_mask: (1, 1, N, M), N is input tokens, M is past input tree tokens
  		- position_ids: (B, N)
  		- past_key_values: Tuple of k and v states, (B, N', C)
  	"""
      inputs_embeds = self.embed_tokens(input_ids)	# (B, N, C)
      inputs_feat = torch.cat((inputs_embeds, hidden_states), dim=-1)
      hidden_states = self.fc(inputs_feat)	# (B, N, C)
      
      attn_mask = self.prepare_attn_mask(inputs_embeds, past_key_values, tree_mask)
      layer_outputs, past_kv_values = self.layer(hidden_states, attn_mask, position_ids, past_key_value)
      
      return layer_outputs, past_kv_values
  ```

- Draft tokens to generate tree prediction

  整个过程都写在了 `Model.topK_generate`

- Build tree mask

- Logit processor

## Layout

- Reranking is less important

- EAGLE1 vs EAGLE2

- EAGLE2 does not support batch inference right now, because the tree attention mask between different batch is not the same

  But EAGLE1 uses fix tree structure, so it supports batch inference, and it can increase 2x throughput at low batch size, because EAGLE is helpful at memory bound, when it becomse compute bound, it would not be so helpful 

- EAGLE1

## Question

- mask index is the same as the draft_parents?

  No

- 在使用 logits_processor 的时候如何处理多个 token input 使得得到的结果和 auto regressive 一致？

  验证过程得用一个 for 循环来验证，逐个验证即可保证结果和 auto regressive 一致