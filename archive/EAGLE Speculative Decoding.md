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

  投机采样的核心思想，就是用一个 draft model 去猜测（speculate）后面的 tokens 是什么，猜完过后用原模型（本文有时候也会称原模型为 base model）去进行验证，如果猜对了，那就正好接受；如果猜错了，就扔掉猜错的 token。加速效果来自于并行计算了多个 input tokens，猜对得越多，加速越明显。具体来说：**投机采样少量的 draft model 猜测时间 + 用一次 decode 的时间用于验证，输出了多个正确的 next tokens，所以节省了多次 decode 的时间，获得了加速效果**

  下面我将用一个具体的例子，来展示整个投机采样的过程。假设 prompt 为三个单词：`How can I`，并且假设 LLM 最终的 decode 输出为 `learn eagle speculative decoding well?`，我用以下图示来表示一般的 LLM decoding 过程

  <img src="EAGLE Speculative Decoding/image-20250328225651677.png" alt="image-20250328225651677" style="zoom:80%;" />
  
  **我们可以简单地把 LLM (i.e. Base Model) 看作是一个 next token prediction machine**，也就是说你给进去任何 token，它都会基于历史状态给你预测下一个 token 是什么。这个想法将会简化 LLM 模型，帮助我们理解整个 speculative decoding 过程。简要描述下上图的过程：在 prefill 中输入 `How can I` 预测出了 3 个 next tokens，但是我们只关注最后一个，即由 `I` 预测得到的 next token `learn`。然后进入 decode 阶段，我们使用 `learn` 去预测得到下一个 token `eagle`，用 `eagle` 去预测得到 `speculative`，...，如此循环下去获得最终完整的句子 `How can I learn eagle speculative decoding well?`
  
  OK，现在来看看用投机采样整个过程可能是什么样的？这里就开始引入 draft model 了，该 draft model 也是一个 LLM，只不过比原来的 baes model 要小很多，**但是不妨碍其本质是一个 next token prediction machine**
  
  <img src="EAGLE Speculative Decoding/image-20250328231636101.png" alt="image-20250328231636101" style="zoom:80%;" />
  
  Draft model 拿到一个初始的 token `learn`，开始了自己的自回归过程，获得了许多 draft tokens `eagle speculative decoding better? ...`。这些 draft tokens 就是 draft model 去猜测 base model 接下来会生成的词。问题来了，怎么知道 draft model 猜得对不对呢？这就需要一个验证过程。验证过程的第一步就是将这些 draft tokens 连同 initial token **一起**输入到 base model 中，如下图所示
  
  <img src="EAGLE Speculative Decoding/image-20250329002507107.png" alt="image-20250329002507107" style="zoom:80%;" />
  
  当这些 tokens 输入到 base model 过后，每一个 token 都会产生自己对应的 next token（再次强调，base model 的本质是一个 next token prediction machine）。此外你可以注意到这个图中没有虚线进行连接，**这表示该过程并非自回归的，而是并行的**。现在我们有了 base model 预测的 next token，我们就可以和由 draft model 所产生的 draft tokens 进行对比，看下是否猜对了
  
  <img src="EAGLE Speculative Decoding/image-20250329002637632.png" alt="image-20250329002637632" style="zoom:80%;" />
  
  可以看到，**绿色的线条代表我们的 draft tokens 和真实的 base model 所预测的 next tokens 是一样的，红色线条就代表二者并不匹配**。所以说在这个 case 当中 draft model 猜对了三个词 `eagle speculative decoding`。但是最后一个词 `well?`没有猜对，猜的是 `better?`，并且一旦一个词猜错过后，后面的所有 draft tokens 我们都认为是错误的，所以直接忽略掉
  
  虽然最后一个词没有猜对，但是由于 `decoding` 这个词是猜对的，那么由 base model 预测得到的 `well?` 就是正确的 next token。正是这个性质，保证了投机采样的下限就是：**一定获得一个正确的 next token**。这种情况我用下图表示
  
  <img src="EAGLE Speculative Decoding/image-20250329002706795.png" alt="image-20250329002706795" style="zoom:67%;" />
  
  可以看到，即使 draft tokens 生成的是 garbage 导致一个都对不上。但由于 `learn` 这个词是 initial token，由 base model 正常生成，所以 `eagle` 这个词一定是正确的 next token
  
  注意：在完成了 verify 过后，我们需要将正确的 tokens 进行接收，错误的 tokens 进行删除。这里主要是对 KV Cache 进行操作，将错误的 KV Cache 进行删除，保留正确 tokens 的 KV Cache
  
  最后，生成还没有结束，因为我们现在获得的最新的 next token 是 `well?`，还没有遇到 `<endoftext>`，所以继续进行下一轮的投机采样：
  
  1. 将 initial token (`well?` in this case) 输入到 draft model 当中，生成 draft tokens
  2. 将 initial token & draft tokens 输入到 base model 中，生成 base model 预测的 next token
  3. 将 base model 预测的 next token 和 draft tokens 进行对比验证
  4. 如果没有出现 `<endoftext>` 回到 Step1
  
  <img src="EAGLE Speculative Decoding/image-20250328235339912.png" alt="image-20250328235339912" style="zoom:80%;" />

## EAGLE-1



## EAGLE-1 Tree

## EAGLE-2 Tree

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