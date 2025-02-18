## EAGLE Speculative Decoding

[github](https://github.com/SafeAILab/EAGLE)

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