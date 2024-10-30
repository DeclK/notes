# Tokenizer

参考 Andrej Karpathy 的视频整理

- 视频中推荐了一个[tokenizer website](https://tiktokenizer.vercel.app/)，可以选择不同的 tokenizer，并查看他们的结果

- 为什么不直接选用 UTF-8 encoding (character level)

  因为词表很小，就会让 token sequence 变得很长，不利于注意力的计算。最近也有一篇不需要 tokenizer 的论文，使用的是层级的注意力来解决，但该方法没有得到更多的验证

- 解释了 GPT2 tokenizer 面对 python scripts 的缺点

  所有的空格都是单独的 token，在 python 中有很多的缩进，这就会产生不必要的 token

- Byte-Pair Encoding [wiki](https://en.wikipedia.org/wiki/Byte_pair_encoding)

  BPE 算法思路：将一些常见的词（也称为 bytes）进行两两合并，合并过后的词作为新词加入到 vocab 当中。wiki 中的例子很好地说明了 BPE 的思想

  接下来就要动手实现 BPE 算法了！

  1. 寻找最常见的 pair

     ```python
     def get_stats(ids):
         counts = {}
         for pair in zip(ids, ids[1:]):
             counts[pair] = counts.get(pair, 0) + 1
         return counts
     ```

  2. 给 pair 一个新的 id，并替换原来分开 token 的 id

     ```python
     def merge(ids, pair, pair_id):
         newids = []
         while i < len(ids):
             if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                 newids.append(pair_id)
                 i += 2
             else:
                 newids.append(ids[i])
                 i += 1
     	return newids
     ```

  3. 循环往复进行融合，知道压缩到期望的 token 数量

  这里 Andrej 举了一个很形象的比喻

  > BPE 有点类似于树，但又不是树。tree 是有一个 root，root 下有 node，而 BPE 是从 node 出发，然后向上生成 parent node

  通过 BPE 这样的压缩算法，很容易受到你的训练集组成的影响，如果你的语料库里英文较多，那么英文被压缩的概率就会更大

  