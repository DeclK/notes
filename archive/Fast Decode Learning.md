# Fast Decode Learning

## pyproject.toml

- 直接参考 python packaging user guide [Writing your `pyproject.toml`](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)

  通常我们使用 setuptools 来作为我们的打包工具 [Configuring setuptools using](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)

  以后打包只用简单写写这个配置文件就行了

  使用 setup.py + 环境变量来管理 csrc extension，用 pyproject.toml 来管理 python dependencies，使用 setuptools.packages 关键字来管理需要打包的文件 [link](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html#setuptools-specific-configuration)

  一个 example pyproject.toml

  ```txt
  [build-system]
  requires = ["setuptools>=61.0.0", "wheel"]
  build-backend = "setuptools.build_meta"

  [project]
  name = "FQT"
  version = "0.1.0"
  authors = [
      {name = "DeclK", email = "xxx@gmail.com"},
  ]
  description = "xxx"
  readme = "README.md"
  requires-python = ">=3.8"

  [tool.setuptools.packages.find]
  where = ["."]
  include = ["xxx"]
  ```

## Datasets of medusa

- raw 数据形式，以及输入到网络中的数据形式

  数据来源：[shareGPT](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered)

  数据格式：json

  当 load json 过后是一个 list of dict，每一个 dict 包含对话内容

  ```yaml
  id: 'QWJhYvA_0'
  conversations:
      - from: 'human'
        value: 'conversation content...'
      - from: 'gpt'
        value: 'conversation content...'
      - from: 'human'
        value: '...'
  ```

  代码使用了 asyncio 来快速进行处理，6.8w 条数据集在一秒多就处理完成了

  代码使用 `Conversation` 类来存储对话

  ```python
  from transformers import Conversation
  
  conv = Conversation()
  conv.add_message({'role': 'system', 'content': 'Hello! How can I help you today?'})
  conv.add_message({'role': 'user', 'content': 'I would like to book a flight.'})
  print(conv.message)
  ---
  [{'role': 'system', 'content': 'Hello! How can I help you today?'},
   {'role': 'user', 'content': 'I would like to book a flight.'}]
  ```

  role 只有两种：user & system & assistant，不能进行更改

  这三种角色的作用：

- tokenizer 常用

  - 配置 tokenizer 的参数

    - tokenizer class

      属于哪个 tokenzier 类，例如 LlamaTokenizer, GPT2Tokenizer

    - 对 tokenizer 长度的配置

      `model_max_length`，最大输出长度

      `padding_side`，left or right

      `truncation_size`，left or right

    - Special tokens 配置

      `add_bos_token`，true or false

      `add_eos_token`，true or false

      `special_tokens_map`，a dict

    - chat_template [link](https://huggingface.co/docs/transformers/chat_templating)

      下面的内容就是使用 vicuna 的 tokenzier chat template

      ```python 
      from transformers import Conversation
      
      conv = Conversation()
      conv.add_message({'role': 'user', 'content': 'Hello! How can I help you today?'})
      conv.add_message({'role': 'assistant', 'content': 'I would like to book a flight.'})
      
      chat = tokenizer.apply_chat_template(conv.messages, tokenize=False)
      print(chat)
      ```

      输出

      ```txt
      <s>[INST] Hello! How can I help you today? [/INST] I would like to book a flight. </s>
      ```

      可以看到加入了 `<s> & </s>` 来表示对话的开始和结束，加入 `[INST] & [/INST]` 表示 user message 的开始和结束，INST 应该是 instruct 的缩写

      除了 `tokenize=False` 参数外还有一个参数 `add_generation_prompt` 也是常用的，其功能就是在 prompt 的末尾加一个/一些 token，以表示接下来的内容将由 gpt 生成

      > This argument tells the template to add tokens that indicate the start of a bot response.

      > When using chat templates as input for model generation, it’s also a good idea to use `add_generation_prompt=True` to add a [generation prompt](https://huggingface.co/docs/transformers/chat_templating#what-are-generation-prompts).

      > When training, you should usually set `add_generation_prompt=False`, because the added tokens to prompt an assistant response will not be helpful during training.

      一个例子

      ```python
      tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
      """<|im_start|>user
      Hi there!<|im_end|>
      <|im_start|>assistant
      Nice to meet you!<|im_end|>
      <|im_start|>user
      Can I ask a question?<|im_end|>
      """
      
      tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
      """<|im_start|>user
      Hi there!<|im_end|>
      <|im_start|>assistant
      Nice to meet you!<|im_end|>
      <|im_start|>user
      Can I ask a question?<|im_end|>
      <|im_start|>assistant
      """
      ```

      可以看到多了一些 prompt：`<|im_start|>assistant`

    - Advanced: how do chat templates work?

      > The chat template for a model is stored on the `tokenizer.chat_template` attribute. If no chat template is set, the default template for that model class is used instead.
      >
      > You can find out what the default template for your tokenizer is by checking the `tokenizer.default_chat_template` attribute.

      使用 Jinja Template 语法来生成模版

  - `tokenizer.__call__`

    这肯定是 tokenizer 最常用的使用方式之一，在传入 text 的同时，还可以传入其他配置

    - `add_spectial_token` 是否加入 special tokens
    - `return_tensors`，tf or pt or np
    - `paddding`，longest or max_length
    - `return_offsets_mapping`，Whether or not to return `(char_start, char_end)` for each token。这会增加一个返回字段 `offset_mapping`，为一个 tensor (B, N, 2)，N 为 token 数量，2 就是该 token 在 input text string 中的位置，同时 special token 的 start & end 是一样的，表明无意义
    - `truncation`
    - `max_length`

    最后会输出一个类字典，可以通过 attr 直接获得 key's value：

    - `input_ids`，(B, N)，N 是 token 数量，如果输入的 text 为字符串（not a list of string），自动默认 B=1
    - `attention_mask` (B, N)，通常由 0,1 组成
    - `offset_mapping` (B, N, 2)，

  - `tokenizer.batch_decode & decode`

    是将 tokenizer id 列表转换成为输入的 text，batch_decode & decode 对 input_ids 形状有要求，前者要求有 batch 维度，后者则要求没有 batch 维度

  - `tokenizer.convert_ids_to_tokens & tokenizer.convert_tokens_to_ids`

    类似与 decode 效果，只是仍然保持了列表形式，例如其返回值为：`['I', 'am']`，而 decode 返回值则为字符串 `'I am'`

  - `tokenizer.tokenize`

    将一个字符串转换为一个 token list (not token id list)，有点类似于先使用 call 方法，然后再用 convert ids to tokens，但是仍然有一定区别，例如 call 过后会加入 special tokens 或者 padding 等等，而 `tokenizer.tokenize` 不会做这些处理，是单纯的 tokeinzer

- how to use transformers

  - safetensors, [link](https://huggingface.co/docs/safetensors/index)

    ```python
    # load
    from safetensors import safe_open
    
    tensors = {}
    with safe_open("model.safetensors", framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
            
    # save
    import torch
    from safetensors.torch import save_file
    
    tensors = {
        "embedding": torch.zeros((2, 2)),
        "attention": torch.zeros((2, 3))
    }
    save_file(tensors, "model.safetensors")
    ```

    和之前的 `.pth` 一样，用于存储 state dict，但是更快

  - `from_pretrained`

    一些常用参数，[link](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained)

    1. `pretrained_model_name_or_path`，所保存的模型路径
    2. `config`，一个 `PretrainedConfig`，会覆盖掉模型原本的 config
    3. `cache_dir`，用于存储从线上下载的模型
    4. `torch_type`，重写模型 dtype
    5. `device_map`，cuda or cpu
    6. `quantization_config`，需要配合 bitsandbytes

  - `save_pretrained`

    

## Question

- 如何给 tokenizer 加入 special tokens？config 中的那些 special tokenizer 是怎么保存的？

- 在 load 模型的过程中有两种数据类型，一种是 bf16 一种是 fp16，似乎 bf16 会在训练里使用，而 fp16 在推理里使用，该 load 哪一个？在训练的时候是否需要开启混合训练？

  [mixed-precision-training](https://huggingface.co/docs/transformers/v4.39.3/perf_train_gpu_one#mixed-precision-training)

  > If you have access to an Ampere or newer hardware you can use bf16 for mixed precision training and evaluation.

  根据 [huggingface discuss](https://discuss.huggingface.co/t/model-pre-training-precision-database-fp16-fp32-bf16/5671) 我们应当根据模型的 config 来选择使用哪个数据类型，保持 fine-tune 和 pretrain 训练格式的一致

- [WARNING: tokenization mismatch: 185 vs. 186. (ignored)](https://github.com/lm-sys/FastChat/issues/1290)

  原因在于不同的词在不同的位置时，分词结果也是不一样的

  `USER` 在 `</s>` 之后的分词就是 `USER` 本身，而在开头时，则会被分为两个子词 `US & ER`

  这就会导致在统计 `turn_len` 的时候出现错误，具体来说当 turn > 1 就会出错，也就是有两轮及以上的对话就会出错

  在 preprocess 代码里有下面一段，这里的 -2 操作让我很是迷惑

  ```python
  # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
  instruction_len = len(tokenizer(parts[0]).input_ids) - 2
  ```

  原因依然是由 tokenizer 引起的，由于 `parts[0]` 的末尾是空格，空格会单独成为一个 token

  但是在合并句子当中，空格被融合到了下一个 token 当中

  ```python
  >>> t.tokenize('I ')
  ['▁I', '▁']
  >>> t.tokenize('I am')
  ['▁I', '▁am']
  ```

  加上开头的 `<s>` 也需要减去，所以 `instruction_len` 要 -2

- transformers 是如何保存 model config 的，即 `save_pretrain & from_pretrain` 二者的流程是怎样？似乎 tokenizer 的保存过程会更复杂一些

- AutoModel 是如何实现寻找对应的模型的（仅限本地）

- How can I utilize ray?

  理解上面三个问题可能需要一两天的时间