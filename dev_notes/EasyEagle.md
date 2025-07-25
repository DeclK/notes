# EasyEagle

EasyEagle 是一个通用的 EAGLE 投机采样训练和推理框架

## HighLight

1. 基于 [Meta Lingua](https://github.com/facebookresearch/lingua)，该项目是一个高效简洁的大模型训练框架，EasyEagle 使用其作为训练引擎，对比原 EAGLE 仓库能够获得 x3 倍以上的硬件利用率。在训练过程中可使用的工具更多：FSDP，`torch.compile`，梯度累积，张量并行，tensorboard 可视化，日志系统
2. 用 ~400 行代码实现了完整的 transformer，相比于 transformers 库来说更加精简、可读、可 hack
3. 支持多模态 EAGLE 推理，这是原 EAGLE 仓库不具备的 
4. 具备强大的兼容性，能原生支持任意的 transformers 库当中的模型作为基础模型进行 EAGLE 投机采样。原 EAGLE 仓库需要对 transformers 库模型进行深度的 kv cache 修改，这对开发效率来说非常不友好

## Code Structure Overview

TODO: 思维导图

## Code Design

### Transformer

代码 `easyeagle/transformer.py`

实现了一个非常简洁的 modern transformer。精度和 transformers 库中的 Qwen2 模型进行过验证：使用相同的权重和输入，输出能够完全对齐

为什么要自己从零实现一个 transformer，而不是直接使用 huggingface or lingua 中的 transformer

1. [A bit philosophical question: Why this instead of HF ecosystem around Trainer?](https://github.com/facebookresearch/lingua/issues/7) 

   原因在于 huggingface 生态并不是针对于高性能而设计的，lingua 没有使用 huggingface Trainer & Model 的原因也在于此

2. Lingua 中的 llama 模型和 transformers 中的 llama 有一些差异，尤其是在 positional embedding 的设计上。由于我对 huggingface 当中的 Qwen 模型已经很熟悉，不希望在之后由于这些细微偏差而导致了一些隐蔽的 bug，所以也不打算使用 lingua llama 实现

综上 EasyEagle 选择从零开始，构建了一个 huggingface Qwen style 的现代 transformer 用于 EAGLE 模型的训练

### Modeling

代码 `easyeagle/modeling.py`

该文件是对 `EagelModel` 的具体实现，其中核心的 `self.model` 直接采用 `easyeagle/transformer.py`，其余还包含一些 norm & embed & fc layer

```python
class EagleModel(nn.Module):
    def __init__(self, config: EagleModelConfig):
        super().__init__()
        self.config = config
        self.model = BaseTransformer(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        if config.small_vocab_size is None:
            self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=True)
        else:
            # TODO: small vocab support
            raise NotImplementedError()
```

`EagleModel` 的 `__init__` 也参考 huggingface 代码风格：从 config 直接构建模型

### Dataset

代码 `easyeagle/dataset.py`

代码 `easyeagle/utils/data_recorder.py`

构建 EAGLE dataset 需要分为两步：

1. 使用 `DataRecorder` 对 base model 当中所产生的 hidden states & input embedding 进行保存。一个最典型的 EAGLE data 保存代码如下

   ```python
   from easyeagle.utils.data_recorder import DataRecorder
   from easyeagle.utils.io import dump
   
   model: Qwen2ForCausalLM
   # use 3 data recorder to save last hidden states & embed & input ids
   dc_hs = DataRecorder(model=model.lm_head)
   dc_embed = DataRecorder(model=model.model.layers[0])
   dc_ids = DataRecorder(model=model)
   
   dc_hs.register_input_hooks()
   dc_embed.register_input_hooks()
   dc_ids.register_input_ids_hooks()
   
   def save_for_eagle(path):
       item = {}
       item["input_embeds"] = dc_embed.inputs["model_input"]
       item["hidden_states"] = dc_hs.inputs["model_input"]
       item["input_ids"] = dc_ids.inputs["model_input_ids"]
       dump(item, path)
       # NOTE: it's important to clear the data recorder
       dc_embed.clear_all()
       dc_hs.clear_all()
       dc_ids.clear_all()
   
   # forward loop
   for i, data in enumerate(dataset):
       model(data)
       save_for_eagle(f"eagle_dataset/{i}.pkl")
   ```

2. 使用 `EagleStreamDataset` 进行模型训练。该数据集也受 lingua 启发，是一个 iterable dataset，能够进行无限地迭代

### Train & Eval

代码 `easyeagle/train.py`

基于 lingua 中的 `apps/main/train.py` 进行修改。主要更改包含：

1. 适配 FSDP EagleModel 
2. 适配 dataloader
3. 去除评测流程、Probe 等代码

另外需要注意的是：在训练过程中，模型会存储为 dcp (distributed checkpoint) 格式，这种格式不利于日常开发和使用，还需要额外使用如下代码进行转换为常规的 state dict 格式。这将在之后的代码中解决：除了 dcp 外，训练过程中会自动存储一份 `pytorch_model.bin` or `model.safetensors`

```python
# easyeagle/utils/hf_format.py
def dcp_to_state_dict(dcp_dir):
    from torch.distributed.checkpoint.format_utils import (FileSystemReader, _EmptyStateDictLoadPlanner, _load_state_dict, STATE_DICT_TYPE)
    sd: STATE_DICT_TYPE = {}
    _load_state_dict(
        sd,
        storage_reader= FileSystemReader(dcp_dir),
        load_planner=_EmptyStateDictLoadPlanner(),
        no_dist=True
    )
    return sd
```

代码 `easyeagle/eval.py`

用于评测所训练 EAGLE 模型的 top3 accuracy，需要使用常规的 `pytorch_model.bin` 模型，不接受 dcp 模型格式

### Generate

代码 `easyeagle/generate.py`

在评测过程中只有对 top3 accuracy 的统计，对于常见的投机采样指标：例如 decode steps per sample，则无法评估。要获得该指标 EasyEagle 设计了 `EagleLLM` 来进行投机采样生成。通过调用 `EagleLLM.generate` API 即可获得投机采样加速

`EagleLLM` 设计核心：

1. **牺牲复杂度以换取可读性和可扩展性。**在 `EagleLLM` 中没有实现 tree-decodeing，以避开对 kv cache & attention mask 的复杂管理，仅通过 temperature = 0 的采样和 chain decoding 进行投机采样。**该过程的算法完全可控，且不失投机采样的本质**，仍可以通过 decode steps per samples 指标来验证小模型的命中率。并且在日常使用过程中，chain decoding 能够满足大部分的加速需求，并且在 serving 场景中由于 batch 的增加，tree-decoding 的收益也许不会太多？
2. **原生支持 transformers 库中的模型作为 base model。**得益于牺牲了 tree-decoding，EasyEalge 可以不去额外实现一个 base model，并修改其 kv cache & attention mask 管理。EasyEagle 可直接使用 `AutoModelForCausalLM.from_pretrained` 方式构建 base model。除了语言模型以外，多模态模型 `AutoModelForVision2Seq` 也能够被直接支持
3. **构建一个通用的 KV Cache 系统以在 EasyEagle 和 transformers 库中共同使用。**在投机采样 generate 过程中必须要对两套 kv cache 进行维护：主模型+小模型，而 transformers kv cache 并为支持 `pop` 操作，所以 EasyEagle 实现了一套简易的静态 kv cache 系统，用于管理主模型和小模型在投机采样过程中的 kv cache

在以上设计的指导下，EasyEagle 成功构建了一个扩展性极强的投机采样推理框架

## User Guide

### Install

```shell
pip install -e . --no-build-isolation
```

### Usage

1. Train

   单卡训练/debug

   ```shell
   python easyeagle/train.py config=easyeagle/config/eagle.yaml
   ```

   多卡训练

   ```shell
   CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run  easyeagle/train.py config=easyeagle/config/eagle.yaml
   ```

   其中 `eagle.yaml` 也很简洁易懂，一个 EAGLE-Qwen2.5VL 的配置如下：

   ```yaml
   dump_dir: "eagle2-qwen2.5-vl-7b"
   name: "EAGLE2-Qwen2.5-VL-7B"
   steps: 10000
   seed: 777
   optim:
       lr: 3e-4
       warmup: 500
       lr_min_ratio: 0.000001
       clip: 10.0
   
   distributed:
       fsdp_type: no_shard
       compile: true
       model_dtype: bf16
       matmul_allow_tf32: false
       selective_activation_checkpointing: false
       tp_size: 1
   
   model:
       hidden_size: 3584
       intermediate_size: 18944
       n_layers: 1
       n_heads: 28
       n_kv_heads: 4
       norm_eps: 1e-6
       rope_theta: 10000
       attn_bias: false
       vocab_size: 152064
       base_model_path: "Qwen2.5-VL-7B-Instruct"
   
   data:
       root_dir: "eagle_dataset/train"
       batch_size: 4
       num_workers: 2
       seed: 777
   
   profiling:
       run: false
   
   checkpoint:
       dump:
           every: 2000
           keep: 1
       eval:
           every: 2000
           keep: 1
   
   logging:
       freq: 50
       tensorboard: true
   ```

2. Eval

   单卡评测/debug

   ```shell
   python easyeagle/eval.py config=easyeagle/config/eval.yaml
   ```

   多卡评测

   ```shell
   CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run  easyeagle/eval.py config=easyeagle/config/eval.yaml
   ```

3. Generate

   EasyEagle 提供了一个推理示例脚本 `projects/qwen2vl.py`，~30行代码

   ```python
   from easyeagle.generate import EagleLLM
   from tqdm import tqdm
   from easyeagle.utils.io import load, dump
   from pathlib import Path
   import torch
   
   def main():
       model = "eagle2-qwen2.5-vl-7b"
       data_path = "eagle_dataset/test"
       LLM = EagleLLM.from_pretrained(model)
       LLM.cuda()
       image_grid_thw = torch.tensor([[1, 36, 92]]).cuda()
   
       all_steps = []
       for item in tqdm(list(Path(data_path).rglob('*.pkl'))):
           data = load(item)
           input_embeds = data["input_embeds"][0].cuda()
           input_ids = data["input_ids"][0].cuda()
   
           output, decode_steps, total_tokens = LLM.generate(
               input_ids=input_ids,
               input_embeds=input_embeds,
               log=True,
               image_grid_thw=image_grid_thw
           )
           all_steps.append((decode_steps, total_tokens))
           text = LLM.tokenizer.decode(output[0])
           print(text)
       dump(all_steps, "all_steps.pkl")
   
   if __name__ == "__main__":
       main()
   ```

   其中的 `image_grid_thw` 是 Qwen2.5VL 在具体任务中的指定参数，以上推理代码具有一般性

## Future Work

1. **小词表原生支持**
2. **EAGLE3 实现**
3. 数据增广
4. 利用 Producer-Consumer 模型实现 Dataloader
5. 多轮对话支持
6. 融入 transformers 生态