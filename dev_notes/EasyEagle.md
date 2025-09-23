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

## Producer-Consumer Dataloader

放弃传统 pytorch dataloader 实现了 producer-consumer dataloader 以更好地适应 iteration based training & 训练 resume，缺点是没有使用多进程，producer 是一个单进程，多进程还需要考虑他们之间的排序。不过好消息是单进程即可满足绝大多数的场景，除非 GPU 的吞吐极其快，超过了 CPU 的能力

1. python multi-processing

   在 Python 中使用多进程需要使用 `multiprocessing ` 模块中的 `Process`

   ```python
   from multiprocessing import Process
   ```

   每创建一个 `Process` 实例就是创建了一个子进程，利用 `start & join` 方法来启动子进程和等待子进程

   ```python
       p = multiprocessing.Process(target=worker, args=[params_needed_by_worker,...])
       p.start()	# non-blocking
       p.join()	# wait until process finish
   ```

   通常可以使用一个 list 来管理多进程

   ```python
   process_list = []
   for i in range(num_process):
       p = Process(target=worker, ...)
       p.start()
       process_list.append(p)
       
   for p in process_list:
       p.join()	# wait for every process to finish
   ```

2. contextmanager combined with generator

   lingua 使用了把 contextmanager 和 generator 合并了起来，二者都使用了 `yeild` 语法，这里会比较容易混淆

   ```python
   @contextlib.contextmanager
   def async_iterator(buffer_size: int, iterator_builder):
       ...	# prepare code
       consumer = consume_buffer(producer, queue)
       yield consumer	# return a generator
       ... # clean code
       
   def consume_buffer(producer: Process, queue: Queue):
       while producer.exitcode is None:
           try:
               item = queue.get(timeout=0.1)  # Tries to get an item from the queue with a timeout
               yield item
           except Empty:
               pass
   
       raise RuntimeError("Data loader quit unexpectedly, real error has been raised previously")
   ```

   在实际使用过程中会使用 `with` 语句或者 `enter_context` 来返回 iterator，在退出 context 的时候就会触发 clean code 逻辑

   ```python
   with async_iterator(producer, queue) as iterator:
       data = next(iterator)
   ```

   使用生成器语法糖的函数，其实可以看做一个特殊的类。我们需要先对这个类进行实例化（即传入参数），构建好生成器，在这个过程中不会执行任何函数中的代码。然后再使用 `next` 方法输出每一次 yield 的返回值

   ```python
   generator = consume_buffer(producer, queue)	# build a generator, will not execute any code
   item = next(generator)
   ```

   区别于 iterator 似乎还需要先调用 `iter` 方法，让 iterator 初始化，然后再使用 `next`

3. 构建 resumable dataloader

   我的思路：记录 rank0 的 step & rng，保证每一个 rank 的 data list 数量是相等的。有 step & rng 可以直接推断得到当前 epoch 的随机顺序，然后通过 `step % len(data)` 来获得当前 iteration index

缺点：

1. 对于 unpickable 的对象无法使用 multi-process。这对于 flex attention block mask 来说就是如此。我把 mask 利用类的形式进行重写使得其能 pickable，从而把编译移动到 dataloader 当中来 overlap latency，在之后发现这样做会降低 flex attention 的运行时间，得不偿失。
2. （已解决）multiprocess 中的 `Queue.get` 真的很慢，这仍然是受制于 python 进程通信问题，data 需要在不同的进程间进行传输，随着数据量的增加（~40MB 量级），这个 overhead 非常大！这个问题在使用了 `threading` 模块后彻底解决，因为线程的资源和主进程是共用的，根据 [zhihu-天清](https://www.zhihu.com/question/516209908) 的回答，科学计算库即使在线程中还是会自动使用多核计算，即使 thread 在 GIL 的限制下，加速效率仍然很高。最后加入 `pin_memory` 优化，async producer-consumer dataloader 和 pytorch multi-worker dataloader 的相同，相比优化前提升了 20x (30ms vs 1.5ms)
3. 改为了 threading 放弃 multiprocess 过后，也就放弃了 multiprocess 的优点。当 GPU 的吞吐能力大于 CPU 时，此时无解
4. （已解决）由于顺序性的 generator，无法利用 num workers 同时加载多个数据。但是我可以通过在 iterable dataset 中添加一个预取数据机制，这样通过多线程的优势，能够把数据 loading 的成本快速下降

优点：

1. 能够完美地进行 resume
2. producer-consumer 逻辑足够简单也足够强大，性能瓶颈并不来源于此

## EAGLE Scaling

在 [issue](https://github.com/sgl-project/SpecForge/issues/93) 中提及了一个相关工作 [Scaling Laws for Speculative Decoding](https://arxiv.org/abs/2505.07858)，另外也有一个 [issue](https://github.com/SafeAILab/EAGLE/issues/220) 提到了模型结构改进可能对 scaling 有所影响。简单来说 Pretrain & SFT & model capacity 在 draft model 上都有 scaling law，然后根据 GPU 的 roofline 模型可以得到投机采样时 batch size 的最优值

在此也记录下在多模态上进行 EAGLE3 scaling law 中遇到的现象

1. loss 在后期有上升

   [zhihu](https://www.zhihu.com/question/415931517) 中的回答我认为比较合理。这种 loss 震荡原因可能是因为有两个极小值点，但是这两个极小值点的 loss 相差非常大，所以在这样的情况下，会出现 loss 的震荡

2. 如何判断收敛

   一般从 grad norm 来判断模型的收敛更加容易，因为这是一条更平滑的曲线（相比于 loss 曲线来说）。在我的实验场景中 grad norm 在 0.5 过后就不再会明显下降了。另外 grad norm 不应该为0，这意味着网络中的梯度消失，而且由于权重衰减的存在，也阻止了这一现象的发生

3. norm before lm head

   在 lm head 之前加了一个 RMSNorm，初始的 loss 直接爆炸了，训练无法收敛。参考链接 [PreNorm & PostNorm](https://kexue.fm/archives/9009) [Transformer 初始化](https://kexue.fm/archives/8620) [Bert 初始标准差为什么是 0.02](https://kexue.fm/archives/8747)

   这可能因为 lm head 是冻住的，对于输入前的分布有比较特别的要求，一般来说最后的 RMSNorm 的 scale 可能会比较大（10+）。而我们初始化的 RMSNorm 全部 scale 为 1，扰乱了分布

4. scaling 结论

   **3.6x of the data, 10% improved the accept length**

   以上是我的实验场景中的结论，效果还是非常显著的。需要注意的是，一些论文在计算 acc len benchmark 时会开启 tree attention，而我的实验中不会开启

   我在一开始的 scaling 实验中得到了错误结论：4x of the data, 3% improved the acc len。其中有三个原因：

   1. 所保存的 checkpoint 处于 loss 的震荡区域，没有完全收敛！这是导致错误结论的最根本原因
   2. 设置的 threashold 过高（0.7）
   3. 学习率过大（1e-3），模型效果没有调整到最优

   现在又出现一个新的结果：我使用了更多的数据 (1.8x)，但是没有获得更好的结果，我改变了 batch size (4->2) & eagle3 length (5->7)，最终的 accept length 变化为 (4.8->4.5)。更改 batch size 主要是显存因素考虑，我认为理论上 batch size 应该没有这么大的影响。我把实验配置改回 batch size 4, eagle3 length 5，结果又出奇的好，accept length 为了 5.5。难道 batch size 的力量真的有这么大？还是 eagle3 length 在起作用？还需要补充一个实验 ablation 实验

5. top1 accuracy 不是绝对的指标

   训练出来的有的模型虽然第一个 token 的命中 accuracy 不是最高的，但是整体的 accept length 还要更高。还是应当综合考虑

6. **EAGLE2 has scaling law too**

   目前从其他人的结论来看：EAGLE2 也具备 scaling 性质，但是在官方 EAGLE2 的实现设置下无法展现。我正在尝试复现其中的 scaling 性质，我先用我当前的代码跑了一下，没有发现 scaling 性质，不过我发现我的实现与有两个显著区别：

   1. `p_loss & v_loss` 的权重

      在我的代码中 `p_loss` 权重为1，`v_loss` 权重为 0.1，但实际上是反过来的。也就是说 EAGLE2 更看重对 `v_loss` 的模仿

   2. data augmentation

      我的代码中没有 data augmentation

   通过修改 loss，最终得到了非常好的结果（accept len 3.5->4.5）。所以我没有进一步实验 data augmentation 的影响。这说明了 loss 的设置对于模型的 scaling 有着深刻的影响，而这个 loss 也说明了 EAGLE2 本质上是对  hidden states 的模仿，对 logits 的学习并没有发挥多大作用
   
   而 EAGLE3 则全面转向了 logits 的学习，而我最初的设置也许可以近似于 EAGLE3 step=1 的情况，在此限制下模型无法对 step > 1 的 token 进行有效预测，因此目标的不匹配也限制了模型的能力，即使 scale up data 也无法获得好的投机采样效果
   
   我认为对 logits 的学习是更难的，但是这解除了对 base model 的特征模仿限制，打开了更多的可能性：使用融合特征作为初始条件进行投机采样（EAGLE3 的另一个优化）

## Future Work

1. **小词表原生支持**
   - 已支持小词表推理，小词表训练需要重构。对不在小词表中的 output token 进行 loss masking
2. ✅**EAGLE3 实现**
3. 数据增广
4. ✅利用 Producer-Consumer 模型实现 Dataloader
5. 多轮对话支持
6. ✅融入 transformers 生态