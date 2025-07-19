# EASY EAGLE

准备构建一个 repo 用来高效且灵活地构建 EAGLE 投机采样

想要完成的事情：

1. 使用 lingua 中高效且简洁的训练代码，scalable & hackable

   顺便学习一下目前训练框架中的常用技巧：FSDP & tensor parallel & torch compile

2. 构建 non-tree eagle decode

   在常用任务中，non-tree eagle 基本能够满足需求，尤其是在特定任务中。并且目前云端 serve engine 对于 tree eagle decode 支持也没有很迫切

   non-tree decode 会极大简化投机采样的实现，方便用户理解和集成

3. 将 lingua 训练格式集成到 huggingface transformers ecosystem

   [A bit philosophical question: Why this instead of HF ecosystem around Trainer?](https://github.com/facebookresearch/lingua/issues/7)

   主要有两个原因：efficient & hackable，这两点都不能够由 huggingface trainer 完成，所以 lingua 选择实现了自己的 transformer 结构。这也是很好的，因为现在 transformer 结构都是固定的，在模型侧没有太多需要更改的部分，如果有，请自己实现！

代码结构

```txt
```

## Lingua Code

- data.py

  没有使用 pytorch dataloader，而是自己实现的 streaming dataloader，专门为 llm pretrain 而设计

  主要功能：

  1. 能够从断点中恢复数据的加载顺序，以保证可复现性
  2. 将 tokens pack 成为 `(seq_len, n_views)`，其中 `n_views` 代表 shift 数量，理解为向未来 shift `n_views` 个 tokens
  3. 自带 buffer，利用这次 pack 中剩余的 tokens
  4. 分配不同数据集之间的比例，例如 arxiv 数据集 20%，wiki 数据集 40%...

  在 llm pretraining 中不考虑不同 sentences/sources 之间的 mask，直接当成一个整体进行 next token prediction task

- generate.py

  似乎其中设计了可以并行解码 + mask 的一些技巧，需要学习

- transformer.py

  定义了 llama transformer，如果我们使用的是 qwen 的话，应该有一定不同。并且根据 [issue](https://github.com/facebookresearch/lingua/issues/64) 转换到 huggingface 格式还需要对 linear 层做一下 permute

  我将其中的 transformer 全部按照 huggingface qwen 的格式进行了重构（模型命名，配置，旋转编码实现），这样在迁移的过程中尽可能减少匹配出错

  用更简单的实现重构了其中的 kv cache，不考虑其中的 packed transformer generate

- train.py

  这里面有很多工作需要完成，omega conf，xformers profiler, tensor parallel settings 
  
- 完成 qwen model 的构建。并且和 hugging face 精度对齐 1e-3 尺度

- tensor parallelize

  如果一个模型太大无法放在一个 GPU 上，就可以通过 TP (tensor parallel) 来解决：将模型切分到多个 GPU 上

  我将以 transformer 中的 Attention & MLP 为例子，解释 tp 过程，从而理解 tp 原理

  1. ColwiseParallel

     将 NK weight 按照 N 维度切分，并且最后不会 gather

  2. RowwiseParallel

     将 MK weight 按照 K 维度切分，最后会 gather，把所有 rank 的结果叠加起来（looks like a Big K iteration）。通常 Rowwise 的输入会是经过 colwise 切分后的输出

  3. SequenceParallel

     在 seq 维度上将 input tensor 进行切分 `(B, N, C) -> DTensor (B, n, C)`，输出显然是 sharded tensor

  4. PrepareModuleInput

     一般用于将 sharded input 进行 gather，通常在 rmsnorm 过后使用，用于将 Sequence Prallel 过后的结果进行 gather

  搭建 transformer tensor parallelism

  figure 示意图说明 matmal 在进行 

- parallelize_model & tp_parallelize

- 把 wandb 换成 tensorboard，wandb 需要联网，如果不能联网的话很难发挥其优势

- preemption 

  > Preemption in computing means interrupting a running process to allow something else to take over the resources. Think of it like this: you’re working on a long task at a shared desk, and someone with a higher-priority task taps you on the shoulder and says, “I need the desk now.” You’d save your work and step away, right? That’s preemption in a nutshell.

  `init_signal_handler(set_preemption_flag)` 的作用：在特殊 signal 被进程接受过后，执行函数 `set_preemption_flag`，将全局变量 `preemption_flag = dict(flag=False)` 设置为 `True`

  在训练代码中会有部分是针对该 flag 的判断，如果 flag 为 True，就会保存 checkpoint 然后停止程序运行

  但尚不清楚这个信号是如何发射的，Grok 说可以由下面的代码发出

  ```shell 
  kill -SIGUSR2 <process_id>
  ```

- device mesh

  [device mesh doc](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html) [DTensor](https://docs.pytorch.org/docs/2.5/distributed.tensor.html)

  why called dp_shard, even though the data is not sharded...难道数据并行的意思不就是多份数据的意思吗？

  fsdp config 是什么作用？[doc](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-to-use-fsdp2)

  dp shard vs tp: 需要先进行 tp 然后再进行 dp shard，dp shard 会在每次 forward 之前进行 all gather，把所有的参数补齐，然后再进行计算，计算完整过后丢弃这些参数，利用通信来交换存储空间。似乎这两个二选一就行了？

- skipping the probing & profiler so we focus on the training, learn it later...

- 构建了一个 streaming dataset (iter-based)，考虑多进程真的很难，需要对临界条件进行仔细的讨论，增加变量来控制进程行为

- flex attention 和 sdpa 速度对比，flex attention 使用方法 [doc](https://pytorch.org/blog/flexattention/)

- enter stack 用于创建多个 context manager 环境

- omega conf 常用方式：command line

  会遇到 Incompatible value 'None' for field of type 'int'

  解决方法有两个：一个是加入 Optional 标签，另一个就是使用 default factory

  What is the position of omega conf? How can it best be used in our project?

- python -m torch.distributed.run to replace torchrun in venv[issue](https://github.com/pytorch/pytorch/issues/92132), uv set pip url [issue](https://github.com/astral-sh/uv/issues/1404)

- fsdp 必须要做两次 fully shard 这样 optimzer 才会正常工作，否则将不会接受梯度更新

  在 mix precision 过程中，会自动使用 bf16 进行训练（把权重 cast to bfloat16），并且保留 RMSNorm 的 float32

  fsdp does not support run submodule!!!

- distributed checkpointing

  与 torch.load 不相同，dcp.load 需要额外提供 `state_dict` 作为容器来承接 DTensor，不能只有 checkpoint path。另外 dcp 是一个目录，而不是一个文件

- a new interface to manage the multi-process GPUs：DeviceMesh

- barrier & all_reduce

  疑惑是否需要在所有 all reduce 之前添加 barrier

  > From DeepSeek
  >
  > No, **you should not call `torch.distributed.barrier()` before `torch.distributed.all_reduce`** to ensure synchronization. 
  >
  > `barrier` is useful in scenarios **unrelated to collective communication**, such as:
  >
  > - **Data Loading**: Ensure only one process downloads/preprocesses data. Other processes wait via `barrier` until the main process completes this task 
  > - **Checkpointing**: Synchronize processes before saving a model to avoid file conflicts

- how to make model init to be consistent in multi rank situations?

  dtensor would take care of all that, and it would seems like a full one tensor, not sharded!

  use `distribute_tensor` to distribute global weights to different ranks! I  use `DTensor.from_local`, it assumes that what you give is a local (not global!) tensor, i.e. shared tensor content

- context manager

  ```python
  from contextlib import contextmanager
  
  @contextmanager
  def my_context_manager():
      print("Entering the context")  # Equivalent to __enter__
      try:
          yield "resource"  # Value assigned to 'as' variable
      finally:
          print("Exiting the context")  # Equivalent to __exit__
  
  # Usage
  with my_context_manager() as resource:
      print(f"Using {resource} inside the context")
  ```

  经过测试，在我的场景下，没有启动多进程的 dataloader loading 时间小于 GPU 时延，所以单独使用一个进程作为 Producer 能够满足吞吐要求