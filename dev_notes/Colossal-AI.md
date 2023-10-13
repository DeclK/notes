# Colossal-AI

[ColossalAI](https://github.com/hpcaitech/ColossalAI)

希望解决的问题：

- [x] 使用 ColossalAI 构建好完整的训练-推理框架
- [ ] 了解 huggingface 在这个框架中所扮演的角色，熟悉 huggingface 的使用方法，包括两个库：transformers & accelerate & huggingface_hub

## Concept

- Colossal-AI

  Colossal-AI 是一个 **Deep Learning System**，用于提供高效的并行训练。其目标是支持分布式 GPU 训练，但不需要对你的单 GPU 代码进行太多的修改

- Linear speedup, scalability of distributed systems

  对于分布式系统来说，我们希望其具有很好的扩展性，这种扩展性通常是线性的。即我们扩大4倍的计算机数量，就能加速4倍。但是实际上由于计算机之间的通信，线性加速可能不能实现，这就需要好的系统设计来帮助实现线性加速甚至超越线性加速。

- Host, port, rank, world size, process group, nodes

  这几个概念在 [link](https://colossalai.org/docs/concepts/distributed_training#basic-concepts-in-distributed-training) 中清楚的解释了

  其中 node 就是一个 machine，一个 machine 可以包含多个 GPU

  一个 GPU 通常只对应一个 process

  默认的 process group 是所有的 GPU，同时也可以创建新的 process group 其只包含部分 GPU，但是其 process id 是保持不变的。所以 process group 是一个逻辑上的概念

  rank 和 world size 的概念都是针对某个 process group 的。对不同的 process group 可能有不同的值

  ![image-20231008171659165](/home/lixiang/Projects/notes/dev_notes/Colossal-AI/image-20231008171659165.png)

- communicate ways

  在一个 process group 中有两种交流方式：peer-to-peer 和 collective

  前者是 process 间的交流，后者是整个 group 的交流（scatter, gather, all-reduce...）

- ZeRO

  ZeRO（Zero Redundancy Optimizer）是一种旨在优化大规模深度学习模型训练的方法。ZeRO 的主要目标是解决模型和优化器状态（如梯度和动量）存储效率低下的问题。

  ZeRO 的主要思想是，通过将模型参数、优化器状态和梯度分布存储在多个计算节点上，降低单个节点的内存负担，从而大大提高存储的效率。

  需要注意的是，ZeRO 增加了通信开销，因为节点之间需要交换更多的数据。

- Parallelism on Heterogeneous System

  GPU Memory 不够大，那就将 GPU 存储的地方换到更大的 CPU 上或者 NVMe Disk 上

- booster, plugin

  > Booster is a high-level API for training neural networks. It provides a unified interface for training with different precision, accelerator, and plugin
  >
  > Plugin is an important component that manages parallel configuration: HybridParallelPlugin, GeminiPlugin, TorchDDPPlugin...

  booster 类似于 pytorch DDP，要将模型、dataloader、optimizer 等组件送进去，然后输出对应的分布式模型、dataloader、optimizer...

  可以说 booster 的主要成分就是 plugin，而 plugin 决定了所使用的分布式训练策略。不同的 Plugin 可以适配不同大小的模型

  | Plugins         | Params |
  | --------------- | ------ |
  | Torch DDP       | <2B    |
  | Torch FSDP      | <10B   |
  | Gemini          | >10B   |
  | Hybrid Parallel | >60B   |

- AMP

  automatic mixed precision，这个技术出来挺久了，能够自动调整模型的参数精度，从而达到节省显存的目的，以下是 GPT 说明的 AMP 过程：

  1. 首先，将模型的参数及输入数据从 FP32 转为 FP16。
  2. 然后进行前向计算，计算损失函数。
  3. 损失乘以给定的缩放因子（Dynamic Loss Scaling），进行反向传播计算梯度。缩放因子用于防止数值下溢问题，因为 fp16 表示范围很小，如果梯度值比较小的话会直接表示为0，所以在 loss 处乘以一个较大的数就能够保留梯度信息，最后再在 fp32 表示时缩放回去就能够得到较准确的梯度值
  4. 如果在反向传播过程中没有发生梯度消失，将梯度的值从 FP16 转回到 FP32，进行权重更新。
  5. 如果在反向传播过程中发生了梯度消失，AMP 会自动增加缩放因子，然后重新进行反向传播计算梯度。
  6. 如果产生了梯度爆炸的情况，会忽略此次更新，并尝试减小缩放因子

- Gemini in Colossal AI

  > A Dynamic heterogeneous memory space manager for parameters, gradients and optimizer states.

- Sequence Parallel

  类似于 data parallel，经常在语言模型中出现，这是由于 self-attention 的平方复杂度导致的。所以为了计算更长的序列，将采用 sequence parallel 并行处理以保持计算高效

  另外 sequence parallel 也可以用于减少 activation recomputation 当中，这可能是其主要用法

- Pipeline Parallel

  简单理解就是 pipeline + model parallelism，下图从 GPipe paper 中展示了。model parallelism 就是将模型进行切分，然后把每个子模型放到一个 GPU 上，这就是 model parallelism。但是这将产生 GPU 空闲的问题，为了减缓这个问题，可以使用 pipeline 来更多的 GPU 同时运行

  ![mp-pp](/home/lixiang/Projects/notes/dev_notes/Colossal-AI/parallelism-gpipe-bubble.png)

- Activation Checkpoint

  也被称为 Gradient Checkpoint

  > *Activation checkpointing* (or *gradient checkpointing*) is a technique to reduce memory usage by clearing activations of certain layers and recomputing them during a backward pass.

  具体来说对于某个模块，将其输入和输出值保存下来（Checkpointing Layer），模块中计算产生的其他中间张量都被清空，这样就能省去一大部分的内存。然后在进行反向传播的时候又重新算一遍

## Layout

### Plugin 与并行训练

目前（2023/10）Booster Plugin 还是一个正在开发的新 API，可能不太稳定。使用 Booster API 能够只用少量的更改就能够加速。API 的更改估计与这个 [discussion](https://github.com/hpcaitech/ColossalAI/discussions/3046) 有关，向 huggingface accelerate 靠近，这也符合原生 pytorch DDP 的代码逻辑

```python
# Following is pseudocode

# init distributed env
colossalai.launch(...)
# create plugin
plugin = GeminiPlugin(...)
# create booster, specify precision
booster = Booster(precision='fp16', plugin=plugin)

# normal pytorch code
# create model, dataset, optimizer, lr_scheduler, loss 
model = GPT2()
dataset = train_dataset
optimizer = HybridAdam(model.parameters())
lr_scheduler = LinearWarmupScheduler()
criterion = GPTLMLoss()

# create dataloader
dataloader = plugin.prepare_dataloader(train_dataset, batch_size=8)

# wrap
model, optimizer, criterion, dataloader, lr_scheduler = booster.boost(model, optimizer, criterion, dataloader, lr_scheduler)

# normal train loop
for epoch in range(max_epochs):
    for input_ids, attention_mask in dataloader:
        outputs = model(input_ids.cuda(), attention_mask.cuda())
        loss = criterion(outputs.logits, input_ids)
        # use booster to backward, instead directly loss.backward()
        booster.backward(loss, optimizer)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

在创建好 `train.py` 的脚本过后就可以通过 [Colossal CI](https://colossalai.org/docs/basics/launch_colossalai) 启动并行训练

```shell
colossal run train.py --nproc_per_node 2 
```

 ### AMP & Grad Accumulation

AMP 直接在 booster 接口使用即可，而 grad accumulation 直接调用 optimizer 的接口 `optimizer.clip_grad_by_norm(max_norm=GRADIENT_CLIPPING)`

Grad Accumulation 还需要在 training loop 实施额外的操作：创建一个 `sync_context = booster.no_sync(model)` 来管理梯度更新

```python
optimizer.zero_grad()
for idx, (img, label) in enumerate(train_dataloader):
    	# create sync_context
        sync_context = booster.no_sync(model)
        img = img.cuda()
        label = label.cuda()
        if idx % (GRADIENT_ACCUMULATION - 1) != 0:
            with sync_context:
                output = model(img)
                train_loss = criterion(output, label)
                train_loss = train_loss / GRADIENT_ACCUMULATION
                booster.backward(train_loss, optimizer)
        else:
            output = model(img)
            train_loss = criterion(output, label)
            train_loss = train_loss / GRADIENT_ACCUMULATION
            booster.backward(train_loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()
```

## Question

- 为什么 port 只指向 master port，其有什么作用，其他的 worker nodes 之间必须要通过这个 port 通信吗？
- 既然都使用了 CPU 的内存，那为什么不直接使用更大的 CPU 内存，从而解决 GPU 的显存墙？
- mixed precision & gradient accumulation 原理是怎样？如何使用 colossal ai 来实现
- 经常看到 O1 O2 O3 的优化水平（Optimization Level），这是否有固定的标准？
- Sequence Parallel 是否会影响前后关系？毕竟在并行的时候进行了切分？
- Sequence Parallel 如何在 activation recomputation 中发挥作用？