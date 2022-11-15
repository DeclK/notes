---
title: Pytorch 分布式训练
tags:
  - Pytorch
  - 分布式
categories:
  - 编程
  - Python
  - Pytorch
abbrlink: b45af34d
date: 2022-07-10 22:20:38
---

# Pytorch 分布式训练

对于分布式训练的原理实在是很不清晰！可能就算整理了也不太能完全明白，但是做了总比没做好😭

参考：[zhihu](https://zhuanlan.zhihu.com/p/113694038) [zhihu](https://zhuanlan.zhihu.com/p/358974461) [zhihu](https://zhuanlan.zhihu.com/p/393648544) [bilibili](https://www.bilibili.com/video/BV1xZ4y1S7dG/?spm_id_from=333.788)

关于指定 GPU, CUDA_VISIBLE_DEVICES [CSDN](https://blog.csdn.net/alip39/article/details/87913543) 

## DistributedDataParallel

### 内部机制的通俗理解

`DistributedDataParallel`通过多进程在多个GPUs间复制模型，每个GPU都由一个进程控制。GPU可以都在同一个节点上，也可以分布在多个节点上。每个进程都执行相同的任务，并且每个进程都与所有其他进程通信。进程或者说GPU之间只传递梯度，这样网络通信就不再是瓶颈

每一个GPU都有自己的前向过程，然后梯度在各个GPUs间进行 All-Reduce（所谓 All-Reduce 可以简单看作是一种平均）。每一层的梯度不依赖于前一层，所以梯度的 All-Reduce 和后向过程同时计算，以进一步缓解网络瓶颈。在后向过程的最后，每个节点都得到了平均梯度，这样模型参数保持同步

### 一些基本概念

**每个进程都需要知道进程总数及其在进程中的顺序，以及使用哪个GPU**，这样进程之间才能正确通信，通常将进程总数称为 `world_size`，其顺序称为 rank or local rank（两者我其实没有很好区分）

一般情况下称进程0（local rank == 0）是 master 进程，比如我们会在进程0中打印信息或者保存模型

Pytorch提供了`nn.utils.data.DistributedSampler`来为各个进程切分数据，以保证训练数据不重叠

## torchrun & torch.distributed.launch

DDP 的启动方式形式上有多种，内容上是统一的：都是启动多进程来完成运算，这里就整理一下 OpenPCDet 使用的一种方法。现在 pytorch 准备使用 `torchrun` 完全替代 `torch.distributed.launch`

### launch

[torch.distributed.aunch](https://pytorch.org/docs/stable/distributed.html#launch-utility)

launch 实际上主要完成的工作：

1. 参数定义与传递。解析环境变量，并将变量传递到子进程中

2. 起多进程。调用subprocess.Popen启动多进程

用 launch 方式需要注意的位置：

1. 需要添加一个解析 local_rank 的参数

   ```python
   parser.add_argument("--local_rank", type=int)
   ```

   运行脚本时 launch 将会自动传入这个参数的值

2. DDP的设备都需要指定 local_rank

   ```python
   net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])
   ```

一般使用单节点多进程脚本

```shell
python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
           YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
           arguments of your training script)
```

### torchrun

[torchrun](https://pytorch.org/docs/stable/elastic/run.html)	[train scripts](https://pytorch.org/docs/stable/elastic/train_script.html)	[quickstart](https://pytorch.org/docs/stable/elastic/quickstart.html)

换到 torchrun 上

```shell
torchrun
   --standalone
   --nnodes=1
   --nproc_per_node=TRAINERS_PER_NODE
   YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

不再需要使用 `--local-rank` 参数解析，而是直接在环境变量中获取 `os.environ['LOCAL_RANK']`

以上两种方法都可以使用 `dist.get_rank() & dist.get_world_size()` 获得 local rank 和 world size

## 完整过程

总结一下 OpenPCDet 分布式训练的逻辑：

1. 使用 `torch.distributed.launch` 调用多进程

   ```shell
   python -m torch.python -m torch.distributed.launch --nproc_per_node={NUM_GPUS} ./train.py
   ```

2. 初始化进程组，并设置 cuda device

   ```python
   mp.set_start_method('spawn')	# mmdet use 'fork' to start, faster but might be unstable
   								# not needed when use torch.distributed.launch or torchrun
   dist.init_process_group(backend='nccl')	# 指定 backend，用于进程间的通信
   torch.cuda.set_device(local_rank)		# 设置了当前 cuda device 过后可以直接用 model.cuda()
   										# 该步骤不是必须的
   ```

3. 使用 DDP 包装 model，并指定 device

   ```python
   model = DDP(model, device_ids=[args.local_rank])
   ```

4. 使用 `DistributedSampler` 作为 sampler

   ```python
   sampler = torch.utils.data.distributed.DistributedSampler(dataset)
   ```

5. 设置 sampler 的 epoch 数，以确定随机种子，保证切割数据不重叠

   ```python
   sampler.set_epoch(epoch)
   ```

### TODO

example with CUDA_VISIBLE_DEVICES, cudnn deterministic，cudnn benchmark  [zhihu](https://zhuanlan.zhihu.com/p/359058486), random seed，dataloader

fixed must prepare steps

## Official

现在回过头来看，分布式训只要熟练运用 API 就好。这样还不如直接去官网看看教程，整理起来更清晰

[Dist Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)	[Writing Dist Apps](https://pytorch.org/tutorials/intermediate/dist_tuto.html)	[**Getting Started with DDP**](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

分布式模块是用 `torch.distributed`，另一个使用最频繁类是 `DistributedDataParallel` （简称 DDP）。现在基本不用 `DataParallel` 因为它使用的是多线程，会受到 GIL 的限制，所以很慢

要使用 DDP 需要先使用 `init_group_process` 进行配置

### Collective Communication

Pytorch 支持 Point-to-Point Communication，也就是任意两个 GPU 之间的交流，这是更灵活的交流方式，这里不做总结。而单平时用的最多的是 Collective Communication，也就是在所有 GPU 之间的交流。举个例子，如果我们需要一个张量在所有进程里的和的时候可以使用 `dist.all_reduce(tensor, op=dist.ReduceOp.SUM)`，除了和以外，Pytorch 总共实现了4中操作：

- `dist.ReduceOp.SUM`,
- `dist.ReduceOp.PRODUCT`,
- `dist.ReduceOp.MAX`,
- `dist.ReduceOp.MIN`.

除了 `dist.all_reduce` 以外，Pytorch 总共有6种 collective method

- `dist.broadcast(tensor, src, group)`: Copies `tensor` from `src` to all other processes.
- `dist.reduce(tensor, dst, op, group)`: Applies `op` to every `tensor` and stores the result in `dst`.
- `dist.all_reduce(tensor, op, group)`: Same as reduce, but the result is stored in all processes.
- `dist.scatter(tensor, scatter_list, src, group)`: Copies the $i^{\text{th}}$ tensor `scatter_list[i]` to the $i^{\text{th}}$ process.
- `dist.gather(tensor, gather_list, dst, group)`: Copies `tensor` from all processes in `dst`.
- `dist.all_gather(tensor_list, tensor, group)`: Copies `tensor` from all processes to `tensor_list`, on all processes.
- `dist.barrier(group)`: Blocks all processes in group until each one has entered this function.

`src & dst` 都是对应的某个 rank，而 `group` 一般不用指定，默认是所有的进程。`rank & world size` 都可以通过 `dist.get_rank() & dist.get_world_size()` 获得，前提是进行了 `dist.init_group_process`

在分布式中需要注意进程之间的同步，具体来说，分布式要求各个进程运行的速度不能相差太大，大家都在某一个时刻执行相同的代码。但是通常主进程要做一些其他的事情，如果各个进程之间的代码进度相差太大，就可能出问题，可以使用 `dist.barrier` 进行一个同步

### Basic Usage

一般在主程序的初始，直接使用最简单的初始化

```python
def setup(rank, world_size, backend='nccl'):
    # initialize the process group
    dist.init_process_group(backend)

def main():
	setup(rank, world_size)
	model = Model().to(rank)	# model = Model().cuda(rank)
    ddp_model = DDP(model, device=[rank])
```

`rank & world_size` 并不是必须的使用的，当然也可以传入

````python
dist.init_process_group("gloo", rank=rank, world_size=world_size)	# use another backend 'gloo'
````

### Save Model

虽然说的是保存模型，但其实任何不同步的操作都需要这么做

```python
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 runs after process 0 saves model.
    dist.barrier()
```