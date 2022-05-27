# Pytorch 分布式训练

对于分布式训练的原理实在是很不清晰！可能计算整理了也不太能完全明白，但是做了总比没做好😭

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

## torch.distributed.launch

DDP 的启动方式形式上有多种，内容上是统一的：都是启动多进程来完成运算，这里就整理一下 OpenPCDet 使用的一种方法。似乎现在 pytorch 写了一个更好用的方法来替代，之后再说吧

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

## 完整过程

总结一下 OpenPCDet 分布式训练的逻辑：

1. 使用 `torch.distributed.launch` 调用多进程

2. 初始化进程组，并设置 cuda device

   ```python
   dist.init_process_group(backend='nccl')
   torch.cuda.set_device(local_rank)
   ```

3. 使用 DDP 包装 model，并指定 device

   ```python
   model = DDP(model, device_ids=[args.local_rank])
   ```

4. 使用 `DistributedSampler` 作为 sampler

   ```python
   sampler = torch.utils.data.distributed.DistributedSampler(dataset)
   ```

5. 设置 sampler 的 epoch 数，以确定随机种子?，保证切割数据不重叠

   ```python
   sampler.set_epoch(epoch)
   ```