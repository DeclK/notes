# TVM Tutorial 1

参考 [TVM Overview](https://tvm.apache.org/docs/tutorial/introduction.html#an-overview-of-tvm-and-model-optimization)  [MLC tutorial](https://mlc.ai/chapter_introduction/index.html)

下面这张图简单描述了 TVM 的框架流程

![A High Level View of TVM](/home/lixiang/Projects/notes/dev_notes/TVM Tutorial 1/overview.png)

整体来讲，TVM (Tensor Virtual Machine) 作为一个 MLC (Machine Learning Compiler) 的核心目的就是将模型编译成为可以部署的库

## Concept

### MLC

下面对核心概念进行定义

**编译（compile）**

这里的“编译”与传统的编译器是相似的，但不完全相同，因为这个过程不一定设计代码生成。更准确的说，机器学习编译是指：将机器学习算法从开发阶段，通过**变换**和**优化算法**，使其变成指定硬件的**部署状态**

**张量函数（Tensor Function）**

描述的是张量之间的映射：输入张量和网络权重进行计算，获得输出张量。张量函数不需要对应于神经网络计算的单个步骤。部分计算或整个端到端计算也可以看作张量函数。这也是一般函数具有的性质，只考虑输入到输出的映射

**抽象/实现**

我们使用抽象/实现，来表示张量函数的计算过程。一个张量函数可以有多个抽象，并且不同的抽象可以包含不同的细节量。例如下图：

![image-20230928113436117](/home/lixiang/Projects/notes/dev_notes/TVM Tutorial 1/image-20230928113436117.png)

一个抽象可以用简单的流程图表示，也可以用 python 代码表示，也可以用 C++ 代码表示。显然 C++ 抽象肯定比流程图包含的细节更多。这里也表明了，“语言”也是一种抽象

这里我不对抽象和实现进行区分

**元张量函数**

元张量函数是一个较“小”的张量函数，当小到一定程度时，这个张量函数就不可继续分割了

**变换**

从一个抽象到另一个抽象的过程，称之为变换。通常变换前后，所表示的张量函数不变。变换是一个范围非常广泛的过程：可以是高级语言到底层语言的变换，也可以是相同语言，但是不同实现方式。**抽象的变换可以说就是 MLC 的核心**

**IR (Intermediate Representation)**

IR 可看做一种语言，就是一种抽象。**IR、语言、抽象、表示，这些似乎都是相同的概念**

### TVM

**Relay**

TVM 用于描述模型的高级语言（high-level）。我们可以在高级抽象的层面上对模型进行优化，例如图优化

**TE**

Tensor Expression，一种描述张量计算的语言，相比于 Relay，TE 语言可认为是更低级的（low-level）。我们可以在更底层的抽象上对算子进行优化，例如分块、并行、循环展开。在 TVM 中可以通过 TOPI 将 relay 表示转换为 TE 表示。或许可以认为 TE 即 TOPI，TOPI 即模板，所以 TE 即模板

**TOPI**

Tensor Operator Inventory，一个预定义的模板库，包含很多常见算子的模板（卷积、转置）。

**AutoTVM**

对模板中可进行的变换进行计算优化

**AutoScheduler**

无需模板，对计算进行优化

**TIR**

Tensor Intermediate Representation，TVM 中最底层的计算描述语言，由 TE 转化而来

## TVM Overview

了解了这些概念过后再来看看这张图

![A High Level View of TVM](/home/lixiang/Projects/notes/dev_notes/TVM Tutorial 1/overview-1695882108590-3.png)

TVM 的流程就是 IR 的不断降级（lowering）的过程，并且在这个过程中不断地对 IR 进行优化，使得实现最优

我们需要对这些工具熟练使用，使用工具来准确表达我们自己的想法（自己的抽象！）

## 问题

1. AutoScheduler 的搜索空间是如何生成的