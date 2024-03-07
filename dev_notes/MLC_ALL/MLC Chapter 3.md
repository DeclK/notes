# MLC Chapter 3

在实践中，我们使用更智能的算法。如果我们对其它设备的优化感兴趣，我们还需要提供额外的工具，例如远程设备上的基准测试。 TVM 的 Meta-Schedule API 提供了这些附加功能。

`meta_schedule` 是支持搜索可能变换空间的命名空间。Meta-Schedule 在幕后做了很多额外的事情：

- 跨越多个进程的并行基准测试。
- 使用**代价模型** (cost model) 来避免每次都进行基准测试。
- 基于历史轨迹进行**遗传搜索** (evolutionary search)，而不是每次都随机采样。