# TensorRT Plugin Tutorial

之前从宏观上把握了 TensorRT，但是具体怎么写 Plugin 没有进行细致整理，只是知道大概流程，现在要结合代码和具体的例子来学习

这是之前所提出的问题

- 怎么从头开始实现一个 Plugin？ 要写哪些类和函数 
- 怎么把 Plugin 接到 TensorRT 网络中去？ 要怎么包装 kernel 以便 TensorRT 识别
- TensorRT 怎么参与 Plugin 的资源管理？ 两者之间要交换些什么信息 
- Plugin 有些什么扩展性？ FP16/INT8，Dynamic Shape，data-dependent-shape，… 
- Plugin 与原生 Layer 相比性能怎么样？





[mmdeploy-plugin](https://mmdeploy.readthedocs.io/zh_CN/latest/tutorial/07_write_a_plugin.html)

**实现：构造函数、析构函数和 TensoRT 中三个基类的方法即可**

cmake & setuptools 都可以生成 .so

supportsFormatCombination 什么用

如何写 enqueue

输入参数的含义

如何支持 fp16 等其他数据格式

## TopDown

有两个类

PluginCreator

Plugin itself

使用 Plugin Creator 来注册你的 Plugin，并且提供 name, version, field paramer

## 定义哪些函数



## Questions

问题：如何通过 setuptools 对 plugin 进行编译，并将 plugin 通过 trt python 借口进行调用（直接使用和 ONNX parser）

问题：写 plugin 到底需要继承哪些类，需要重写哪些函数？这些函数有什么作用？

问题：为什么要先生成 .o 文件，然后再生成 .so 文件？因为 make file 自动删了这些中间文件

问题：写 plugin 的时候 trt.PluginField("scalar", np.float32(scalar), trt.PluginFieldType.FLOAT32)，其中的 data type 如何确定的？

问题：如何构建高效写 plugin 的流程，标准的 debug 流程。目前卡在了 build_engine 阶段，检查了 shared memory 和 workspace 大小问题

问题：DimsExprs *inputs 有什么用？inputs[0,1,2,...] 代表了什么

问题：如何使用 serialized data 来 plugin？TensorRT 如何知道哪个构造函数是使用了 serialized data？deserialize_value 怎么使用？

```cpp
QKVToContextInterleavedPlugin::QKVToContextInterleavedPlugin(std::string const& name, void const* data, size_t length)
    : mLayerName(name)
{
    deserialize_value(&data, &length, &mNumHeads);
    deserialize_value(&data, &length, &mHeadSize);
    deserialize_value(&data, &length, &mHiddenSize);
    deserialize_value(&data, &length, &mSM);
    deserialize_value(&data, &length, &mS);
    deserialize_value(&data, &length, &mB);
    deserialize_value(&data, &length, &mDqProbs);
    deserialize_value(&data, &length, &mUseInt8ScaleMax);
}
```



问题：getOutputDimensions 中 outputIndex 基本上用不上？

问题：DynamicPluginTensorDesc & PluginTensorDesc 用来干啥？

问题：PluginFieldCollection 怎么使用

问题：如何对多个文件进行编译？

问题：CUDA Stream_t 就只是一个 int ？

问题：如何将输入输出表示为 Tensor？ getOutputDimensions

问题：template instanciate 的必要性？

问题：如何从 buffer 中获得数据？read or deserialize_value

问题：DimsExprs 应该怎么使用，其应用场景如何？一种维度表示，并不代表维度数值本身

问题：双重指针的作用：1. 在函数中修改指针的值，通常出现在动态内存当中；2. 多维数组，或者换句话说，指针数组。指针运算所跳过的字节为指针所指向的对象

最后发现出发 segmentation fault 的问题

1. onnx 导出错误，因为对模型的理解出现了偏差！！factor 使用的是 1-D tensor，而实际上应该使用 4-D tensor，即使用具体形状的向量，而不是形状向量本身
2. tensorrt input 设置错误

## Profile

1. 使用 tensorrt Python api 进行测速

   直接在 context 中加入 profiler 即可，但这会对每一层进行 profile，python 文档 [Profiler](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Profiler.html)

   ```python
   context.profiler = trt.Profiler()
   ```

2. 使用 trtexec 进行测速

   ```shell
   # static shapes
   trtexec --loadEngine=xxx --plugins=xxxx
   
   # dynamic shapes
   trtexec --loadEngine=xxx --plugins=xxx --shapes=input:1x3x256x256,factor:1x1x512x512
   ```

   关于 trtexec 的输出

   ```shell
   [09/18/2023-09:54:19] [I] === Performance summary ===
   [09/18/2023-09:54:19] [I] Throughput: 858.971 qps
   [09/18/2023-09:54:19] [I] Latency: min = 1.53241 ms, max = 2.40356 ms, mean = 1.57121 ms, median = 1.56714 ms, percentile(90%) = 1.58105 ms, percentile(95%) = 1.58569 ms, percentile(99%) = 1.60509 ms
   [09/18/2023-09:54:19] [I] Enqueue Time: min = 0.0640869 ms, max = 0.246704 ms, mean = 0.151355 ms, median = 0.160156 ms, percentile(90%) = 0.16687 ms, percentile(95%) = 0.171387 ms, percentile(99%) = 0.184448 ms
   [09/18/2023-09:54:19] [I] H2D Latency: min = 0.15625 ms, max = 0.230347 ms, mean = 0.166099 ms, median = 0.16687 ms, percentile(90%) = 0.167908 ms, percentile(95%) = 0.168945 ms, percentile(99%) = 0.178223 ms
   [09/18/2023-09:54:19] [I] GPU Compute Time: min = 1.12231 ms, max = 1.99365 ms, mean = 1.16186 ms, median = 1.15918 ms, percentile(90%) = 1.17041 ms, percentile(95%) = 1.17554 ms, percentile(99%) = 1.19501 ms
   [09/18/2023-09:54:19] [I] D2H Latency: min = 0.24234 ms, max = 0.25293 ms, mean = 0.243255 ms, median = 0.24292 ms, percentile(90%) = 0.24408 ms, percentile(95%) = 0.244385 ms, percentile(99%) = 0.250977 ms
   [09/18/2023-09:54:19] [I] Total Host Walltime: 3.00359 s
   [09/18/2023-09:54:19] [I] Total GPU Compute Time: 2.99759 s
   ```

   其解释如下（By GPT）

   > 1. Throughput: This metric measures the number of queries processed per second (qps). In this case, the model achieved a throughput of 858.971 qps.
   > 2. Latency: Latency measures the time it takes for a query to be processed. The output provides various statistics related to latency:
   >    - Min: The minimum latency observed (1.53241 ms).
   >    - Max: The maximum latency observed (2.40356 ms).
   >    - Mean: The average latency across all queries (1.57121 ms).
   >    - Median: The middle value of the latency distribution, separating the lower half from the upper half (1.56714 ms).
   >    - Percentile: These values indicate the latency below which a given percentage of queries falls.
   >      - percentile(90%): 90% of the queries had a latency of 1.58105 ms or lower.
   >      - percentile(95%): 95% of the queries had a latency of 1.58569 ms or lower.
   >      - percentile(99%): 99% of the queries had a latency of 1.60509 ms or lower.
   > 3. Enqueue Time: Enqueue time refers to the time taken to enqueue a query on the GPU for execution. The statistics provided are similar to latency statistics and include values such as min (0.0640869 ms), max (0.246704 ms), mean (0.151355 ms), etc.
   > 4. H2D Latency: H2D stands for Host-to-Device, representing the time taken to transfer data from the CPU (host) to the GPU (device). The statistics are similar to latency statistics and provide insights into data transfer timings.
   > 5. GPU Compute Time: GPU Compute Time measures the time taken by the GPU to execute the inferencing or computation for a query. Similar to other metrics, it provides statistics such as min (1.12231 ms), max (1.99365 ms), mean (1.16186 ms), etc.
   > 6. D2H Latency: D2H stands for Device-to-Host, representing the time taken to transfer data from the GPU (device) back to the CPU (host). The statistics provided are similar to latency statistics seen before.
   > 7. Total Host Walltime: This metric represents the total time spent by the host (your system) in running the entire inference workload. In this case, it took 3.00359 seconds.
   > 8. Total GPU Compute Time: This metric represents the total time spent by the GPU in executing the inference workload. In this case, it took 2.99759 seconds.
