# ONNX Basic Concept

参考资料 [zhihu](https://zhuanlan.zhihu.com/p/477743341)	[torch.onnx doc](https://pytorch.org/docs/stable/onnx.html#)

一直都在提模型部署，到底什么是模型部署？这里给了一个定义

> 模型部署指让训练好的模型在特定环境中运行的过程

训练好的模型：pytorch 模型（python & GPU 环境）。起初还有很多不同的训练框架：pytorch, tf, caffe, mxnet。但目前 pytorch 已经占据了绝对主流

特定环境：不同的芯片（intel, qualcomm, nvidia） & 不同的系统（linux, windows, macos, android, ios）& 不同的编程语言（python, cpp, java）

开发者可以使用 pytorch 来定义网络结构，并通过训练确定网络参数，之后**模型结构和参数**会被转换成一种**只描述网络结构的中间表示**，一些针对网络结构的**优化会在中间表示上进行**，即：用面向硬件的**高性能推理引擎**把中间格式转换成特定的文件格式，并在对应硬件平台上高效运行模型。

发展到如今，这种中间表示即为：ONNX！没有第二家！推理引擎则有多家：ONNXRuntime , TensorRT, NCNN...

## 将模型转换为 ONNX 格式

在 pytorch 中，理论上讲只需要一个命令

```python
import torch

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    verbose=True,
    input_names=input_names,
    output_names=output_names)
```

但实际上会遇到各种各样的问题。这是因为 `torch.onnx.export` 需要接收一个 `torch.jit.ScriptModule`，而不是一个原装 `nn.Module`。如果输入的是 `nn.Module` 那 `torch.onnx.export` 将会尝试将其转换为 `ScriptModule`，这个过程调用的是 `torch.jit.trace` 方法，但是 tracing 过程会将模型中的 loop & if 语句展开，并生成静态图，该静态图将与 trace 运行过程一模一样不会更改

整个过程可以描述为：给定一组输入，再实际执行一遍模型，即把这组输入对应的静态计算图记录下来，保存为 ONNX 格式

这里就遇到了第一个问题：面对动态过程如何处理，这里我们仅介绍处理动态输入的方法

## 处理动态输入

实际上最常见的就是动态过程就是动态输入问题，在 pytorch 中已经很好地支持了动态输入了

一个小例子即可说明，这是一个两层的卷积模型，其输入和输出都是动态的

```python
# https://pytorch.org/docs/stable/onnx.html#torch.onnx.export
import torch 
 
class Model(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.conv0 = torch.nn.Conv2d(3, 3, 3) 
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
 
    def forward(self, x): 
        x = self.conv0(x)
        x = self.conv1(x)
        return x 
 
 
model = Model() 
dummy_input = torch.rand(1, 3, 10, 10) 

torch.onnx.export(model=model,
                  args=dummy_input,
                  f='dynamic_input_model.onnx',  
                  input_names=['in'],
                  output_names=['out'],
                  dynamic_axes={'in' : {2: 'h', 3: 'w'},
                                'out' : {2: 'h', 3: 'w'}}
                ) 
```

查看 [doc](https://pytorch.org/docs/stable/onnx.html#torch.onnx.export) 就知道 `dynamic_axes` 应该怎么写了

## 避免使用

1. 避免在模型中使用 numpy & built-in python types
2. 避免使用 `tensor.data`，请使用 `tensor.detach()`
3. 避免对 `tensor.shape` 使用 in-place 计算

前两个操作会让这些变量在 trace 中以常量记录，最后一个操作会让之后的程序访问错误的 shape 值

## Limitations

1. 在 trace 过程中，对于模型的输入和输出，只有 tensor, list of tensor, tuple of tensor 才能正常地被作为 **tensor 变量**看待，而对于 dict & string，其内容则会被作为**常量**看待

2. 对于任意 dict 输出，都将会被默认转变为序列，其 key 将会被抹去

   ```python
   {"foo": 1, "bar": 2} -> (1, 2)
   ```

3. 对于任意的 string 输出，都将会被默认删除

4. 在 trace 过程中，嵌套列表是不被支持的，其会被默认展开为无嵌套序列

5. 在使用 index 对 tensor 进行读取/修改时无法使用负的索引

   ```python
   # Tensor indices that includes negative values.
   data[torch.tensor([[1, 2], [2, -3]]), torch.tensor([-2, 3])]
   # Workarounds: use positive index values.
   ```

6. 在使用 index 对 tensor 进行修改时

   1. 无法使用多个 ranks >=2 的 index

      ```python
      # Multiple tensor indices if any has rank >= 2
      data[torch.tensor([[1, 2], [2, 3]]), torch.tensor([2, 3])] = new_data
      # Workarounds: use single tensor index with rank >= 2,
      #              or multiple consecutive tensor indices with rank == 1.
      ```

   2. 无法使用多个不连续的 index

      ```python
      # Multiple tensor indices that are not consecutive
      data[torch.tensor([2, 3]), :, torch.tensor([1, 2])] = new_data
      # Workarounds: transpose `data` such that tensor indices are consecutive.
      ```

## 算子支持

通常在导出的时候不会有顺利的时候😂你一定会遇到 `RuntimeError`

这里就要特殊处理这些算子了：TODO

[自定义算子-以超分辨模型为例](https://zhuanlan.zhihu.com/p/479290520)

## 如何加入 NMS 后处理

这个案例应该能研究好几个问题：

1. 算子支持
2. 动态过程
3. 计算图整合

如果我能够将 NMS 加入到 ONNX 当中，那么就也能够处理更多的问题

[自定义算子-进阶](https://zhuanlan.zhihu.com/p/513387413)

RT-DETR

## ONNX 调试与修改

[ONNX-调试与修改](https://zhuanlan.zhihu.com/p/516920606)
