# Learn TorchPilot 1 

## Concept

- exception hook

  `sys.excepthook`

  利用这个 hook 来获得更显眼的 error message

- MPI command world, dist manager

  感觉 api 有好几种，但是核心的几个概念就几个，到时候只需要对应不同的 api 来获得关键信息就行

  1. rank: global and local
  2. world size
  3. barrier: done by mpi
  4. broadcast: done by mpi
  5. abort: done by mpi

- `ClassVar`

  一个 typing 标志，表示这个对象是一个类变量，属于类，不属于类的实例

- `@lru_cache`

  该装饰器修饰一个函数

  ```python
  @lru_cache
  def fun(*args)
  ```

  当相同的 args 输入到 fun 中，不会进行重新计算，而是使用最近的缓存结果来替代计算

  ```python
  import functools
  import time
  
  @functools.lru_cache(maxsize=None)
  def fib(n):
      if n < 2:
          return n
      return fib(n-1) + fib(n-2)
  
  s = time.time()
  print(fib(30)) # prints: 55
  e = time.time()
  print(e - s) # prints: 0.0
  ```

- logger 设计

  每一个文件都有自己单独的 logger，这样能够快速查看 logger 所在文件

- `load_plus()`

  通过 `entry_points` & `import_module` & `registry.register_module` 完成对模块的注册，从而避免显式地在类上添加 `@register` 装饰器

## Layout

- `run`

  这是 torchpilot 的运行接口

  这是一个嵌套运行的过程有好几个函数：

  1. `runtime_program`
  2. `spawn_launcher`
  3. `spawn_run`

  下面逐个介绍

  - `run_program`

    初始化了 `FileManager` 创建了一些文件夹用于记录

    使用 `load_plus` 导入模块

    获得命令行输入的 args

    调用 `spawn_launcher`，其中 `spawn_launcher` 使用了 `runtime_program` 作为 `runtime_function`

    - `runtime_program` 做了两件事

      1. 构建了 `PyConfig`，其实就是解析 `config.py`
      2. 构建并运行 `Runner`

    - `spawn_launcher` 中使用了 `torch.multiprocessing.spawn` 来发起多进程，但我们还有一件事没有干：初始化 proces group 以及设置 GPU device

      ```python
      dist.init_process_group(...)
      torch.cuda.set_device(local_rank)
      ```

      所以还使用了 `spawn_run` 作为一个 wrapper，来把 `runtime_program` 包起来，我觉得二者可以写到一个函数里，都是 OK 的

- `analyze_model`

  用于对一个 pytorch 模型进行 Profile，可查看其参数量，flops，以及 pytorch 模型 latency

  似乎是使用了 Zen-NAS 的方法，其中有一个 Zen score 用于评价网络的效果

- `PlainNetBasicBlockClass`

  定义了一个模板函数，似乎是为卷积而定制的，因为定义了 in out channels, stride

  但不同的是，这里还定义了从字符串获取模型的方法，以及一些辅助方法。这些方法都是模板，之后都会在子类中重新定义

  - `create_from_str`

    给一个字符串，字符串包含了模型的名字和初始化参数，通过参数将模型实例化。其实真正重要的是初始化参数，字符串包含的模型名字必须要匹配真正的模型名字，不然会报错。匹配检查方法为 `is_instance_from_str`

    基本上我们不直接使用 `__init__` 方法生成实例，而是调用 `create_from_str` 这个类方法来生成实例

  - 辅助方法
    1. `get_FLOPs`
    2. `get_model_size`
    3. `get_output_resolution`
    4. `set_input_channels`

- `PlainNet`

  从字符串生成 module

  调用上面 `PlainNetBasicBlockClass` 中定义的 `create_from_str` 生成实例，使用一个 list 进行存储

  因为要调用 `create_from_str` 需要对应的模块，所以需要一个字典存储所有定义的模块，然后用字符串名字的字典，去取到模块，在用模块的 `create_from_str` 方法

- ResNet block

  `SuperConvKXBNRELU`，KX 代表的是 kernel size is X

  `SuperResK3K3`，代表有两个卷积层的残差块

  `SuperResK1K3K1` 代表有三个卷积层的残差块

  `SuperResIDWE1K3` 代表有两个 DWConv 块，每一个 DWConv block 由 Conv1x1 + Depthwise3x3 + Conv1x1 组成


- Framework of torchpilot

  在设计每一个功能时，先设计了一些模版类 `Basexxx`，这些模版函数需要提供文件的基本功能定义，之后的子类可在这些定义的功能上进行拓展

  **Data**

  - DataLoader

  - Dataset

    dataset 的最终目标就是定义 `__getitem__` 方法，为了这个方法需要构建一个样本列表以及数据增强，通过采样和数据增强获得训练/测评的完整数据

    同时，为了符合规范还需要定义 `__len__` 方法

  **Model**

  - Common Modules

  - Loss

    - `BaseLoss` 在初始化时需要定义 `name`
    - `forward` 参数需要定义 `model_outputs & model_inputs` 用于计算最终损失函数

  - Task

    一开始本来准备寻找 model 文件夹的，一般这个文件夹用于定义完整的模型。但是我没有找到 model 文件夹，只看到了 task 文件夹，这个 task 文件夹就是我想找的。这给了我概念的更新：model is task, task is model。Task 实际上包含了三个组成部分

    - Task Model
      - 需要根据 `sub_module_cfg` 把模型组装起来
      - 定义 `train_forward & infer_forward`
    - Loss calculator
    - Postprocess

    我觉得把它定义为 task 比定义为 model 更为合适！相比于 mmlab 是一个更好的抽象

    根据是 train 和 val 可以使用上述组件，在 `forward` 中完成相应的任务逻辑

  Optimizer

  Evaluator

  Tools/ Utils

## Question

- Loss 的定义是否不够灵活，因为输入都统一成为了一个 dict
- Train 和 Evaluation 没有集成到一起，单独 eval 初始化时间长
- 没有单卡的 debug 模式，ddp 无法使用 pdb 进行 debug
- logger 还是可以参考 mmlab 的形式，比较直观
- 没有看到 tensorboard 文件
- logger 记录间隔不受控制
- 优化器的定义不够多

学习点：

- pytorch profiler `torch.profiler`
- setuptools entry usage
- Need a template config so we can modify easily
- pipenv shell
- cli tools
- pre-commit regulations, clean code base, typing, flake, black, pylint
- engine and project needs to be separate
- Runner of mmengine is still heavy, need to be reformed, maybe we can use optimizaiton hook explicitly
- mmengine BaseModel should rename to TaskFlow
- when to use typing?