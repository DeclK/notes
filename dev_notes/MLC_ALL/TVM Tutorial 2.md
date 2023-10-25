# TVM Tutorial 2

[Compiling and Optimizing a Model with TVMC](https://tvm.apache.org/docs/tutorial/tvmc_command_line_driver.html)

这个教程介绍了使用 tvmc 来对模型进行编译、运行、优化，但我更关心使用 tvm python api，所以不做整理

## Concepts

TVMC

这是 tvm 中的高级 api，能够比较省事完成简单模型的编译、运行、优化

AutoTVM

autotvm 是 tvm 中的一个模块，其作用是优化模型。我们可以使用 autotvm 来完成 tvmc 中的事情，不过粒度更细

IRModule

IRModule 就是 tvm 中用来承载计算中的 function 和 type difinition 的类，IRModule 主要对接于 relay，而不对接于其他 IR，例如 TE。IRModule 实际上是一个张量函数的容器（container），可以通过函数名获得定义在该模块下的张量函数

```python
type(MyModule)
# tvm.ir.module.IRModule
type(MyModule["mm_relu"])
# tvm.tir.function.PrimFunc
```

## Layouts

### TVMC

我们可以使用 tvmc 接口对一个 ONNX 模型进行编译和优化。这个过程可以用下面5步来概括

1. 从 ONNX 加载模型，并以 relay 表示

   ```python
   from tvm.driver import tvmc
   
   model = tvmc.load('my_model.onnx') #Step 1: Load
   
   # print(type(model.mod))
   ```

2. 指定硬件，编译 Relay 模型

   ```python
   package = tvmc.compile(model, target="llvm") #Step 2: Compile
   ```

3. 运行编译好的 package

   ```python
   import numpy as np
   inputs = {'input': np.random.randn(1, 10).astype('float32')}
   result = tvmc.run(package, inputs=inputs, device='cpu')
   result.outputs
   ```

4. 优化模型，可启用 autoscheduler 自动生成搜索空间

   ```python
   tuning_records = tvmc.tune(model, target="llvm", enable_autoscheduler=True， tuning_records='log.json') #Step 1.5: Optional Tune
   ```

5. 利用优化记录重新编译模型

   ```python
   opt_package = tvmc.compile(model, target='llvm', tuning_records=tunine_records)
   ```

其中还提供了对 relay 模型，编译 package 的保存

### AutoTVM

使用 autotvm 相当于要手动实现 tvmc 当中的高级 api 接口。我们同样也是做5件事

1. 从 ONNX 加载模型，返回的 mod 就是 Relay 模型（IRModule）

   ```python
   import onnx
   import tvm.relay as relay
   
   onnx_model = onnx.load('model.onnx')
   
   shape_dict = {"data": input.shape}
   
   mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
   
   # params is a dict {}
   ```

2. 指定硬件，编译模型

   ```python
   with tvm.transform.PassContext(opt_level=3):
       lib = relay.build(mod, target=target, params=params)
   ```

3. 使用 TVM Runtime 运行编译好的 lib。这里的运行不涉及特定 device

   ```python
   dev = tvm.device(str(target), 0)
   module = graph_executor.GraphModule(lib["default"](dev))
   
   module.run(input=input)
   # module.set_input("input", input)
   module.get_output(0).asnumpy()
   ```

4. 优化模型。这里相对 tvmc 就要复杂一些，我们会指定相关的优化配置，使用 `LocalRunner & LocalBuilder & tasks & XGBTuner` 。其中 Builder 和 Runner 用于编译和运行代码，tasks 是一个迭代器，迭代器中的每一个 item 代表了 TOPI 中的一个计算声明。这说明一个 program 有多种不同的实现模板。`config_space` 就是模板中可变换的一些操作，例如 tile 大小

   ```python
   from tvm.autotvm.tuner import XGBTuner
   from tvm import autotvm
   
   number = 10         # number of different configurations that we will test
   repeat = 1          # how many measurements we will take of each configuration
   min_repeat_ms = 0   # since we're tuning on a CPU, can be set to 0
   timeout = 10        # in seconds
   
   # create a TVM runner
   runner = autotvm.LocalRunner(
       number=number,
       repeat=repeat,
       timeout=timeout,
       min_repeat_ms=min_repeat_ms,
       enable_cpu_cache_flush=True,
   )
   
   tuning_option = {
       "tuner": "xgb",
       "trials": 4,
       "early_stopping": 100   ,
       "measure_option": autotvm.measure_option(
           builder=autotvm.LocalBuilder(build_func="default"), runner=runner
       ),
       "tuning_records": "resnet-50-v2-autotuning.json",
   }
   
   # begin by extracting the tasks from the onnx model
   tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
   
   # Tune the extracted tasks sequentially.
   for i, task in enumerate(tasks):
       prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
   
       # choose tuner
       tuner_obj = XGBTuner(task, loss_type="reg")
   
       # print(len(task.config_space))
       # print(task.config_space)
       tuner_obj.tune(
           n_trial=min(tuning_option["trials"], len(task.config_space)),
           early_stopping=tuning_option["early_stopping"],
           measure_option=tuning_option["measure_option"],
           callbacks=[
               autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
               autotvm.callback.log_to_file(tuning_option["tuning_records"]),
           ],
       )
   ```

5. 利用调优过后的参数，重新编译模型

   ```python
   with autotvm.apply_history_best(tuning_option["tuning_records"]):
       with tvm.transform.PassContext(opt_level=3, config={}):
           lib = relay.build(mod, target=target, params=params)
   
   dev = tvm.device(str(target), 0)
   module = graph_executor.GraphModule(lib["default"](dev))
   ```

之后就可以对优化前后的模型进行速度比较，教程使用了 timeit 模块

```python
import timeit

def func(): pass

timer = timeit.Timer(lambda: func()) # must use lambda
timer.timeit(number=1000)
timer.repeat(repeat=3, number=1000)
```

## Questions

1. What is a context? 之前在 GPU 教程里也见到过，在 TensorRT 中也见到过，现在又出现了 `PassContext`

2. XGBoost 算法是如何实现的

3. 在 autotvm 中各个参数的含义是什么？

   ```python
   measurements
   number
   repeat
   trials
   early_stopping	# only works when < trials
   ```

4. 在 autotvm 的调优过程中 `LocalRunner & LocalBuilder` 是要编译什么？运行什么？