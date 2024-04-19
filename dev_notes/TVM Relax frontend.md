# TVM Relax frontend

现在 TVM 学习已经进入了深水区，有几个难点：

1. TVM 没有足够清楚的文档来介绍新特性，这也和 TVM 开发速度很快，文档基本上没有更新，甚至出现了好几种文档，非常的混乱
2. 目前大部分的介绍都是针对 RelayIR 的介绍，但是 RelaxIR 在今后将会是绝对的主流，资料更少
3. 看源码看不懂，源码会调用 C++，我 C++ 基础薄弱



策略：从源码入手，去推理其代码的步骤，结合 GPT 进行合理猜测

如何从源码入手：从一个简单的代码块入手，逐步深入，先搞懂一个特例，从中获取通用信息，再去做推广



## Tensor

Tensor 自己的注释

```python
class Tensor(_TensorOp):
    """A wrapper on top of relax.Expr whose struct_info is a TensorStructInfo, providing more
    convenient access shape and dtype information. Tensor is always symbolc and not bound to any
    concrete values. Shape and dtype inference is done eagerly upon tensor creation, i.e. when
    operators are applied on tensors, the shape and dtype information is already available.
    """

    _expr: rx.Expr
```

1. 什么是 StructInfo or TensorStructInfo，是一个 Node 还是 Node 的一个属性

2. rx.const 是什么

   通过 numpy data or python scalar 创建一个 `rx.Constant` 对象

3. _expr 到底是什么，也是一个节点吗？显然 `_expr=rx.const(data)`，说明 `rx.const` 是 expr 的一种形式

   除此之外还可以使用 `_expr=rx.Var`，那么这里的 `rx.Var` 也是一种 expr，它又代表哦了什么

4. `rx.Var & tir.Var` 似乎不是一个东西

   ```python
   @tvm._ffi.register_object("relax.expr.Var")
   class Var(ExprWithOp):
       """The variable class for all Relax bindings
       """
   
   @tvm._ffi.register_object("tir.Var")
   class Var(PrimExprWithOp):
       """Symbolic variable.
       """
   ```

   所以区别在于 Primitive & non-Primitive

## Node & Expression

- `ExprOp & ExprWithOp`

  > - `ExprOp` is more focused on the basic arithmetic and bitwise operations for primitive expressions.
  > - `ExprWithOp` extends the functionality to include relax-specific operations and tensor manipulation.
  > - `ExprWithOp` provides a higher-level interface for working with relax expressions, including casting, tensor properties, and function call syntax.

- Primitive expressions and non-primitive expressions

  `PrimExpr`是一个表示不可进一步简化的操作的节点。它通常对应于硬件指令集中的一个原子操作，如加法、乘法、加载、存储等

  非原始表达式是由其他表达式组合而成的复杂表达式。它们可以包含一个或多个子表达式，这些子表达式可以是`PrimExpr`或其他非原始表达式。

  这也是为什么 `tir.Var` 是一个 PrimExpr 节点。我的理解是 `tir.Var` 就像是一个 python scalar，表示一种最原始的变量。而 `rx.Var` 就像一个 torch tensor，表示更复杂的变量。`ExprOp` 就是对普通 python scaler 的运算重载，例如加减乘除；而 `EpxrWithOp` 就是对张量运算的算符重载，具有更复杂的操作

- Expression and Expr

  > An expression, in a general programming and computer science sense, is a combination of values, variables, operators, and/or function calls that produces a value when evaluated.
  >
  > `Expr` is a class in TVM that represents an expression node in the AST, used to construct and manipulate parts of the computational graph.

- Statement & Expression

  > A statement is a distinct unit of code that performs some action or declares something. Unlike expressions, which compute a value, statements do not necessarily evaluate to a value. Statements are used to alter the state of the program, control the flow of execution, or declare variables and functions. Examples of statements include:
  >
  > - Variable declarations (`int x;`)
  > - Assignments (`x = 10;`)
  > - Control structures (`if`, `while`, `for` loops)
  > - Function definitions (`void foo() { ... }`)
  > - Class and struct definitions
  >
  > In the context of an AST, a `Statement` node represents a statement from the source code. ASTs are tree structures that represent the syntactic structure of the source code. Each `Statement` node in the AST corresponds to a statement in the source code.

  > 疑问为什么在 Relax 里面没有看到任何跟 Statement 的相关节点
  >
  > TVM's Relax DSL (Domain Specific Language) is designed for expressing high-level tensor operations and optimizations. It is not a general-purpose programming language, which is why you might not see the term "statement" used as frequently as "expression" in Relax.
  >
  > Your confusion might stem from the fact that in TVM's Relax DSL, the term "expression" is used more prominently because the DSL is centered around defining mathematical computations and operations on tensors. These expressions are the core of the Relax language, and they are what get compiled into optimized code for execution on various hardware targets.

## rx.Expr

```python
import tvm.relax as rx
import tvm.tir as tir

rx.Expr -> RelayExpr -> BaseExpr -> Node
rx.TensorStructInfo -> StructInfo -> Node

rx.Var -> ExprWithOp -> rx.Expr
rx.Constant -> ExprWithOp	(Tensor)
class Node(Object):
    """Base class of all IR Nodes."""
    
tir.Var -> PrimExprWithOp -> ExprOp, PrimExpr -> BaseExpr 

rx.Function -> BaseFunc -> RelayExpr
rx.Call -> ExprWithOp
```

## BlockBuilder

- IRModule

  获得了 IRModule 过后我们可以如何使用它？什么是 `R.function`，它和 `prim_func` 有什么区别

  似乎 `R.function` 只是一种标记，它和 `blockbuilder.function()` 是类似的作用

  同样的 `R.dataflow` 和 `blockbuilder.dataflow()` 也是相同的作用

- BlockBuilder 的作用

  似乎这是第二种方式用来构建 IRModule，与使用 `R.function` 似乎是相同的

  

- `bb.emit` && `effect.emit_init` emit 是什么概念

  emit 就是创建一个 variable，然后将 Expr binding 到这个 variable 上。上面这句话是 emit 的官方文档解释，我的理解：emit 将 expr node 加入到当前的 block 当中，更具体的说，加入到当前的 `block.dataflow or block.function` scope 之下

- TensorOp 的 wrap 

  > A wrapper on top of relax.Expr whose struct_info is a TensorStructInfo, providing more convenient access shape and dtype information.

- Tensor & Parameter

  之前一直在纠结为什么 Tensor 没有 bound to any concrete value，但我们实际上可以使用 `Tensor.from_const` 来创建常量张量，这令我很困惑

  > Tensor is always symbolc and not bound to any concrete values.

  为了不再继续纠结，我在这里做简要的理解：这里实际上这里是为了和 Parameter 做区分，因为 Parameter 是可以 bound to concrete value

  > A parameter represents the weight of a neural network layer. It is a special tensor which could be bound or not bound to concrete values.

  Parameter 在进行编译之前一定会寻找 `self.data` 来初始化自己的权重，所以从这个意义上说，bound to any concrete values 指的是在进行编译时必须要绑定的值。Tensor 通常使用在 forward 当中，在编译时不会要求有绑定值

  绑定似乎是发生在 TorchModule or VirtualMachine 当中的

  ```python
  method = self.vm[method_name]
  
  if self.effects is not None:
      outputs, self.effects = method(*args, *self.effects, *self.params)
  else:
      outputs = method(*args, *self.params)
  ```

- `rx.Call` & `rx.Function`

  > 每个 value 对应一个 `relax.Call` 节点，表示对元函数的调用

  rx.Function 的概念和 BuildBlock.function & R.function 的概念是非常接近的，而 `rx.Call` 所代表的是调用一个 rx.Function

  似乎链路是这样的：Expr --(compose)--> Function --(compose)---> IRModule -> mod

  同时 `Expr` 能够调用 Function

  Function must be global variable? or bind to it, set a test

  Binding block 有什么作用？binding

  > binding often refers to the process of associating a variable or function name with a value or implementation. For example, in languages like Python or JavaScript, variable assignment is a form of binding.
  >
  > It refers to the process of creating a mapping from a name (or identifier) to a value or function.

  似乎 func 和 seqe 非常接近，经过 `func = rx.Function(self._func._params, seqe)` 过后，会将 seqe 正式包装成为一个函数

  ```python
  (Pdb) func
  # from tvm.script import relax as R
  
  @R.function(private=True)
  def main() -> R.Tuple(R.Object):
      with R.dataflow():
          _io: R.Object = R.null_value()
          lv: R.Tuple(R.Object) = (_io,)
          gv: R.Tuple(R.Object) = lv
          R.output(gv)
      return gv
  
  (Pdb) seqe
  with R.dataflow():
      _io: R.Object = R.null_value()
      lv: R.Tuple(R.Object) = (_io,)
      gv: R.Tuple(R.Object) = lv
      R.output(gv)
  ```

  一个更复杂的例子

  ```python
  (Pdb) func
  # from tvm.script import relax as R
  
  @R.function(private=True)
  def main(x: R.Tensor((2, 4), dtype="float32"), _io: R.Object, weight: R.Tensor((2, 4), dtype="float32")) -> R.Tuple(R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")), R.Tuple(R.Object)):
      with R.dataflow():
          add: R.Tensor((2, 4), dtype="float32") = R.add(x, x)
          add1: R.Tensor((2, 4), dtype="float32") = R.add(weight, x)
          gv1: R.Tuple(R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")), R.Tuple(R.Object)) = (add, add1), (_io,)
          R.output(gv1)
      return gv1
  
  
  (Pdb) seqe
  x: R.Tensor((2, 4), dtype="float32")
  weight: R.Tensor((2, 4), dtype="float32")
  _io: R.Object
  with R.dataflow():
      add: R.Tensor((2, 4), dtype="float32") = R.add(x, x)
      add1: R.Tensor((2, 4), dtype="float32") = R.add(weight, x)
      gv1: R.Tuple(R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")), R.Tuple(R.Object)) = (add, add1), (_io,)
      R.output(gv1)
  gv1
  
  (Pdb) self._func._blocks
  [x: R.Tensor((2, 4), dtype="float32")
  weight: R.Tensor((2, 4), dtype="float32")
  _io: R.Object
  with R.dataflow():
      add: R.Tensor((2, 4), dtype="float32") = R.add(x, x)
      add1: R.Tensor((2, 4), dtype="float32") = R.add(weight, x)
      gv1: R.Tuple(R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")), R.Tuple(R.Object)) = (add, add1), (_io,)
      R.output(gv1)]
  
  (Pdb) self._func._params
  [x, _io, weight]
  ```

  Var is mostly used for binding!!! binding is basically asignment, and decide the scope of the variable, like C++, all the variables need to have it's Type, Var must have StructureInfo in this case

  1. emit

     Bind the expression to variable

     If you need to use the expression's results, you need to bind it!!! emit is binding!!! binding is assignment!!!

     block builder would give binding name automatically 

  2. emit output
     output of the block must be binding to var, 在 [教程](https://mlc.ai/zh/chapter_integration/index.html#blockbuilder-api) 里说如果不 binding output 那么就无法在 block 外面使用这个变量，这个说法是不对的，经过实验是完全可以在 block 之外使用任意变量的

     但是我认为这对 dataflow block 的优化是有帮助的，在编译的时候去掉 side effect 运算

     目前发现，emit_output 的 binding scope 对 CUDA 编译是成立的。在最后 emit_func_output 必须使用 emit_output 标注好的变量，否则会报错
     
     ```python
     TVMError: Do not have a default for relax.expr.DataflowVar
     ```
     
  3. emit func output

     每一个 function 必有有且仅有一个 `emit_func_output`，这一步所完成的工作是比较多的

     1. normalze 所有的 output
     2. **binding function output**，`rx.SeqExpr`
     3. 将 block 中的输入 params 标出来，生成最终的 rx.Function
     4. 生成 function 加入到 IRModule 当中

- `ir.Op.get`

- `ir.extern`

- mod 编译以及 jit 编译应该如何使用，其区别是什么？

- ExprWithOp is different from Expr... Mostly it is referred to a Var, which is for binding. But it can also being with simple operation, because the basic operation is integraed, like add multiply...

  it returns a `rx.Call` node, which needs to be bound to a `rx.Var` node

## Block & Function

如何使用 Block 去构造 Function 并且灵活地调用 function，去调用其他 block 中的 function

这其中还涉及到 `call_tir & call_packed_func & rx.Call`，我认为他们之间是有联系的

**另一个点需要弄清楚的：`T.prim_func & R.function` 的区别** 

可以在 block 中直接调用 nn.Linear..[github](https://gist.github.com/YuchenJin/43729fc6f5ddf30e339258a7de5155c9)

this tells you how to add a tir to bb [discuss](https://discuss.tvm.apache.org/t/blockbuilder-emitting-tir-straight-away-solved/14863)

- R.function can use another R.function

  ```python
  from tvm import tir
  import tvm.relax as rx
  import tvm.relax.op as op
  m = tir.Var("m", "int64")
  n = tir.Var("n", "int64")
  x = rx.Var("x", rx.TensorStructInfo([m, n], "float16"))
  y = rx.Var("y", rx.TensorStructInfo([n], "float16"))
  bb = rx.BlockBuilder()
  with bb.function("func", [x, y]):
      with bb.dataflow() as df:
          lv0 = bb.emit(op.add(x, y)) # lv0 means local variable 0
          lv1 = bb.emit(op.multiply(lv0, y))
          # gv0 = bb.emit_output(lv1)
      lv4 = bb.emit(op.add(lv0, y))
      with bb.dataflow() as df:
          lv3 = bb.emit(op.add(lv1, y))
      op.print(lv3)
      bb.emit_func_output(lv1)
  mod = bb.finalize()
  
  new_block = rx.BlockBuilder()
  with new_block.function("func1", [x, y]):
      # use mod['func'] to get the function from the previous block
      gvar = new_block.add_func(mod['func'], func_name='')
      output = gvar(x , y)
      new_block.emit_func_output(output)
  
  new_mod = new_block.finalize()
  print(new_mod)
  ```

  script

  ```python
  # from tvm.script import ir as I
  # from tvm.script import tir as T
  # from tvm.script import relax as R
  
  @I.ir_module
  class Module:
      @R.function
      def func(x: R.Tensor(("m", "n"), dtype="float16"), y: R.Tensor(("n",), dtype="float16")) -> R.Tensor(("m", "n"), dtype="float16"):
          m = T.int64()
          n = T.int64()
          with R.dataflow():
              lv: R.Tensor((m, n), dtype="float16") = R.add(x, y)
              lv1: R.Tensor((m, n), dtype="float16") = R.multiply(lv, y)
              R.output()
          gv: R.Tensor((m, n), dtype="float16") = R.add(lv, y)
          with R.dataflow():
              lv2: R.Tensor((m, n), dtype="float16") = R.add(lv1, y)
              R.output()
          return lv1
  
      @R.function
      def func1(x: R.Tensor(("m", "n"), dtype="float16"), y: R.Tensor(("n",), dtype="float16")) -> R.Tensor(("m", "n"), dtype="float16"):
          m = T.int64()
          n = T.int64()
          cls = Module
          gv: R.Tensor((m, n), dtype="float16") = cls.func(x, y)
          return gv
  ```

  如果想要使用的话，可以用 `.get` 来获得当前的 `IRModule` 然后再获得 function
  
  ```python
  new_block = rx.BlockBuilder()
  with bb.function("func1", [x, y]):
      # use mod['func'] to get the function from the previous block
      gvar = bb.add_func(bb.get()['func'], func_name='')
      output = gvar(x , y)
      bb.emit_func_output(output)
  
  new_mod = bb.get()
  print(new_mod)
  ```

## Run IRModule on Devices

当构建好了 IRModule 过后，如何通过编译，让其在各个设备上运行？

参考 [mlc-tutorial](https://mlc.ai/zh/chapter_end_to_end/index.html#id9) [notebook](https://github.com/mlc-ai/notebooks)

- Run on CPU with torch & numpy

  1. `relax.build` 生成制定硬件的 VM executable，并构建 VirtualMachine
  2. 构建 tvm nd array
  3. 运行函数

  ```python
  import tvm
  import tvm.relax as relax
  from tvm.target import Target
  import numpy as np
  
  device = tvm.cpu()  # cpu(0)
  target = Target.from_device(device)  # "llvm"
  
  exec = relax.build(mod, target)
  vm = relax.VirtualMachine(exec, device)
  raw_data1 = np.arange(5).reshape((1, 5)).astype("float16")
  raw_data2 = np.arange(5).astype("float16")
  data1 = tvm.nd.array(raw_data1, device)
  data2 = tvm.nd.array(raw_data2, device)
  output = vm["func"](data1, data2)
  print(output)
  ```

- Run on GPU with torch

  在 GPU 上跑会比较麻烦，似乎需要 bind。但现在有了更好的 pipeline 流程，不需要走那么多 bind，就可以测试下效果

  ```python
  from tvm import tir
  import tvm.relax as rx
  import tvm.relax.op as op
  m = tir.Var("m", "int64")
  n = tir.Var("n", "int64")
  x = rx.Var("x", rx.TensorStructInfo([m, n], "float32"))
  y = rx.Var("y", rx.TensorStructInfo([n], "float32"))
  bb = rx.BlockBuilder()
  with bb.function("func", [x, y]):
      with bb.dataflow() as df:
          lv0 = bb.emit(op.add(x, y)) # lv0 means local variable 0
          lv1 = bb.emit(op.multiply(lv0, y))
          gv0 = bb.emit_output(lv1)
  
      bb.emit_func_output(gv0)
  
  mod = bb.get()
  
  def optimize_and_deploy(mod):
    # Use default graph optimization pipeline
    mod = relax.pipeline.get_pipeline()(mod)
  
    # Use default tensor function scheduling
    with tvm.target.Target("cuda"):
      mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
  
    # Step 3. deploy to GPU
    ex = relax.build(mod, "cuda")
    vm = relax.VirtualMachine(ex, tvm.cuda())
    return vm
  
  import tvm
  import tvm.relax as relax
  from tvm.target import Target
  import numpy as np
  
  device = tvm.cuda()
  target = Target.from_device(device)  # "cuda"
  
  vm = optimize_and_deploy(mod)
  # exec = relax.build(mod, target)
  # vm = relax.VirtualMachine(exec, device)
  
  
  raw_data1 = np.arange(50000 * 50).reshape((50, 50000)).astype("float32")
  raw_data2 = np.arange(50000).astype("float32")
  data1 = tvm.nd.array(raw_data1, device)
  data2 = tvm.nd.array(raw_data2, device)
  output = vm["func"](data1, data2)
  print(output)
  print(output.device)
  ```

- Run in JIT mode, what is difference with compiled library and JIT?

  JIT 模型目前可以方便地构建 CPU 上的模块验证，但是 GPU 上仍然无法完成，会报错

  ```python
  Did you forget to bind?
  Variable `B` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
  Variable `A` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
  Variable `T_add` is directly accessed by host memory (it is not contained in a thread environment or in the function arguments.
  ```

  估计还是因为没有设置好 shedule 的原因...

  ```python
  # test JIT with nn.Module
  import tvm
  import tvm.relax.frontend.nn as nn
  import tvm.relax as rx
  import torch
  
  class Module(nn.Module):
      def __init__(self):
          self.linear = nn.Linear(10, 5)
  
      def forward(self, x):
          return self.linear(x)
  
  module = Module()
  spec = {"forward": {"x": nn.spec.Tensor([1, 10], dtype="float32")}}
  state_dict = {"linear.weight": torch.rand((5, 10), dtype=torch.float32),
                "linear.bias": torch.rand((5,), dtype=torch.float32)}
  print(module.state_dict())
  module.load_state_dict(state_dict, strict=True)
  
  torch_module = module.jit(spec=spec, device="cpu")
  # ir_module, param = module.export_tvm(spec=spec)   # get the IR
  
  x = torch.rand((1, 10), dtype=torch.float32)
  
  y = torch_module["forward"](x)
  ```

  我还可以使用上面原始的方式，重新进行compile

  ```python
  ir_module, param = module.export_tvm(spec=spec)   # get the IR
  print(ir_module)
  vm = optimize_and_deploy(ir_module)
  linear_weight = tvm.nd.array(state_dict["linear.weight"].numpy(), device)
  linear_bias = tvm.nd.array(state_dict["linear.bias"].numpy(), device)
  data = tvm.nd.array(x.numpy(), device)
  output = vm["forward"](data, linear_weight, linear_bias)
  ```

  mlc_llm 中自己写了一个更复杂的 `mlc_llm` pipeline，但是步骤实在是比较多，目前没办法持续深挖

visualizing the graph [discuss](https://discuss.tvm.apache.org/t/rfc-visualizing-relay-program-as-graph/4825)

## Torchy frontend

[discuss](https://discuss.tvm.apache.org/t/design-torchy-productive-model-definition-in-tvm-unity/15404)

## Paper

- 阅读 [Relax paper](https://arxiv.org/abs/2311.02103)

  Relax 特点有两个：1. 支持动态 shape；2. 支持 cross-level optimization

  > 1. Relax uses variables to represent symbolic shape dimensions and uses a “best-effort” approach to infer dynamic shapes across tensor operators and function calls statically when possible, using a dynamic fallback otherwise.
  >
  > 2. Traditionally, ML compilers focus on optimizations within each abstraction level and do a uni-directional single-shot lowering to the next level. Relax brings computational graphs, tensor programs, and libraries into a single cross-level abstraction. We design the interaction of those components to enable cross-level interactions and partial lowering of programs within the same abstraction.

  

  Relax 的三大组件 (Language Constructs)：

  1. **structural annotations**

     每一个 value/object 在 relax 当中都有其结构信息，该结构信息被称为 annotations。value 代表的是任何能被程序操作的对象，包括：Tensors, Shape, Tupple, Callable

     > Each value in Relax is associated with an annotation that conveys structural information

     各个 value/object 的 annotations 的简要例子如下：

     - `Object`: A generic annotation for values that do not have a specific structure or shape information.
     - `Shape`: Represents a known shape of a tensor, such as `Shape([n, 4])` indicating a two-dimensional tensor with dimensions n and 4.
     - `Tensor`: Specifies that a value is a tensor with a particular shape and data type, such as `Tensor((n, 4), "f32")` for a float32 tensor with shape (n, 4).
     - `Tuple`: Represents a collection of heterogeneous values, like a pair or a list, for example, `Tuple[Tensor((n, 4), "f32"), Object]`.
     - `Callable`: Annotates a value as a function that can be called with certain input types and returns specific output types, such as `Callable([Tensor((n, 4), "f32")], Tensor((n * 4,), "f32"))`.

  2. **dataflow blocks**

     dataflow blocks 代表着一个子图区域，该区域是 side effect-free 的计算图，即：该部分计算图没有 control-flows，仅由一系列纯操作组成（a sequence of pure-operations）

     当一块子图被标记为 dataflow blocks，我们可以对其进行特定的优化，例如能够安全地去除一些未被使用的算子操作 (dead code elimination)，这些算子可能具有 side-effect

     > a "side-effect" refers to any operation that modifies something other than its own output or has an impact beyond the computation of its result. **Side-effects can include actions like writing to memory, reading from a file, printing to the console, or modifying global state.**

  3. **function calls**

     在 Relax 当中能够调用函数，该函数可以是同级别函数 (same level function, i.e. graph-level function to invoke anohter graph-level function)，也可以是跨级别函数 (cross-level function, i.e. graph-level function to invoke another tensor program functions or external library)

     通常 tensor program functions or external library functions 也被称作 loop-level functions

     > loop-level functions refer to functions or operations that are implemented at a lower abstraction level, specifically designed to handle iterative computations such as loops over tensor elements.

     之所以被称为 loop-level，我认为是因为在实现这些张量程序的时候，loop 的定义和顺序通常是优化的关键，所以这些更低层次的函数也被称为 loop-level



- 论文花了大量的内容介绍 **First-Class Symbolic Shape Abstraction**

  symoblic shape 作为 first-class citizen 是 relax 的重要设计，该设计使得 relax 能够处理动态形状的张量

- Cross-level abstraction

  为什么 cross-level 之间需要优化？loop-level function 需要 shape information 来进行定制化的优化操作，从而使得并行运算更快。所以如果可以将 graph-level 的形状信息传递给 loop-level function，那将更有助于优化

  > most tensor programs and libraries deal with low-level computations and often mutate their inputs. For these cases, we introduce a requirement to pass in input and output memory explicitly, via the destination-passing style

  Relax 为了完成 cross-level optmization，可以在 graph-level 中调用 loop-level function，调用方式为 `call_tir & call_packed_func`，在调用 `call_tir & call_packed_func` 时就会将形状信息一并传入

- 



- First-class citizen

  > When a concept is considered a first-class citizen, it means that it is fully integrated into the language and has the same level of support and flexibility as any other basic element, such as numbers or strings.

  更通俗的理解，当一个概念被作为 first-class citizen 时意味着该概念能够支持更广泛的操作（一般化操作）：

  1. Assignability，first-class citizens 能够赋值给变量，并且能被存储在数据结构中，例如存储在列表、字典中
  2. Passibility，first-class citizens 能够被作为参数传入到函数当中
  3. Returnability，能够作为函数返回值
  4. Constructable，能够动态地创建
  5. Closure，能够创建闭包





# Question



除此之外还要回答问题：

1. blockbuilder 是什么用
2. R.function 是什么用，应该和 block 有联系
3. call_tir & call_packed 区别





- Node

  既然 TensorIR 本质是一种树，那么节点的概念也就自然而然的诞生了。

  是 IR (TensorIR?) 的基本组成元素，也就说 IR 是由 Node 组成的

  **Node 是由什么组成的呢？**（Expr & Stmt）

- `call_tir & call_packed_fun`

- `ffi_api`

  `ffi` 代表 "Foreign Function Interface"，这是一个允许你调用外部函数或者库的机制。在编程语言中，FFI 通常用于实现语言之间的互操作性，它允许你在一个语言编写的程序中调用另一个语言编写的代码。

- Side effect

  我猜测 Effect 也只是一个名称而已...但是更重要的是经常用到一个东西叫做 `BlockBuilder` 这到底是怎么回事