# TVM Tutorial

1. 安装 llvm

   ```shell
   wget https://github.com/llvm/llvm-project/releases/download/llvmorg-9.0.1/llvm-project-9.0.1.tar.xz
   
   tar xf llvm-project-9.0.1.tar.xz
   
   cd llvm-project-9.0.1/llvm
   
   mkdir build && cd build
   
   cmake ..
   make -j8
   make install
   ```

2. 安装 tvm

   ```shell
   git clone --recursive https://github.com/apache/tvm tvm
   cd tvm
   git checkout 7fd47
   git apply xxx.patch
   
   mkdir build 
   cp cmake/config.cmake build
   cd build
   
   cmake ..
   make -j8 # multi process to speed up
   ```

   添加 tvm python path

   ```shell
   export TVM_HOME=/path/to/tvm
   export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
   ```

3. 编译模型

   下载 resnet50 onnx

   ```shell
   wget https://github.com/onnx/models/raw/b9a54e89508f101a1611cd64f4ef56b9cb62c7cf/vision/classification/resnet/model/resnet50-v2-7.onnx
   ```

   编译模型

   ```shell
   python -m tvm.driver.tvmc compile \
   --target "llvm" \
   --input-shapes "data:[1,3,224,224]" \
   --output resnet50-v2-7-tvm.tar \
   resnet50-v2-7.onnx
   ```

   输出一个 `.tar` 文件，可以解压

   ```shell
   mkdir model
   tar -xvf resnet50-v2-7-tvm.tar -C model
   ls model
   ```

   解压后有三个文件：

   - `mod.so` 是可被 TVM runtime 加载的模型，表示为 C++ 库。
   - `mod.json` 是 TVM Relay 计算图的文本表示。
   - `mod.params` 是包含预训练模型参数的文件。

4. 运行模型

   在运行前需要通过 `preprocess.py` 脚本成输入图片 `imagenet_cat.npz`

   ```shell
   python -m tvm.driver.tvmc run \
   --inputs imagenet_cat.npz \
   --output predictions.npz \
   resnet50-v2-7-tvm.tar
   ```

   获得预测结果过后，需要通过 `postprocess.py` 脚本查看输出

   ```shell
   class='n02123045 tabby, tabby cat' with probability=0.621104
   class='n02123159 tiger cat' with probability=0.356378
   class='n02124075 Egyptian cat' with probability=0.019712
   class='n02129604 tiger, Panthera tigris' with probability=0.001215
   class='n04040759 radiator' with probability=0.000262
   ```

5. 调优模型

   ```shell
   python -m tvm.driver.tvmc tune \
   --target "llvm" \
   --output resnet50-v2-7-autotuner_records.json \
   resnet50-v2-7.onnx
   ```

   可以使用调优参数重新编译模型

   ```shell
   python -m tvm.driver.tvmc compile \
   --target "llvm" \
   --tuning-records resnet50-v2-7-autotuner_records.json  \
   --output resnet50-v2-7-tvm_autotuned.tar \
   resnet50-v2-7.onnx
   ```

   测试调优模型

   ```shell
   python -m tvm.driver.tvmc compile \
   --target "llvm" \
   --tuning-records resnet50-v2-7-autotuner_records.json  \
   --output resnet50-v2-7-tvm_autotuned.tar \
   resnet50-v2-7.onnx
   ```

6. 模型对比

   ```shell
   python -m tvm.driver.tvmc run \
   --inputs imagenet_cat.npz \
   --output predictions.npz  \
   --print-time \
   --repeat 100 \
   resnet50-v2-7-tvm_autotuned.tar
   
   # Execution time summary:
   #  mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   #   81.1100      81.0930      83.1741      80.0667       0.5271   
   python -m tvm.driver.tvmc run \
   --inputs imagenet_cat.npz \
   --output predictions.npz  \
   --print-time \
   --repeat 100 \
   resnet50-v2-7-tvm.tar
   
   # Execution time summary:
   #  mean (ms)   median (ms)    max (ms)     min (ms)     std (ms)  
   #   95.4207      95.4308      97.3485      94.1116       0.6824   
   ```


## Grid Sample

1. 使用的 `trt_grid_sampler_kernel.cu` 中对 grid sampler 的实现，创建了 helper 文件夹，主要针对 trt_grid_sampler.

2. 使用了 cnpy 对 numpy 的读取，这样能够方便对照。编译时需要使用 `-lz` flag

   问题：哪些是需要使用 cpu 数据，而哪些不需要？





问题：有哪些可能的 attribute type，Integer，Bool，如何表示 string？

```c++
/*! \brief Attributes used for grid sample operator */
struct GridSampleAttrs: public tvm::AttrsNode<GridSampleAttrs>{
    std::string interpolation; // what are the possible type values?
    std::string padding_mode;
    Bool align_corners;
    TVM_DECLARE_ATTRS(GridSampleAttrs, "relay.attrs.GridSampleAttrs"){
        TVM_ATTR_FIELD(interpolation).describe("The interpolation mode to calculate output values.");
        TVM_ATTR_FIELD(padding_mode).describe("The padding mode to deal with border pixels.");
        TVM_ATTR_FIELD(align_corners).describe("Whether to align the corner pixels of input and output.");
    }
};
```

问题：如何获得 BevpoolRel 当中的 types？这应该是函数的输入？

函数的输入是由 xxx_api.cc 决定，types 问题还不太清楚

问题：为什么在 cast data 的时候需要使用 byte_offset

```c++
auto feat_data = reinterpret_cast<float*> (static_cast<char*>(feat->data)+feat->byte_offset);
```

问题：attribute 为什么没有在 api 中出现？那不是白定义了？发现了，是在 MakeGridsample 中有的，在 transform.py 里面是有的

问题：似乎 Bool 的 attr 行不通



问题：目前能够编译 tvm，导入 tvm plugin，但是在构建 function 时出现了 segmentation fault 

```shell
  27: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::TypedPackedFunc<tvm::runtime::String (tvm::runtime::ObjectRef const&)>::AssignTypedLambda<tvm::runtime::String (*)(tvm::runtime::ObjectRef const&)>(tvm::runtime::String (*)(tvm::runtime::ObjectRef const&), std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
  26: tvm::relay::PrettyPrint(tvm::runtime::ObjectRef const&)
  25: tvm::relay::TextPrinter::PrintFinal(tvm::runtime::ObjectRef const&)
  24: tvm::relay::RelayTextPrinter::PrintFinal(tvm::runtime::ObjectRef const&)
  23: tvm::relay::RelayTextPrinter::PrintScope(tvm::runtime::ObjectRef const&)
  22: tvm::relay::RelayTextPrinter::Print(tvm::runtime::ObjectRef const&, bool, bool)
  21: tvm::relay::RelayTextPrinter::PrintExpr(tvm::RelayExpr const&, bool, bool, bool)
  20: tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)
  19: void tvm::relay::ExpandDataflow<tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.0]
  18: tvm::relay::RelayTextPrinter::VisitLeaf(tvm::RelayExpr const&)
  17: _ZZN3tvm5relay11ExprFunctorIFNS0_3DocERKNS_9RelayExprEEE10InitVTableEvENUlR
  16: tvm::relay::RelayTextPrinter::VisitExpr_(tvm::relay::FunctionNode const*)
  15: tvm::relay::RelayTextPrinter::PrintFunc(tvm::relay::Doc const&, tvm::relay::Function const&)
  14: tvm::relay::RelayTextPrinter::PrintBody(tvm::runtime::ObjectRef const&, int)
  13: tvm::relay::RelayTextPrinter::PrintScope(tvm::runtime::ObjectRef const&)
  12: tvm::relay::RelayTextPrinter::Print(tvm::runtime::ObjectRef const&, bool, bool)
  11: tvm::relay::RelayTextPrinter::PrintExpr(tvm::RelayExpr const&, bool, bool, bool)
  10: tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)
  9: void tvm::relay::ExpandDataflow<tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}, tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#2}, tvm::relay::ExpandDataflow, tvm::relay::ExpandDataflow<{lambda(tvm::RelayExpr const&)#1}, tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1}>(tvm::RelayExpr, {lambda(tvm::RelayExpr const&)#1}, tvm::relay::RelayTextPrinter::VisitExpr(tvm::RelayExpr const&)::{lambda(tvm::RelayExpr const&)#1})::{lambda(tvm::RelayExpr const&)#1}) [clone .isra.0]
  8: tvm::relay::RelayTextPrinter::VisitLeaf(tvm::RelayExpr const&)
  7: _ZZN3tvm5relay11ExprFunctorIFNS0_3DocERKNS_9RelayExprEEE10InitVTableEvENUlR
  6: tvm::relay::RelayTextPrinter::VisitExpr_(tvm::relay::CallNode const*)
  5: tvm::relay::RelayTextPrinter::PrintCallAttrs(tvm::Attrs const&, tvm::RelayExpr const&)
  4: tvm::relay::RelayTextPrinter::AppendGenericAttrs(std::vector<tvm::relay::Doc, std::allocator<tvm::relay::Doc> >*, tvm::Attrs const&, bool)
  3: tvm::AttrsNode<tvm::relay::GridSampleAttrs>::VisitNonDefaultAttrs(tvm::AttrVisitor*)
  2: tvm::relay::RelayTextPrinter::AttrPrinter::Visit(char const*, tvm::runtime::ObjectRef*)
  1: tvm::relay::RelayTextPrinter::PrintAttributeValue(tvm::runtime::ObjectRef const&, bool)
  0: tvm::runtime::Object::DerivedFrom(unsigned int) const
  File "/Projects/cuda_tutorial/tvm/src/runtime/object.cc", line 73
InternalError: Check failed: child_tindex < type_table_.size() (1768712546 vs. 654) :
```





onnx 

注册 onnx

tvm/python/tvm/relay/frontend/onnx.py

auto tune

https://tvm.apache.org/docs/how_to/tune_with_autotvm/index.html

https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/index.html

pass

https://tvm.apache.org/docs/how_to/extend_tvm/index.html

fp16



