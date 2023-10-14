# TVM 中迁移 Grid Sample

## Install TVM

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


## 迁移 Grid Sample

1. 使用的 `trt_grid_sampler_kernel.cu` 中对 grid sampler 的实现，创建了 helper 文件夹，主要针对 trt_grid_sampler.

2. 使用了 cnpy 对 numpy 的读取，这样能够方便对照。编译时需要使用 `-lz` flag

   问题：哪些是需要使用 cpu 数据，而哪些不需要？

## 问题

1. 有哪些可能的 attribute type，Integer，Bool，如何表示 string？

2. 如何获得 BevpoolRel 当中的 types？这应该是函数的输入？

3. 目前能够编译 tvm，导入 tvm plugin，但是在构建 function 时出现了 segmentation fault。这似乎是 bool type 所导致的，改成了 Integer
