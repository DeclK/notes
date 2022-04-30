# SA-SSD 复现

主要对于 SA-SSD 的辅助网络架构感兴趣，参考 [CSDN](https://blog.csdn.net/qq_39732684/article/details/105147497)

## SingleStageDetector

### Backbone

Backbone 是 `SimpleVoxel`

### SpMiddleFHD

读不下去了，还是需要实际运行看看怎么做

现在重新搭建 SA-SSD 环境

首先先下载pytorch！！

基础要先编译 spconv 再安装 SA-SSD	

下载 pybind11 `pip install pybind11`

准备使用 pytorch 1.4, spconv 1.1

```python
ImportError: libtorch_cpu.so: cannot open shared object file: No such file or directory
```

怀疑 pytorch 版本问题，尝试 pytorch 1.5，并重新 build

现在遇到新问题

```python
RuntimeError: Error compiling objects for extension
```

尝试 [issue](https://github.com/skyhehe123/SA-SSD/issues/79#issuecomment-820341500)

新问题

```python
ImportError: /opt/conda/lib/python3.7/site-packages/mmcv/_ext.cpython-37m-x86_64-linux-gnu.so: undefined symbol: _ZN2at5emptyEN3c108ArrayRefIlEERKNS0_13TensorOptionsENS0_8optionalINS0_12MemoryFormatEEE
```

考虑重新更改 mmcv 版本

```python
pip install mmcv-full==1.3.8 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.5.0/index.html
```

同样遇到 undefined symbol 问题，无法进行 spconv 编译

## 重新开始

使用 repo 所说的 torch 1.1

```shell
docker pull pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
```



```python
docker run --gpus all --shm-size=8g -it -v /home/chk/data:/shared -v /home/chk/.Xauthority:/root/.Xauthority -e DISPLAY --net=host --name
```

### spconv 1.0

[github](https://github.com/tyjiang1997/spconv1.0)

原 spconv 仓库已经停止使用 spconv1.0 版本了

查看 ubuntu 版本

```shell
uname -a 
Linux psdz 4.15.0-142-generic #146~16.04.1-Ubuntu SMP Tue Apr 13 09:27:15 UTC 2021 x86_64 x86_64 x86_64 GNU/Linux
```

**替换 apt source** 

`/etc/apt/`

遇到问题

```python
E: The method driver /usr/lib/apt/methods/https could not be found.
N: Is the package apt-transport-https installed?
E: The method driver /usr/lib/apt/methods/https could not be found.
N: Is the package apt-transport-https installed?
E: The method driver /usr/lib/apt/methods/https could not be found.
N: Is the package apt-transport-https installed?
E: The method driver /usr/lib/apt/methods/https could not be found.
N: Is the package apt-transport-https installed?
E: Failed to fetch https://mirror.nju.edu.cn/ubuntu/dists/xenial/InRelease  
E: Failed to fetch https://mirror.nju.edu.cn/ubuntu/dists/xenial-updates/InRelease  
E: Failed to fetch https://mirror.nju.edu.cn/ubuntu/dists/xenial-backports/InRelease  
E: Failed to fetch https://mirror.nju.edu.cn/ubuntu/dists/xenial-security/InRelease  
E: Some index files failed to download. They have been ignored, or old ones used instead.
```

解决方法

先使用原始的 source.list 下载 apt-transport-https `apt install apt-transport-https` 然后再 apt update

**更新 Pip 源，北师大**

```python
pip install pip -U
pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
```





下载 `apt-get install libboost-all-dev`



下载 `pip install cmake`



**python setup.py bdist_wheel** 

遇到如下报错，更新一下 CUDA 路径

```shell
CMake Error at /opt/conda/lib/python3.6/site-packages/cmake/data/share/cmake-3.22/Modules/CMakeDetermineCUDACompiler.cmake:179 (message):
  Failed to find nvcc.

  Compiler requires the CUDA toolkit.  Please set the CUDAToolkit_ROOT
  variable.
Call Stack (most recent call first):
  CMakeLists.txt:2 (project)
```

是下载的 docker 镜像的问题，需要下载 devel 版本的镜像，我下的是 runtime 镜像，重新来过吧



pip install xxx.wheel

成功！

## SA-SSD

1. `python3 setup.py build_ext --inplace`

   报错

   ```shell
   gcc -pthread -B /opt/conda/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/opt/conda/lib/python3.6/site-packages/torch/include -I/opt/conda/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -I/opt/conda/lib/python3.6/site-packages/torch/include/TH -I/opt/conda/lib/python3.6/site-packages/torch/include/THC -I/usr/local/cuda/include -I/opt/conda/include/python3.6m -c src/iou3d.cpp -o build/temp.linux-x86_64-3.6/src/iou3d.o -g -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=iou3d_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
   cc1plus: warning: command line option '-Wstrict-prototypes' is valid for C/ObjC but not for C++
   src/iou3d.cpp: In function 'int boxes_overlap_bev_gpu(at::Tensor, at::Tensor, at::Tensor)':
   src/iou3d.cpp:38:24: error: 'CHECK_INPUT' was not declared in this scope
        CHECK_INPUT(boxes_a);
                           ^
   src/iou3d.cpp: In function 'int boxes_iou_bev_gpu(at::Tensor, at::Tensor, at::Tensor)':
   src/iou3d.cpp:59:24: error: 'CHECK_INPUT' was not declared in this scope
        CHECK_INPUT(boxes_a);
                           ^
   src/iou3d.cpp: In function 'int nms_gpu(at::Tensor, at::Tensor, float)':
   src/iou3d.cpp:79:22: error: 'CHECK_INPUT' was not declared in this scope
        CHECK_INPUT(boxes);
                         ^
   src/iou3d.cpp: In function 'int nms_normal_gpu(at::Tensor, at::Tensor, float)':
   src/iou3d.cpp:129:22: error: 'CHECK_INPUT' was not declared in this scope
        CHECK_INPUT(boxes);
                         ^
   error: command 'gcc' failed with exit status 1
   root@psdz:/sa-ssd/mmdet/ops/iou3d# ls
   build  iou3d_utils.py  setup.py  src
   root@psdz:/sa-ssd/mmdet/ops/iou3d# rm -rf build/
   root@psdz:/sa-ssd/mmdet/ops/iou3d# ls
   iou3d_utils.py  setup.py  src
   root@psdz:/sa-ssd/mmdet/ops/iou3d# python setup.py build_ext --inplace
   running build_ext
   building 'iou3d_cuda' extension
   creating build
   creating build/temp.linux-x86_64-3.6
   creating build/temp.linux-x86_64-3.6/src
   gcc -pthread -B /opt/conda/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/opt/conda/lib/python3.6/site-packages/torch/include -I/opt/conda/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -I/opt/conda/lib/python3.6/site-packages/torch/include/TH -I/opt/conda/lib/python3.6/site-packages/torch/include/THC -I/usr/local/cuda/include -I/opt/conda/include/python3.6m -c src/iou3d.cpp -o build/temp.linux-x86_64-3.6/src/iou3d.o -g -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=iou3d_cuda -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11
   cc1plus: warning: command line option '-Wstrict-prototypes' is valid for C/ObjC but not for C++
   src/iou3d.cpp: In function 'int boxes_overlap_bev_gpu(at::Tensor, at::Tensor, at::Tensor)':
   src/iou3d.cpp:38:24: error: 'CHECK_INPUT' was not declared in this scope
        CHECK_INPUT(boxes_a);
                           ^
   src/iou3d.cpp: In function 'int boxes_iou_bev_gpu(at::Tensor, at::Tensor, at::Tensor)':
   src/iou3d.cpp:59:24: error: 'CHECK_INPUT' was not declared in this scope
        CHECK_INPUT(boxes_a);
                           ^
   src/iou3d.cpp: In function 'int nms_gpu(at::Tensor, at::Tensor, float)':
   src/iou3d.cpp:79:22: error: 'CHECK_INPUT' was not declared in this scope
        CHECK_INPUT(boxes);
                         ^
   src/iou3d.cpp: In function 'int nms_normal_gpu(at::Tensor, at::Tensor, float)':
   src/iou3d.cpp:129:22: error: 'CHECK_INPUT' was not declared in this scope
        CHECK_INPUT(boxes);
                         ^
   error: command 'gcc' failed with exit status 1
   ```

   怀疑是不是 repo 的问题，重新 git clone 一个 repo 试一试，这个 repo 有 500 MB 是真滴大

   成功

2. **终于到了最重要的运行阶段！！！！！！**

   现在遇到了 mmcv 方面的问题，我尝试装一下 mmdetection

   ```shell
   pip install openmim
   mim install mmdet
   ```

   似乎不需要安装完整的 mmdetection 项目 [CSDN](https://blog.csdn.net/qq_38316300/article/details/110161110)

   ```shell
   pip install mmcv==0.5.0
   ```

   继续运行遇到报错

   ```python
   Traceback (most recent call last):
     File "train.py", line 5, in <module>
       from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
     File "/opt/conda/lib/python3.6/site-packages/mmcv/__init__.py", line 5, in <module>
       from .image import *
     File "/opt/conda/lib/python3.6/site-packages/mmcv/image/__init__.py", line 2, in <module>
       from .colorspace import (bgr2gray, bgr2hls, bgr2hsv, bgr2rgb, gray2bgr,
     File "/opt/conda/lib/python3.6/site-packages/mmcv/image/colorspace.py", line 2, in <module>
       import cv2
     File "/opt/conda/lib/python3.6/site-packages/cv2/__init__.py", line 8, in <module>
       from .cv2 import *
   ImportError: libGL.so.1: cannot open shared object file: No such file or directory
   ```

   尝试下载 `pip install opencv-python-headless`

   遇到报错

   ModuleNotFoundError: No module named 'mmdet'

   修改 PYTHONPATH

   ```python
   import sys
   sys.path.append('/SA-SSD')
   ```

   接下来就是却啥安装啥

   修改 data root 

   现在能运行一些了，遇到报错

   ```python
   Traceback (most recent call last):
     File "train.py", line 127, in <module>
       main()
     File "train.py", line 117, in main
       log_interval = cfg.log_config.interval
     File "/SA-SSD/tools/train_utils/__init__.py", line 99, in train_model
       log_interval = log_interval
     File "/SA-SSD/tools/train_utils/__init__.py", line 40, in train_one_epoch
       for i, data_batch in enumerate(train_loader):
     File "/opt/conda/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 582, in __next__
       return self._process_next_batch(batch)
     File "/opt/conda/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 608, in _process_next_batch
       raise batch.exc_type(batch.exc_msg)
   numba.core.errors.TypingError: Traceback (most recent call last):
     File "/opt/conda/lib/python3.6/site-packages/torch/utils/data/_utils/worker.py", line 99, in _worker_loop
       samples = collate_fn([dataset[i] for i in batch_indices])
     File "/opt/conda/lib/python3.6/site-packages/torch/utils/data/_utils/worker.py", line 99, in <listcomp>
       samples = collate_fn([dataset[i] for i in batch_indices])
     File "/SA-SSD/mmdet/datasets/kitti.py", line 131, in __getitem__
       data = self.prepare_train_img(idx)
     File "/SA-SSD/mmdet/datasets/kitti.py", line 206, in prepare_train_img
       self.augmentor.noise_per_object_(gt_bboxes, points, num_try=100)
     File "/SA-SSD/mmdet/core/point_cloud/point_augmentor.py", line 334, in noise_per_object_
       valid_mask, loc_noises, rot_noises)
     File "/opt/conda/lib/python3.6/site-packages/numba/core/dispatcher.py", line 420, in _compile_for_args
       error_rewrite(e, 'typing')
     File "/opt/conda/lib/python3.6/site-packages/numba/core/dispatcher.py", line 361, in error_rewrite
       raise e.with_traceback(None)
   numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
   Internal error at <numba.core.typeinfer.CallConstraint object at 0x7fe300c9f048>.
   Failed in nopython mode pipeline (step: nopython mode backend)
   scipy 0.16+ is required for linear algebra
   
   
   Traceback (most recent call last):
     File "train.py", line 127, in <module>
       main()
     File "train.py", line 117, in main
       log_interval = cfg.log_config.interval
     File "/SA-SSD/tools/train_utils/__init__.py", line 99, in train_model
       log_interval = log_interval
     File "/SA-SSD/tools/train_utils/__init__.py", line 40, in train_one_epoch
       for i, data_batch in enumerate(train_loader):
     File "/opt/conda/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 582, in __next__
       return self._process_next_batch(batch)
     File "/opt/conda/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 608, in _process_next_batch
       raise batch.exc_type(batch.exc_msg)
   numba.errors.TypingError: Traceback (most recent call last):
     File "/opt/conda/lib/python3.6/site-packages/torch/utils/data/_utils/worker.py", line 99, in _worker_loop
       samples = collate_fn([dataset[i] for i in batch_indices])
     File "/opt/conda/lib/python3.6/site-packages/torch/utils/data/_utils/worker.py", line 99, in <listcomp>
       samples = collate_fn([dataset[i] for i in batch_indices])
     File "/SA-SSD/mmdet/datasets/kitti.py", line 131, in __getitem__
       data = self.prepare_train_img(idx)
     File "/SA-SSD/mmdet/datasets/kitti.py", line 206, in prepare_train_img
       self.augmentor.noise_per_object_(gt_bboxes, points, num_try=100)
     File "/SA-SSD/mmdet/core/point_cloud/point_augmentor.py", line 334, in noise_per_object_
       valid_mask, loc_noises, rot_noises)
     File "/opt/conda/lib/python3.6/site-packages/numba/dispatcher.py", line 401, in _compile_for_args
       error_rewrite(e, 'typing')
     File "/opt/conda/lib/python3.6/site-packages/numba/dispatcher.py", line 344, in error_rewrite
       reraise(type(e), e, None)
     File "/opt/conda/lib/python3.6/site-packages/numba/six.py", line 668, in reraise
       raise value.with_traceback(tb)
   numba.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
   Internal error at <numba.typeinfer.CallConstraint object at 0x7fc22ccaec88>.
   Failed in nopython mode pipeline (step: nopython mode backend)
   scipy 0.16+ is required for linear algebra
   
   File "../mmdet/core/bbox3d/geometry.py", line 537:
   def box2d_to_corner_jit(boxes):
       <source elided>
           rot_mat_T[1, 1] = rot_cos
           box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]
           ^
   
   [1] During: lowering "$176.35 = $176.33 <built-in function matmul> rot_mat_T" at /SA-SSD/mmdet/core/bbox3d/geometry.py (537)
   [2] During: resolving callee type: type(CPUDispatcher(<function box2d_to_corner_jit at 0x7fc24a0e8950>))
   [3] During: typing of call at /SA-SSD/mmdet/core/point_cloud/point_augmentor.py (80)
   
   Enable logging at debug level for details.
   
   File "../mmdet/core/point_cloud/point_augmentor.py", line 80:
   def noise_per_box(boxes, valid_mask, loc_noises, rot_noises):
       <source elided>
       num_tests = loc_noises.shape[1]
       box_corners = box2d_to_corner_jit(boxes)
       ^
   
   ```



下载 scipy

新报错

AttributeError: 'Tensor' object has no attribute 'bool'

根据 [issue](https://github.com/skyhehe123/SA-SSD/issues/61) 修改成功！终于能顺利跑起来了，但由于目前实验室 GPU 别人还在用，先等一手
