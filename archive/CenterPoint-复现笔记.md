---
title: CenterPoint 复现笔记
tags:
  - CenterPoint
categories:
  - papers
abbrlink: ba7995ee
date: 2021-07-25 22:49:44
---

# CenterPoint 复现笔记

## Installation

1. pytorch下载如果很慢，请采用镜像！

2. 下载cmake使用apt，但要更新下载源为阿里云，这样版本才够新

3. 安装spconv时报错

   ```shell
   The following packages have unmet dependencies:
    gsettings-desktop-schemas : Breaks: mutter (< 3.31.4) but 3.28.4+git20200505-0ubuntu18.04.2 is to be installed
   E: Error, pkgProblemResolver::Resolve generated breaks, this may be caused by held packages.
   ```

   网络上一个解决方案是使用aptitude

   ```SHell
   # download aptitude
   sudo apt install aptitude
   sudo aptitude install <packagename>
   ```

   血泪教训，不要轻易使用aptitude，系统很多东西都被改写了，经过一番折腾升级到了ubuntu 20.04，也不知道之前安装的东西有没有被动

   印象中提到"held packages"，应该按照那种方式解决

4. 在下载 spconv 报错

   ```shell
   CMake Error at CMakeLists.txt:2 (project):
     No CMAKE_CUDA_COMPILER could be found.
     
   raise CalledProcessError(retcode, cmd)
   subprocess.CalledProcessError: Command '['cmake', '/home/declan/CenterPoint/spconv', '-DCMAKE_PREFIX_PATH=/home/declan/anaconda3/envs/centerpoint/lib/python3.6/site-packages/torch', '-DPYBIND11_PYTHON_VERSION=3.6', '-DSPCONV_BuildTests=OFF', '-DCMAKE_CUDA_FLAGS="--expt-relaxed-constexpr"', '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/home/declan/CenterPoint/spconv/build/lib.linux-x86_64-3.6/spconv', '-DCMAKE_BUILD_TYPE=Release']' returned non-zero exit status 1.
   ```

   尝试设置 CUDA 路径

   ```shell
   export PYTHONPATH="${PYTHONPATH}:/home/declan/CenterPoint"
   export PATH=/usr/local/cuda-10.0/bin:$PATH
   export CUDA_PATH=/usr/local/cuda-10.0
   export CUDA_HOME=/usr/local/cuda-10.0
   export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
   ```

   没有效果。#241 issues和我问题一模一样 https://github.com/traveller59/spconv/issues/241

   尝试增加路径 

   try adding '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.1/bin/nvcc' in cmake_args

   依旧失败

   尝试使用不同版本的g++ sudo apt install g++-7，显示已经下载，我们需要切换版本

   ls name* 可以列出所有name开头的包

   https://blog.csdn.net/FontThrone/article/details/104279224

   --slave 使得g++ gcc版本保持一致 70 90为重要权重priority

   ```shell
   sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 70 --slave /usr/bin/g++ g++ /usr/bin/g++-7
   sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9
   ```

   对g++和gcc进行管理

   ```shell
     Selection    Path            Priority   Status
   ------------------------------------------------------------
   * 0            /usr/bin/gcc-9   90        auto mode
     1            /usr/bin/gcc-7   70        manual mode
     2            /usr/bin/gcc-9   90        manual mode
   sudo update-alternatives --config gcc
   ```

   成功

5. 选择了 nuScenes mini 数据集

6. 运行测试

   ```shell
   python ./tools/dist_test.py infos_val_10sweeps_withvelo_filter_True.json  --work_dir work_dirs/CONFIG_NAME --checkpoint work_dirs/CONFIG_NAME/latest.pth --speed_test 
   ```

   报错

   ```shell
   ModuleNotFoundError: No module named 'torchie'
   ```

   尝试将torchie文档直接加入到PYTHONPATH当中

   ```shell
   export PYTHONPATH=$PYTHONPATH:/home/delcan/CenterPoint/det3d
   ```

   新报错

   ```shell
   Traceback (most recent call last):  File "./tools/dist_test.py", line 211, in <module>    main()  File "./tools/dist_test.py", line 102, in main    logger = get_root_logger(cfg.log_level)  File "/home/declan/CenterPoint/det3d/torchie/utils/config.py", line 146, in __getattr__    return getattr(self._cfg_dict, name)  File "/home/declan/CenterPoint/det3d/torchie/utils/config.py", line 29, in __getattr__    raise exAttributeError: 'ConfigDict' object has no attribute 'log_level'
   ```

   可能是因为 nuscenes_mini 测试集原因

   https://github.com/tianweiy/CenterPoint/issues/106

   我根据上面的内容对源码进行了修改，先尝试 train 一下

   依旧报错，再检查可能是因为配置文件没有对的原因

   https://github.com/open-mmlab/mmdetection/issues/3093

   通过正确配置 `nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py` 文件解决上面报错

   `./configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py`

   由于没有完整的 nuScenes 数据集，又有新的错误报错

   ```shell
   FileNotFoundError: [Errno 2] No such file or directory: 'nuScenes/samples/LIDAR_TOP/n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385105950634.pcd.bin'
   ```

   跳过这里，尝试跑一个 demo

   ```shell
   python ./tools/dist_test.py ./configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py --work_dir work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z --checkpoint work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/latest.pth --speed_test 
   ```

   上面的报错

   ```shell
   FileNotFoundError: [Errno 2] No such file or directory: 'nuScenes/samples/LIDAR_TOP/n015-2018-10-02-10-50-40+0800__LIDAR_TOP__1538448754047572.pcd.bin'
   ```

   查找目录发现这个文件其实是存在的，说明路径没有对，在 CentePoint 目录下添加 nuScenes 的软链接就可以解决了

7. 最终 RuntimeError: CUDA out of memory 即使 batch_size = 1 也直接炸了，放弃！

