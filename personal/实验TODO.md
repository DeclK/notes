# Experiment

TODO：

0. ~~更改模型载入方法~~
1. ~~（deprecate）构建 ODIoU loss, 在单阶段中加入 iou 预测头并进行预测~~
2. ~~**构建 Consistency loss**。对 cls 和 reg 进行 consistency loss~~
3. ~~构建 ODIoU loss~~
3. ~~se-ssd 中的 data augmentation，已经在 OpenPCDet 中实现~~

刚刚测试了一下，似乎 voxel rcnn 的 roi head 并不是一个即插即用的头，看来不同的 second 所提取出来的 feature 是完全不一样的，不能够使用一个 roi head 来做预测。除非是 Lidar R-CNN。所以要使用一个 end-to-end 的方式来训练模型，看看这样的结果能否提升单阶段/两阶段

TODO:

1. ~~ODIoU loss，需要使用不一样的 loss 流程~~
2. ~~Consistency loss。还差一个权重的设置~~

现在已经将所有的设置都弄好了，基本上放弃了 sa-ssd 的实现，可能 sa-ssd 陷入了一个局部最优解，没有找到一个比较平滑的方法，而且训练起来很慢

TODO:

1. ~~查看自己复现的 cia-ssd 和原版的差距~~

   复现笔记：1. 在 docker hub 上搜索一个 det3d 镜像；2. git clone cia-ssd，根据提示进行安装，遇到报错 `pytest_runner` 相关，直接 pip install 就好

   在创建数据集的时候遇到错误 `RuntimeError: builtin cannot be used as a value:` 似乎是 torchvision 版本的问题，docker 里安装的是 torch 1.3 和 torchvision 0.5，现在安装了 torchvision 0.4.2。现在遇到新错误 `ImportError: cannot import name 'PILLOW_VERSION'`，尝试卸载 Pillow 然后重新 `pip install Pillow`，建议先更新一下 pip `pip install --upgrade pip`。遇到 `ModuleNotFoundError: No module named 'terminaltables'`，直接 pip 下载，之后都这样处理...都是使用 pip 安装，除了个别 scikit-image, opencv-python-headless(ImportError: libSM.so.6: cannot open shared object file: No such file or directory) 是需要更换一下名字的

   现在已经能够跑通 cia-ssd 的模型了，将 dataloader 的 shuffle 关闭了，发现竟然使用了相同的 random seed 也是无法复现...不清楚为什么，现在对比我的实现和原来的实现有什么不同。检查过了 ssfa 的权重是相同的

   由于各种随机性的原因，我认为是不太可能复现两个框架下的结果的...这将花费相当大的调试，只能够通过一些分析比较，看两个框架哪些地方不一样，然后尽力去更改。这一次调整了参数初始化，也调整了 epoch，基本上整个框架都对应起来了，也有可能是因为中途的一些改动而导致效果不好，可能之后再尝试一下用 OpenPCDet v0.5 的版本跑一下，已经尽力了😢

2. ~~考虑更新 OpenPCDet 到 v0.5 的版本，支持了 docker 安装，而且支持 spconv 2.0 速度应该更快了。并且由于自己 git commit 的次数过于多了，似乎需要更稳定的代码支持~~

3. ~~（deprecate）将两阶段的网络分离出来训练，单阶段的权重冻结，看看效果如何，这样的代价更小，不过参数会难以调整~~

## ONCE

有点激动，刚刚看到了 [ONCE_Benchmark](https://github.com/PointsCoder/ONCE_Benchmark) 的 repo，之前还以为没有发放！而且是在 OpenPCDet 上进行实验的，这不是天公作美？这下直接奠定了 OpenPCDet 在点云检测的主要地位了，这下可有的忙活了，既然在 OpenPCDet 下已经实现了 Mean Teacher 的方法，拿过来用岂不是美滋滋

解压多个文件 `for tar in *.tar;  do tar xvf $tar; done`

把 ONCE Benchmark 中有关半监督的结构熟悉了一下，真的是非常的干净漂亮，那么之前的两个阶段的想法实现可能性大大提高！今天还获得一个消息就是辅助网络真的有用，即使是在 pointpillars 上面，也能够有 3% 的提升

ImageSet 需要通过 [once-devkit](https://github.com/once-for-auto-driving/once_devkit) 下载，并且在处理数据集的时候修改一下 `splits = ['train', 'val', 'test', 'raw_small', 'raw_medium', 'raw_large'] `

TODO:

1. Baseline of ONCE
2. CIA reproduce
3. SA-SSD reproduce (if possible, replace CIA structure with auxilary structure)
4. Voxel R-CNN reproduce
5. CIA + SESS reproduce
6. SA-SSD reproduce
7. CenterPoint + SESS reproduce
8. SA-SSD + CenterPoint + SESS reproduce
9. CIA + SESS R-CNN (Final Goal, process still need to discuss)