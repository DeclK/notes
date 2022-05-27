# Autowise Intern

## Evaluation SimpleTrack with CenterPoint

（已经完成）使用 SimpleTrack 进行推理，并获得评估结果

1. 使用 simple track 作为 tracker，对我们的 detection results 进行跟踪：

   1. 问题1：nuscenes detection results 是以 json 文件存在的吗？是怎么表示的。回答：是的，`nusc_result.json`。具体表示未知，需要查看 json 文件
   1. 问题2：怎样得到 track 结果？回答：按照 repo 中的说明设置即可，最后 track 的结果以 results.json 文件存在
   2. 问题3：还有几个 json 文档什么意思：metrics_details.json & metrics_summary.json。回答：这两个 json 是最终的 evaluation 的结果

2. 得到了 tracking results 过后怎么使用 nuscenes 进行评估？直接使用 pub_test 里面的 eval 部分就可以

   有两个 issue 大概提了一下怎么 evaluation，但是没有细说：

   1. [How to get the json format detection files of NuScenes](https://github.com/TuSimple/SimpleTrack/issues/9)

   2. [Evaluation result](https://github.com/TuSimple/SimpleTrack/issues/10)

   从 centerpoint 中的 eval 代码进行了测试出现报错

   ```
   ValueError: Length of names must match number of levels in MultiIndex.
   ```

   尝试 [Tracking eval - Fix motmetrics version](https://github.com/nutonomy/nuscenes-devkit/pull/300) 通过降级 `pip install motmetric==1.1` 解决

   之后又遇到 [AssertionError: assert unachieved_thresholds + duplicate_thresholds...](https://github.com/tianweiy/CenterPoint/issues/292) 降级 numpy==1.19.2 解决

## nuScenes & SimpleTrack

深入了解了一下 nuScenes 数据集

tran-val 数据集一共有 850 个 scenes，每一个场景大概有 40 帧的数据（大部分）被标注，每一个场景大约有 20 秒。可以计算标注频率为 2 Hz。实际上数据的记录帧数是更长的，nuScenes 保留了这些未标注的数据，我们称之为 sweeps，为是 20 Hz，我们经常取 10 Hz 作为保留 sweep 的频率（9个未标注 sweep + 1个标注场景）作为数据输入

nuScenes 官方有一个库用于管理整个数据集，需要进一步了解！**如果要把 track 和 detection 结合起来的话，数据需要是时序的，时序数据的加载和获取需要进一步了解**

需要和 Learder 讨论一下下一步具体应该作什么：

0. 整个 pipline 敲定，对于每一个细节，根据 offboard 的思路，从 track extraction 开始？

1. ~~重构数据集，因为我们需要对时序序列进行预测，以此细化检测表现~~
   1. 需要进一步了解 nuScenes 的管理机制，即 nuscenes-devkit 一定是要整理一下的
   2. ~~并且了解 nuScenes 最终 output 的形式（好像是以 json 格式进行的），evaluation 的需要的 inference 结果形式是怎么样的？~~最好的方法就是把 CenterPoint 的代码拉通看一遍，知道 CenterPoint 的 batch_dict 表现形式，剩下的形式转换：pred_dict -> results.json 就交给现成代码去处理好了，不要深入太多
2. **对 SimpleTrack 的代码进行细致化了解，尤其是输入、输出**，对于 id switch 这种怎么计算的？

刚才和 Learder 讨论了一下：

1. 每周有一个会议的形式，更新一下进度
1. 以模块化的形式进行工作，先完成各个模块，并对各个模块进行评估，最后再穿起来
1. 路径：detection -> tracking -> classification -> static auto labeling -> dynamic labeling

## Training classification

问题：训练时无法收敛

初步思路，使用一个小的数据集，看能不能 Overfit

检查 label 是否正确

使用 gt 检查了一下，基本上正确，不管是 label 还是 one hot

## TODO

1. nuscenes-devkit 整理
2. SimpleTrack 接口整理
3. 完成 classification 正负样本划分
