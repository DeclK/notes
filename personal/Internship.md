# 实习记录

1. 蔚来：3.10 一面，与面试官方向不符。需要精简对项目的描述，感觉当时讲解地很乱。需要了解更多的前沿论文，不同方向的。需要实现 NMS 算法。3.20 重新投递点云检测方向
2. 美团：3.5 笔试，3.17 晚上7点一面
3. 华为：3.23 晚上 7 点笔试
4. 字节：3.14 hr 打电话把简历推给游戏 ai，3.20 早10点笔试
5. 阿里：3.14 笔试 3.15 晚上7点一面
6. 商汤：等待面试通知...
7. 旷视：3.16 上午11点
8. 腾讯：3.14 hr打电话来询问意向，3.20 下午两点半腾讯面试
9. 百度：3.22 晚上 7 点笔试
10. 网易：3.27 下午3点笔试
10. 地平线
10. 仙途：3.25 下午三点面试，**4.1 下午三点面试**
10. 元戎
10. 文远
10. autox
10. 小马智行
10. 大疆
10. 博世

## 未投

1. 快手
2. 京东
3. 微软
6. 腾讯优图

## 项目介绍

1. 项目属性：是正在做的毕业设计，实验室条件有限，想要改进单阶段检测器
2. 出发点：希望单阶段检测器能够有两阶段检测器的准确度，但是单阶段检测器没办法像两阶段一样利用精细的**空间信息进行细化**，所以我使用了辅助网络执行**辅助任务**，将我们认为有帮助的空间信息更新到 backbone 当中，作为**信息的补偿**。具体来说辅助网络使用了 backbone 中的 feature 进行预测
3. 未来方向：
   1. 熟悉 center point。摆脱 anchor-based 的各种缺陷，feature misalign，长宽超惨，方向，可以尝试进行改进
   2. 了解 transformer 在点云目标检测中的运用。因为 transformer 是更强大的特征提取器，这是显然的发展方向。可以将 transformer 和 center point 结合起来
   3. domain adaption，网络已经能够非常好地处理 dense scene，但是对于稀疏场景识别能力很差，怎样让 dense scene 的知识提升稀疏场景知识
   4. 模型压缩，SE-SSD 两阶段压缩到一阶段
4. 尝试过：不同的辅助任务 & 将辅助任务结合到两阶段中。没有非常明显的提升，可能尝试一下多个 channel 会更合理一些

### NMS

参考 [知乎](https://zhuanlan.zhihu.com/p/64423753)

```python
def nms(data, thresh):
    """Pure Python NMS baseline. TODO: update with more concise codes"""
    x1 = data[:, 0]
    y1 = data[:, 1]
    x2 = data[:, 2]
    y2 = data[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    index = np.argsort(-data[:, -1], axis=-1)    # 重点，使用 index 保存仍需要计算的 boxes
    keep = []
    while index.size > 0:
        i = index[0]
        keep.append(data[i])
        xx1 = np.maximum(x1[i], x1[index])	# 仅计算 index 中的 boxes
        yy1 = np.maximum(y1[i], y1[index])
        xx2 = np.minimum(x2[i], x2[index])
        yy2 = np.minimum(y2[i], y2[index])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[index] - inter)
        index = index[ovr < thresh]
    return np.stack(keep, axis=0)
```
