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
11. 地平线
12. 仙途：3.25 下午三点面试，**4.1 下午三点面试**
13. 元戎
14. 智己
15. 文远
16. autox
17. 小马智行
18. 大疆
19. 博世

## 未投

1. 快手
2. 京东
3. 微软
6. 腾讯优图

## 项目介绍

1. 项目属性：是正在做的毕业设计，实验室条件有限，想要改进单阶段检测器
2. 出发点：希望单阶段检测器能够有两阶段检测器的准确度，但是单阶段检测器没办法像两阶段一样利用精细的**空间信息进行细化**，所以我使用了辅助网络执行**辅助任务**，将我们认为有帮助的空间信息更新到 backbone 当中，作为**信息的补偿**。具体来说辅助网络使用了 backbone 中的 feature 进行预测
3. 未来方向：
   1. 熟悉 center point。摆脱 anchor-based 的各种缺陷，feature misalign，长宽超参，方向，可以尝试进行改进
   3. domain adaption，网络已经能够非常好地处理 dense scene，但是对于稀疏场景识别能力很差，怎样让 dense scene 的知识提升稀疏场景知识
   4. 模型蒸馏

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


def nms(boxes, score, thresh):
    """
    NMS scripts based on numpy.
    Params:
        - data: (N, 5)
        - thresh: float
        - score: (N,)
    """
    lt = boxes[:, :2]
    rb = boxes[:, 2:]
    area = np.prod(rb - lt, axis=1)
    index = np.argsort(score)
    keep = []
    while len(index) > 0:
        cur_i = index[0]
        cur_box = boxes[cur_i]
        keep.append(cur_box)

        lt_inter = np.maximum(cur_box[:2], boxes[index, :2])
        rb_inter = np.minimum(cur_box[2:], boxes[index, 2:])
        wh_inter = np.clip(rb_inter - lt_inter, a_min=0)
        inter_area = np.prod(wh_inter, axis=1)

        iou = inter_area / (area[cur_i] + area[index] - inter_area)
        index = index[iou < thresh]
    return np.stack(keep, axis=0)
```
