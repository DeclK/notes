# Daily Progress

为了给自己形成良好的监督和记录习惯，创建这个 repo，用于记录每次的进展


## Dec, 2022
### Week 53

> Sat.

- 完成 C++ 基础知识总结，思考部署的具体情况：对于工业部署应当选择 trt，而对于移动端可尝试 onnxruntime 部署
- 还有一小时就是 2023 了，再见 2022。希望拥有更好管理思维的我，能够更快速的成长，前进！

> Mon.

- 将 deformable attention 用到了地铁项目上，达到了 ~90% 的准确率
- 整理完成 DETR 系列的笔记

### Week 52

Coming Home & Covid19...😢

### Week 51

>Sun.

- 完成对 DeNoising 技巧理解，需要了解 pytorch attention 的用法（`attn_mask`，`key_padding_mask`），以及如何快速搭建 transformer，detrex 是一个非常好的示范代码
- 初步完成对 DINO 的整理，放弃 contrastive denoising 代码阅读

> Sat.

- 基本完成对 two stage & iterative box refinement 的理解
- 对比了 deformable detr 和 DINO，差异主要存在于 DeNoising 这个技巧，可以直接跳过 DAB DETR，后续需要深入理解 query 该怎么用

>Fri.

- 深入阅读 Deformabel DETR 源码，完成对 Deformable Attention 的理解和整理
- 初步完成对 Two Stage Decoder 的理解

> Thu.

- 广泛地看了一下 ONNX 部署
- 完整阅读 DETR 代码，基本掌握过程
- 大概阅读了 Deformable-DETR, DN-DETR, DAB-DETR, DINO 的思想，并重新复习了 focal loss
- 了解 detectron2 的框架，对比了一下其与 mmengine，各有优略
- 发现自己在完成没有时间规划的任务时，容易转移注意力去处理一个子问题。没有能运用二八法则

> Mon.

- 了解 ONNX，计划使用 ONNX Runtime 进行模型部署
- 对英文论文的前半部分进行修改。因为是在实习期间写的，和最近写的有一些不匹配，造成行文不通顺

### Week 50

> Sun.

- 安装 detrex

> Sat.

- 完成英文论文 Loss，Experiment 撰写
- 将工作流转移到 redmi 上，了解了一下 Quillbot 和飞书妙计

> Fri.

- 完成 Manim 整理文档，对于一些使用的操作进行了熟悉，包括 Animation & Updater & Camera
- 完成了 BEVDemo 示意图，以及 SECA 速度分析

> Thu.

- 完成 Manim 复习工作，其中花费了较多时间去理解渲染流程

> Wed.

- 完成所有样本的可视化工作，找到了 vedo 无法可视化的真正原因
- 重新整理了 matplotlib 脚本，并决定以 manim 作为最终的 3D 渲染库去替代 vedo，原因：具有方便的旋转接口，自由度更高，有更完善的文档，并且能扩展到动画

> Tue.

- 完成角点校正超参数搜索，提交了一个大的版本更新
- 正在完成可视化脚本，基本有了思路，但是在 remote 无法运行代码，接下来需要调整一些后处理参数，以及相机位置，将可视化展示出来

>  Mon.

- 思考该如何进行写作。在面对字数要求时该如何写
- 整理 TODO Box，计划先迅速翻译出小论文，然后再逐渐扩写毕业论文

### Week  49

> Sun.

-  使用 LayerNorm 来替代 SubwayFormer 原来的归一化，收敛速度明显加快，减少了 MLP ratio 和 training epoch，大幅度缩短了训练时间 ~10 min
- 完成 Subway Health Classification 文档
- 完成实习报告

