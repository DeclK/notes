# Daily Progress

为了给自己形成良好的监督和记录习惯，创建这个 repo，用于记录每次的进展

## Mar, 2023

### Week 10

> Fri.

- 完成所有可视化图像，包括改进样本，角点热力图，角点偏移说明
- 初步过完赵老师的 revised version

> Thu.

- 可视化的数据基本准备，包括获得数据、搭建本地环境

> Wed.

- 大论文字数达到5.4k
- 决定放弃对 KITTI 数据集的测试，认为本质原因是数据量和调参经验的不足
- 重新思考了决策原则，必须专注于完成紧急且重要的事情，在完成的过程中可能会想做其他不紧急但重要的事情，这必须禁止，可以先记录下来之后有空再解决，因为这是无限游戏。结合几个关键词：损失厌恶、复利、赢者通吃

> Tue.

- 大论文字数到达3.9k
- 获得了 ONCE 数据集的测试集结果

> Mon.

- 大论文字数到达3.5k
- 进行了 KITTI 实验，包括小 batch size（x）更强的数据增强（x）

## Feb, 2023

### Week 9

> Sun.

- 初步获得 mean teacher on kitti 的结果，在 Ped & Cyc 应该是可以获得接近 SOTA 的结果，但是 Car 依然不好，决定使用更深（x）的网络和单独类（x）别训练
- ChatGPT 更适合用于填充子论点，如果直接让其构思整个段落，并不会符合你的思路。人是线，ChatGPT 是点

> Sat.

- 完成 kitti semi dataset 搭建，开始训练 mean teacher。这个过程主要是对 kitti 的配置文件进行 debug，关键在于 kitti 和 once 使用的数据增强并不相同
- 尝试解决 tensorbard bug，但是失败，移动到本地查看成功
- 开始写中文论文

> Fri.

- 查看 ONCE Benchmark test split，发现网站已经不再运营。又回到了 KITTI 数据集上
- 了解了 conference 和 journal 的区别，conference 是更合适的选择
- 尝试将 OpenPCDet 的环境移动到自己的电脑上，尝试失败
- 搭建了 kitti semi dataset 初步

> Thu.

- 完成论文的所有表格的 latex 格式，论文进行了初级的排版
- 规划了可视化的一些内容

> Wed.

- 听 maki 的认知视频，思考如何调动起自己的积极性。即使是面对自己感觉无聊的事情，也要找到能够学习和探索的点来刺激自己
- 认清楚什么是真实的阻碍，什么是自己给自己设定的阻碍。自己想了一个比喻：真实的阻碍就是面前的迷宫，需要不断地碰壁才能解决；而自己设定的阻碍就像不断从耳边吹过的杂风，让你感到不适和泄气
- 和赵老师沟通，感觉赵老师对我的要求更高。和童俊文沟通，他打算之后推出茅台项目，重新回到3D视觉的研究

> Tue.

- 完成小论文图片的 resize，并整理了作者相关的格式
- 完成公式的整理和笔记
- 注册 chatgpt，我感觉这个东西也是一旦使用就回不去的，问了一些 latex 问题，回答得相当不错
- 将小论文推到了 Method-Loss Function 章节

> Mon.

- 完成小论文摘要和总结的撰写
- 开学第一次网球复健

### Week 8

> Mon.

- 到达学校，整理了一下房间和办公室
- 去校医院查看了手指，原因是免疫力下降导致的疱疹，开了阿昔洛韦
- 开始阅读《穷查理宝典》
- 了解了一下 ChatGPT，初步打算用于论文的润色和扩写

> Tue.

- Valentines Day with Bestey🥰

> Wed.

- 修改英文论文，又感觉遇到了抵触情绪，更多的是害怕的情绪。改得比较慢，目前改了一小半

> Thu.

- 继续修改英文论文，只剩下了角点模块的修改
- 找了两份知网的毕业论文范文，对扩写有了初步的把握

> Fri.

- 完成了英文论文对角点模块的修改
- 完成了中文论文的扩写指南，基本结构按照初稿写，并添加一些小节，这些小节在一般期刊里本是一笔带过的
- 整理了一半费曼读书笔记

> Sat.

- 完成费曼读书笔记整理
- 继续学习 latex tutorial，决定先凑一个 aaai23 的模板出来以学习

> Sun.

- 完成 tex tutorial，补充了对图片的排版，利用 zotero 引用参考文献，分章节撰写等方法
- 撰写了 1/3 摘要

## Jan, 2023

- 获得 C2 驾照
- 完成《别逗了，费曼先生》阅读

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

