# VideoMAE Code Read

github: https://github.com/MCG-NJU/VideoMAE

## Concept

- 在查看 VideoMAE 代码的时候发现了 [decord](https://github.com/dmlc/decord)，一个可以替代 OpenCV 来读取视频的库，在查看 decord 时候又发现了 [GluonCV 的文档](https://cv.gluon.ai/build/examples_action_recognition/decord_loader.html)，这相对于 Dive Into Deep Learning 似乎是一个更全面的视频任务介绍

  处理 ssv2 数据集需要将 webm 格式转换为 mp4 格式，这样才好用 decord 来进行处理，[issue](https://github.com/MCG-NJU/VideoMAE/issues/62)

- 

## Question

- 在 GluonCV 的文档里说：训练一个 Kinetics-400 大约会花费 8 个 V100 GPU 10 天的时间。但是一个 ImageNet1K 的数据量也有 1Million 张图像，而用 8卡 A100 训练也只需要数小时，或许里面还有上升空间