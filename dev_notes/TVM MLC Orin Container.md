# TVM MLC Orin Container

## What is L4T & JetsonPack

From ChatGPT: what is L4T?

> L4T, which stands for Linux for Tegra, is a software platform provided by NVIDIA designed specifically for the NVIDIA Tegra processor family. It includes a highly optimized version of the Linux operating system along with drivers, APIs, and developmental tools that are tailored to support the features and capabilities of Tegra processors. For devices like the NVIDIA Jetson Orin, which is part of the NVIDIA Jetson series aimed at enabling AI and machine learning applications at the edge, L4T provides the necessary foundation to develop, deploy, and run complex AI models and applications directly on the device.

From ChatGPT: what is JetPack?

> JetPack SDK is a software development kit provided by NVIDIA for the Jetson platform, which includes a suite of software tools and technologies for developing applications on NVIDIA Jetson embedded computing boards.

从二者的关系上来说：JetPack 包含 L4T，也就是说 JetPack 是 L4T 的一个超集，其还包括一些深度学习、CUDA 相关的库/SDK

参考资料：

[Jetson Containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch/tags)

[Pytorch L4T](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch/tags)

## How to build containers manually?

目前正在尝试手动 build 一个 jetson containers，但是我发现我在 apt install 的时候会受到很多的阻力，根本就找不到包，尝试了 r35~r36 的所有 l4t-base 环境都会出现问题

jetson containers 主要通过 packages 来构建 docker image，我们可以通过查看 package 中的 Dockerfile 来获得参考。通常 Dockerfile 会调用一个 `install.sh & build.sh` 来构建镜像，同时通过 `config.py` 来管理不同版本的 package 配置

我最需要的可能是找到匹配的 os-release version 基础镜像，并以此为基础构建所需要的 custome package
```shell
cat /etc/os-release
```
这是因为 container 中的一些 lib 是直接依赖于 host 中的 lib。从表现上来看，我发现 container 会直接调用 host 中的 `/usr/local/cuda` 以及 `apt`。所以如果不匹配的话，经常会造成依赖缺失或者运行失败