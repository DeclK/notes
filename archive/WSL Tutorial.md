---
title: WSL Tutorial
cagegories:
  - 编程
  - tools
date: 2022-12-5
---

# WSL Tutorial

## 安装

参考 [wsl official](https://learn.microsoft.com/en-us/windows/wsl/install)

如果之前没有安装 wsl 过直接

```cmd
wsl --install
```

还可以尝试手动安装 [wsl manual install](https://learn.microsoft.com/en-us/windows/wsl/install-manual#step-3---enable-virtual-machine-feature)，完成两个步骤即可：开启虚拟环境，下载升级固件

安装完成了的话，先将 wsl 版本设置为 wsl2

```cmd
wsl --set-default-version 2
wsl --set-version <distro name> 2
```

然后查看可下载的 linux 版本，并下载

```cmd
wsl -l -o 	# --list --online
wsl --install -d Ubuntu-20.04
```

下载完后可查看下载的版本，并进行切换

```cmd
wsl -l -v
############
  NAME            STATE           VERSION
* Ubuntu-20.04    Stopped         2
############
wsl -s Ubuntu-20.04
```

安装完成后可以直接在 cmd 中输入 `wsl` 打开你的 linux 子系统，也可以在应用列表里找到。如果想要删除的话需要先卸载对应的应用（一般在 Microsoft Store App 里），然后再 unregister

```cmd
wsl --unregister Ubuntu
```

可以直接在子系统里打开 vscode，在对应文件夹输入 `code .`，可以下载 WSL 插件，插件名就叫 WSL

网络问题尽可能使用镜像解决，实在不行就先使用本地下载，然后再搬运到 WSL 中。文档中说明，如果文件保存在本地的话，WSL 获取的速度会比较慢

参考 [博客](https://www.cnblogs.com/lepeCoder/p/wsl_dir.html) wsl 文件系统可以直接映射到 windows 中。先用 win + R 打开运行窗口，然后输入 `\\wsl$` 就能够打开 wsl 文件系统，然后右键映射网络驱动器

设置 cmd 的语言可以设置 语言和区域，添加英文，并把英文置顶即可（设为首选语言）

## Docker

安装 Docker 的过程就不多叙述，我之前也整理过 Docker 相关的笔记。可以直接在 WSL 2 里安装 docker，也可以下载 Docker Desktop，但目前我不清楚 Docker Desktop 支不支持调用显卡，所以我选择前者！

遇到下面的问题

```shell
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?
```

我们需要打开 docker 服务

```shell
sudo service docker start
# 重启
sudo service docker restart
```

为了让开发更方便，直接上 [pytorch docker](https://hub.docker.com/r/pytorch/pytorch/tags) 镜像可以极大减少安装环境的时间，特别是不用担心 CUDA 的问题了

直接选一个版本，然后一个命令搞定

```cmd
docker pull pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
```

注意一定要下载 **devel** 版本的镜像才包含 CUDA

## Proxy

直接使用脚本一件开启代理，参考 [zhihu](https://zhuanlan.zhihu.com/p/153124468)

```shell
#!/bin/zsh
host_ip=$(cat /etc/resolv.conf |grep "nameserver" |cut -f 2 -d " ")
export ALL_PROXY="http://$host_ip:7890"
```

update 2025/07/19 From DeepSeek

```shell
# work with clash allow LAN on
export http_proxy="http://$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):7890"
export https_proxy="http://$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):7890"
```
