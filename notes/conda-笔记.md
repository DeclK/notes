---
title: Conda 笔记
tags:
  - Conda
categories:
  - 编程
  - Tools
abbrlink: c8895fc
date: 2021-08-08 21:26:07
---

# Conda Cheat Sheet

记录一些常用的 conda 命令帮助快速管理环境，整理自官方 [CONDA CHEAT SHEET](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)

## Install in Linux

安装 [miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html)，可以不 care 第二步 Verify，下载好 .sh 文件后直接运行，然后还要将其加入 PATH 中

```python
export PATH=$PATH:/home/.../miniconda3/bin
```

## Basic

### conda info

这个命令非常管用，基本上能够看到所有的 conda 配置信息

### conda install

`conda install PACKAGENAME` 下载指定包

### conda update

`conda update PACKAGENAME` 更新指定包 

## Environment

### conda create -n

`conda create -n py36 python=3.6` 创造一个 python 3.6 的环境 py36

注意：一般都要加上 `python=x.x` 否则使用的是 base 环境的 python 解释器

### conda env list

列出目前有的环境

### conda env remove 

`conda env remove -n env_name ` 移除环境

###  conda activate/deactivate

`conda activate/deactivate env_name` 激活/退出环境

## Package

### conda install

`conda install PACKAGENAME` 下载包

`conda install --file requirements.txt` 通过 requirements 文件下载包

### conda remove

`conda remove PACKAGENAME` 移除包

### conda list

`conda list PACKAGENAME` 查看环境的某个包，如果不加 PACKAGENAME 则列出所有环境

### conda clean

如果不清理的话，anaconda 还是很吃存储的，会逐渐积累很多下载包

`conda clean --all`  Remove index cache, lock files, unused cache packages, and tarballs.

## 其他

### 禁用自动启动 base 环境

每次打开新的 shell 都会自动进入 base 环境，用下面的命令行禁用

`conda config --set auto_activate_base false`

### 镜像源设置

编辑 `~/.condarc` 文件，设置镜像源，例如 [北京外国语大学镜像源](https://mirrors.bfsu.edu.cn/help/anaconda/)，如没有该文件可使用 `conda config` 创建。使用 `conda config --show` 进行查看当前配置

**建议使用 pip 下载包，而不用 conda，仅使用 conda 管理环境**，更换 pypi 源脚本如下

```shell
# 可能需要先更新 pip，一般不用
pip install pip -U	# Linux
pip install pip -U --user	# Windows

# 设置镜像
pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
```

