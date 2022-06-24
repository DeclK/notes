---
title: VSCode 教程
tags:
  - VSCode
  - 教程
categories:
  - 编程
  - Tools
abbrlink: ca50f614
date: 2021-07-25 22:52:02
---

# VSCode note

学习视频 [bilibili](https://www.bilibili.com/video/BV1ty4y1S7mC?p=1)

## 安装

直接在官网一键下载，宇宙第一开发工具，而且还是免费的:laughing: 使用最新版本的 vscode，有时出现一些莫名其妙的 bug:cry:，可以使用前几个版本的，如果用熟了某一个版本那暂时不要更新了吧。如果想要完整卸载的话需要删除 `usr/.vscode` 和 `AppData/Code` 文件，感觉整体来看 vscode 依然是在发展当中的开发工具

### Getting started

1. vscode 支持更换主题皮肤

2. vscode 支持插件扩展，能够实现多种功能来提高编程效率，如下载不同语言，高亮代码等等，推荐的插件：Python, Gitlens, Remote SSH

3. 通过快捷键 palette 查找文件 Ctrl + P，查找命令 Ctrl + Shift + P，折叠左侧工具栏 Ctrl + B

   可以修改一些你常用命令的快捷键，我修改了如下命令：

   1. Run in python file: Ctrl + enter
   2. Start debugging: Shift +enter
   3. Close panels: Ctrl + Alt + P
   4. Insert Line Above/Below: Ctrl + Shifg + \\ or Enter
   5. Collapse folders: Ctrl + Shift + F

4. 设置面板快捷键 Ctrl + ,

5. 新建一个文件 Ctrl + N

## 交互式演练场 Interactive Playground

一些编辑的小技巧

1. 同时选中同名字段 Ctrl + Shift + L，但这个功能没有 Refactoring 智能

2. IntelliSense 自动补全 api 功能 Ctrl + space
3. 行操作
   1. 复制行，在没有任何东西选中的时候直接 Ctrl + C
   2. 上下移动行，Alt + 上下键
   3. 删除行 Ctrl + Shift + K

4. Formatting 规范代码，需要自己定义 format

## 一些推荐的设置

打开 settings

1. 在熟悉之后关闭 welcome/startup 界面
2. 建议使用英文界面
3. 设置字体 JetBrains Mono，还可以顺便设置一下字号。安装字体[教程](https://blog.csdn.net/HUSTHY/article/details/104023077)
4. 设置是否显示缩略图 minimap
5. 设置 restore windows 是否直接恢复上一次的项目

6. 设置开启 Trims final newlines 自动消除文件末尾多余的空行

## Python

下载 Python 插件

### Get Started

1. 在底部的 status bar 添加 python 解释器，也可以使用 palette 来添加 interpreter。会自动检测到电脑上的 python interpreter (包括 anaconda 中创建的环境)

2. 建立 debug 环境 craete a launch.json file，直接使用默认值就好，详细说明在 [知乎](https://zhuanlan.zhihu.com/p/142642410)

   补充：默认配置只 debug 当前 folder 下的代码（“自己”的代码），如果是 pip/conda install 安装的代码（“别人”的代码），是不会进行 debug 的。如果想要 debug 所有代码则配置文件需要加上 `"justMyCode": false` [CSDN](https://blog.csdn.net/g534441921/article/details/102743393)

   补充：对于需要传参数的 python 脚本调试可以参考 [知乎](https://www.zhihu.com/question/50700473)

3. Jupyter notebook 是插件里自带的功能，直接打开 ipynb 文件就可以运行代码块！通过选择 notebook 中的 kernel (具体一点说就是选择 conda 中的环境) 就能在你想要的环境中运行了，非常方便

4. 当 workspace 比较大的时候 Pylance 可能加载得很慢，参考 [Stack Overflow](https://stackoverflow.com/questions/50389852/visual-studio-code-intellisense-not-working) 在 Folder 下新建 pyrightconfig.json 进行配置（注意不是在 .vscode/ 下新建，而是直接在 workspace root 下！

   ```json
   {
     "exclude": [
       "**/data",
       ".git"
     ]
   }
   ```

### 连接到远程服务器

由于要跑一些模型，自己的电脑显卡根本跑不动，那就~~白嫖~~连接到实验室的服务器​

通过 vscode Extensions: `Remote-SSH` 完成，[知乎](https://zhuanlan.zhihu.com/p/141205262)

在 Remote Expolorer -> SSH TARGETS -> config 中添加配置

```config
# Read more about SSH config files: https://linux.die.net/man/5/ssh_config
Host random_name
    HostName host_ip
    User user_name
```

遇到了报错 `Resolver error: Error: Running the contributed command: '_workbench.downloadResource' failed.` 无法连接到服务器。原因可能在于服务器是在内网，没有办法在服务器上自动下载好 vscode-server，于是只能本地下载，然后上传

参考 [CSDN](https://blog.csdn.net/ibless/article/details/118610776) 解决问题

