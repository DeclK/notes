---
title: Linux/Windows 安装笔记
tags:
  - Linux
  - Windows
  - 安装教程
categories:
  - 编程
  - Tools
abbrlink: 53e3a160
date: 2021-07-12 14:36:42
---

# Linux

## Install Linux Subsystem

安装一个双系统可能对于初学者来讲是比较友好的，随着对 Linux 的了解越来越多，更多的 Linux 使用转移到了服务器上，对双系统的需求越来越少。并且 windows 现在支持了 linux 子系统（WSL），也可以直接使用 docker，这样安装双系统就显得是一个很复杂的选项了。我之前就算安装了双系统，现在也将其删除了，释放了 100G 空间，删除参考：[bilibili](https://www.bilibili.com/video/BV1Ba411z75z/)，再贴一个 [WSL](https://docs.microsoft.com/en-us/windows/wsl/install) 官网

这个 up 的系列视频都教得非常好,：[bilibili](https://www.bilibili.com/video/BV1aA411s7PJ)，教你如何安装漂亮的 WSL，配合 vscode + zsh 变为强力开发环境，下面简要总结 zsh 的安装：

1. 安装 zsh `sudo apt install zsh`

2. 根据 [ohmyzsh](https://github.com/ohmyzsh/ohmyzsh) 项目进行安装，可以使用网络安装：

   ```shell
   bash -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
   ```

   也可以把上面链接中的脚本复制下来，保存为 `install.sh`，然后执行

   ```shell
   sh install.sh
   ```

3. 安装插件。插件推荐 [zsh-autusuggestions](https://github.com/zsh-users/zsh-autosuggestions)，把这个项目 clone 到 `~/.oh-my-zsh/custom/plugins` 中

   ```shell
   cd ~/.oh-my-zsh/custom/plugins
   git clone https://github.com/zsh-users/zsh-autosuggestions.git
   ```

   然后在 `~/.zshrc` 里配置 plugin

   ```.zshrc
   plugins=(git zsh-autosuggestions)
   ```

## Install Ubuntu

（如果仍需要）安装双系统：移步 [bilibili](https://www.bilibili.com/video/BV18W41137XB)（建议安装最新版，美观且体验更友好）或者阅读 [ubuntu 官方教程](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview)

## Common Settings

1. 设置 root 密码。然后创建新用户，并设置新用户密码以及 sudo 权限 `/etc/sudoers` 

   ```shell
   username ALL=(ALL) NOPASSWD: ALL
   ```

2. 修改 /etc/hostname，reboot 后永久更改主机名

3. 可能出现启动 windows 的时候有 bitlocker，禁用 bitlocker 安全协议

4. 配置代理 clash，从 youtube 上学的（迷途小书童），要点就是将配置文件 config.yaml 和 Country.mmdb 移动到 ~/.config/clash 文件夹下面，配置文件通过 clash for windows 生成，文件目录为 User/.config/clash(/profiles) 。通过 clash dashboard 切换节点 http://clash.razord.top/

   让Terminal走代理的方法(desktop上的settings中设定会改写terminal端，使用export改写则不会影响desktop)，参考 [知乎链接](https://zhuanlan.zhihu.com/p/46973701)

5. 官网下载git anaconda chrome typora chrome baiduyun vscode软件并安装

   conda install, pip install 下载速度慢时，请使用国内镜像源，例如：

   1. [北京外国语大学镜像源]( https://mirrors.bfsu.edu.cn/help/anaconda/)（截至2021/6/15下载速度很快）
   2. [清华大学镜像源](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/)
   3. [南京大学镜像源](https://mirror.nju.edu.cn/help/anaconda)（南大本家推荐）

6. 配置 nvidia driver: 根据 [知乎链接](https://zhuanlan.zhihu.com/p/59618999) ，在命令行里下载推荐的driver。如果在配置 nvidia driver 的过程中出现连接不上显卡，可能需要关闭 security boot。参考 [稚晖君](https://zhuanlan.zhihu.com/p/336429888) 的教程，下载安装 CUDA，选择 runfile。

   如果想移除所有 cuda, cudnn, nvidia driver

   ```shell
   sudo apt-get remove --purge nvidia*
   ```

   设置 cuda path

   ```shell
   export CUDA_HOME=/usr/local/cuda
   export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
   export PATH=${CUDA_HOME}/bin:${PATH}
   ```

   (2022/1/31 更新) 尝试使用命令行在 ubuntu 16.04 上更新驱动，不太顺利，因为 ppa 中好像没有对这 16.04 进行支持，最新仅支持到 430，通过其他方法可能成功，但我就不进行过多尝试了。最终使用 `sudo apt install nvidia-418` 恢复了之前的驱动版本，其中遇到的报错 `NVIDIA NVML Driver/library version mismatch`，参考了 [StackOverflow ](https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch) 中的第二个回答解决

   教程里还教了如何更新 apt source 为阿里云镜像源，镜像中的软件会持续而且下载速度很快，这里我选择更换为 [南京大学镜像源](https://mirror.nju.edu.cn/help/ubuntu) 如下所示

     ```source.list
   sudo cp /etc/apt/sources.list /etc/apt/sources.list.back
   sudo sed -i 's/archive.ubuntu.com/mirrors.nju.edu.cn/g' /etc/apt/sources.list
   sudo sed -i 's/security.ubuntu.com/mirrors.nju.edu.cn/g' /etc/apt/sources.list
     ```

   同时教程里也设置了 sudo，让每一次 sudo 都不需要输入密码

8. pip install 遇到问题 enter your password to unlock your login keyring

   解决方法，直接cancel，或者在passwd and key中更改密码

# Windows

实验室有一个空的主机，比较老，想要重新清理一下自己用。我并没有选择重装整个系统，而是选择重置，即恢复出厂设置

资源下载：[MSDN](https://msdn.itellyou.cn/) [rufus](https://rufus.ie/zh/) MSDN 提供了需要的各个 Windows 版本的 iso，使用 rufus 将 iso 烧入到U盘里

Win10 安装教程：[bilibili](https://www.bilibili.com/video/BV1DJ411D79y/?spm_id_from=333.788.recommend_more_video.-1)

Windows 激活：[github](https://github.com/TGSAN/CMWTAT_Digital_Edition/releases)

github 如果下载不够快，自行搜索 github 镜像，这里留一个参考 [link](https://ghproxy.com/)

磁盘管理：[bilibili](https://www.bilibili.com/video/BV1Uj411f7wj)

Office 下载：[Office Tool plus](https://otp.landian.vip/zh-cn/)

Office Tool plus [使用方法](https://www.coolhub.top/archives/11)：

1. 卸载原有的 office wps，并清除旧版本激活信息（激活页面 -> 许可证管理 -> 清除激活状态）

2. 推荐 Microsoft 365 企业应用版

3. 使用一键安装代码

   ```
   deploy /addProduct O365ProPlusRetail_zh-cn_Access,Bing,Groove,Lync,OneDrive,OneNote,Outlook,Publisher,Teams /channel Current /downloadFirst
   ```

4. 在之后使用过程中可能遇到许可证问题，可以使用工具箱中的**修复Office许可证问题**。此时需要一个 [KMS 地址](https://www.coolhub.top/tech-articles/kms_list.html)，填入即可

好用的 windows terminal: [github](https://github.com/microsoft/terminal)

