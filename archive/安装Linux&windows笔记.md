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

## Install WSL & zsh

安装一个双系统可能对于初学者来讲是比较友好的，随着对 Linux 的了解越来越多，更多的 Linux 使用转移到了服务器上，对双系统的需求越来越少。并且 windows 现在支持了 linux 子系统（WSL），也可以直接使用 docker，这样安装双系统就显得是一个很复杂的选项了。我之前就算安装了双系统，现在也将其删除了，释放了 100G 空间，删除参考：[bilibili](https://www.bilibili.com/video/BV1Ba411z75z/)，再贴一个 [WSL](https://docs.microsoft.com/en-us/windows/wsl/install) 官网

这个 up 的系列视频都教得非常好：[bilibili](https://www.bilibili.com/video/BV1aA411s7PJ)，教你如何安装漂亮的 WSL，配合 vscode + zsh 变为强力开发环境，下面简要总结 zsh 的安装：

1. 安装 zsh `sudo apt install zsh`

2. 根据 [ohmyzsh](https://github.com/ohmyzsh/ohmyzsh) 项目进行安装，可以使用网络安装：

   ```shell
   bash -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
   ```

   也可以把上面链接中的脚本复制下来，保存为 `install.sh`，然后执行

   ```shell
   sh install.sh
   ```

   如果由于网络原因可以选择使用 gitee 镜像，更改 REMOTE

   ```install.sh
   REMOTE=${REMOTE:-https://gitclone.com/github.com/${REPO}.git}
   ```

3. 安装插件。插件推荐 [zsh-autusuggestions](https://github.com/zsh-users/zsh-autosuggestions)，把这个项目 clone 到 `~/.oh-my-zsh/custom/plugins` 中

   ```shell
   cd ~/.oh-my-zsh/custom/plugins
   git clone https://github.com/zsh-users/zsh-autosuggestions.git
   # git clone https://gitclone.com/github.com/zsh-users/zsh-autosuggestions.git
   ```

   然后在 `~/.zshrc` 里配置 plugin

   ```.zshrc
   plugins=(git zsh-autosuggestions)
   ```

   上面的步骤可以用下面脚本统一替换

   ```shell
   cd ~/.oh-my-zsh/custom/plugins && git clone https://gitclone.com/github.com/zsh-users/zsh-autosuggestions.git
   sed -i 's/(git)/(git zsh-autosuggestions)/g' ~/.zshrc
   source ~/.zshrc
   ```

4. 打开个人目录下的配置文件:  `~/.zshrc` 

   找到 `auto_update` 相关行，将注释去掉，则可禁用 ohmyzsh 自动检查更新。可通过命令 `upgrade_oh_my_zsh` 手动升级

5. 如果由于 git 文件太大，oh-my-zsh 会比较卡顿，可以使用 `git config --add oh-my-zsh.hide-dirty 1` 来禁止其读取文件变化信息，如果还觉得慢则用 `git config --add oh-my-zsh.hide-status 1`

## Install Ubuntu

（如果仍需要）安装双系统：移步 [bilibili](https://www.bilibili.com/video/BV11k4y1k7Li/?spm_id_from=333.788&vd_source=65e80258e57b5ae307bd30541465a0be)（建议安装最新版，美观且体验更友好）或者阅读 [ubuntu 官方教程](https://ubuntu.com/tutorials/install-ubuntu-desktop#1-overview)

## Common Settings

1. 设置 root 密码。然后创建新用户，并设置新用户密码以及 sudo 权限 `/etc/sudoers`，在文件最后面写入

   ```shell
   username ALL=(ALL) NOPASSWD: ALL
   ```

2. 修改 /etc/hostname，reboot 后永久更改主机名

3. （如果没有中文输入法）下载中文输入法，需要严格按照以下步骤！**不要按照官方的教程去安装搜狗拼音！！！**

   ```shell
   # 1. open regions & language -> manage installed language, make sure you choose fcitx
   # 2. 
   sudo apt remove fcitx-ui-qimpanel
   # 3. install sogou deb
   sudo dpkg -i sogoupinyin_4.2.1.145_amd64.deb &reboot
   # this might give magic: sudo apt install sogoupinyin && reboot
   # when reinstall it you need to install this
   sudo apt install fcitx-config-gtk
   ```

   之后就可以在 fcitx configure 中看到 sogou pinyin 了，把其移到第一个位置即可！如果不行，就按照下面方法彻底卸载 fcitx，重复以上步骤，一定能行！

   ```shell
   # zsh, if batch sudo apt purge fcitx*
   sudo apt purge 'fcitx*'
   sudo apt autoremove
   sudo rm -rf /opt/sogoupinyin
   # config 
   rm -rf ~/.config/fcitx
   # config sogou
   rm -rf ~/.config/sogoupinyin
   
   # restart the fcitx so it is completely exit
   ```

   安装好过后取消一些快捷键：

   1. fcitx 的 `ctrl+alt+P` 的快捷键设置，因为我平常习惯用这个快捷键在 vscode 中 close panel。方式是 `Configure -> Global config -> Show advanced options -> Switch embedded preedit`
   2. 设置 sogou 输入法的简体繁体快捷键，以及 fcitx 的简体繁体快捷键 `Configure -> Addon -> Advanced -> Simpliflied Chinese To Traditional Chinese `

4. 时过境迁，现在已经有了图形化界面的 [clash](https://github.com/zzzgydi/clash-verge) 啦！目前该软件还在开发当中，所以可能会遇到一些 bug，多试几个版本，我正在使用 1.3.6，1.3.7 无法显示图形界面

5. 重要需求 [miniconda](https://docs.conda.io/projects/miniconda/en/latest/#quick-command-line-install) [typora](https://typoraio.cn/) chrome vscode 软件

   conda install, pip install 下载速度慢时，请使用国内镜像源，例如：

   1. [北京外国语大学镜像源]( https://mirrors.bfsu.edu.cn/help/anaconda/)（截至2021/6/15下载速度很快）
   2. [清华大学镜像源](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/)
   3. [南京大学镜像源](https://mirror.nju.edu.cn/help/anaconda)（南大本家，但不推荐🤣）

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

   教程里还教了如何更新 apt source 为阿里云镜像源，镜像中的软件会持续而且下载速度很快（但现在好像默认的源速度也不错了）。这里我选择更换为 [北外镜像源](https://mirrors.bfsu.edu.cn/help/ubuntu/)

   如果是新的系统，一定要记得 `sudo apt update && apt upgrade` 这样在安装其他环境的时候会避免一些莫名其妙的错误

   同时教程里也设置了 sudo，让每一次 sudo 都不需要输入密码

7. 安装 JetBrains Mono 字体，[Download Link](https://www.jetbrains.com/lp/mono/)，安装命令

   ```shell
   sudo unzip -d /usr/share/fonts JetBrainsMono-xxx.zip
   sudo fc-cache -f -v
   ```

8. 安装 MiSans 字体，ttf 字体可以通过直接双击文件进行安装。安装好字体过后可以修改 typora 的渲染字体

   ```css
   /* referene: https://zhuanlan.zhihu.com/p/684183797 */
   /* Change Code Font */
   .CodeMirror-wrap .CodeMirror-code pre {
      font-family: "JetBrains Mono"
   }

   /* Change Inline Code Font */
   .md-fences,
   code,
   tt {
      border: 1px solid #e7eaed;
      background-color: #f8f8f8;
      border-radius: 3px;
      padding: 0;
      padding: 2px 4px 0px 4px;
      font-size: 0.9em;
      font-family: "JetBrains Mono";
   }
   ```

9. 安装 utools，[Download Link](https://www.u.tools/)，打开 startup applications preference，add 一个 utools command 即可开机启动

10. 安装 flameshot，[Download Link](https://flameshot.org/#download)，通过 keyboards shortcut 添加 flameshot gui 命令完成快捷键设置，参考 [CSDN](https://blog.csdn.net/u013171226/article/details/107717009)

11. 安装 electerm 作为更好的 sftp 传输工具，[github](https://github.com/electerm/electerm)

12. 安装 fsearch，[Download Link](https://github.com/cboxdoerfer/fsearch#download)

13. 修改 Files 侧栏 [StackOverflow](https://unix.stackexchange.com/questions/207216/user-dirs-dirs-reset-at-start-up)

14. 安装 foxit pdf reader，[Download Link](https://www.foxitsoftware.cn/pdf-reader/)

## Typora scripts

🤔🤨🧐

```python
from pathlib import Path
import os
import re

licence_dir = '/usr/share/typora/resources/page-dist/static/js'
licence_dir = Path(licence_dir)

# check if the directory exists
if not licence_dir.exists():
    raise Exception('cannot find the directory')

# change the permission
print(f"Doing sudo chmod 777 -R for {str(licence_dir)}, might need to enter password")
os.system('sudo chmod 777 -R ' + licence_dir)

prefix = 'LicenseIndex'

licence_dir = Path(licence_dir).iterdir()

licence_file = None
for file in licence_dir:
    if file.name.startswith(prefix):
        licence_file = file

print(f"Found the licence file: {licence_file.name}")
if licence_file is None:
    raise Exception('cannot find licence file')

print("Overwriting the licence file...")
# read file content
with open(licence_file, 'r') as f:
    content = f.read()

# replace the pattern
target = 'e.hasActivated="true"==e.hasActivated'
replacement = 'e.hasActivated="true"=="true"'
content = re.sub(target, replacement, content)

# write the content to original file
with open(licence_file, 'w') as f:
    f.write(content)

print("Done!")
```

激活完成后关闭激活弹窗。首先定位文件：进入 Typora 安装目录，找到 `resources\page-dist\license.html`。然后修改代码：用文本编辑器打开 license.html，将 `</body></html>` 替换为以下代码：

```html
</body><script>window.onload=function(){setTimeout(()=>{window.close();},5);};</script></html>
```

Typora custom css，请确保下载了 JetBrains Mono 和 MiSans 字体

```css
:root {
    --side-bar-bg-color: #fafafa;
    --control-text-color: #777;
}

@include-when-export url(https://fonts.googleapis.com/css?family=Open+Sans:400italic,700italic,700,400&subset=latin,latin-ext);

/* open-sans-regular - latin-ext_latin */
@font-face {
    font-family: 'Open Sans';
    font-style: normal;
    font-weight: normal;
    src: local('Open Sans Regular'), local('OpenSans-Regular'), url('./github/open-sans-v17-latin-ext_latin-regular.woff2') format('woff2');
    unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD, U+0100-024F, U+0259, U+1E00-1EFF, U+2020, U+20A0-20AB, U+20AD-20CF, U+2113, U+2C60-2C7F, U+A720-A7FF;
  }
  /* open-sans-italic - latin-ext_latin */
  @font-face {
    font-family: 'Open Sans';
    font-style: italic;
    font-weight: normal;
    src: local('Open Sans Italic'), local('OpenSans-Italic'), url('./github/open-sans-v17-latin-ext_latin-italic.woff2') format('woff2');
    unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD, U+0100-024F, U+0259, U+1E00-1EFF, U+2020, U+20A0-20AB, U+20AD-20CF, U+2113, U+2C60-2C7F, U+A720-A7FF;
  }
  /* open-sans-700 - latin-ext_latin */
  @font-face {
    font-family: 'Open Sans';
    font-style: normal;
    font-weight: bold;
    src: local('Open Sans Bold'), local('OpenSans-Bold'), url('./github/open-sans-v17-latin-ext_latin-700.woff2') format('woff2'); 
    unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD, U+0100-024F, U+0259, U+1E00-1EFF, U+2020, U+20A0-20AB, U+20AD-20CF, U+2113, U+2C60-2C7F, U+A720-A7FF;
  }
  /* open-sans-700italic - latin-ext_latin */
  @font-face {
    font-family: 'Open Sans';
    font-style: italic;
    font-weight: bold;
    src: local('Open Sans Bold Italic'), local('OpenSans-BoldItalic'), url('./github/open-sans-v17-latin-ext_latin-700italic.woff2') format('woff2');
    unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD, U+0100-024F, U+0259, U+1E00-1EFF, U+2020, U+20A0-20AB, U+20AD-20CF, U+2113, U+2C60-2C7F, U+A720-A7FF;
  }

html {
    font-size: 16px;
    -webkit-font-smoothing: antialiased;
}

body {
    font-family: "HarmonyOS Sans SC","Clear Sans", "Helvetica Neue", Helvetica, Arial, 'Segoe UI Emoji', sans-serif;
    color: rgb(51, 51, 51);
    line-height: 1.6;
}

#write {
    max-width: 860px;
  	margin: 0 auto;
  	padding: 30px;
    padding-bottom: 100px;
}

@media only screen and (min-width: 1400px) {
	#write {
		max-width: 1024px;
	}
}

@media only screen and (min-width: 1800px) {
	#write {
		max-width: 1200px;
	}
}

#write > ul:first-child,
#write > ol:first-child{
    margin-top: 30px;
}

a {
    color: #4183C4;
}
h1,
h2,
h3,
h4,
h5,
h6 {
    position: relative;
    margin-top: 1rem;
    margin-bottom: 1rem;
    font-weight: bold;
    line-height: 1.4;
    cursor: text;
}
h1:hover a.anchor,
h2:hover a.anchor,
h3:hover a.anchor,
h4:hover a.anchor,
h5:hover a.anchor,
h6:hover a.anchor {
    text-decoration: none;
}
h1 tt,
h1 code {
    font-size: inherit;
}
h2 tt,
h2 code {
    font-size: inherit;
}
h3 tt,
h3 code {
    font-size: inherit;
}
h4 tt,
h4 code {
    font-size: inherit;
}
h5 tt,
h5 code {
    font-size: inherit;
}
h6 tt,
h6 code {
    font-size: inherit;
}
h1 {
    font-size: 2.25em;
    line-height: 1.2;
    border-bottom: 1px solid #eee;
}
h2 {
    font-size: 1.75em;
    line-height: 1.225;
    border-bottom: 1px solid #eee;
}

/*@media print {
    .typora-export h1,
    .typora-export h2 {
        border-bottom: none;
        padding-bottom: initial;
    }

    .typora-export h1::after,
    .typora-export h2::after {
        content: "";
        display: block;
        height: 100px;
        margin-top: -96px;
        border-top: 1px solid #eee;
    }
}*/

h3 {
    font-size: 1.5em;
    line-height: 1.43;
}
h4 {
    font-size: 1.25em;
}
h5 {
    font-size: 1em;
}
h6 {
   font-size: 1em;
    color: #777;
}
p,
blockquote,
ul,
ol,
dl,
table{
    margin: 0.8em 0;
}
li>ol,
li>ul {
    margin: 0 0;
}
hr {
    height: 2px;
    padding: 0;
    margin: 16px 0;
    background-color: #e7e7e7;
    border: 0 none;
    overflow: hidden;
    box-sizing: content-box;
}

li p.first {
    display: inline-block;
}
ul,
ol {
    padding-left: 30px;
}
ul:first-child,
ol:first-child {
    margin-top: 0;
}
ul:last-child,
ol:last-child {
    margin-bottom: 0;
}
blockquote {
    border-left: 4px solid #dfe2e5;
    padding: 0 15px;
    color: #777777;
}
blockquote blockquote {
    padding-right: 0;
}
table {
    padding: 0;
    word-break: initial;
}
table tr {
    border: 1px solid #dfe2e5;
    margin: 0;
    padding: 0;
}
table tr:nth-child(2n),
thead {
    background-color: #f8f8f8;
}
table th {
    font-weight: bold;
    border: 1px solid #dfe2e5;
    border-bottom: 0;
    margin: 0;
    padding: 6px 13px;
}
table td {
    border: 1px solid #dfe2e5;
    margin: 0;
    padding: 6px 13px;
}
table th:first-child,
table td:first-child {
    margin-top: 0;
}
table th:last-child,
table td:last-child {
    margin-bottom: 0;
}

.CodeMirror-lines {
    padding-left: 4px;
}

.code-tooltip {
    box-shadow: 0 1px 1px 0 rgba(0,28,36,.3);
    border-top: 1px solid #eef2f2;
}

.md-fences,
code,
tt {
    border: 1px solid #e7eaed;
    background-color: #f8f8f8;
    border-radius: 3px;
    padding: 0;
    padding: 2px 4px 0px 4px;
    font-size: 0.9em;
    font-family: "JetBrains Mono";
}

code {
    background-color: #f3f4f4;
    padding: 0 2px 0 2px;
}

.md-fences {
    margin-bottom: 15px;
    margin-top: 15px;
    padding-top: 8px;
    padding-bottom: 6px;
}


.md-task-list-item > input {
  margin-left: -1.3em;
}

@media print {
    html {
        font-size: 13px;
    }
    pre {
        page-break-inside: avoid;
        word-wrap: break-word;
    }
}

.md-fences {
	background-color: #f8f8f8;
}
#write pre.md-meta-block {
	padding: 1rem;
    font-size: 85%;
    line-height: 1.45;
    background-color: #f7f7f7;
    border: 0;
    border-radius: 3px;
    color: #777777;
    margin-top: 0 !important;
}

.mathjax-block>.code-tooltip {
	bottom: .375rem;
}

.md-mathjax-midline {
    background: #fafafa;
}

#write>h3.md-focus:before{
	left: -1.5625rem;
	top: .375rem;
}
#write>h4.md-focus:before{
	left: -1.5625rem;
	top: .285714286rem;
}
#write>h5.md-focus:before{
	left: -1.5625rem;
	top: .285714286rem;
}
#write>h6.md-focus:before{
	left: -1.5625rem;
	top: .285714286rem;
}
.md-image>.md-meta {
    /*border: 1px solid #ddd;*/
    border-radius: 3px;
    padding: 2px 0px 0px 4px;
    font-size: 0.9em;
    color: inherit;
}

.md-tag {
    color: #a7a7a7;
    opacity: 1;
}

.md-toc { 
    margin-top:20px;
    padding-bottom:20px;
}

.sidebar-tabs {
    border-bottom: none;
}

#typora-quick-open {
    border: 1px solid #ddd;
    background-color: #f8f8f8;
}

#typora-quick-open-item {
    background-color: #FAFAFA;
    border-color: #FEFEFE #e5e5e5 #e5e5e5 #eee;
    border-style: solid;
    border-width: 1px;
}

/** focus mode */
.on-focus-mode blockquote {
    border-left-color: rgba(85, 85, 85, 0.12);
}

header, .context-menu, .megamenu-content, footer{
    font-family: "Segoe UI", "Arial", sans-serif;
}

.file-node-content:hover .file-node-icon,
.file-node-content:hover .file-node-open-state{
    visibility: visible;
}

.mac-seamless-mode #typora-sidebar {
    background-color: #fafafa;
    background-color: var(--side-bar-bg-color);
}

.md-lang {
    color: #b4654d;
}

/*.html-for-mac {
    --item-hover-bg-color: #E6F0FE;
}*/

#md-notification .btn {
    border: 0;
}

.dropdown-menu .divider {
    border-color: #e5e5e5;
    opacity: 0.4;
}

.ty-preferences .window-content {
    background-color: #fafafa;
}

.ty-preferences .nav-group-item.active {
    color: white;
    background: #999;
}

.menu-item-container a.menu-style-btn {
    background-color: #f5f8fa;
    background-image: linear-gradient( 180deg , hsla(0, 0%, 100%, 0.8), hsla(0, 0%, 100%, 0)); 
}

```

# Windows

实验室有一个空的主机，比较老，想要重新清理一下自己用。我并没有选择重装整个系统，而是选择重置，即恢复出厂设置

资源下载：[MSDN](https://msdn.itellyou.cn/) [rufus](https://rufus.ie/zh/)  [balena](https://www.balena.io/etcher/) MSDN 提供了需要的各个 Windows 版本的 iso，使用 rufus or balena 将 iso 烧入到U盘里

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
