# TeX Tutorial

参考 [bilibili](https://www.bilibili.com/video/BV1Rd4y1X7AL) [overleaf](https://www.overleaf.com/learn)

## 简介

tex 是一个专业的排版系统，也常有人认为 tex 是一种语言。我的理解也是类似的，就像 python 一样需要先下载一个解释器 `python.exe` 然后再执行 python 脚本；tex 也需要先下载其“解释器”，这个解释器有多个发行的版本，例如 Texlive or MikTex

优点：处理公式非常强大

缺点：麻烦，麻烦，麻烦

## 安装

参考  [zhihu](https://zhuanlan.zhihu.com/p/166523064)

1. 软件/解释器 Texlive or MikTex or Overleaf。
   1. Texlive 是推荐的选项，因为其包含的宏包是最全的，但是也是最大的（GB 级别），需要更长时间的下载。既然想要用 tex 就不要怕麻烦了😢许多学校都有镜像站，可提供高速下载，[NJU mirror](https://mirror.nju.edu.cn/tex-historic/systems/texlive/2021/) [THU mirror](https://mirrors.tuna.tsinghua.edu.cn/CTAN/systems/texlive/Images/)，在 cmd 中使用 `latex -v` 检查安装是否成功，可考虑重启一下
   2. MikTex 则仅包含一些基础的宏包，在需要其他宏包时会自动下载
   3. Overleaf 是一个网站，无须下载，**并且里面有许多模板可以使用**

2. 编辑器 vscode。使用插件 **LaTeX Workshop** 即可，推荐对该插件进行如下配置：

   ```json
   {
       "latex-workshop.latex.autoBuild.run": "never",
       "latex-workshop.latex.autoClean.run": "onFailed",
   }
   ```

   这三个选项分别代表：1. 不自动编译，需要手动启动编译；2. 当编译失败时自动清除缓存文件

**tex 中的很多功能需要宏包完成，我们需要在导言区实现声明。**个人认为宏包可以理解为 python 中的三方库，都需要事先安装然后才能 import 并使用。例如，显示中文需要使用 ctex 宏包

## 第一个 tex 脚本

下面看一个最精简的 tex 脚本

```tex
\documentclass{article}
\begin{document}
First document. This is a simple example, with no 
extra parameters or packages included.
\end{document}
```

<img src="Tex tutorial/image-20221027151659500.png" alt="image-20221027151659500" style="zoom:50%;" />

可以看到编译结果还带自动缩进。**tex 的命令格式如下，**代码中注释用 `%`

```tex
\command[optinal params]{params}	% comment
```

每一个 tex 文档必须要两个部分

1. 文档类型 `\documentclass{class_name}`，类别常用 article, book, report, beamer...
2. 正文内容，即 `\begin{document}` 和 `\end{document}` 之间的内容

## 导言区 Preamble

导言区就是文档类型和正文内容之间的区域。我们在这里声明需要的宏包和一些配置，一个最简单的导言区如下

```tex
\documentclass[12pt, letterpaper]{article}
\usepackage{graphicx}	% 导入图片必须使用该宏包
```

这里对 article 类通过可选参数进行更详细的定义：

1. 12pt 设置了字体大小
2. letterpaper 设置了页面大小，还可选 a4paper, legalpaper

可选参数必须由逗号隔开，也可以使用关键字指定，例如 `[fontset=windows]` 这个参数经常在使用中文时设置

标题，作者，日期，这三个元素也是在导言区定义的

```tex
\title{My first LaTeX document}
\author{Hubert Farnsworth\thanks{Funded by the Overleaf team.}}
\date{August 2022}
```

这里的 `\thanks` 会在作者后面加一个 `*` 号，并在页脚下显示相应内容。定义好后需要在正文内容里使用 `\maketile` 生成真正的内容

```tex
\begin{document}
\maketitle
We have now added a title, author and date to our first \LaTeX{} document!
\end{document}
```

## 加粗，斜体，下划线

加粗，斜体，下划线，三个命令分别为

```tex
\textbf{hello world}
\textit{hello world}
\underline{hellow world} 

\textbf{\textit{hello world}} % 加粗斜体
\emph{hellow world}	% 根据周围环境，自适应突出内容
```

## 加入图片

参考 [overleaf](https://www.overleaf.com/learn/latex/Inserting_Images) [overleaf](https://www.overleaf.com/learn/latex/Positioning_images_and_tables)

加入图片需要做3件事情：

1. 把图片放到 tex 项目文件夹中
2. 在 tex 脚本中导入宏包，并指定图片所在文件夹
3. 使用图片

我们的 tex 项目整理如下

```txt
- tex_projcet
	- Figure
		- fig.jpg
	- main.tex
```

我们想要使用 `Figure/fig.jpg`，按照上述步骤来

```tex
\documentclass{article}
\usepackage{graphicx}
\graphicspath{{Figures/}}
\begin{document}
\includegraphics{fig}
\end{document}
```

### 使用 figure 环境

一般推荐使用 `figure` 环境来插入图片

```tex
\begin{figure}[h]
    \includegraphics[width=0.5\textwidth, center]{fig}
    \caption{Caption}
    \label{fig:figure2}
\end{figure}
```

其中 `[h]` 是一个可选参数，表示图像会放在页面的哪个地方，这里表示 `here`，还有其他的比如 `t: top, b: bottom`。并且可以同时放多个 `[ht]` 表示先尝试放在这里，再尝试放在顶部，这样能够避免编译出错。

1. `\includegraphics` 的可选参数中 `width` 表示宽度，当然还可以调节 `height=3cm` ，也可以直接使用如 `scale=0.8` 来进行缩放。`\textwidth`  表示文本的宽度，`center` 表示图片的相对位置，相对位置还有 `left, right, center, outer and inner` 最后两个是为双列的文本准备的。
2. `\caption` 中放对图像的描述，其位置根据代码顺序，决定是在图的上方还是下方
3. `\label` 类似于图像的 id，方便在之后使用 `\ref{fig:figure2}` 进行引用。因为可能会有许多引用，所以使用 `fig:` 作为一个标记，其并不是必须的

### 使用 subfigure 环境

我们可以在 `figure` 环境中插入 `subfigure` 环境，以达到创建子图的效果，子图环境中使用的命令是类似的，同时需要使用 `subcaption` 宏包 

```tex
\documentclass{article}
\usepackage{graphicx}
\usepackage{subcaption}
\graphicspath{{Figures/}}
\begin{document}
	\begin{figure}[h]
		\centering 
		\begin{subfigure}{0.2\textwidth}
			\includegraphics[width=\textwidth]{fig1}
			\caption[]{fig a}
		\end{subfigure}
		\hfill
		
		\begin{subfigure}{0.3\textwidth}
			\includegraphics[width=\textwidth]{fig1}
			\caption[]{fig b}
		\end{subfigure}
		\hfill

		\begin{subfigure}{0.4\textwidth}
			\includegraphics[width=\textwidth]{fig1}
			\caption[]{fig c}
		\end{subfigure}
		\caption{This is big figure}
	\end{figure}
\end{document}
```

<img src="Tex tutorial/image-20221028200631130.png" alt="image-20221028200631130" style="zoom:50%;" />

可以看到 `subfigure` 有一个必选参数需要设置图像的大小，其余的命令都与 `figure` 对应。这里 `\hfill` 代表水平对齐，放置在各个子图之间，希望图像尽量在同一横排，对应的 `\vfill` 代表垂直对齐

## 创建 Lists

使用 `itemize` 创建无序列表，使用 `enumerate` 创建有序列表，还有一种列表为 `description` 这里不介绍，下面是简单代码

```tex
\documentclass{article}
\begin{document}
\begin{itemize}	% or use \begin{enumerate}
  \item The individual entries are indicated with a black dot, a so-called bullet.
  \item The text in the entries may be of any length.
\end{itemize}
\end{document}
```

我们还可以使用可选参数改变标号的形式

```tex
\documentclass{article}
\usepackage{amssymb}
\begin{document}

\begin{itemize}
   \item This is an entry \textit{without} a label.
   \item[!] A point to exclaim something!
   \item[$\blacksquare$] Make the point fair and square.
   \item[NOTE] This entry has no bullet
\end{itemize}
\end{document}
```

<img src="Tex tutorial/image-20221027213440700.png" alt="image-20221027213440700" style="zoom: 33%;" />

还可以使用嵌套的列表环境，例如在 `enumerate` 中嵌套 `enumerate` or `itemize`，这甚至会自适应选择不同的序号

## 数学公式

使用 `$formula$` 表示行内公式，使用 `$$formula$$` 表示单独一行公式，但现在推荐使用 `\[formula\]` 来替代 `$$formula$$`，这也等价于 `equation*` 环境，绝大部分公式都需要在此环境中渲染，并且像图片一样接受 `\label` 命令，方便之后引用

贴一个 [CSDN](https://blog.csdn.net/ViatorSun/article/details/82826664) 的总结已经非常全面了，建议按需求查找，熟能生巧。公式的选软通常需要 `amsmath` 这个宏包，下面整理一下其中的常用环境

1. 使用 `align` or `split` 环境对等号对齐，环境中的语法是用 `&=` 来标记对齐的等号，使用 `\\` 换行。区别在于，`align` 不需要在 `equation` 环境中渲染，并且是对每一行都会进行自动标号
2. 使用 `*` 取消标号。例如 `equation, align` 等环境都会自动标号，在环境名后加上 `*` 即可，如 `equation*, align*`
3. 使用 `cases` 环境来渲染分段函数。语法为 `&` 标记条件，`\\` 分段
4. 使用 `matrix` 环境来渲染矩阵。语法是使用 `&` 分隔元素，`\\` 换行

另外一个常用的宏包是 `amssymb` 提供丰富的符号和字体

## 文章结构

1. 摘要需要写在 `abstract` 环境中

2. 使用 `\\` 进行换行（还是同一个段落），使用两个回车创建一个新的段落

3. article 提供三级文章层次结构，并可以自动编号

   ```tex
   \section{}
   \subsection{}
   \subsubsection{}
   
   \section*{}	# no number
   ```


## 插入表格

可以先使用 Excel 将表格的基本数据填入，然后使用 [TableGenerator](https://www.tablesgenerator.com/) 在线生成 tex 表格。

Table Generator 使用技巧：

1. 可以给单元加入/消除边框。通过按住 shift 可以对一行一列进行操作
2. 可以选择使用 booktabs table style。这样能够对边框进行加粗
3. 可以设置 centering，将表格放在中间
4. 可以设置表格整体宽度
5. 可以设置 caption

其中如果想要双线的话可以使用两个 `\hline`，想要粗线的话先导入宏包 `booktabs`，然后使用 `\toprule, \downrule, \midrule`

## 参考文献的插入

TODO

两种方式：手动插入和自动插入

手动插入需要自己定义参考文献格式

```tex
\begin{thebibliography}{99}
\bibitem{ref1}
\bibitem{ref2}
\end{thebibliography}
```

自动插入需要创建一个 `.bib` 文件，该文件包含 bibitem

## 命令与环境

### newcommand

在 tex 中我们可以自定义一些简单的命令，使用如下形式的命令即可

```tex
\newcommand{\CommandName}[NumberofParams][OptionParam]{expressions}
```

可选参数最多只能有一个，并且在表达式中使用 `#` + 数字表示引用第几个参数，如果使用了可选参数，那么可选参数一定为 `#1`，下面举个例子

```tex
\documentclass{article}
\usepackage{amssymb}
\newcommand{\water}{H_2O}
\newcommand{\power}[3][2]{$(#2+#3)^#1$}
\begin{document}
$\water$
\power{x}{y}	% 可选参数的值没有给，使用默认值2
\power[3]{x}{y}
\end{document}
```

上面的代码会得到 $H_2O (x+y)^2(x+y)^3$，简单看一下 `\power` 命令：

1. 总共有3个参数
2. 可选参数的默认值为2，使用 `#1` 进行引用
3. 接收2个必须参数

### Environment

tex 中的环境能够对环境中的内容提供定制化的渲染，以 `\begin{name}` 开始 `\end{name}` 结束，其中 `name` 为环境名。同时环境也接收参数，包括可选参数和必须参数，跟在环境名之后

## 补充

这里整理一些零碎的 trick

[Paper Writing Tips](https://github.com/MLNLP-World/Paper-Writing-Tips)

[English Writing](https://github.com/yzy1996/English-Writing)

[latex writing tips](https://github.com/guanyingc/latex_paper_writing_tips)