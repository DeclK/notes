---
title: Tex Tutorial
date: 2023-04-05
categories:
  - 编程
  - Tools
---

# TeX Tutorial

参考 [bilibili](https://www.bilibili.com/video/BV1Rd4y1X7AL) [overleaf](https://www.overleaf.com/learn) [latex writing tips](https://github.com/guanyingc/latex_paper_writing_tips) [Paper Writing Tips](https://github.com/MLNLP-World/Paper-Writing-Tips)

## 简介

tex 是一个专业的排版系统，也常有人认为 tex 是一种语言。我的理解也是类似的，就像 python 一样需要先下载一个解释器 `python.exe` 然后再执行 python 脚本；tex 也需要先下载其“解释器”，这个解释器有多个发行的版本，例如 Texlive or MikTex

优点：最强大的排版软件，写公式是比较方便的

缺点：麻烦，麻烦，非常麻烦！

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

3. vscode 也可以实现 latex 和 pdf 之间的跳转

   1. 在 LaTeX 文件中，按 Ctrl + Alt + J 跳转到对应的 PDF 文件位置
   2. 在 PDF 文件中，按下 Ctrl + 鼠标单机，跳转到对应的 LaTeX 文件位置


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

`\centering` 是指让所有子图图片居中对齐，左对齐和右对齐分别是 `\flushleft & flushright`

### 使用 minipage 打包

当有多组图像的时候，可以使用 minipage 进行打包处理，参考 [bilibili](https://www.bilibili.com/video/BV1bi4y1D7Z3)，同样和 subfigure 一样需要指定 minipage 的宽度，此时再在 minipage 中设置 subfigure 宽度 `\linewidth` 就指的是 minipage 的宽度。当然也可以使用多个 `\includegraphics` 来创造多图

```tex
\documentclass{article}
\usepackage{graphicx}
\usepackage{subcaption}
\graphicspath{{Figures/}}
\begin{document}
	\begin{figure}[h]
		\centering
        \begin{minipage}{0.4\linewidth}
            \flushleft
            \begin{subfigure}{\textwidth}
                \includegraphics[width=\textwidth]{fig1}
                \caption[]{fig a}
            \end{subfigure}
        \end{minipage}
        \begin{minipage}{0.4\linewidth}
            \flushright
            \begin{subfigure}{0.5\textwidth}
                \includegraphics[width=\textwidth]{fig1}
                \caption[]{fig b}
            \end{subfigure}

            \vspace{10pt}
            \begin{subfigure}{0.5\textwidth}
                \includegraphics[width=\textwidth]{fig1}
                \caption[]{fig c}
            \end{subfigure}
        \end{minipage}
		\caption{This is big figure}
	\end{figure}
\end{document}
```

如果两个 subfigure 之间有一个空行将会自动竖向排列，没有空行则是横向排列。并且可以使用 `\vspace, \hspace` 来进行微调

<img src="Tex tutorial/image-20230219162041107.png" alt="image-20230219162041107" style="zoom:50%;" />

实际上如果图片过于复杂，图片的排列还可以直接在 PPT 当中完成，许多文章也是这样做的，latex 中的排列多为简单的排列

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

1. 使用 `align` or `split` 环境对等号对齐，环境中的语法是用 `&` 来标记对齐的地方，使用 `\\` 换行。区别在于，`align` 不需要在 `equation` 环境中渲染，并且是对每一行都会进行自动标号
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
2. 可以设置 centering，将表格放在中间
3. 可以设置表格整体宽度
4. 可以设置 caption

推荐的设置

<img src="Tex tutorial/image-20230223112713116.png" alt="image-20230223112713116" style="zoom:50%;" />

**微调技巧**

其中如果想要双线的话可以使用两个 `\hline`，想要粗线的话先导入宏包 `booktabs`，然后使用 `\toprule, \downrule, \midrule` 去替换 `\hline`，并且可以使用参数调整粗细 `\toprule[pt]`

更细致的调整可以使用 `\specialrule{<thickness>}{<abovespace>}{<belowspace>}`，三个都是必填参数，这种一般用不上

## 参考文献的插入

两种方式：手动插入和自动插入

手动插入需要自己定义参考文献格式

```tex
\begin{thebibliography}{99}	% 99 为最大参考文献数量
\bibitem{ref1}auther, title
\bibitem{ref2}auther, title
\end{thebibliography}
```

自动插入需要创建一个 `.bib` 文件，该文件包含 bibitem

推荐使用 [zotero better bibtex](https://retorque.re/zotero-better-bibtex/) 插件生成 `refs.bib` 文件，然后放到 latex 项目目录下。一般在 `main.tex` 文件中的末尾加入 `\bibliography{refs}`，然后在正文中使用 `\cite{key}` 方式完成引用。参考 [bilibili](https://www.bilibili.com/video/BV1ug411W7nY)，视频里还下载了 vscode 插件，不是必须的。注意导出格式一定要选择 `BibTex` 而不是 `BibLaTex`

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

## 分章节撰写

就像写代码一样，我们不希望所有的函数都放在一个文件里，模块化才是更好的选择。所以用 latex 写文章也可以进行模块化，一个章节用一个 `.tex` 文件进行，然后使用 `\input{path/xxx.tex}` 即可，该命令相当于把 `xxx.tex` 中的代码复制到当前文件中。下面是一个 `main.tex` 例子

```tex
\begin{document}
%\maketitle
\begin{abstract}
ABSTRACT
\end{abstract}

%******************BODY TEXT
\input{1_intro.tex}
\input{2_related.tex}
\input{3_method.tex}
\input{4_exp.tex}
\input{5_con.tex}

\clearpage

\section{Acknowledgements}

\bibliography{ref}

\end{document}
```

文件夹结构如下

```txt
- Figure
	- fig1.pdf
	- fig2.pdf
- main.tex
- 1_intro.tex
- 2_related.tex
- 3_method.tex
- 4_exp.tex
- 5_con.tex
- ref.bib
```

## 实战补充

### 作者

撰写作者通常要注意三点：

1. 是否为通讯作者 corresponding author
2. 是否为相同贡献 equal contribute
3. 所在机构与联系邮箱

下面用一个例子来解决

```tex
\urlstyle{rm} % DO NOT CHANGE THIS
\def\UrlFont{\rm}  % DO NOT CHANGE THIS

\title{My Publication Title --- Multiple Authors}
\author {
    % Authors
    First Author Name,\textsuperscript{\rm 1,\rm 2}
    Second Author Name\thanks{Correspoingding author},
    Third Author Name \textsuperscript{\rm 1},
    Fourth Author Name \equalcontrib
}
\affiliations {
    % Affiliations
    \textsuperscript{\rm 1} Institution 1\\
    \textsuperscript{\rm 2} Institution 2\\
    firstAuthor@affiliation1.com, 
    secondAuthor@affilation2.com,
    thirdAuthor@affiliation1.com
}
```

解释：

1. `\textsuperscript` 代表上标，`\rm` 代表数字正体
2. `\equalcontrib` 代表同等贡献
3. `\thanks` 会生成脚注，并显示其中的内容，通常用于通讯作者。也会有使用信封标志表示 `$\textsuperscript{\Letter}$`

### 副标题

在使用副标题的时候也经常使用加粗的形式表示，`\noindent \textbf{xxx}`，这里也取消了缩进

### 限制换行符 `~`

`~` 可以起到限制换行的作用，例如在引用的时候期望换行则使用 `~\cite`，一般也使用在 `Figure~\ref{}` 以及 `Table~\ref{}`，注意图片和表格都是大写

图片最好是裁切得刚刚好，可以使用在线的 pdf 裁切工具 [pdf resizer](https://pdfresizer.com/crop) 来完成

### 调整图片位置

图片表格插入到下一页了怎么办？没什么办法，多尝试一下，放在哪里合适

有时候图片太大了，latex 将其排版到最后一页，这里可以使用 `[H]` 参数，强制其在此页/下一页出现

```tex
\begin{figure}[H]
\end{figure}
```

### 图片距离内容太远

使用 `\vspace{-0.3cm}` 在 `\begin{figure} \end{figure}` 前后均可调整

### 双栏大图片

`\begin{figure*}` 代表生成一个覆盖双栏的图片，不支持 `[h] or [b]`，一般使用 `[t]`

### 调整公式大小

如果内联公式比较长，不够紧凑怎么办？

1. 可以使用 `\resizebox{0.4\textwidth}{!}{ formula }` 的方法调整公式宽度，`{!}` 表示自动调整高度
2. 可以使用 `\!` 来去除间隙

### 强调

`\emph{words}` 来完成斜体强调

### 算法语法

在许多论文当中都有算法的需求，规范的算法流程可以用latex语法表达，需要使用两个包，`algorithm & algorithmic` 前者用于 caption 后者用于书写代码。参考 [overleaf](https://www.overleaf.com/learn/latex/Algorithms)，下面是 AdamW 的算法：

```tex
\usepackage{algorithm,algorithmic}

\begin{algorithm}[ht]
    \caption{AdamW algorithm}
    \label{algo:adamw}
    
    \textbf{Input:} $\gamma(\text{lr}), \beta 1,\beta 2, \epsilon, \lambda(\text{weight decay}),f(\theta)(\text{objective}),\theta_0(\text{parameters})$
    \vspace{2pt} \hrule
    
    \begin{algorithmic}[0]  % [n]代表每n行有标号
      \STATE $m_0 \leftarrow 0$ (Initialize first moment vector)
      \STATE $v_0 \leftarrow 0$ (Initialize second moment vector)
      \FOR {t=1,...}
        % \STATE $t \gets t + 1$
        \STATE $g_t = \nabla_\theta f (\theta_{t-1})$ % (Get gradients w.r.t. stochastic objective at timestep t)
        \STATE $m_t = \beta_1 · m_{t-1} + (1 - \beta_1) · g_t$ % (Update biased first moment estimate)
        \STATE $v_t = \beta_2 · v_{t-1} + (1 - \beta_2) · g_t^2$ % (Update biased second raw moment estimate)
        \STATE $\hat{m}_t = m_t / (1 - \beta_1^t)$ % (Compute bias-corrected first moment estimate)
        \STATE $\hat{v}_t = v_t / (1 - \beta_2^t)$ % (Compute bias-corrected second raw moment estimate)
        \STATE $\theta_t = \theta_{t-1} - \gamma · (\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda \theta_{t-1})$ % (Update parameters with weight decay)
      \ENDFOR
      \STATE \textbf{return} $\theta_t$ % (Resulting parameters)
    \end{algorithmic}
    
\end{algorithm}
```

技巧：

1. 使用`\STATE`表示**所有语句**，这样才有缩进
2. 使用 `\IF...\ELSIF...\ELSE...\ENDIF` 表示条件
3. 使用 `\FOR...\ENDFOR` 或者 `\WHILE...\ENDWHILE` 表示循环
4. 使用 `\RETURN` 表示返回值。也可以使用 `\STATE \textbf{text}` 表示任何语句

### 编译错误

1. 写公式和表格前一定要检查是否使用对应的包！不然会报错，有时候将难以察觉！

2. `\&` 等特殊符号 

## NJU中文毕设

在 github 上找到了 [nju-lug/NJUThesis](https://github.com/nju-lug/NJUThesis) 这是有组织在维护的，所以推荐！该项目的 [用户手册](https://mirror-hk.koddos.net/CTAN/macros/unicodetex/latex/njuthesis/njuthesis.pdf)，非常长，仅作为查询

**使用技巧，每一条都要看！**

1. 设置 `\documentclass`，包含：本科/硕士/博士，学术型/专业型，单页/双页，盲审，字体设置win/mac/linux
2. 设置 `info`，在 `njuthesis-setup` 文件当中，包括：标题，作者，学号，导师，专业等
3. 设置 `.bib` 参考文献，依然在 `njuthesis-setup` 文件当中。但是使用默认方法在进行参考文献的引用时无法自动提示，根据 [discussion](https://github.com/nju-lug/NJUThesis/discussions/126) 解决。在 `njuthesis-setup` 中使用 `\addbibresource{.bib}` 即可
4. 增加了 `\chapter{}` 作为文章结构，该结构等级高于 `\section`
5. 使用 `\include{chapter.tex}` 来导入章节。区别于 `\input`：
   - `\include` 命令会在插入文件之前和之后自动换页，而 `\input` 命令不会
   - `\include` 命令只能在主文件的正文部分使用，不能在导言区或其他环境中使用，而 `\input` 命令没有这个限制
6. 必须使用 xelatex 来进行编译，即 build latex project 选项下面有一个：Recipe: latexmk (xelatex)
7. 模板使用 `unicode-math` 宏包配置数学字体， 该方案目前不兼容传统的 amsfonts，amssymb 等宏包。需要使用新方案提供的相应命令。可能会出现有的数学公式或者表格中的  `\checkmark` 打不出来，可以查阅  [用户手册](https://mirror-hk.koddos.net/CTAN/macros/unicodetex/latex/njuthesis/njuthesis.pdf)。例如使用 `\ensuremath{\checkmark}` 就可打出✔
8. 参考文献超宽了，可以在 `\printbibliography` 之前加入 `\sloppy` 命令是最简单的方法，没人会在乎参考文献美观与否，参考 [issue](https://github.com/nju-lug/NJUThesis/issues/57)
9. 如果参考文献中出现了 `\\`，这是符合规定的。如果非要修改那么可以参考 [issue](https://github.com/nju-lug/NJUThesis/issues/152)
10. 在一个段落内插入图片时，需要使用 `\\` 进行换行处理，使用该方式换行不会有缩进。同时有时候会出现渲染超出文本范围的问题，也需要手动使用`\\`进行换行处理，或者加入连字符`-`，这种情况通常出现在公式/英文与中文混合的时候

其他问题就见招拆招吧，应该会比较顺利
