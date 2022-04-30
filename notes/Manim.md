---
title: Manim 教程
tags:
  - Manim
  - 教程
categories:
  - 编程
  - Python
  - Package
abbrlink: 5654e2fe
date: 2021-08-07 23:16:43
---

# Manim note

官方 [github 项目](https://github.com/3b1b/manim)

> Note, there are two versions of manim. This repository began as a personal project by the author of [3Blue1Brown](https://www.3blue1brown.com/) for the purpose of animating those videos, with video-specific code available [here](https://github.com/3b1b/videos). In 2020 a group of developers forked it into what is now the [community edition](https://github.com/ManimCommunity/manim/), with a goal of being more stable, better tested, quicker to respond to community contributions, and all around friendlier to get started with. See [this page](https://docs.manim.community/en/stable/installation/versions.html?highlight=OpenGL#which-version-to-use) for more details.

这里提到有两个版本的 manim，推荐使用 community edition，这个版本更稳定，更容易上手，下面是两个参考链接

1. [ManimCE github](https://github.com/ManimCommunity/manim/)

2. [ManimCE documentation](https://www.manim.community/)

强烈推荐根据官方文档进行学习，因为网上的很多资源都是过时的，包括我这篇笔记也会可能会很快过时。这篇笔记主要记录如何安装 ManimCE，以及其代码逻辑，更多实用的动画方法另外再做整理

## Install

### Install dependencies

官方给出了一个[ jupyter notebook](https://hub.gke2.mybinder.org/user/behackl-725d956-b7bf9b4aef40b78-cns6ckcq/notebooks/basic%20example%20scenes.ipynb) 预先装载好了 manim 环境，如果你现在不想安装的话，可以在这个 notebook 里面进行小实验

ManimCE 需要预先下载两个软件 ffmpeg 和 LaTex。文档中建议使用 Scoop, Chocolatey 等包管理软件来下载 dependencies，但我太懒了，不想再下一个包管理软件，而且我自己的电脑上本来就安装了 LaTex 和 ffmpeg 所以只要确保 ManimCE  能够调用这些 dependencies 即可，具体来说就是让 ffmpeg 和 LaTex 命令加入环境变量

给两个参考链接：[ffmpeg 知乎教程](https://zhuanlan.zhihu.com/p/118362010)  [LaTex简介 bilibili](https://www.bilibili.com/video/BV11h41127FD?from=search&seid=5330798070960440671)

最后在命令行窗口输入 `ffmpeg -version` 和  `tex -version` 检查看看能否成功运行

### Install ManimCE

> Manim Community runs on Python 3.7+. If you’d like to just use the library, you can install it from PyPI via pip:

```shell
pip install manim
```

因为习惯了用 conda，所以我选择在 conda 中创建一个 Manim 环境，然后再 pip install

```shell
conda create -n manim python=3.8
```

(然而最终却稀里糊涂的安装在了 base 环境)

使用 `conda list manim` 检查是否安装成功，通过 [Quickstart](https://docs.manim.community/en/stable/tutorials/quickstart.html) 简单运行一个程序，检查整个流程是否能够运行

## Tutorials

### Quickstart

下面是一段简单的代码，能够实现从矩形到圆形的变换

<img src="Manim/SquareToCircle_ManimCE_v0.8.0.gif" alt="SquareToCircle_ManimCE_v0.8.0" style="zoom: 25%;" />

代码如下

```python
# scene.py

from manim import *


class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.rotate(PI / 4)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation
```

在 terminal 中执行文件

```shell
manim -pql scene.py SquareToCircle
```

从上面的代码能够看出 Manim 的一些基本框架，首先要有 class 类别作为场景 Scene，在这个 class 中定义 construct() 函数来实现动画效果，然后通过命令行渲染出动画。文档中提到的 Tip:

> Every animation must be contained within the [`construct()`](https://docs.manim.community/en/stable/reference/manim.scene.scene.Scene.html#manim.scene.scene.Scene.construct) method of a class that derives from [`Scene`](https://docs.manim.community/en/stable/reference/manim.scene.scene.Scene.html#manim.scene.scene.Scene). Other code, for example auxiliary or mathematical functions, may reside outside the class.

### A deeper look

主要来分析一下上面的命令

```shell
manim -pql scene.py SquareToCircle
```

文档中解释

> First, this command executes manim on the file `scene.py`, which contains our animation code. Further, this command tells manim exactly which `Scene` is to be rendered, in this case, it is `SquareToCircle`. This is necessary because a single scene file may contain more than one scene. Next, the flag -p tells manim to play the scene once it’s rendered, and the -ql flag tells manim to render the scene in low quality.

下面是几个常见的参数：

1. `-ql, -qm, -qh, -qk` 分别代表不同分辨率，从低到高，再到4k
2. `-a` 渲染所有的 Scene
3. `-i` 输出为 gif 格式
4. `-f` 渲染完成后打开所在文件夹

### Manim’s building blocks

Manim 中有3个重要概念

1. **mathematical object** (or **mobject** for short)
2. **animation**
3. **scene**.

> As we will see in the following sections, each of these three concepts is implemented in manim as a separate class: the [`Mobject`](https://docs.manim.community/en/stable/reference/manim.mobject.mobject.Mobject.html#manim.mobject.mobject.Mobject), [`Animation`](https://docs.manim.community/en/stable/reference/manim.animation.animation.Animation.html#manim.animation.animation.Animation), and [`Scene`](https://docs.manim.community/en/stable/reference/manim.scene.scene.Scene.html#manim.scene.scene.Scene) classes.

#### Mobject

> Any object that can be displayed on the screen is a `mobject`, even if it is not necessarily *mathematical* in nature.

通过 scene 类中的方法 add(), remove() 来在场景中加入 mobject

```python
from manim import *

class CreatingMobjects(Scene):
    def construct(self):
        circle = Circle()
        self.add(circle)
        self.wait(1)
        self.remove(circle)
        self.wait(1)
```

对于 mobject 属性的调整，是通过调 mobject 类的方法来完成

1. Placing mobject location. 选择在哪里加入物体

   .shift()  .move_to()  .next_to()  .align_to()

2. Styling mobjects. 对物体进行风格渲染

   .set_stroke()  .set_fill()

3. Mobject on-screen order. 添加入场景的 mobject 是有顺序的，后添加的物体会覆盖到图层的上方

#### Animation

> At the heart of manim is animation. Generally, you can add an animation to your scene by calling the `play()` method.

一般来讲通过 play() 方法来加入动画

> Animations are procedures that interpolate between two mobjects.

动画的基本原理可以理解为，使用不同的函数在两个关键帧之间进行插值，然后使用 play() 方法进行播放，比如在 Quickstart 中的动画

```python
        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation
```

只要可变的属性，都可以使用动画，通过 `Mobject.aminate` 实现

```python
from manim import *

class ApplyMethodExample(Scene):
    def construct(self):
        square = Square().set_fill(RED, opacity=1.0)
        self.add(square)

        # animate the change of color
        self.play(square.animate.set_fill(WHITE))
        self.wait(1)

        # animate the change of position
        self.play(square.animate.shift(UP))
        self.wait(1)
```

<img src="Manim/ApplyMethodExample_ManimCE_v0.8.0.gif" alt="ApplyMethodExample_ManimCE_v0.8.0" style="zoom: 25%;" />

#### Scene

> The [`Scene`](https://docs.manim.community/en/stable/reference/manim.scene.scene.Scene.html#manim.scene.scene.Scene) class is the connective tissue of manim. 

所有的 mobject 和 animation 都必须加入到 scene 中才能被展现出来，并且 scene 中必须包含 construct() 方法

### Configuration

> Manim provides an extensive configuration system that allows it to adapt to many different use cases. There are many configuration options that can be configured at different times during the scene rendering process. 

在渲染动画的过程中还可以设置更多的参数，比如视频渲染质量的高低 `-ql, -qh`

能够设置 Configuration 的方法有几种

1. The ManimConfig class

   > The most direct way of configuring manim is via the global `config` object, which is an instance of [`ManimConfig`](https://docs.manim.community/en/stable/reference/manim._config.utils.ManimConfig.html#manim._config.utils.ManimConfig).

   ```python
   from manim import *
   config.background_color = WHITE
   config["background_color"] = WHITE
   ```

2. Command-line arguments

   > The following example specifies the output file name (with the `-o` flag), renders only the first ten animations (`-n` flag) with a white background (`-c` flag), and saves the animation as a .gif instead of as a .mp4 file (`-i` flag). It uses the default quality and does not try to open the file after it is rendered.

   ```shell
   manim -o myscene -i -n 0,10 -c WHITE <file.py> SceneName
   ```

3. The config files

   > Manim can also be configured using a configuration file. A configuration file is a file ending with the suffix `.cfg`. To use a configuration file when rendering your scene, you must create a file with name `manim.cfg` in the same directory as your scene code.

   ```python
   [CLI]
   # my config file
   output_file = myscene
   save_as_gif = True
   background_color = WHITE
   ```

#### A list of all config options

```config
     aspect_ratio              assets_dir        background_color       background_opacity
           bottom          custom_folders         disable_caching                 dry_run
  ffmpeg_loglevel             flush_cache            frame_height              frame_rate
       frame_size             frame_width          frame_x_radius          frame_y_radius
from_animation_number          fullscreen              images_dir              input_file
        left_side                 log_dir             log_to_file        max_files_cached
        media_dir             media_width       movie_file_extension    notify_outdated_version
      output_file       partial_movie_dir            pixel_height             pixel_width
          plugins                 preview            progress_bar                 quality
       right_side             save_as_gif         save_last_frame               save_pngs
      scene_names       show_in_file_browser                sound                 tex_dir
     tex_template       tex_template_file                text_dir                     top
      transparent       upto_animation_number   use_opengl_renderer     use_webgl_renderer
        verbosity               video_dir       webgl_renderer_path       window_position
   window_monitor             window_size               write_all          write_to_movie
```

### Using Text

There are two different ways by which you can render **Text** in videos:

1. Using Pango ([`text_mobject`](https://docs.manim.community/en/stable/reference/manim.mobject.svg.text_mobject.html#module-manim.mobject.svg.text_mobject))
2. Using LaTeX ([`tex_mobject`](https://docs.manim.community/en/stable/reference/manim.mobject.svg.tex_mobject.html#module-manim.mobject.svg.tex_mobject))

一般如果不用公式的话，直接使用 text_mobject 就可以了

```python
from manim import *

class HelloWorld(Scene):
    def construct(self):
        text = Text('Hello world').scale(3)
        self.add(text)
```

<img src="Manim/image-20210729160507790.png"  style="zoom:25%;" />

使用 tex_mobject 的话，如下

```python
from manim import *

class HelloLaTeX(Scene):
    def construct(self):
        tex = Tex(r"\LaTeX").scale(3)
        self.add(tex)
```

<img src="Manim/image-20210729160532548.png" style="zoom: 33%;" />

注意需要使用 raw string `r('...')` 因为 Latex 中很多特殊字符需要进行转义

还有不同的方法都能返回 text_mobject & tex_mobject，比如 MarkupText, MathTex 

这篇笔记就到这里了，还有好多好玩的功能就请自行探索吧 😎
