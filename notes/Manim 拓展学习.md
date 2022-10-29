---
title: Manim Kindergarten 教程
abbrlink: 84b26b6c
date: 2021-09-06 18:23:38
tags:
- Manim
- 教程
categories:
- 编程
- Python
- Package
---

# Manim 拓展学习

由于 Manim-Kindergarten 的 [视频教程](https://www.bilibili.com/video/BV1p54y197cC?from=search&seid=1264961070490764259) 非常友好，所以根据他们的视频进行整理和学习。但是由于教程中使用的 manim 版本并不是社区版本，所有整理的内容和原教程有所出入

## 第一讲 物体的位置与坐标变换

怎样确定物体的坐标？首先要理解 manim 的坐标体系。

1. 在 manim 中，使用三维 ndarray 表示一个点的坐标 `np.array([x, y, z])`，二维场景中设 `z = 0`
2. 单位长度取决于 constants.py 中的 FRAME_HEIGHT，画面的宽度由高度和长宽比同时决定。FRAME_HEIGHT 默认值为8，y 的变化范围只能是 [-4, 4]
3. 在 manim 中，二维画面以中心为坐标原点，向右为 x 轴正方向，向上为 y 轴正方向

manim 中定义了一些常用的单位方向常量比如：

LEFT = `np.array([-1, 0, 0])`

RIGHT = `np.array([1, 0, 0])` 

UP = `np.array([0, 1, 0])`

DOWN = `np.array([0, -1, 0])`

### shift+move_to

两个方法都可以根据传入的 vector 移动物体，shift 是相对移动，move_to 是移动到坐标系中的点

```python
# create a mobject 'mob' first
mob.shift(*vectors)
mob.move_to(*vectors, aligned_egde, coor_mask)
# corr_mask 可以屏蔽指定方向的移动
```

### scale & stretch & set_width/height

对物体进行放大

```python
mob.scale(factor, about_egde=None, about_point=None)
```

对物体进行拉伸

```python
mob.stretch(factor, dim)
mob.set_width(width, stretch=True)
mob.set_height(height, stretch=True)
```

### rotate

根据右手定律进行旋转 

```python
# angle 需要用 manim 中定义的 DEGREES or PI 进行计算
mob.rotate(angle, axis=OUT, about_point)
```

### flip

能够指定对称轴进行镜像翻转

```python
mob.flip()
mob.flip(axis=vector, about_point=None)
```

### align_to & next_to

坐标和某个物体对齐，或者在某个物体的相邻位置

```python
# direction is a vector like LEFT, RIGHT, UL
mob.align_to(mob_or_point, direction)
mob.next_to(mob_or_point, direction, aligned_edge, buff)
```

## 第二讲 manim常用几何类

### line & arrow

```python
line = Line(start_point, end_point, buff=0)
arrow = Aroow(start_point, end_point, buff=0, tip_length)
# buff 调整的是到目标点的距离
```

还有其他更多的变化，比如 Dashline Vector

### arc

```python
arc = Arc(arc_center, radius, start_angle, angle)
```

### circle & dot & ellipse

这三类几何图形都是继承于 arc 类，所有参数都有相似之处

```python
circle = Circle(arc_center, radius, stroke_width)
dot = Dot(arc_center, radius)
ellipse = Ellipse(arc_center, width, height)
```

### annulus & sector

```python
annulus = Annulus(outer_radis, inner_radius)
sector = Sector(outer_radius, inner_radius)
```

### polygon & triangle

```python
triangle = Polygon(point_1, point_2, point_3)
triangle = Triangle()
Hexagon = RegularPolygon(6)
```

对于多边形还有一个特别的方法，将顶点变为圆弧

```python
# mob is a polygon mobject instance
mob.round_corners(radius)
```

### rectangle & square

```python
rectangle = Rectangle(height, width)
square = Square(side_length)
```

### VGroup

该方法能够将多个 mobject 放到一个组中，能够实现类似 list 的功能，例如管理成员、嵌套，同时还能使用 mobject 通用方法

```python
vgroup = VGroup(mob0, mob1, mob2)
vgroup.add(mob3)
vgroup.add_to_back(mob4)
vgroup.remove(mob2)
vgroup.shift(UP)
# 将成员按照某一方向对齐，本质上是实现了 next_to 方法
vg.arrange(DOWN, aligned_edge=LEFT)
```

### AnimationGroup

TODO

## 第三讲 颜色的表示、运算与设置

### 颜色的表示

有三种表示方法

1. 定义的常量，如下图

2. 十六进制，形如 #66CCFF

3. RGB数组，形如 np.array([255, 104, 100])

<img src="Manim 拓展学习/image-20210729201211734.png" style="zoom: 20%;" />

但所有的表示方法，在 manim 中最终都会转化为 Color 类

**推荐使用常量或者十六进制来表示颜色**

### 颜色的运算

列几个常见的运算

```python
# 反色
invert_color(color)
# 插值
interpolate_color(color_1, color_2, ratio)
# 平均
average_color(*colors)
# 梯度
color_gradient([color_1, color_2,], length)
# 随机
random_color()
```

<img src="Manim 拓展学习/image-20210729203331953.png" alt="color_gradient" style="zoom:25%;" />

### 物体颜色设置

stroke 代表边框着色，fill 代表内部着色

| stroke         | fill         |
| -------------- | ------------ |
| stroke_color   | fill_color   |
| stroke_opacity | fill_opacity |

上面的属性都可以通过 set_stroke/fill 来更改

```python
mob.set_stroke(color, width)
mob.set_fill(color, opacity)
mob.set_fill([color_1, color_2,], opacity)
```

### 给 VGroup 子物体上色

```python
# vg is a VGroup instance
vg.set_color(color)
vg.set_color_by_gradient(*colors)
vg.set_colors_by_radial_gradient(radius, inner_color, outer_color)
```

值得一提的是，这些操作都是 `VMobject` 的内置方法，是一种通用的方法，而且这些方法不仅仅可以设置 `stroke` 的颜色，也可以设置 `width, opacity...`

## 第四讲 插入SVG、图片与文字

### 图片

在 manim 插入图片需要先将图片放在当前文件夹下，或者使用 `config.assets_dir` 指定素材文件夹。不同的图片类型则使用不同的对象进行存储

```python
# SVG
mob = SVGMobject('file')

# image: jpg, png, gif
mob = ImageMobject('file')
```

`SVGMobject` 是 `VMobject` 的子类，可以使用其所有动画，但 `ImageMobject` 仅能使用部分动画，如：FadeIn

### 文字与公式

在 manim 中可以使用 `Text` 创建普通文字对象

```python
text = Text(*strings, color, font)
```

可以传入多个字符串，同样 `Text` 也可以使用所有动画。如果想要使用 LaTeX 语法书写文字和公式，则需要使用 `Tex, MathTex` 类

```python
text = Tex(*raw_strings)
formula = MathTex(*raw_strings)
```

所有的文字和公式都是一个 `VGroup`，可以通过索引来对每个字符单独操作

## 第五讲 坐标系统与方程

### 坐标轴

在 manim 中可以插入坐标轴

```python
# NumberLine
# x_range 标明数周的范围以及步长
line = NumberLine(
        numberline = NumberLine(
            x_range=[-10, 10, 2],
            length=10,
            include_numbers=True,
            include_ticks=False
        )

# Axes
# 分别设置 x, y 坐标轴，具体参数打包为字典，关键字同 NumberLine
axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            axis_config=dict(
                include_numbers=True
            )
        )

# NumberPlane
# 注意这里的步长是指数轴的数量
number_plane = NumberPlane(
            x_range=[-10, 10, 2],
            y_range=[-10, 10, 1],
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 4,
                "stroke_opacity": 0.6
            }
        )
```

### 方程

使用 `ParametricFunction` 可以显示函数

```python
# 单变量方程，返回三维 ndarray
def func(self, t):
	return np.array((np.sin(2 * t), np.sin(3 * t), 0))

def construct(self):
    func = ParametricFunction(self.func, t_range = [0, TAU].set_color(RED)
    self.add(func.scale(3))
```

## More

ManimCE 文档有一个 [reference manual](https://docs.manim.community/en/stable/reference.html)，官方描述这个手册的功能：

> This reference manual details modules, functions, and variables included in Manim, describing what they are and what they do. 

里面包含了各种模块和函数，更多的 `VMobject` 和更多的动画操作，文档中还有 [Example Gallery](https://docs.manim.community/en/stable/examples.html) 提供参考

物理模拟，AnimationGroup，三维场景

我判断了一下，manim 更加适合用于制作数学教学视频，而我可能之后将少有机会接触。最后可能把这个文档完善一下，找一些感兴趣的视频，看一下他们的实现，然后制作一个动画放在视频开头或者自己网站吧

目前可以用于制作一些非常简单的示意动画，已经够用了