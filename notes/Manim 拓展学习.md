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

## 物体的位置与坐标变换

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

## Manim常用几何类

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
vg.arrange_in_grid(rows, cols)
```

## 颜色的表示、运算与设置

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

## 插入SVG、图片与文字

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

TODO: draw Axes like matplotlib

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

## 三维场景与 Camera

三维场景更需要的是对 `Camera` 的理解。实际上 `ThreeDScene` 就是将 `Camera` 换成了 `ThreeDCamera`，并实现了方便的接口来控制 Camera

```python
from manim import *

class ThreeDCameraIllusionRotation(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        circle=Circle()
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(circle,axes)
        self.begin_3dillusion_camera_rotation(rate=2)
        self.wait(PI/2)
        self.stop_3dillusion_camera_rotation()
```

[ThreeDCamera](https://docs.manim.community/en/stable/examples.html#threedcamerarotation) 是通过控制各个 ValueTracker 来操作相机的位置以及参数。可以通过这个特点实现对其灵活操作

[MovingCamera](https://docs.manim.community/en/stable/examples.html#followinggraphcamera) 是通过控制 `self.frame` 来操作相机的拍摄范围

ZoomedScene 复杂一些，可认为其是由一个 Camera + 一个 MovingCamera 组合而成，本质是利用了 [MultiCamera](https://docs.manim.org.cn/cairo-backend/camera/multi_camera.html) 其目前没有比较好的文档。依然可以通过调整 MovingCamera 来控制这个场景，即调整 `zoomed_camera_config`

## 实用技巧

### UpdateFromAlphaFunc

通过这个动画能够实现自己的 Animation，其中 alpha 代某个 Animation 时间运行的进度，取值为0~1

```python
class CountingScene(Scene):
    def construct(self):
        # Create Decimal Number and add it to scene
        number = DecimalNumber(mob_class=Text).set_color(WHITE).scale(5)
        # Add an updater to keep the DecimalNumber centered as its value changes
        def func(obj, alpha):
            value = alpha * (10)
            obj.set_value(value)
        ani = UpdateFromAlphaFunc(number, func)

        number.add_updater(lambda number: number.move_to(LEFT * 3))
        # Play the Count Animation to count from 0 to 100 in 4 seconds
        self.play(Count(number, 0, 100), run_time=4, rate_func=linear)
```

### Updater

既然这里提到了 updater，那么也必须要把 updater 给将明白。每一个 mobject 都有一个 updater list，所谓的 updater 实际上是一个函数：

1. 函数的第一个参数必须是 mobject 本身
2. updater 的第二个参数可有可无，被成为 `dt`，表示动画在渲染时每隔 `dt` 时间会渲染一帧，帧率是由 config 决定的，如果是30帧的帧率，`dt=1/30`

可以简单认为，在渲染的每一帧时，都会调用每个 mobject 的  updater list，对其进行更新

通常可以使用 `ValueTracker` 来作为一个中间桥梁，各个 updater 可以 `ValueTracker` 中的值为基准进行自己对应的更新，[example](https://docs.manim.community/en/stable/examples.html#movingdots)

### AnimationGroup

将多个动画进行组合是非常使用的功能，并且实现了简单控制动画的开始时间，能够通过 `lag_ratio` 来实现延迟

```python
ag = AnimationGroup(Wait(1), circle.animate.set_fill(RED), lag_ratio=1)
```

上方就实现了先等1秒，然后再进行其他动画

### 使用代码渲染

通常来说会需要使用命令行来进行渲染，但当想要更灵活的使用时，直接调用代码渲染会更方便

```python
scene = DemoScene()
scene.render()
```

同时有时候我想获得当前 camera 的拍摄结果可以通过 `pixel_array` 获得

```python
class Demo(ThreeDScene):
    def construct(self):
        self.renderer.update_frame(self)	# 拍摄图像
        cur_picture = self.renderer.camera.pixel_array	# 获取图像
```

## More

[Reference manual](https://docs.manim.community/en/stable/reference.html) [物理模拟](https://www.bilibili.com/read/cv15424290)

总体来说，Manim 非常适合用于制作**示例**，对于三维示例也能够完成大部分的任务，但是对于长视频的制作将非常考验项目的组织能力，可以参考 [issue](https://github.com/3b1b/manim/issues/1086#issuecomment-1272617831) 来完成对多个场景的组合，建议分为多个场景分别渲染以节省不必要的时间