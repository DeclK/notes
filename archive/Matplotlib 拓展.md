---
title: Matplotlib 拓展
tag:
  - Matplotlib
categories:
  - 编程
  - Python
  - Package
abbrlink: ea21061e
date: 2021-11-16 00:00:00
---

# Matplotlib 拓展

[pyplot api doc](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html#module-matplotlib.pyplot) 放一个 api 链接，便于查找 api

## Axis

有两个推荐的方法

1. `plt.xlim(left, right)`，[xlim doc](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xlim.html?highlight=pyplot%20xlim#matplotlib.pyplot.xlim)

2. `plt.axis([xmin, xmax, ymin, ymax])`，[axis doc](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.axis.html#matplotlib.pyplot.axis) 控制坐标轴范围

3. `plt.axis(False)` 控制是否显示坐标轴

4. `plt.xticks(ticks=, labels=)` [xticks doc](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html#matplotlib.pyplot.xticks)

5. `plt.grid(bool, axis='x')` 显示格点

6. 通过 Axes 控制坐标轴，`ax = plt.gca()`

   - `ax.invert_yaxis()`将 y 轴进行反转 

   - `ax.yaxis.set_label_position` & `ax.yaxis.set_ticks_position`，传入参数 `left/right`

   - [ax.spines](https://matplotlib.org/stable/api/spines_api.html) 设置 "x, y" 轴的位置，以及上下两边的是否显示边框

     ```python
     ax.spines['left'].set_position(('data', 0))
     ax.spines['bottom'].set_position(('data', 0))
     ax.spines['top'].set_visible(False)
     ax.spines['right'].set_visible(False)
     ```

## Text

所有文字相关的接口都会返回 `Text` 文字对象，一般都可传入如下属性

1. `color` & `backgroundcolor`
2. `fontsize`
3. `fontweight` 调整字体粗细 'ultralight', 'light', 'normal', 'regular', 'bold', 'extra bold', 'black'
4. `fontfamily` 规定字体家族，可以更改字体
5. `fontstyle` 可以使用斜体 `italic`
6. `alpha` 透明度，0~1之间，1为完全不透明
7. `bbox` 给文字增加外框，其值为一个字典，常用 `dict(boxstyle='', facecolor='', edgecolor='')`，其中 `boxstyle` 取值请参考 [link](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyBboxPatch.html#matplotlib-patches-fancybboxpatch)，默认为 square 也常用 round

下面看看这些 API 有哪些必要参数，通过一个例子了解

```python
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()

# Set titles for the figure and the subplot respectively
plt.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
plt.title('pltes title')

plt.xlabel('xlabel')
plt.ylabel('ylabel')
# Set both x- and y-pltis limits to [0, 10] instead of default [0, 1]
plt.axis([0, 10, 0, 10])

plt.text(3, 8, 'boxed italics text in data coords', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5})

plt.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)

plt.text(3, 2, 'unicode: Institut für Festkörperphysik')
plt.show()
```

<img src="Matplotlib 拓展/image-20211115155638742.png" style="zoom:67%;" />

## Legend & annotate

`label` 其实是各个 `Artist` 对象都拥有的属性，在使用 `plt.plot()` 类似的方法来绘图的时候，可以直接在参数里使用 `label=` 以创造该绘图对象的标签。而 `legend` 可以用于将 `label` 以图例形式加入到 figure 当中，参考 [知乎](https://zhuanlan.zhihu.com/p/111108841), [legend doc](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend),  [about matplotlib.rcParams](https://matplotlib.org/stable/tutorials/introductory/customizing.html#matplotlib-rcparams)

也可以对图像中的某些点进行标记，使用 `annotate()` 方法即可，[annotate doc](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html#matplotlib.pyplot.annotate)

```python
import matplotlib.pyplot as plt
import numpy as np
# 设置默认字体以显示中文
plt.rcParams['font.family'] = ['HarmonyOS Sans SC']

n = np.linspace(-5, 4, 30)
m1 = 3 * n + 2
m2 = n ** 2

plt.xlabel('时间')
plt.ylabel('心情')

line1, = plt.plot(n, m1, color='r', linewidth=1.5, linestyle='-', label='女生购物欲望')
line2, = plt.plot(n, m2, 'b', label='男生购物欲望')

plt.legend(handles=[line1, line2], labels=['girl购物欲望','boy购物欲望'], loc='best')

plt.annotate('bottom', xy=(0, 0), xytext=(0, -5), arrowprops=dict(color='black', shrink=0.05))

plt.show()
```

<img src="Matplotlib 拓展/image-20211115163146776.png" style="zoom:67%;" />

## Image

在目标检测中，经常使用 bbox 对目标进行框选，并进行类别标注，这些都是可以通过 matplotlib 做到的。一般的图像在计算机视觉中，被处理为一个 (H, W, C) 的三维张量，其中 C 通常为 3，在 matplotlib 中可以使用 `matplotlib.image` 包处理图像，然后使用 `plt.imshow()` 绘制图像，[imshow doc](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html#matplotlib.pyplot.imshow)

```python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

img = mpimg.imread('test.png')
# img = plt.imread('test.png')
plt.imshow(img)
plt.show()

# 修改原点
plt.imshow(img, origin='lower')
plt.show()

# 修改透明度
plt.imshow(img, alpha=0.5)
# 在 img 上绘图
x = np.arange(img.shape[1])
k = img.shape[0] / img.shape[1]
y = k * x
plt.plot(x, y, color='red')
plt.show()
```

<img src="Matplotlib 拓展/image-20211115174301946.png" style="zoom: 67%;" />

`plt.imshow` 也可以用于展示 heatmap，此时输入为 (H, W)，而不是 (H, W, 3)。顺便加一个 colorbar

```python
heatmap = np.random.randn(3, 4)
plt.imshow(heatmap, cmap='rainbow')
plt.colorbar()
```

<img src="Matplotlib 拓展/image-20221207204258909.png" alt="image-20221207204258909" style="zoom: 33%;" />

除了绘制函数，一般的几何图形也能够绘制，一般使用 patches 对象，具体操作可参考 [简书](https://www.jianshu.com/p/8d14238d402a)

## With Jupyter Notebook

在 notebook 中绘画矢量图，并且保存。参考 [zhihu](https://www.zhihu.com/question/59392251/answer/403124614)

```python
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

# 输出为 pdf 可以轻松放入 latex 中
plt.savefig('tmp.pdf', bbox_inches='tight')
plt.show()
```
