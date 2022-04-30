# Manim 项目

## Graph

图形的中心，不是其几何中心，如何设置中心

## Updater

value tracker

dt

.become

.replace

## Camera

本质是什么

四个对象：zoom_camera, frame,

zoomed_display, zoomed_frame

使用 frame 对准想要放大的相框，使用 zoomed_display 放置 camera 位置，性质类似于一个 Mobject。一般不调整 camera & zoomed_frame

zoom_camera 做一些属性设置， zoom display 做移动、变形操作

删除 frame 和 zoomed_frame 都会删除其对应 camera 的显示内容。除非使用 FadeOut 删除 frame 会保留 camera 内容。所以只有当两个 frame 重合时，删除各自的 frame （其中删除 frame 使用 FadeOut）才有还原的动画效果 

而且 frame 和 zoomed_frame 要同时进行缩放，不然不能改变其长宽比。且仅能用 scale([scale_factor]) 进行缩放。scale_factor 可传入一个3维列表，分别代表对每个维度的放缩大小。建议在初始化的时候就定义好 zoom_frame 的比例

zoomed_frame 放大后长宽是有变化的

## 3D

图层

z_index

## Animation

组合

rate_func

run_time

## Text

text 仅支持输入一个字符串

tex 支持输入多个字符串

