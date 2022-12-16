---
title: Numpy Cheat Sheet
tags:
  - Python
  - Numpy
categories:
  - 编程
  - Python
  - Package
abbrlink: 19c6ccbf
date: 2021-10-08 22:20:38
---
# Numpy Cheat Sheet

## Basics 基础

### 简单创建

```python
np.arange(start, stop, step)
np.array([1, 2, 3])
```

### Placeholders 创建

包括创建 linespace, zeros, ones, random, empty 数组

```python
np.linespace(start, stop, num)
np.zeros(shape)
np.ones(shape)
np.random.random(shape)	# 0-1 uniform
np.random.randn(*dimensions)	# normal/gaussian
np.random.randint(low, high, shape)	# high is exclusive
np.empty(shape)
```

## Array 数组

### Array Properties 属性

```python
array.shape
array.ndim	# dimensions of array
array.size
array.dtype
array.astype(type)	# convert data type
len(array)
type(array)
```

### Copy 复制

```python
np.copy(array)
other = array.copy()
```

### Sort 排序

```python
array.sort()
array.sort(axis=0)
```

## Array Operations 数组操作

### Adding & removing

```python
np.append(a, b)
np.insert()	# not common
np.delete()	# not common
```

### Combining

```python
np.concatenate(arrays, axis)
np.vstack(arrays)
np.hstack(arrays)
np.stack(arrays, axis)	# This will create a new axis
```

### Shaping

```python
array.reshape(shape)
array.flatten()
array.transpose()	# equals array.T
array.transpose(axes)	# permute axes
```

## Math 数学

### Operations 基础运算

```python
# basic
np.multiply(x, y)	# equals x @ y
np.dop(x, y)		# dot product of 1D array
np.sqrt(x)
np.sin(x)
np.cos(x)
np.log(x)
np.exp(x)
np.power(x1, x2)	# x1 & x2 have the same shape or x2 can broadcast
np.ceil(x)
np.floor(x)

# preprocess
np.isnan(x)
np.isinf(x)
np.round(x, ndigits)
np.nan_to_num(x, nan)	# Replace NaN with zero and infinity with large finite numbers (default behaviour) or with the numbers defined by the user using the nan, posinf and/or neginf keywords.
np.all(x, axis)		# test whether all be true along a axis
np.any(x, axis)

# compare
np.max(x, axis)
np.min(x, axis)
np.maximum(x, y)	# Element-wise maximum of array elements.
np.minimum(x, y)

# cumulative
np.cumsum(x, axis)	# cumulative sum, if axis=None, flatten x
np.diff(x, axis)	# differences
np.prod(x, axis)	# product along given axis

# arg-relative
np.argmax(x, axis)
np.argsort(x, axis)
np.bincount()		# Count number of occurrences of each value in array of non-negative ints.

# linear algebra
np.linalg.det(x)
np.linalg.inv(x)

# Statistics
np.sum(x, axis)
np.mean(x, axis)
np.std(x, axis)
np.corrcoef(x, y=None)
```

## Slicing 切片

```python
# n dimensions
array[:3,:,...]	# upper bound is exclusive
array[:,-1]		# reverse slicing
array[::2,:]	# step is 2

# bool
array[array > 5]
array[(array>5) & (array%2==0)]

# Fancy indexing
array[[2, 1, 3],:]	# could be any iterable int array
```

提一些注意点：

1. 在多个维度使用花式索引时，注意维度变化（下面代码写的是 pytorch，但 numpy 也是一样的）

   ```python
   import torch
   a = torch.randn(1,3,4,5)
   
   b = a[:, [0, 1],:, [0, 1]]    	# (2, 1, 4)
   # equal with concat a[:, 0, :, 0] & a[:, 1, :, 1]
   
   index = torch.tensor([0, 1])[None, :].expand((6, -1))
   c = a[:, [0, 1],:, index]    	# (6, 2, 1, 4)
   d = a[:, index, :, index]       # (6, 2, 1, 4)
   
   print(b.shape)
   print(c.shape)
   print(d.shape)
   ```

   可以看到，但多个维度都使用花式索引的时候，所有维度的花式索引的 tensor/iterable 必须形状相同，或能够广播到同一个维度上。并且花式索引的维度会被消除，移动到前方

2. 通常使用的布尔索引其形状和被索引张量是一致的，最后生成一个单维度的张量。但也可以用布尔索引在某个维度上进行索引（需要保持布尔索引的形状和该维度的形状一致），此时可把布尔索引当作花式索引来看待

   ```python
   e = a[:, [True, False, True], :, [0, 1]]
   ```

3. 当传入 `index` 是一个元组/列表，且仅传入该 `index` 时，会自动将其分解

   ```python
   index = (index1, index2)
   e = a[index]	# eq a[index1, index2]
   ```

   但是，当对多个维度使用花式索引时，依然按照(1)中的广播方法处理

   ```python
   index = (index1, index2)
   f = a[[:, [0, 1], :, index]	# eq a[[:, [0, 1], :, torch.stack(index)]
   ```

## Broadcast 广播

广播数组维度需要满足以下要求任意一个：

1. 从后往前比，两个数组各个维度大小相同

   ```python
   A = np.zeros((2, 3, 4, 5))
   B = np.zeros((4, 5))
   C = A + B	# B will broadcast to (2, 3, 4, 5)
   ```

2. 两个数组存在维度大小不相等时，其中一个不相等维度大小为1

   ```python
   A = np.zeros((2, 3, 4, 5))
   B = np.zeros((4, 1))
   C = np.zeros((2, 1, 1, 5))
   D = A + B + A * C	# B & C will broadcast to (2, 3, 4, 5)
   ```

## 存储为 .bin

使用 `tofile & fromfile` 将 ndarray 存储为二进制文件

```python
import numpy as np
from pathlib import Path

bin_file = Path('./xxx.bin')
a = np.array([1, 2, 3, 4])
with open(binfile, mode='w') as f:
    a.tofile(f)
b = np.fromfile(binfile, dtype=a.dtype)
print(b)
```

numpy 还可以设置打印参数

```python
np.set_printoptions(precision=4, linewidth=200, suppress=True)
```

precision：控制输出结果的精度(即小数点后的位数)，默认值为8

threshold：当数组元素总数过大时，设置显示的数字位数，其余用省略号代替

linewidth：每行字符的数目，其余的数值会换到下一行

suppress：小数是否需要以科学计数法的形式输出

