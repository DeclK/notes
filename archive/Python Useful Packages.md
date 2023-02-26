---
title: Python Useful Packages
tag:
  - Python
categories:
  - 编程
  - Python
  - Package
abbrlink: d99f910b
date: 2022-03-21 00:00:00
---

# Python Useful Packages

## pathlib

### 取路径

```python
from pathlib import Path

# 当前工作目录
Path.cwd()

# 当前文件路径
Path(__file__)

# 任意字符串路径
Path('abc/file.py')

# 获取绝对路径
Path('abc/file.py').resolve()
```

### 获取路径组成部分

```python
file = Path('abc/file.py')

# 文件名
file.name

# 文件名，不含后缀
file.stem

# 后缀
file.suffix

# 父级目录
file.parent

# 获得所有父级目录
file.parents
file.parents[0]	# 上级 abc
file.parents[1]	# 上上级 .
```

对 `file` 获得父级目录时，仅对所输入的字符串进行操作 `abc/file.py`，如果想要获得绝对路径下的父级目录，请先使用 `.resolve()` 获得绝对目录

### 子路径扫描

```python
path = Path('.')

# 遍历目录下的所有文件/子目录, 但不会递归遍历子目录
files = [f for f in path.iterdir()]

# 查找目录下的指定文件, 通常查找某后缀名文件
files = [f for f in path.glob('*.txt')]
# 子目录递归查询
files = [f for f in path.rglob('*.txt')]
```

### 路径拼接

重载除法算符，非常好用👍

```python
path = Path('.')
new_file = file / 'dir' / 'file.txt'
print(new_file)
# ./dir/file.txt
```

### 路径判断

```python
file = Path(any_str)

# 是否为文件
file.is_file()

# 是否为目录
file.is_dir()

# 是否存在
file.exists()
```

### 文件操作

```python
file = Path('hello.txt')

# 创建文件 touch
file.touch(exist_ok=True)
# exist_ok = False 文件不存在时才能创建, 如果文件存在则报错
file.touch(exist_ok=False)

# 读取与写入文本
# pathlib 对读取和写入进行了简单封装, 不用 open 操作
file.read_text()
file.write_text()

# 打开文件
with file.open() as f:
    pass

# 重命名文件
file.rename(new_name)

# 创建目录
path = Path('dir/')
# parents = True 可以创建多级目录
path.mkdir(parents=True, exist_ok=True)

# 删除目录
# 一次只能删除一级目录, 且目录必须为空
path.rmdir()

# 其实 pathlib 的功能并不是为了删除文件
# 可以使用下面的方法删除文件和目录
import shutil
import os
os.remove(file_path)
shutil.rmtree(dir_path)
```

下面写一个简单的代码，因为有时候想要删除 pycache & gnu.so 文件，把这个文件放在 root dir 就可以了

```python
# clean pycache & gnu.so
from pathlib import Path
import shutil, os
delete_file = ['__pycache__', '*gnu.so']
root_dir = Path(__file__).parent
print(f'root dir: {root_dir}')

for file in delete_file:
    for item in root_dir.rglob(file):
        if item.is_dir():
            shutil.rmtree(item)
        else:
            os.remove(item)
        print(f'deleting {item}')
```

## tqdm

参考 [zhihu](https://zhuanlan.zhihu.com/p/163613814)

tqdm 主要有两种使用方式：

1. 基于迭代对象，你可以把 tqdm 当成是一个装饰器，不影响原来迭代对象的使用

   ```python
   from tqdm import tqdm
   import time
   dic = ['a', 'b', 'c', 'd', 'e']
   for item in tqdm(dic):
       time.sleep(0.1)
   ```

2. 手动进行更新

   ```python
   pbar = tqdm(dic)
   for item in dic:
       time.sleep(0.1)
       pbar.update(n=1)
   ```

   手动更新还能有更多的功能

   1. `pbar.set_description(desc)`
   2. `pbat.set_postfix(dict)`
   3. `pbar.refresh()` 强制更新

除此之外还经常使用 trange 来快速创建

```python
from tqdm import trange
pbar = trange(10)
tbar = tqdm(total=10)	# 不可迭代，仅支持手动更新
```

同时如果有嵌套进度条的话需要指定 `leave` 参数，这样在循环完成后进度条不会留在 shell 输出

```python
with trange(10, leave=False) as tbar:
    for i in tbar:
        pbar = trange(20, leave=False)
        for t in pbar:
            time.sleep(0.1)
```