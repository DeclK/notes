---
title: Python 2 数据结构
tags:
  - Python
  - 数据结构
  - 教程
categories:
  - 编程
  - Python
  - Basic
abbrlink: 760ade0a
date: 2021-08-28 21:52:40
---

# Python String & Data Structure

## String 字符串

python 用单引号 `''` 或者双引号 `""` 来表示字符串，三引号 `'''` 允许跨多行的字符串

```python
s1 = 'hello, world!'
s2 = "hello, world!"
# 以三个双引号或单引号开头的字符串可以折行
s3 = """
hello, 
world!
"""
print(s1, s2, s3, end='')
```

字符串中可以使用反斜杠 `\` 表示转义，如 `\n, \t` 等转义字符，如果希望输入原字符串可在引号前加 r 字母

```python
s1 = '\n\\hello, world!\\\n'
s2 = r'\n\\hello, world!\\\n'
print(s1, s2, end='')
```

### 字符串运算

字符串接受 `+, *, [:], in` 等运算，对字符串内容进行合并、重复、截取、成员运算

### 字符串格式化

当字符串中需要表示变量时，格式化表示将会非常有用，基本用法如下

```python
num, pai = 1, 3.1415926
s = 'here is a number %d, and pai is %.2f' % (num, pai)
print(s)
```

基本逻辑就是，一个萝卜一个坑，想要在字符串中插入变量，就在需要插入的地方用相应的符号替代，然后在字符串后用 `%` 指明变量

这里列几个常用的替代符号：`%d, %f, %s, %e`，最后一个 `%e` 表示科学计数

### 字符串内建函数

字符串自带了很多方便的方法，这里列举几个常用的，string 代表某个字符串

1. `sting.count(str)`，字符串中有多少个 str
2. `string.find(str)`，str 在字符串中的哪个位置，如不存在返回-1
3. `string.format(*args, **kwargs)`，**增强版格式化方法，非常推荐使用该函数来格式化字符串**，参考 [菜鸟教程](https://www.runoob.com/python/att-string-format.html)
4. `string.split(str)`，**按照 str 分隔字符串，返回一个列表**
5. `string.join(Iterable)`，将多个字符串使用 string 连接起来
6. `string.replace(str1, str2)`，将 str1 替换为 str2

## List 列表

列表是最基本的数据结构，可以顺序存储不同的对象。在 python 中创建列表：

1. 直接赋值 `LIST = []`
2. 类型转换 `list(tuple)`

### 列表运算

列表也接受：`+, *, [:], in` 等运算符

### python 内置函数操作列表

1. `len(list)`
2. `max(list)` & `min(list)`
3. `for i in list` & `for index, value in enumerate(list)`，对列表进行循环遍历

### 列表内建函数

1. `list.append(obj)` & `list.insert(index, obj)`
2. `list.remove(obj)` & `list.pop()`
3. `list.sort(reverse=False)`，将列表按照上升值排列，返回 None
4. `list.copy()`

## Tuple 元组

元组和列表类似，也是线性表的一种，不过在元组建立过后，其成员是不能够被修改的，增加和删除成员也是不支持的，在 python 中创建元组：

1. 直接赋值 `TUPLE = (1,)`
2. 类型转换 `tuple(list)`

### 元组运算

元组支持 `+, *, [:], in` 等运算符，但注意，由于元组成员不能被修改的性质，所以不能够对索引值进行新的赋值。虽然不能增加成员，但是仍可以使用 `+` 运算，将两个元组合并

### python 内置函数操作元组

在列表当中能用的操作也能用作元组，毕竟元组可以看作不可修改成员的列表

### 元组内建函数

元组的内建函数就不如列表的丰富了，可以说元组有的，列表都有。下面列举两个

1. `tuple.count(obj)`

2. `tuple.index(obj)`，返回 obj 在元组中的索引数，若元组中不存在 obj 则报错

最后提一句，为什么有了列表还需要有元组这样的数据类型呢？部分原因是元组可以用于保护一些数据，让其不被修改，并且创建元组在时间和空间上的代价更小

## Dict 字典

字典也是一种可变容器，其不是通过 index 索引获得容器内的对象，而是通过 key: value 对，将 key 映射到其对应的 value。换句话说就是通过 key 来获得容器内的对象。很明显这样的映射性质需要 key 是唯一的，同时 python 也要求 key 是不可变的，如数字，字符串，元组。创建字典的方式：

1. 直接赋值 `DICT = {key_1：value_1}`

2. 通过内置函数 `dict()` 创建：

   - `dict(**kwargs)`，传入 key=value 对，例如 `DICT = dict(a=1, b=2)`，注意这里 key 不用转化为字符串形式

   - `dict(iterable, **kwargs)`，传入可迭代对象，其中可迭代对象中的成员必须为二元元组，同时也可任意传入 key=value 对，下面举一个例子

   ```python
   ITER = [('a', 1), ('b', 2)]
   # ITER 也可以通过 python 内置函数 zip() 生成
   # ITER = zip(['a', 'b'], [1, 2])
   DICT = dict(ITER, c=3)
   ```

   - `dict.fromkeys(iterable, value)`，传入可迭代对象，比如列表、元组都可以，将这些顺序表的成员作为 key 创建字典，并且所有的 value 都统一初始化

### 字典运算

字典支持 `[], in` 运算，显然 `+, *` 运算对于字典是没有意义的。而 `[]` 运算区别于列表和元组，索引值不是数字而是 key

### python 内置函数操作字典

1. `len(dict)`

2. `for i in dict`，注意这样形式的循环只会对字典中的 key 进行循环遍历。如果想要同时对 key 和 value 循环遍历则需要调用字典内建函数 `dict.items()` 返回可遍历的 (key, value) 元组数组

   ```python
   DICT = dict(a=1, b=2)
   for i in DICT:
   	print(i)
   # result: a b
   for key,value in DICT.items():
   	print('%s:%d' % (key, value))
   # result: a:1 b:2
   ```

### 字典内建函数

1. `dict.get(key, default=None)`，返回指定键的值，如果 key 值不在字典中则返回 default 值。如果希望不存在 key 值时，不仅返回 default 值，还创建该 key 值，则使用 `dict.setdefault(key, default=None)`。比较类似的，如果字典中不存在某个 key，那么使用 `dict[key] = value` 也会自动在字典中创建 key, value 对
2. `dict.items()` 返回可遍历的 (key, value) 元组组成的列表
3. `dict.keys()` & `dict.values()` 返回 key/value 列表

4. `dict.pop(key, default)` & `dict.clear()` 前者删除 key 值，并返回该 key 对应 value，若不存在该 key 则返回 default。后者清空字典所有内容
5. `dict.update(dict_new)` 将新字典里的 (key, value) 更新到本字典中
6. `dict.copy()`

## Set 集合

集合是一个无序的不重复元素序列，且元素性质是不可变的，如数字，字符串，元组。创建集合有如下方法：

1. 直接赋值 `SET = {1, 2, 3}` 这里必须有元素，如果没有则会创建一个字典

2. 类型转换 `set(iterable)`，传入可迭代对象，可以是列表、元组，也可以不传参数以创建空集合

### 集合运算

集合运算相比于前面的数据结构就要更多了，除了基础的 `[], in` 这样的索引和成员运算，还有 `&, |, -, ^` 交集、并集、差集、异或（对称差集）

### python 内置函数操作集合

在列表当中能用的操作也能用作集合，毕竟集合可以粗略看作要求元素唯一的列表

### 集合内建函数

1. `set.add()`
2. `set.remove(obj)` & `set.clear()` 前者删除指定元素，后者清空所有
3. `set.issubset(set_new)` & `set.issuperset(set_new)` & `set.isdisjoint(set_new)` 判断是否是集合的子集、超集，以及是否是不相交的





## 补充：迭代器，生成式，生成器

这一部分将会涉及到部分面向对象的内容，以及 python 的特殊方法。之前就是因为缺少这两部分的内容的了解，一直对迭代器、生成器不理解，现在进行整理。参考资料：[迭代器与生成器](https://www.runoob.com/python3/python3-iterator-generator.html) [浅析 yield](https://www.runoob.com/w3cnote/python-yield-used-analysis.html) 

### iterator 迭代器

其实在之前就已经见到了迭代器了，列表、元组、字典这些数据结构都能够在 for-in 循环的时候自动生成迭代器对象。在一个类中如何实现迭代器对象？python 提供了特殊方法 `__iter__()` 和 `__next__()` 实现创造迭代器对象，当类中有 `__iter__()` 方法时，就意味着这个类的实例不是一个普通的对象，而是一个迭代器对象

这里提一下 python 类中 `__function__()` 形式的函数被成为称为特殊函数，在特定条件发生时运行，而 `__iter__()` 和 `__next__()` 则会在 for-in 循环中被调用。下面通过迭代器实现一个简单 `ListDemo` 了解其内部逻辑

```python
class ListDemo:
    def __init__(self, *args):
        self.args = args
        self.count = len(args)

    def __iter__(self):
        # 函数内可以做一些初始化操作
        self.index = 0
        # 最后必须返回迭代器对象本身
        return self

    def __next__(self):
        if self.index < self.count:
            result = self.args[self.index]
            self.index += 1
            return result
        else:
            # 循环结束的标志
            raise StopIteration

# 运行一下
LIST =  ListDemo(1, 2 ,3)
for i in LIST:
    print(i)
```

整个迭代器在循环中到底在怎么运行呢？首先 `ListDemo` 先运行了 `__iter__()` 进行了初始化，返回迭代器对象本身。在循环当中， `ListDemo` 类在重复调用 `__next__()` 函数，并返回结果给 `i`，最后遇到 `StopIteration` 结束循环

python 还有内置函数 `iter()`  & `next()` 用于直接调用 `__iter__()` & `__next__()` 而不需要遇到循环才触发

```python
LIST =  ListDemo(1, 2 ,3)
iterator = iter(LIST)
while True:
    try:
        print(next(iterator))
    except StopIteration:
        break
```

### generator 生成器

如果说迭代器是定义在类当中的，那么生成器就是可以直接定义在普通函数中的迭代器。通过 `yield` 关键字能够将普通函数变为生成器，它将实现之前的  `__iter__()` 和 `__next__()` 功能。每次遇到 yield 时函数会暂停并保存当前所有的运行信息，并返回一个 `yield` 的值, 并在下一次执行 next() 方法时从当前 `yield` 位置继续运行。下面实现一个斐波那契数列函数，来进一步理解其内部逻辑

```python
def fibonacci(n):
    a, b = 1, 1
    count = 0 
    while count < n:
        yield a
        a_ = a
        a = b
        b = a_ + b
        count += 1

iterator = fibonacci(10)
for i in iterator:
    print(i)
```

类比一下类中的迭代器，能够更快速的理解。通过 `yield` 声明这是一个生成器，这样会在遇到 for-in 循环的时候不断地调用 next() 方法并返回 `yield` 值 `a` 赋给 `i`。区别于类中的 `__next__()`  方法是重复 `__next__()` 函数内的代码，生成器运行 `next()` 函数，则会从当前 `yield` 处继续运行，直到碰到下一个 `yield` 

### 生成式

生成式是生成器的一个简单应用，可以用于生成列表、字典

```python
# 使用生成式创建列表和字典
f = [x**2 for x in range(10)]
f = [x + y for x in 'ABCDE' for y in '1234567']
f = {x:x**2 for x in range(10)}
# 生成式本质上是一个生成器
f = x for x in range(10)
for i in f:
    print(i)
```

