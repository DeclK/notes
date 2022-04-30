---
title: Python 4 拓展
tags:
  - Python
  - 拓展
  - 教程
categories:
  - 编程
  - Python
  - Basic
abbrlink: c19cb72c
date: 2021-08-28 21:52:57
---

# python 拓展

## format输出

这里总结一下 format 输出的形式，主要以 `{position:format}` 形式对变量进行引用

```python
name = 'Declan'
age = 23
# 按照顺序引用
print("{0} {1} {0}".format(name, age))
# Declan 23 Declan

# 按照关键字引用
print("{name} {age}".format(name=name, age=age))
# Declan 23

# 数字格式化
print("{:.2f}".format(age))
# 23.00

# 输出左右对齐
print("{:0>10d}".format(age))   # 向右对齐，默认宽度为10，不够以0补位
# 0000000023
print("{:x<10b}".format(age))   # 向左对齐，默认宽度为10，不够以x补位
# 10111xxxxx
```

## 读写文件

Python open() 方法用于打开一个文件，并返回文件对象，在对文件进行处理过程都需要使用到这个函数，如果该文件无法被打开，会抛出 OSError。**注意：**使用 open() 方法一定要保证关闭文件对象，即调用 close() 方法

open() 函数常用形式是接收两个参数：文件名(file)和模式(mode)，更多参数查询  [菜鸟教程](https://www.runoob.com/python/file-methods.html)。现在介绍一下常用的 `mode` 参数：

1. `r`，只读模式，为默认设置
2. `w`，写入模式，会擦除文件中之前的内容，重新写入。如果没有文件会自动创建
3. `a`，附加模式，会在文件末尾加入新内容
4. `r+`，读写模式，指针位于文件开头

下面介绍操作文件的基本功能：读和写。我们先在 python 脚本所在文件夹新建一个文本文档 `test.txt` 内容为 `a, b, c`

### 读文件

```python
f = open('test.txt', mode='r')

# 使用 read(size) 输出
# 如果不传入输出字符数量，则全部输出
print(f.read())
# a, b, c

# 读完文件过后需要移动指针到文件开头才能重新再读一遍
# 使用 seek(offset)，默认0位置从文件开头算起
f.seek(0)

# 使用 readlines() 逐行输出
# 该方法返回值是由每一行内容组成的一个列表
for i in f.readlines():
    print(i)
# a, b, c
f.close()
```

### 写文件

```python
f = open('test.txt', mode='a')

# 使用 write(str) 在指针处写入内容
f.write('\nadd something')
# a, b, c
# add something
f.close()
```

如果需要写入中文，建议设置编码为 utf-8，`open(filename, mode, encoding='utf-8')`

### with 关键字

一些对象定义了标准的清理行为，无论系统是否成功的使用了它，一旦不需要它了，那么这个标准的清理行为就会执行。只需要该对象定义了 `__enter__ & __exit__` 魔术方法，关键字 with 就可以保证诸如文件之类的对象在使用完之后一定会正确的执行他的清理方法

```python
with open(file, 'r') as f:
    # statement
```

## os & sys 模块

### os

**os** 模块提供了非常丰富的方法用来处理文件和目录。关于 os 模块更多操作可以查询 [菜鸟教程](https://www.runoob.com/python3/python3-os-file-methods.html)，这里列举一些常用的部分。由于返回的这些路径多为字符串类型，所以字符串的操作在这里也会经常使用

#### 系统相关

```python
import os
print(os.name)  # 操作系统名称
print(os.environ)   # 环境变量，返回一个“字典”
print(os.sep, os.pathsep)   # 路径分割符号，windows 为 \ ;

# 查看一下 PATH 环境变量
# PATH一般为系统指定可执行文件的搜索路径
PATH = os.environ['PATH'].split(';')
for path in PATH:
    print(path)
```

#### 文件和目录

```python
import os
path = '.'  # 当前文件夹

os.mkdir('folder')
os.makedirs('folder\subfolder')
os.remove(path)   # 删除文件
os.removedirs(path)   # 删除空文件夹
os.listdir(path)    # 返回文件列表
os.getcwd() # 返回当前路径

# os.path 是一个常用模块，重点介绍
os.path.abspath(path)   # 返回绝对路径
os.path.dirname(path)   # 返回路径的 dirname
os.path.basename(path)
os.path.split(path) 	# 把路径分割成 dirname 和 basename，返回一个元组
os.path.splittext(path)	# 将路径的扩展名分离出来
os.path.exists(path)
os.path.isdir(path)
os.path.join(path, *path)	# 将多个名称合成为一个路径名
```

### sys

sys 即 system，该模块提供了一些接口，用于访问 Python 解释器自身使用和维护的变量。下面列举一些常用成员

```python
import sys

sys.version # 版本号
sys.path    # python 解释器搜索路径
sys.argv    # 传入当前 python 文件的参数列表
sys.exit(0) # 退出状态码
```

## 错误和异常

当 Python 脚本发生异常时我们需要捕获处理它，否则程序会终止执行

### try相关

异常处理的方法

```python
try:
    statement
except Exception:
    # 发生异常时执行的代码
    print('Something Wrong!')
    statement
# 可以有多个 exception 表示面对不同异常的处理
except Excepthion_2:
    # 发生异常时执行的代码
    statement
else:
    # 没有异常时执行的代码
    statement
finally:
    # 无论有没有异常都会执行的代码
    statement
```

### raise

Python 使用 raise 语句抛出一个指定的异常。更多 python 内置异常查看 [菜鸟教程](https://www.runoob.com/python/python-exceptions.html)

### assert

Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。其基本语法为

```python
assert expression [, arguments]

# 与以下代码等价
if not expression:
    raise AssertionError(arguments)
```

之后的参数为抛出异常的补充信息，也可以不用加。下面举一个例子

```python
assert 1==2, 'Somenthing Wrong!'

# 以下为异常
Traceback (most recent call last):
  File "d:/...", line 1, in <module>
    assert 1==2, 'Somenthing Wrong!'
AssertionError: Somenthing Wrong!
```

## argparse 模块

argsparse 是 python 的命令行解析的标准模块，内置于python，不需要安装。这个库可以让我们**直接在命令行中就可以向程序中传入参数**，同时argparse会自动生成帮助文档，传入参数 `-h` 显示。下面写一个 python 脚本

```python
# arg.py
import argparse

# 创建 parser 并添加描述，通常为该脚本的用途
parser = argparse.ArgumentParser(description='this is a parser')
# 添加参数
parser.add_argument('params')
# 解析参数
args = parser.parse_args()

print('show the params:', args.params)
```

由于 argparse 模块是从命令行向文件传入参数，所以现在在 cmd 下给文件名后面传入参数 123 得到结果如下

```cmd
(base) D:\VScodeProjects\LearnManim>python arg.py 123
show the params: 123
# 参数 123 传入到了文件中 parser 的位置参数 params 中
```

从 `arg.py` 中看出使用 argparse 模块的步骤：

1. 创建 `ArgumentParser` 对象
2. 给对象中加入参数 `add_argument`
3. 解析对象，也即将命令行中的参数传给 `args`，之后能够像类的属性一样调用这些参数 `args.params`

下面来详细了解一下方法 `add_argument`，该方法能够方便我们从命令行输入参数。先来看看该函数自己的参数

1. 位置参数：不带 `-` 的参数。从命令行传入的参数会按照顺序传给 parser 添加的参数，比如上面的例子中，传入的第一个参数为 123，而 parser 中添加的第一个参数是 `params`，所以 123 传给了 `params`。正如 python 自身函数传参规则一样，位置参数是必须要传入的参数，如果不传入则会报错

   ```python
   (base) D:\VScodeProjects\LearnManim>python arg.py         
   usage: arg.py [-h] params
   arg.py: error: the following arguments are required: params
   ```

2. 可选参数：带 `-` 的参数。可选参数有两种方式，

   - `-` 指定短参数，如 `-h`
   - `--` 指定长参数，如 `--help`

   两种参数可以同时存在，可选参数就像 python 函数中使用关键字传参一样。修改一下上面的 python 脚本

   ```python
   # arg.py
   import argparse
   
   parser = argparse.ArgumentParser(description='this is a parser')
   parser.add_argument('--params', '-p')
   args = parser.parse_args()
   
   print('show the params:', args.params)
   ```

   使用短参数和长参数都可以，不传入参数也不会报错

   ```cmd
   (base) D:\VScodeProjects\LearnManim>python arg.py -p 1
   show the params: 1
   
   (base) D:\VScodeProjects\LearnManim>python arg.py --params 1
   show the params: 1
   ```

3. 类型 `type`：指定输入参数的类型

4. 默认值 `default`：指定默认值

5. 帮助 `help`：描述该参数的功能

6. 必须参数 `required`：该参数是否为必须传入的参数

   ```python
   parser.add_argument('--params', '-p', type=int, default=0, help='optional number', required=True)
   ```

7. 动作 `action`：不需要在命令行传入参数。常用值为 `action='store_true'` 也即当传入关键字时，该参数值为 `True` 

   ```python
   parser.add_argument('--params', '-p', action='store_true')
   ```

   ```cmd
   (base) D:\VScodeProjects\LearnManim>python arg.py -p  
   show the params: True
   ```

当然还有更多的参数，就不列举了，遇到了就去查文档

## re 模块

re 为正则表达式的缩写，所以需要了解正则表达式的一些基础知识。正则表达式可以匹配某些特定模式的数据，在编辑文件、爬虫等方面有着广泛应用，以下内容参考 [bilibili](https://www.bilibili.com/video/BV19t4y1y7qP/) & [菜鸟教程](https://www.runoob.com/regexp/regexp-syntax.html) 学习，这里记录一些基本的语法规则，仅供复习

### 正则化

#### 字符组

`[]`，字符组，匹配集合中的任意字符，如果想表示特殊字符则需要转义字符 `\`

 `^`，在字符组内表示取反，在字符组外表示匹配开头

`$`，匹配字符串的末尾

#### 快捷表示

`\d`，数字

`\s`，空格

`\w`，字符（字母、数字、下划线）

大写为其取反 `\D` `\S` `\W` 表示非数字、非空格、非字符

`.`，为任意单个字符，换行符除外

#### 限定符（控制重复次数）

`{n}`，`{n,}`，`{n, m}` n, m 为非负整数，表示匹配重复次数或者重复区间

`+`，匹配前面的子表达式一次或多次，相当于 `{1,}`

`*`，匹配前面的子表达式零次或多次，相当于 `{0,}`

`?` 可选字符，匹配0次或1次。也可表示非贪婪模式（最小匹配），常用于 `+ or *` 之后。例如需要匹配 `<mark>.*<mark>` 这样的模式，如果遇到字符串 `<mark>1<mark> <mark>2<mark>` 将会匹配整个字符串，而不能分 `<mark>1<mark>` 和 `<mark>2<mark>`，这时使用最小匹配即可完成 `<mark>.*？<mark>`

#### 定位符

`$`，匹配字符串的末尾

`^`，匹配字符串的开头，但当 `^` 和 `$` 同时出现时不代表开头和末尾分别匹配，而是代表整个字符串必须匹配该表达式，例如 `^python$` 仅能匹配 python 这个字符串，而不能匹配开头末尾为 python 的字符串，像 python ... python 是不被匹配的

`\b`，匹配单词边界，即字与空格的位置

#### 分组

`()`，表示捕获分组，`()` 会把每个分组里的匹配的值保存起来， 多个匹配值可以通过转义符+数字 n 来查看，举个例子 `(expression_1)(expression_2)\2\1`

`(?:)`，不捕获，只使用分组功能

`(express_1|express_2|express_3)`，选择，满足任一表达式即匹配

`(?=express)`，正向先行断言，顺序从左往右看，判断在该位置的字符是否匹配表达式，如满足表达式则匹配整个表达式，如 `re(?=gular)` 只匹配 regular 中的 re 而不匹配其他单词中的 re

`(?<=express)`，正向后行断言，顺序从右往左，如 `(?<=re)open` 只匹配 reopen 中的 open，不匹配其他单词中的 open

`(?!express)`，逆向先行断言，不满足表达式则匹配

`(?<!express)`，逆向后行断言

### 模块方法

1. `re.match(pattern, string, flags=0)`，尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配成功的话，match()就返回none，匹配成功则返回匹配对象 MatchObject

   `flags` 为标志位，用于控制正则表达式的匹配方式，如：是否区分大小写，多行匹配等等，参考 [正则表达式修饰符 - 可选标志](https://www.runoob.com/python/python-reg-expressions.html#flags)

   使用 group(num) 或 groups() 匹配对象函数来获取匹配表达式中的分组，即表达式中要有括号

   ```python
   import re
   
   line = "Cats are smarter than dogs"
    
   a = re.match( r'(.*) are (.*?) .*', line)
   
   print(a.group(1), a.group(2))
   # Cats smarter
   print(a.groups())
   # ('Cats', 'smarter')
   ```

   group() 也可以一次输入多个组号，在这种情况下它将返回一个包含那些组所对应值的元组

   如果没有分组，则直接使用 group() 调用，不用传参，默认传入0

   ```python
   import re
   
   line = "Cats are smarter than dogs"
    
   a = re.match( r'.* are .*? .*', line)
   print(a.group())
   # print(a.group(0)) is the same
   # Cats are smarter than dogs
   ```

2. `re.search(pattern, string, flags=0)`，扫描整个字符串并返回第一个成功的匹配对象 MatchObject

3. `re.findall(pattern, string, flags=0)`，在字符串中找到正则表达式所匹配的所有**子串**，并返回一个列表，如果没有找到匹配的，则返回空列表

4. `re.sub(pattern, repl, string, count=0, flags=0)` 用于替换字符串中的匹配项

5. `re.compile(pattern, flags=0)` ，根据一个模式字符串和可选的标志参数生成一个正则表达式对象，该对象能够调用一系列的方法用于正则表达式匹配和替换

   ```python
   import re
   
   line = "re short for regular expression"
    
   pattern = re.compile( r're')
   all_list = pattern.findall(line)
   sub = pattern.sub('RE', line)
   
   print(all_list)
   # ['re', 're', 're']
   print(sub)
   # RE short for REgular expREssion
   ```

## 发送邮件

参考链接：[廖雪峰][https://www.liaoxuefeng.com/wiki/1016959663602400/1017790702398272], [CSDN](https://blog.csdn.net/qq_24285815/article/details/98945385)

直接上代码

```python

import smtplib
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr

msg = MIMEText('hello, send by Python...', 'plain', 'utf-8')

# 输入Email地址和口令:
from_addr = 'hongkun20sme@smail.nju.edu.cn'
password = 'your_passwd'
# 输入收件人地址:
to_addr = 'hongkun20sme@smail.nju.edu.cn'
# 输入SMTP服务器地址: smtp.exmail.qq.com(使用SSL，端口号465)
smtp_server = 'smtp.exmail.qq.com'
port = 465

# 编码名字和地址
def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, 'utf-8').encode(), addr))

msg = MIMEText('hello, send by Python...', 'plain', 'utf-8')
# 发件人（仅作内容，不实际参与 SMTP 服务）
msg['From'] = _format_addr('MailBot <%s>' % from_addr)
# 收件人
msg['To'] = _format_addr('Declan Chen <%s>' % to_addr)
# 主题
msg['Subject'] = Header('来自SMTP的问候……', 'utf-8').encode()

# 这里与原廖雪峰教程中使用的 SMTP() 方法不一样，使用了 SMTP_SSL() 方法
server = smtplib.SMTP_SSL(smtp_server, port)
server.set_debuglevel(1)
server.login(from_addr, password)
server.sendmail(from_addr, [to_addr], msg.as_string())
server.quit()
```

## TODO

### 时间和日期

### 进程、线程

### Python 爬虫

