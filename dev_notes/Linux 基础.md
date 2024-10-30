# Linux 基础

## 命令

推荐 [tldr](https://tldr.sh/) 项目，能够快速上手命令，把 repo 里的 `pages` 文件夹移动到 `~/.cache/tldr` 之下即可

## Shell 编程

### Intro

`#!` 是一个约定的标记，它告诉系统这个脚本需要什么解释器来执行，即使用哪一种  Shell

运行 shell 脚本有两种方式

1. 作为可执行程序

   ```shell
   chmod +x ./test.sh  #使脚本具有执行权限
   ./test.sh  #执行脚本
   ```

2. 作为解释器参数

   ```shell
   bash test.sh
   ```

### 变量

定义变量时，变量名不加美元 `$` 符号，并且等号前后没有空格

```shell
your_name="runoob.com"
```

使用变量时，只要在变量名前加 `$` 符号

```shell
your_name="qinjx"
echo $your_name
echo ${your_name}
```

删除变量使用 `unset xxx` 

字符串操作：

1. 获取字符串长度

   ```shell
   string="abcd"
   echo ${#string}   	# 输出 4
   echo ${#string[0]}  # 输出 4，二者等价
   ```

2. 提取字符串

   ```shell
   string="runoob is a great site"
   echo ${string:1:4} # 输出 unoo
   ```

定义数组，仅支持一维数组

```shell
array=(item_1 itme_2 itme_3 ...)
```

读取和修改方式

```shell
array[0]=value0
echo ${array[0]}
```

获得数组长度

```shell
length=${#array[*]}
```

### 向脚本传递参数

不是向函数传递参数

```shell
bash test.sh arg1 arg2 ...
```

在 shell 变成中通过数字 `$0, $1, $2` 来获得参数，其中 `$0` 为脚本名称

### 运算符

expr 是一款表达式计算工具，使用它能完成表达式的求值操作

点注意：

- 表达式和运算符之间要有空格，例如 2+2 是不对的，必须写成 2 + 2，这与我们熟悉的大多数编程语言不一样。
- 完整的表达式要被 `` 包含，注意这个字符不是常用的单引号

常见的 `+ - * / % = == !=` 都可以运算

**注意：**条件表达式要放在方括号之间，并且要有空格，例如: **[$a==$b]** 是错误的，必须写成 **[ $a == $b ]**

下面的运算通常用于 if 条件中

用于数值的关系运算：`-eq -ne -gt -lt -ge -le`

布尔运算：`-a -o !` 分别代表与或非

逻辑运算：`&& ||` 分别代表 and 和 or

字符串运算：`-z -n $` 分别代表字符串是否为0，是否不为0，是否为空

文件测试运算：用于检测 Unix 文件的各种属性，例如 `-r -w -x -e` 分别代表可读，可写，可执行，存在

### echo

使用 `\` 来进行转义

单引号里的任何字符都会原样输出，单引号字符串中的变量是无效的，转义也不行。如果需要变量以及转义，需要使用双引号

```shell
#!/bin/sh
echo -e "OK! \c" # -e 开启转义 \c 不换行
echo "It is a test"
```

将结果定向到文件，没有文件会创造文件

```shell
echo "It is a test" > myfile
```

显示命令执行结果

```shell
echo `date`
```

多个变量

```shell
echo "word1" "word2"
```

### printf

用于输出格式化输出

### if

```shell
if condition
then
    command
else
	command
fi
```

命令可以像 C++ 一样用 `;` 进行分隔

`fi` 是 `if` 倒过来，标志结束

使用 elif

```shell
if condition1
then
    command1
elif condition2 
then 
    command2
else
    commandN
fi
```

### for &while

一般格式

```shell
for var in item1 item2 ... itemN
do
    command1
done
```

使用 while

```shell
while condition
do
    command
done
```

## tmux

TODO