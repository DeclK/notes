# Makefile & CMake

## Makefile

Makefile 文件通常用于编译和链接程序的源代码文件，并可以定义目标、依赖关系和命令等



注意将你的 vscode 右下角从 `Space: 4` 调整为 `Tab Size:4`，否则会一直报错：`*** missing separator.  Stop.`



这是一个 Makefile & CMake 的极简教程，能够让你快速上手使用 make 命令，从而编译一些基本项目

https://makefiletutorial.com/#the-shell-function

```makefile
target: prerequrest
	command
```

target 文件存在时，命令行中使用 `make target` 会先看看需不需要运行 command

变量：变量只能字符串，并且单引号和双引号不需要使用，使用了也会被当做简单字符处理

使用变量必须要用 `$() or ${}`

`:=` 代表简单赋值其逻辑跟我们正常编程时一样，`=` 代表循环赋值，并且只在变量被使用时才会运行该赋值操作，不建议对同一个变量同事使用 `:= & =`

`?=` 代表当变量不存在时进行赋值

`+=` 对变量进行附加字符串

target 和 prerequest 都可以有多个

`$@` 代表当前 target 名称

`$<` 代表该 target 第一个 request

`$^` 代表该 target 所有的 request

`$?` 代表所有的 prerequest newer than the target

`$+` 能够代表重复的 request，基本不用

多个 target 等价于拆开写

```makefile
all: f1.o f2.o

f1.o f2.o:
	echo $@
# Equivalent to:
# f1.o:
#	 echo f1.o
# f2.o:
#	 echo f2.o
```





`%` 是 makefile 里的通配符，`*` 则表示搜索你文件系统中符合匹配的文件名，并且 `*` 需要搭配 `wildcard` 关键字使用



加 `@` 在命令前表示 silent command，其不会在 terminal 中显示

命令是在新的 shell 中跑的，可以用过 `SHELL` 变量来指定所使用 shell 的路径

加 `-` 在命令前表示当命令出错时继续 make

```makefile
one:
	# This error will be printed but ignored, and make will continue to run
	-false
	touch one
```



`$$` 使用环境变量，`$` 使用 makefile 中的变量

```makefile
# Run this with "export shell_env_var='I am an environment variable'; make"
all:
	# Print out the Shell variable
	echo $$shell_env_var

	# Print out the Make variable
	echo $(shell_env_var)
```

也可以在 makefile 中使用 export 关键字设置环境变量，并且还可以将其作为 makefile 变量进行使用

`include` 能够 include 其他 makefile 作用和 C 的 include 一样

`.PHONY` 指令表示下面的 file 无论是否存在 file 都要执行

```makefile
.PHONY: clean
clean:
  rm -rf *.o
```

> Now `make clean` will run as expected even if you do have a file named `clean`.

`.EXPORT_ALL_VARIABLES` exports all variables to the environment variable

`.SECONDARY` 保留中间编译的结果

替换 

```makefile
$(var: %.so=%.cu)
# or suffix onlyl
$(var: .so=.cu)
$(text:pattern=replacement)
```



funtions

function 必须在变量环境中使用 `$(function)`

`if` 用法为

`if $(var_is_not_empty), $(then_var_1), $(else_var_2)`

```makefile
foo := $(if this-is-not-empty,then!,else!)
empty :=
bar := $(if $(empty),then!,else!)

all:
	@echo $(foo)
	@echo $(bar)
```



`shell` 关键字使得能够在 makefile 中执行 shell 命令，并获得其输出的字符串，但是这个字符串将把原字符串中的换行符换为空格

## CMake

https://zhuanlan.zhihu.com/p/534439206

https://github.com/wzpan/cmake-demo/

CMake 允许开发者编写一种平台无关的 CMakeList.txt 文件来定制整个编译流程，然后再根据目标用户的平台进一步生成所需的本地化 Makefile 和工程文件，如 Unix 的 Makefile 或 Windows 的 Visual Studio 工程。从而做到 “Write once, run everywhere”
