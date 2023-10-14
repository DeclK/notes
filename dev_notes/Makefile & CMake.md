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

f1.o f2.o: a1.o a2.o
	echo $@

a1.o:
	@echo a1.o

a2.o:
	@echo a2.o
	
# Equivalent to:
# f1.o: a1.o a2.o
#	 echo f1.o
# f2.o: a1.o a2.o	# but a1.o and a2.o is already satisfied during f1.o, so it won't build again
#	 echo f2.o

# output
a1.o
a2.o
echo f1.o
f1.o
echo f2.o
f2.o
```





`%` 是 makefile 里的通配符，`*` 则表示搜索符合匹配的文件名，并且 `*` 需要搭配 `wildcard` 关键字使用。

`%` 能够让你使用更简洁的语句来构建 Rule，通常的用法是这样

```makefile
%.o: %.c
	gcc -c $< -o $@
# it is a rule for every .c file in your folder
```

具体来说如果你的文件如下

```txt
- Project
	- a.c
	- b.c
	- Makefile
```

运行上面的内容相当于运行

```makefile
a.o: a.c
	gcc -c $< -o $@
b.o: b.c
	gcc -c $< -o $@
# not mean below!!!
# a.o b.o: a.c b.c
```

二者匹配的 makefile 所在文件夹的文件，而不是 makefile 中的变量！

```makefile
make: *** No rule to make target '%.o', needed by 'all'.  Stop.
```

二者的区别在于：% 仅能代表某一个文件，而 * 能够代表多个

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

https://subingwen.cn/cmake/CMake-primer/index.html

https://github.com/wzpan/cmake-demo/

CMake 允许开发者编写一种平台无关的 CMakeList.txt 文件来定制整个编译流程，然后再根据目标用户的平台进一步生成所需的本地化 Makefile 和工程文件，如 Unix 的 Makefile 或 Windows 的 Visual Studio 工程。从而做到 “Write once, run everywhere”



补充的 c++ 知识

```c++
int main(int argc, char *argv[])
```

可以给 main 传入参数，代表参数数量，和其他参数内容（必须是字符串）





一个最简单的 CMkeLists.txt

```cmake
# version check
cmake_minimum_required (VERSION 2.8)

# project name, use ${PROJECT_NAME} to get the value
project (Demo1)

# compile main.cc to Demo executable file
add_executable(Demo main.cc)
```

内部构建和外部构建

mkdir build

cmake ..

一般用外部构建

命令不区分大小写，变量区分大小写

在cmake里定义变量需要使用set

语法

```cmake
SET(VAR [VALUE] [CACHE TYPE DOCSTRING [FORCE]])
```

VAR：变量名
VALUE：变量值

```cmake
set(SRC_LIST add.c;div.c;main.c;mult.c;sub.c)
add_executable(app  ${SRC_LIST})
```



指定 C++ 标准

C++标准对应有一宏叫做 `DCMAKE_CXX_STANDARD`。在CMake中想要指定C++标准有两种方式

1. 在 CMakeLists.txt 中通过 set 命令指定

   ```cmake
   set(CMAKE_CXX_STANDARD 11)
   set(CMAKE_CXX_STANDARD 17)
   ```

2. 在执行 cmake 命令的时候指定

   ```cmake
   cmake CMakeLists.txt文件路径 -DCMAKE_CXX_STANDARD=11
   ```

通常 CXX 就是指代 C++ 相关的东西，是不是发音很像，C叉叉，C加加



指定输出路径

在CMake中指定可执行程序输出的路径，也对应一个宏，叫做EXECUTABLE_OUTPUT_PATH，它的值还是通过set命令进行设置:

```cmake
set(HOME /home/robin/Linux/Sort)
set(EXECUTABLE_OUTPUT_PATH ${HOME}/bin)
```

如果此处指定可执行程序生成路径的时候使用的是相对路径 ./xxx/xxx，那么这个路径中的 ./ 对应的就是 makefile 文件所在的那个目录



搜索文件

如果一个项目里边的源文件很多，在编写CMakeLists.txt文件的时候不可能将项目目录的各个文件一一罗列出来，这样太麻烦也不现实。所以，在CMake中为我们提供了搜索文件的命令，可以使用aux_source_directory命令或者file命令

多个源文件

```cmake
aux_source_directory(<dir> <variable>)
```

dir：要搜索的目录
variable：将从dir目录下搜索到的源文件列表存储到该变量中
CMAKE

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALC)
include_directories(${PROJECT_SOURCE_DIR}/include)
# 搜索 src 目录下的源文件
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_LIST)
add_executable(app  ${SRC_LIST})
```

但是什么是源文件呢？这不太好界定...

### file



还有一种方法，那就是使用 file 来搜索

搜索当前目录的src目录下所有的源文件，并存储到变量中

```cmake
file(GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB MAIN_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
```

CMAKE_CURRENT_SOURCE_DIR 宏表示当前访问的 CMakeLists.txt 文件所在的路径。

关于要搜索的文件路径和类型可加双引号，也可不加



 包含头文件

在CMake中设置要包含的目录也很简单，通过一个命令就可以搞定了，他就是include_directories

```txt
$ tree
.
├── build
├── CMakeLists.txt
├── include
│   └── head.h
└── src
    ├── add.cpp
    ├── div.cpp
    ├── main.cpp
    ├── mult.cpp
    └── sub.cpp

3 directories, 7 files
```

cmake list 如下

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALC)
set(CMAKE_CXX_STANDARD 11)
set(HOME /home/robin/Linux/calc)
set(EXECUTABLE_OUTPUT_PATH ${HOME}/bin/)
include_directories(${PROJECT_SOURCE_DIR}/include)
file(GLOB SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(app  ${SRC_LIST})
```

PROJECT_SOURCE_DIR 宏对应的值就是我们在使用cmake命令时，后面紧跟的目录



制作动态库或静态库

有些时候我们编写的源代码并不需要将他们编译生成可执行程序，而是生成一些静态库或动态库提供给第三方使用，下面来讲解在cmake中生成这两类库文件的方法



制作库

静态

```cmake
add_library(lib_name STATIC source_files...) 
```

在Linux中，静态库名字分为三部分：lib+库名字+.a

动态

在cmake中，如果要制作动态库，需要使用的命令如下：

```cmake
add_library(lib_name SHARED source_files...) 
```

在Linux中，动态库名字分为三部分：lib+库名字+.so



给动态库和静态库指定输出路径

1. 仅适用于动态库

   由于在Linux下生成的动态库默认是有执行权限的，所以可以按照生成可执行程序的方式去指定它生成的目录：其实就是通过set命令给EXECUTABLE_OUTPUT_PATH宏设置了一个路径，这个路径就是可执行文件生成的路径

   ```cmake
   cmake_minimum_required(VERSION 3.0)
   project(CALC)
   include_directories(${PROJECT_SOURCE_DIR}/include)
   file(GLOB SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
   # 设置动态库生成路径
   set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
   add_library(calc SHARED ${SRC_LIST})
   ```

2. 都使用

   由于在Linux下生成的静态库默认不具有可执行权限，所以在指定静态库生成的路径的时候就不能使用EXECUTABLE_OUTPUT_PATH宏了，而应该使用LIBRARY_OUTPUT_PATH，这个宏对应静态库文件和动态库文件都适用




包含静态库、动态库

在编写程序的过程中，可能会用到一些系统提供的动态库或者自己制作出的动态库或者静态库文件，cmake中也为我们提供了相关的加载库的命令

静态库

```cmake
link_libraries(<static lib> [<static lib>...])
# if not
link_directories(<lib path>)
```

link_libraries 指定出要链接的静态库的名字
可以是全名 libxxx.a
也可以是掐头（lib）去尾（.a）之后的名字 xxx

有时候光靠这个名字找不到这个库，就需要 link_directories 把这个库的路径加进来

动态库

动态库的制作、使用以及在内存中的加载方式和静态库都是不同的，在此不再过多赘述，如有疑惑请参考 [Linux静态库和动态库](https://subingwen.cn/linux/library/)

在cmake中链接动态库的命令使用 target_link_libraries:

```cmake
target_link_libraries(
    <target> 
    <PRIVATE|PUBLIC|INTERFACE> <item>... 
    [<PRIVATE|PUBLIC|INTERFACE> <item>...]...)
```

target：指定要加载动态库的文件的名字

该文件可能是一个源文件
该文件可能是一个动态库文件
该文件可能是一个可执行文件
PRIVATE|PUBLIC|INTERFACE：动态库的访问权限，默认为PUBLIC

如果各个动态库之间没有依赖关系，无需做任何设置，三者没有没有区别，一般无需指定，使用默认的 PUBLIC 即可。

动态库的链接具有传递性，如果动态库 A 链接了动态库B、C，动态库D链接了动态库A，此时动态库D相当于也链接了动态库B、C，并可以使用动态库B、C中定义的方法

```cmake
target_link_libraries(A B C)
target_link_libraries(D A)
```

动态库的链接和静态库是完全不同的：

静态库会在生成可执行程序的链接阶段被打包到可执行程序中，所以可执行程序启动，静态库就被加载到内存中了。
动态库在生成可执行程序的链接阶段不会被打包到可执行程序中，当可执行程序被启动并且调用了动态库中的函数的时候，动态库才会被加载到内存
因此，在cmake中指定要链接的动态库的时候，应该将命令写到生成了可执行文件之后：

```cmake
cmake_minimum_required(VERSION 3.0)
project(TEST)
file(GLOB SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
# 添加并指定最终生成的可执行程序名
add_executable(app ${SRC_LIST})
# 指定可执行程序要链接的动态库名字
target_link_libraries(app pthread)
```

在target_link_libraries(app pthread)中：

app: 对应的是最终生成的可执行程序的名字
pthread：这是可执行程序要加载的动态库，这个库是系统提供的线程库，全名为libpthread.so，在指定的时候一般会掐头（lib）去尾（.so）

在 link 自己的库时，也要注意加上 link_directories 指定动态库路径



### 日志

message，当成 print 来用

```cmake
message([STATUS|WARNING|AUTHOR_WARNING|FATAL_ERROR|SEND_ERROR] "message to display" ...)
```

message project name

```cmake
message(STATUS "**** PROJECT NAME ****" ${PROJECT_NAME})
```



### 变量操作

有时候项目中的源文件并不一定都在同一个目录中，但是这些源文件最终却需要一起进行编译来生成最终的可执行文件或者库文件。如果我们通过file命令对各个目录下的源文件进行搜索，最后还需要做一个字符串拼接的操作，关于字符串拼接可以使用set命令也可以使用list命令

1. set

   ```cmake
   set(变量名1 ${变量名1} ${变量名2} ...)
   ```

2. list

   list命令的功能比set要强大，字符串拼接只是它的其中一个功能，所以需要在它第一个参数的位置指定出我们要做的操作，APPEND表示进行数据追加，后边的参数和set就一样了

   ```cmake
   cmake_minimum_required(VERSION 3.0)
   project(TEST)
   set(TEMP "hello,world")
   file(GLOB SRC_1 ${PROJECT_SOURCE_DIR}/src1/*.cpp)
   file(GLOB SRC_2 ${PROJECT_SOURCE_DIR}/src2/*.cpp)
   # 追加(拼接)
   list(APPEND SRC_1 ${SRC_1} ${SRC_2} ${TEMP})
   message(STATUS "message: ${SRC_1}")
   ```

   在CMake中，使用set命令可以创建一个list。一个在list内部是一个由分号;分割的一组字符串。例如，set(var a b c d e)命令将会创建一个list:a;b;c;d;e，但是最终打印变量值的时候得到的是abcde	

   将字符串移除也可以用 list

   ```cmake
   list(REMOVE_ITEM <list> <value> [<value> ...])
   ```

   list 还有许多功能，基本上和 Python 的 list 一样，这里不赘述



在 cmake 中可以直接定义宏

```cmake
add_definitions(-D宏名称)
```



```cmake
cmake_minimum_required(VERSION 3.0)
project(TEST)
# 自定义 DEBUG 宏
add_definitions(-DDEBUG)
add_executable(app ./test.c)
```

这和下面的方法是等价的

```shell
gcc test.c -DDEBUG -o app
```



### 嵌套 CMakeList.txt

如果项目很大，或者项目中有很多的源码目录，在通过CMake管理项目的时候如果只使用一个CMakeLists.txt，那么这个文件相对会比较复杂，有一种化繁为简的方式就是给每个源码目录都添加一个CMakeLists.txt文件（头文件目录不需要），这样每个文件都不会太复杂，而且更灵活，更容易维护



```txt
$ tree
.
├── build
├── calc
│   ├── add.cpp
│   ├── CMakeLists.txt
│   ├── div.cpp
│   ├── mult.cpp
│   └── sub.cpp
├── CMakeLists.txt
├── include
│   ├── calc.h
│   └── sort.h
├── sort
│   ├── CMakeLists.txt
│   ├── insert.cpp
│   └── select.cpp
├── test1
│   ├── calc.cpp
│   └── CMakeLists.txt
└── test2
    ├── CMakeLists.txt
    └── sort.cpp
```

include 目录：头文件目录
calc 目录：目录中的四个源文件对应的加、减、乘、除算法
对应的头文件是include中的calc.h
sort 目录 ：目录中的两个源文件对应的是插入排序和选择排序算法
对应的头文件是include中的sort.h
test1 目录：测试目录，对加、减、乘、除算法进行测试
test2 目录：测试目录，对排序算法进行测试
可以看到各个源文件目录所需要的CMakeLists.txt文件现在已经添加完毕了。接下来庖丁解牛，我们依次分析一下各个文件中需要添加的内容

众所周知，Linux的目录是树状结构，所以嵌套的 CMake 也是一个树状结构，最顶层的 CMakeLists.txt 是根节点，其次都是子节点。因此，我们需要了解一些关于 CMakeLists.txt 文件变量作用域的一些信息：

根节点CMakeLists.txt中的变量全局有效
父节点CMakeLists.txt中的变量可以在子节点中使用
子节点CMakeLists.txt中的变量只能在当前节点中使用

#### 子目录

cmake 中父子节点的关系使用命令

```cmake
add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL])
```

source_dir：指定了CMakeLists.txt源文件和代码文件的位置，其实就是指定子目录
binary_dir：指定了输出文件的路径，一般不需要指定，忽略即可。
EXCLUDE_FROM_ALL：在子路径下的目标默认不会被包含到父路径的ALL目标里，并且也会被排除在IDE工程文件之外。用户必须显式构建在子路径下的目标



在目录中 test1 要完成计算相关的测试

test2 要完成排序相关的测试

所以对于 calc sort 可以把它们先编译成库（静态动态都行），然后再加入到根节点中



根目录的 cmake

```cmake
cmake_minimum_required(VERSION 3.0)
project(test)
# 定义变量
# 静态库生成的路径
set(LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)
# 测试程序生成的路径
set(EXEC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)
# 头文件目录
set(HEAD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include)
# 静态库的名字
set(CALC_LIB calc)
set(SORT_LIB sort)
# 可执行程序的名字
set(APP_NAME_1 test1)
set(APP_NAME_2 test2)
# 添加子目录
add_subdirectory(calc)
add_subdirectory(sort)
add_subdirectory(test1)
add_subdirectory(test2)
```

在根节点对应的文件中主要做了两件事情：定义全局变量和添加子目录。

定义的全局变量主要是给子节点使用，目的是为了提高子节点中的CMakeLists.txt文件的可读性和可维护性，避免冗余并降低出差的概率。
一共添加了四个子目录，每个子目录中都有一个CMakeLists.txt文件，这样它们的父子关系就被确定下来了。

add_subdirectory 应该就是告诉根目录，还有目录需要使用 cmake 编译

CMake构建工程会自动将该子目录添加到编译和链接的搜索目录中，以保证整个构建工程能满足依赖，这也是为什么使用 add_subdirectory 后不需要将子文件夹加入到头文件或库文件搜索目录也能搜索到子目录的头文件或库文件



子节点的 cmake

calc cmake

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALCLIB)
aux_source_directory(./ SRC)
include_directories(${HEAD_PATH})
set(LIBRARY_OUTPUT_PATH ${LIB_PATH})
add_library(${CALC_LIB} STATIC ${SRC})
```

第4行include_directories：包含头文件路径，HEAD_PATH是在根节点文件中定义的
第5行set：设置库的生成的路径，LIB_PATH是在根节点文件中定义的
第6行add_library：生成静态库，静态库名字CALC_LIB是在根节点文件中定义的



sort cmake 基本一样，就是替换了名字，并且生成的是动态库

```cmake
cmake_minimum_required(VERSION 3.0)
project(SORTLIB)
aux_source_directory(./ SRC)
include_directories(${HEAD_PATH})
set(LIBRARY_OUTPUT_PATH ${LIB_PATH})
add_library(${SORT_LIB} SHARED ${SRC})
```



test1 cmake 也和上面的 cmake 基本一样，就是替换了名字，使用 EXECUTABLE_OUTPUT_PATH 指定输出位置，并且最重要的一点：加入 link_libraries，这样才能使用库

```cmake
cmake_minimum_required(VERSION 3.0)
project(CALCTEST)
aux_source_directory(./ SRC)
include_directories(${HEAD_PATH})
link_libraries(${CALC_LIB})
set(EXECUTABLE_OUTPUT_PATH ${EXEC_PATH})
add_executable(${APP_NAME_1} ${SRC})
```

所以可以看到，在 C++ 中使用三方库最重要的两件事：

1. 声明。要让编译器知道某函数/类的存在，由 `include_directoires` 完成
2. 链接库。要让编译器能使用某函数/类，由 `link_libraries` 完成





test2 cmake 和 test1 cmake 基本一致，但是链接的是动态库！

```cmake
cmake_minimum_required(VERSION 3.0)
project(SORTTEST)
aux_source_directory(./ SRC)
include_directories(${HEAD_PATH})
set(EXECUTABLE_OUTPUT_PATH ${EXEC_PATH})
# link_directories(${LIB_PATH})
add_executable(${APP_NAME_2} ${SRC})
target_link_libraries(${APP_NAME_2} ${SORT_LIB})
```

> 在生成可执行程序的时候，动态库不会被打包到可执行程序内部。当可执行程序启动之后动态库也不会被加载到内存，只有可执行程序调用了动态库中的函数的时候，动态库才会被加载到内存中，且多个进程可以共用内存中的同一个动态库，所以动态库又叫共享库



完成过后直接 camke .. && build，一键构建！！

问题：仍然是头文件与 object file 的联系。当我们 include a function 过后，编译器如何寻找到对应的 object file 中的对应函数？这个过程又如何体现在 cmake file 当中？

并且如何将 头文件和具体的实现分离开来？

add_executable or add_library 本质上就是在用编译器生成 object

```shell
g++ -o main main.cpp -Iinclude ...
```



### 控制语句

cmake 有比较灵活的控制语句

if

for

while

比较重要的是条件的表示

### config.h.in

TODO

### install & test

TODO

