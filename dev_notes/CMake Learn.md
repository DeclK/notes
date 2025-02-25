# CMake Learn

这是第二次进攻 CMake 学习了🤣第一次在学习 makefile & cmake 的时候感觉啥也没学进去 (2023/10)，看着网上零零碎碎的文档以及博客，收效甚微...现在已经是 5205 年了！有了更好的工具和资源，这次不得拿下？😎

我先通看的 [bilibili-CMake](https://www.bilibili.com/video/BV1Mw411M761)，这个讲解我认为非常通俗易懂，学习曲线很低，让我对 CMake 有一个整体的清晰认知。我先根据视频整理整个 CMake 的知识框架/基本概念，然后再看下 [modern cmake](https://modern-cmake-cn.github.io/Modern-CMake-zh_CN/README_GitBook.html) 作为补充

## Chapter1-编写 CMake 的底层逻辑

- 底层逻辑 ep1 03:52

  这是教程最核心的观点：**一个厨师做菜，抛开味道不谈，他至少需要知道需要什么原材料以及这些原材料在哪**

  个人理解：菜对应的就是可执行程序，原材料就是运行程序需要的文件和库

- Cpp/C语言 为什么会有头文件？

  教程给了一个常用习惯：**当存在多个 cpp/c 文件时，需要为每一个 cpp/c 文件都配置一个头文件**。这是为什么？教程没有马上给出答案，而是直接进入了后面的教学。

  我之前在学头文件的时候也是云里雾里的。这里我只说一个我理解的 `include` 的作用（最表层）：把文件 `<file.h>` 里的内容原封不动地插入到当前 `#include <file>` 所在位置

- A hello world cmake 05:00

  ```cmake
  cmake_minimum_required(VERSION 3.15)	# minimum version required
  project(cmake_study)		# name for the project
  add_subdirectory(lesson1_1)	# sub project directory
  ```

  `add_subdirectory` 的作用是 **CMake 中用于管理模块化项目结构的重要命令**，通常形式为

  ```cmake
  add_subdirectory(source_dir [binary_dir] [EXCLUDE_FROM_ALL])
  ```

  - **source_dir**：子目录路径（相对或绝对），必须包含 `CMakeLists.txt`
  - **binary_dir**（可选）：指定子项目的构建输出目录，默认使用 `source_dir` 的相对路径
  - **EXCLUDE_FROM_ALL**（可选）：若设置，子目录的默认目标（如 `all`）不会自动构建，需显式依赖

  其作用如下：

  1. **引入子项目**：将指定子目录的 `CMakeLists.txt` 纳入当前项目的构建系统，实现多目录项目的统一管理。后面两点会将子项目和子目录进行混淆，二者是同一个东西
  2. **作用域隔离**：子目录中的变量、目标（如库或可执行文件）通常仅在自身作用域内有效，但可通过 `PUBLIC/INTERFACE` 等关键字暴露给父项目。
  3. **构建顺序控制**：父目录的构建会在子目录之后，确保依赖关系正确。

  DS 列出了典型的使用场景方便我们理解

  ```txt
  根项目/
  ├── CMakeLists.txt
  ├── src/           # 主程序代码
  ├── include/       # 主程序头文件
  └── libs/
      ├── math/      # 数学库模块
      │   ├── CMakeLists.txt
      │   ├── src/
      │   └── include/
      └── utils/     # 工具库模块
          ├── CMakeLists.txt
          ├── src/
          └── include/
  ```

  根项目 `CMakeLists.txt`

  ```cmake
  cmake_minimum_required(VERSION 3.10)
  project(MyProject)
  add_subdirectory(libs/math)   # 引入数学库模块
  # add_subdirectory(libs/math build/math)  # 将编译输出定向到 build/math
  add_subdirectory(libs/utils)  # 引入工具库模块
  # add_subdirectory(tests EXCLUDE_FROM_ALL)  # 测试代码仅在明确构建时编译
  
  add_executable(main src/main.cpp)
  target_link_libraries(main PRIVATE math utils)  # 链接子模块的库
  
  ```

  子模块 `libs/math/CMakeLists.txt`：

  ```cmake
  add_library(math STATIC src/math.cpp)  # 生成静态库
  target_include_directories(math PUBLIC include)  # 暴露头文件路径给父目录
  ```

  个人理解：当我们在写 `add_subdirectory` 的时候，就会将立即这个子目录中的 cmake lists 导入，并且会对其中的 target 进行编译。但这些编译完成的 sub target 还没有被链接到当前项目中的 target 中，所以想要使用的话，还得用 `target_link_libraries` 进行链接。

  并且由于我们在子目录中使用了 `target_include_directories`，并用 `PUBLIC` 进行了标识，该操作使得这个头文件搜索路径不仅对 `math` 库本身可见，还会传递给任何依赖 `math` 库的目标，所以在父目录的 `main.cpp` 中我们可以直接使用子目录中的头文件，而不需要写完整路径

  ```c++
  #include "math.h"  // 无需写完整路径（如 `libs/math/include/math.h`）
  ```

  这是因为 cmake 会自动生成类似如下的编译命令将头文件目录包含

  ```shell
  g++ -I<root_dir>/libs/math/include -o main.o -c src/main.cpp
  ```

  头文件的搜索路径：

  1. 使用头文件的 cpp 文件的当前文件夹
  2. 系统 include 路径 `/usr/include & /usr/local/include`
  3. 通过 `-I` 选项指定的目录

  **编译器查找头文件的顺序取决于 #include 的形式：**

  - 对于 #include "header.h"（引号形式）

    1. 源文件所在目录。
    2. 通过 -I 指定的目录（按命令行中指定的顺序）。
    3. 系统 include 路径。

  - 对于 #include <header.h>（尖括号形式）

    1. 通过 -I 指定的目录（按命令行中指定的顺序）。
    2. 系统 include 路径。

    - **注意**：不包括源文件所在目录。

- 浅析 include 作用

  写程序完全可以不需要头文件。因为头文件的意义仅仅在于 copy & paste，把你在头文件里写的东西粘贴到当前文件当中。如果只有声明没有实现会发生什么？能够通过编译（狭义），但是不能通过 linking，因为 linking 会去寻找函数实现，找不到就会报错。问题就来了：linking 是如何寻找实现的？如果有两个 library 同时实现了某个函数，那么 linking 是如何确定使用哪一个函数的？（through CMakeLists🤔）

- 相同头文件问题

  解决方案，在 include directory 增加一层目录来增加标识，我在 tvm & flashinfer 项目里都观察到了这样的现象

- 相同 object 问题

  static library 会发生这样的问题，但是 shared library 不会，namespace 能够完全解决这样的问题。注意需要在头文件和实现文件中同时声明命名空间才行

- 点题：厨师做菜 12:00

- 回答最初的问题：为什么需要头文件 18:37

  **减少重复，减轻依赖**

  19:00 每一个需要使用该函数的文件，都需要再声明一次，而且当函数进行修改过后，每一个文件的声明都可能需要进行改动

  头文件只是减少了重复，减轻依赖并没有做到

- 头文件是否需要加入到 CMakeLists 当中？21:10

  编译单元只编译 cpp/c 文件，头文件是不进行编译的

- 另一个项目想要使用该项目中的函数 23:00

  教材制造了一个 include 的 linking error，通过修改代码 `#include "../lesson1_1/add.h"` 修复

  但是这个加入方式很难受

  有没有什么方法让 cmake 帮助你找到呢？有的兄弟，有的。教程使用了 `include_directories` 帮助你找到头文件位置，但是教程并不推荐使用这个方法，他认为 modern cmake 中 `include_directories` 应该永远不会用到，可能违背了减少依赖原则。这我并不是特别理解