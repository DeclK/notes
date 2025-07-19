# CMake Learn

这是第二次进攻 CMake 学习了🤣第一次在学习 makefile & cmake 的时候感觉啥也没学进去 (2023/10)，看着网上零零碎碎的文档以及博客，收效甚微...现在已经是 5205 年了！有了更好的工具和资源，这次不得拿下？😎

我先通看的 [bilibili-CMake](https://www.bilibili.com/video/BV1Mw411M761)，这个讲解我认为非常通俗易懂，学习曲线很低，让我对 CMake 有一个整体的清晰认知。我先根据视频整理整个 CMake 的知识框架/基本概念，然后再看下 [modern cmake](https://modern-cmake-cn.github.io/Modern-CMake-zh_CN/README_GitBook.html) 作为补充

## Chapter1-编写 CMake 的底层逻辑

- 底层逻辑 ep1 03:52

  这是教程最核心的观点：**一个厨师做菜，抛开味道不谈，他至少需要知道需要什么原材料以及这些原材料在哪**

  个人理解：菜对应的就是完整的可执行程序，原材料就是运行程序需要的文件和库

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

  DeepSeek 列出了典型的使用场景方便我们理解

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

  另外对于 `PUBLIC & PRIVATE` 我认为可以用两种常用的使用方式来理解：

  1. `PUBLIC` 说明：这些头文件其实是想给别人用的（可能自己不用），别人在用的时候只需要 `target_link_libraries` 就够了，头文件会自动加入到 build 过程当中
  2. `PRIVATE` 说明：这些头文件其实是给自己用的（别人不用）

  总结：`PUBLIC` 为把自己的头文件暴露（API 暴露），`PRIVATE` 为自己所需要使用的头文件

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

  linking 的具体过程会在之后进行总结，对于上面的问题答案是：会按照 `target_link_libraries` 的顺序决定使用哪一个

  文件结构如下

  ```txt
  .
  ├── lesson1_1
  │   ├── include
  │   │   └── add.h
  │   ├── CMakeLists.txt
  │   └── add.cpp
  ├── lesson1_2
  │   ├── include
  │   │   └── add.h
  │   ├── CMakeLists.txt
  │   └── add.cpp
  ├── CMakeLists.txt
  ├── main.cpp
  └── utils.py
  ```

  在 `lesson1_1 & lesson1_2` 都实现了一个 `add.h`，然后用如下 cmake 代码添加到项目当中

  ```cmake
  cmake_minimum_required(VERSION 3.15)
  project(cmake_study)
  add_subdirectory(lesson1_2)	# target is add_2
  add_subdirectory(lesson1_1) # target is add_1
  
  add_executable(main main.cpp)
  target_link_libraries(main PRIVATE add_2 add_1)
  ```

  此时会使用 `add_2` 中的 `add.h`，即谁先被 link 到项目当中，就会使用谁的头文件和 target

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

  有没有什么方法让 cmake 帮助你找到呢？有的兄弟，有的。教程使用了 `include_directories` 帮助你找到头文件位置，但是教程并不推荐使用这个方法，他认为 modern cmake 中 `include_directories` 应该永远不会用到，可能违背了减少依赖原则。这我并不是特

## Chapter2-静态库与动态库

- 静态库与动态库的定义

  所谓“库”（library），对于编程的意义是：方便代码的分享。例如当别人写好了一些功能代码，而你需要使用时，别人可以将其源码编译为“库”分享给你，这样你就能直接使用这些功能，而不需要接触到别人所写的源码

  - 静态库：在编译时，库的代码被直接复制到最终的可执行文件中
  - 动态库：在运行时，库的代码由操作系统加载到内存，供程序调用

  这里库的代码不是源码 (like cpp files)，而是编译过后的二进制文件 (like .so or .a)。他们包含了机器码和符号表等信息：机器码包含二进制指令集，而符号表包含函数和变量列表。一个直观的理解是：我们在使用（动态）库中的函数时，就是通过函数的声明找到对应的符号表，通过符号表去找到对应的二进制指令，然后实现功能运行

  总之，不管是静态库还是动态库，他们两者都是库，我们在 cmake 使用他们的方式其实是一样的

  ```cmake
  target_link_directoires(main PRIVATE lib_include_path)	# this could be relative path
  target_link_libraries(main PRIVATE lib_path)	# this must be absolute path
  ```

  我尝试了在 `target_link_libraries` 中使用相对路径，永远都会报错，即时通过编译了也会在运行中报错找不到 lib 文件。虽然说 **`target_link_libraries` 一般会去 `build` 文件夹下找**，但是即使找到了通过编译，`ld.so` 也找不到对应的 lib，所以仍然报错，只有写成绝对路径才能二者都找到。另外 **`target_link_directoires` 一般会去当前 `CMakeList.txt` 所在目录路径下去寻找**（如果你写的相对路径的话），而不会去 `build` 文件夹下去找

  一个实例的 cmake 文件如下

  ```cmake
  cmake_minimum_required(VERSION 3.15)
  project(cmake_study)
  
  add_executable(main main.cpp)
  # include the directories
  target_include_directories(main PRIVATE lesson1_1/include)
  # link the library from lesson1_1
  target_link_libraries(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/build/libadd_1.a)
  ```

  以上讨论都是基于如下情形：别人只给你提供了头文件和库，没有给源码。如果你的库已经有源码了，那么则参考上面的 `math` 子项目构建方式即可：由于 `add_subdirectory` 会执行子项目中的 cmake，而子项目中的 cmake 文件使用了 `add_library` 命令，所以我们只需要使用对应 library 的名字就行了，cmake will take care of it

  > FROM DeepSeek
  >
  > **`add_library(math ...)` 的核心功能是创建一个名为 `math` 的 CMake 目标（Target）。这个目标不仅仅代表最终的库文件，它是一个包含完整构建信息的对象：**
  >
  > - 源文件列表（`math.cpp`, `math.h`）。
  > - 编译选项（`-Wall`, `-O2` 等）。
  > - 包含目录（`target_include_directories(math PUBLIC include)`）。
  > - 链接依赖（`target_link_libraries(math PUBLIC some_other_lib)`）。
  > - **最重要的是：它知道最终生成的库文件的名称（`libmath.a`, `math.lib`, `libmath.so` 等）和路径（在构建目录下的相应位置，比如 `build/libs/math/`）。**
  >
  > **`add_library()` 创建的目标（Target）**:
  >
  > - 这个 `math` 目标成为了 CMake 构建系统中的一个**一等公民**。
  > - 它不是简单的文件名，而是一个**代表构建任务和产物**的实体，携带了所有构建和链接它所需的信息。

- 如何制作一个静态库 04:15

  制作静态库的方式其实在上面 `math` lib cmake 中已经有看到，其实非常简单

  ```cmake
  add_library(<target_name> 
      [STATIC | SHARED] 
      [EXCLUDE_FROM_ALL]
      source1 [source2 ...]
  )
  ```

  其实和 `add_executable` 非常类似，之前没整理，这里我也贴上

  ```cmake
  add_executable(<target_name> 
      [EXCLUDE_FROM_ALL]
      source1 [source2 ...]
  )
  ```

  就少了一个是否为 shared or static flag 标识。二者也都有 `EXCLUDE_FROM_ALL` flag，再次强调其作用：标识这些文件不会被默认 build 方式所构建 `cmake .. && make`，必须要显式构建 `cmake .. --target xxx && make`，但该命令也会导致其他默认构建的 target 不会被构建。所以 `EXCLUDE_FROM_ALL` 通常配合 `add_dependencies`，将 target 作为 target A 的依赖项，只有当 target A 被构建时，该 target 才会被同时构建

  教程一直在强调：编译的最小单元是 cpp，头文件不是“原材料”

  8:30 我们在交付库给别人的时候，最好写一个头文件用于说明库的接口。虽然这个头文件并不是编译必须的，但是除非别人知道这个库的接口，否则他将无从下手

  14:30 `link_libraries` 已经不推荐使用了，请使用 `target_link_libraries`。同样的我们也会看到 `link_directoires` 和 `target_link_directoires`，**仍然只推荐使用带 `target` 的命令**。原因都是一样的：不带 `target` 的命令是全局影响，会给所有的 target 都添加库/目录，这就会导致命名冲突以及以来混乱问题

  他们的使用方式都是在 `add_executable` or `add_library` 之后

  ```cmake
  add_executable(app main.cpp)
  target_link_libraries(app PRIVATE OpenSSL::SSL)
  
  add_executable(app main.cpp)
  target_include_directories(app PRIVATE include)
  ```

  不过有一种情况是推荐使用 `include_directories` 的：当所构建的项目是一个 header-only library 的时候，其没有构建任何的 target，此时直接使用 `include_directories` 是非常方便的！你可以看到在使用 cutlass or flashinfer 这样的 header-only library 的时候，大家都是直接使用 `include_directories`

  > From DeepSeek
  >
  > **集成方式就是包含路径：** 使用 Header-only 库的唯一要求就是让编译器在搜索路径中找到它们的头文件。这正是 `include_directories()` 所做的。在 sglang 项目中，include_directories() 是在顶层调用的，影响的是之后定义的所有目标（如 common_ops, flash_ops）。这通常是可接受的，因为该项目的主要目标都需要这些头文件路径。如果项目结构更复杂，有部分目标不需要这些路径，才更需要避免全局的

  21:00 讲解了 windows 中编译和使用动态库的约束，这个约束在 linux 中没有。这里我不太关注，不深入整理

- 总结

  那么构建一个 C++ 项目到底需要什么呢？我总结需要三个核心要素：

  1. target 源文件
  2. target 源文件所使用的头文件位置，使用 `target_include_directories` 完成
  3. 所使用的头文件包含的声明可能包含外部库，我们需要把这些库链接到 target 当中，使用 `target_link_libraries` 完成

  关注这三个东西就能够顺利构建出 target or executable 了😀

  以下是 DeepSeek 所总结的 Modern Cmake 的最佳实践示例

  ```cmake
  cmake_minimum_required(VERSION 3.15)
  project(MyProject)
  
  # 1. 定义可执行文件目标（源文件）
  add_executable(MyApp main.cpp)
  
  # 2. 添加私有头文件路径
  target_include_directories(MyApp PRIVATE src)
  
  # 3. 链接公共库（自动传递头文件+链接库）
  find_package(Boost 1.70 REQUIRED COMPONENTS filesystem)
  target_link_libraries(MyApp PRIVATE Boost::filesystem)
  
  # 4. 设置C++标准+编译选项
  target_compile_features(MyApp PRIVATE cxx_std_17)
  target_compile_options(MyApp PRIVATE -Wall -Wextra)
  
  # 5. 根据构建类型调整选项
  target_compile_definitions(MyApp PRIVATE 
      $<$<CONFIG:Debug>:_DEBUG>
  )
  ```

  基于我上面的理解 DeepSeek 再给出了3个重要维度：

  1. 编译选项（比如C++标准版本、警告级别、优化标志）会直接影响二进制结果
  2. 依赖管理（特别是现代CMake的target-based依赖传递机制）
  3. 构建类型（Debug/Release等不同配置）

## Chapter3-CMake 大一统

实际上这一节课是在讲解 c++ 的编译是怎么做的，是正确理解 cmake 的关键

- 编译四流程 03:20

  C++ 编译一般要经过四个流程：

  1. 预处理

     用预处理器展开头文件，宏替换，去掉注释

     ```shell
     g++ -E xxx.cpp -o xxx.i
     ```

  2. 编译

     使用编译器编译文件，生成汇编代码（人类可读、机器仍不可读）

     ```shell
     g++ -S xxx.i -o xxx.s
     ```

  3. 汇编

     使用汇编器生成机器码（二进制文件，人类不可读、机器可读）

     ```shell
     g++ -c xxx.s -o xxx.o
     ```

  4. 链接

     调用链接器对程序需要调用的库进行链接

     ```g++
     g++ xxx.o yyy.o -o zzz
     ```

     11:20 如果你的程序只有声明没有实现，那么在前三步都不会报错的。但是在链接这里会报错，找不到文件

> 为什么需要有汇编代码的中间存在？不能够一次生成机器码吗？
>
> From DeepSeek
>
> | 工具   | 输入        | 输出 | 职责                 |
> | :----- | :---------- | :--- | :------------------- |
> | 编译器 | `.cpp`/`.i` | `.s` | 生成平台无关汇编逻辑 |
> | 汇编器 | `.s`        | `.o` | 翻译为平台相关机器码 |
>
> **解耦编译器与汇编器**：
>
> - **编译器前端**（语法分析、语义检查、中间代码生成）只需关注**语言逻辑**。
> - **编译器后端**（代码优化、目标代码生成）负责输出**与CPU架构无关的中间表示**（如LLVM IR）。
> - **汇编器** 则专注于将**通用汇编指令**翻译成**特定CPU的机器码**（x86, ARM, RISC-V等）。
>
> **优势**：更换CPU架构时，只需替换汇编器或后端，无需重写整个编译器。
>
> 🌟 **本质**：汇编代码是**高级语言与机器码之间的“双向翻译层”**，它牺牲了少量编译时间（可忽略），换取了**灵活性、可维护性和透明度**

也就是说汇编语言其实仍然是一个 IR，lower 到最后的机器码还需要汇编器完成

- 利用 cmake 在 linux 和 windows 中完成项目构建 26:00

  在 Linux 下 cmake -> makefile -> executable

  在 windows 下 cmake -> sln -> executable

  这凸显了 cmake 的跨平台特性：你只需要熟悉 cmake 语法，就能在不同平台上构建项目

- Chapter4 26:00 使用 `ldd` 命令可以查看 target 所链接的库

- Chapter4 教程推荐使用 `-DCMAKE_VERBOSE_MAKEFILE=ON` 来查看编译和 linking 的详细过程，很容易就看出问题

- linux 中动态库的搜索路径顺序

  1. `LD_LIBRARY_PATH` 环境变量中定义的目录
  2. 系统默认共享库目录 `/lib`
  3. 系统库目录 `/usr/lib`
  4. `RPATH / RUNPATH` 标志指定的目录

- Chapter5 18:00 教程分享了一个让人感到疑惑的情况

  ```cmake
  add_library(add STATIC add.cpp)
  target_include_directories(add PRIVATE ../common)
  target_link_libraries(add PRIVATE common)
  ```

  他在创建一个静态库 `add`，同时这个静态库使用到了 `common ` 静态库，然后将 `common` 的头文件和 lib 都放进来，最后发现：`add` 库没有链接上 `common` 库

  26:00 教程通过 verbose 编译发现，我们在做 `target_link_libraries(add PRIVATE common)` 时，没有办法将一个 `.a` 文件链接到另外一个 `.a` 文件上。这些 `.a` 文件是在最后生成 `.o` 文件时统一链接进去了。具体的依赖情况是这样：

  ```txt
  common.a -> add.a --> main.cpp
  ```

  我们在最后生成 `main` 的 executable 的时候会把 `common.a` 给直接链接进来，即使我们在给 `main.cpp` 写 cmake 文件的时候只链接了 `add` 库

  表层原因：`.a` 文件接收 `.o` 文件的链接。当我们使用 `target_link_libraries` 来将一个 `.a` link 到另一个 `.a` 时，其中的机器码和符号表是不会被复制过去的。此时该命令所起到的其实是一个标记作用：如果有谁使用到了 `add.a`，那么一定要把 `common.a` 给传递过来

  ```shell
  g++ -g main.cpp.o -o output add.a common.a
  ```

  33:00 这样做有一个好处：无法从 `main.cpp` 使用 `common` 当中的头文件。即：保护了 `main.cpp` 不受 `common` 头文件的侵入。但是由于 `common.a` 当中是有所有的机器码和符号表的，如果你手动写一个声明，而这个声明也确实能够在符号表中找得到，那么仍然可以执行其中的机器码。**总结：只隐藏了头文件，没有隐藏符号表和机器码**

## Chapter4-Find Package

实际上是教程的 chapter6 了，这里我直接询问 DeepSeek

在 CMake 中，`find_package()` 是一个关键命令，用于**查找和加载外部库（第三方依赖）的设置**。它简化了在项目中集成外部库（如 OpenCV、Boost、Qt 等）的过程，自动处理头文件路径、库文件路径、编译选项等依赖关系

**核心作用**

1. **自动定位库**
   搜索系统或指定路径中的库文件（`.lib`/`.a`）和头文件（`.h`/`.hpp`）。
2. **导入预定义变量**
   设置变量（如 `OpenCV_INCLUDE_DIRS`, `Boost_LIBRARIES`）供后续使用。
3. **导入目标（现代用法）**
   创建 CMake 目标（如 `OpenCV::core`），可直接链接到你的目标。

**基本语法**

```cmake
find_package(<PackageName> 
    [version]      # 如 3.2.1 或 >=2.0
    [EXACT]        # 要求精确匹配版本
    [QUIET]        # 静默模式，不输出错误
    [REQUIRED]     # 必需，找不到则报错终止
    [COMPONENTS components...] # 指定子组件
)
```

一个示例

```cmake
find_package(Boost 1.75 REQUIRED COMPONENTS system filesystem)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE Boost::system Boost::filesystem)
# 头文件路径等自动传递，不需要 include_directories
```

**找不到库？**

1. 设置 `<PackageName>_DIR` 指向包含 `Config.cmake` 的目录

   ```cmake
   set(OpenCV_DIR "/opt/opencv/lib/cmake/opencv4") # 手动指定路径
   find_package(OpenCV REQUIRED)
   ```

2. 通过 CMake 命令传递

   ```bash
   cmake -D OpenCV_DIR=/path/to/opencv_config_dir ..
   ```

在上述例子中还出现了 `::` 符号，也是 cmake 当中的命名空间缪包的标识符，通常与 `find_package` 配合使用

1. **命名空间目标**
   `PackageName::Component` 表示由外部库通过 `find_package()` 导出的预定义目标（库或模块），例如：
   - `OpenCV::core`
   - `Boost::system`
   - `Qt5::Core`
2. **自动依赖传递**
   这些目标**自带头文件路径、链接库、编译选项等元数据**，链接时会自动传递依赖关系。
3. **避免命名冲突**
   `::` 提供了命名空间隔离（如 `zlib::zlib` 和 `Boost::zlib` 可共存）。

## TODO

未来需要以实际的项目作为学习材料，进一步整理现代 CMake 用法的最佳实践（优先考虑 SGLang？🤔）