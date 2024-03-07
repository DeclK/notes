# CUDA Programming 2

## CUDA 程序基本框架

一个典型的 CUDA 程序基本内框架如下

```txt
1 头文件包含
2 常量定义（或者宏定义）
3 C++ 自定义函数和 CUDA 核函数的声明（原型）
4 int main(void)
5 {
6 分配主机与设备内存
7 初始化主机中的数据
8 将某些数据从主机复制到设备
9 调用核函数在设备中进行计算
10 将某些数据从设备复制到主机
11 释放主机与设备内存
12 }
13 C++ 自定义函数和 CUDA 核函数的定义（实现）
```

作者用最简单的 VectorAdd 作为例子，说明了这个过程。程序就不在此写出，但一些注意点如下：

1. 当 CUDA 程序开始运行时，你并不需要显式地调用一个函数来初始化 GPU 设备。CUDA 运行时会在后台自动处理这一过程。这意味着，当你第一次调用一个非设备管理（如 cudaSetDevice）或非版本查询（如 cudaGetDeviceProperties）的运行时 API 函数时，CUDA 运行时会检查是否已经有设备被初始化了。如果没有，它将自动选择一个 GPU 并对其进行初始化，让它准备好执行后续的 CUDA 核函数等

   > From GPT

2. CUDA runtime API 全部都以 cuda 开头，而且都有一个类型为 cudaError_t 的返回值，代表了一种错误信息例如 `cudaMalloc`，该函数的原型如下

   ```c++
   cudaError_t cudaMalloc(void **address, size_t size);
   ```

   其中：

   - 第一个参数 address 是待分配设备内存的指针。注意这里是双重指针，用于修改指针的值
   - 返回值是一个错误代号。如果调用成功，返回 cudaSuccess，否则返回一个代表某种 错误的代号（下一章会进一步讨论）

3. 为了区分主机和设备中的变量，我们（遵循 CUDA 编程的传统）用 d_ 作为所有设备变量的前缀， 而用 h_ 作为对应主机变量的前缀

4. **核函数要求**

   - **可以向核函数传递非指针变量**（如 int N），其内容对每个线程可见（这个变量将存储在哪儿？）
   - 除非使用统一内存编程机制（将在第 13 章介绍），否则**传给核函数的数组（指针）必须指向设备内存**
   - **核函数不可成为一个类的成员**。通常的做法是用一个包装函数调用核函数，而将包装函数定义为类的成员

5. **设备函数要求**

   - 必须在设备中调用，在设备中执行
   - 可以用 __host__ 和 __device__ 同时修饰一个函数，使得该函数既是一个 C++ 中的 普通函数，又是一个设备函数。这样做可以减少冗余代码。编译器将针对主机和设备 分别编译该函数
   - 不能同时用 __device__ 和 __global__ 修饰一个函数，也不能同时用 __host__ 和 __global__ 修饰一个函数
   - 编译器决定把设备函数当作内联函数（inline function）或非内联函数，但可以用修饰 符 __noinline__ 建议一个设备函数为非内联函数（编译器不一定接受）

## CUDA 程序的错误检测

有的错误在编译期间没有被发现，但在运行的时候出现，称为运行时刻的错误。一般来说，运行时刻的错误更难排错

一般来说，只有返回值为 cudaSuccess 时才代表成功地调用了 CUDA rumtime API 函数，根据这个返回值，作者写了一个宏函数 `CHECK` 来帮助 debug

```c++
#pragma once
#include <stdio.h>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)
// __LINE__ is a preprocessor macro that expands to the line number as a decimal constant
// __FILE__ is a preprocessor macro that expands to the name of the current input file, in the form of a C string constant
```

上方代码的解释如下：

1. 在定义宏时，如果一行写不下，需要在行末写 \，表示续行
2. cudaGetErrorString() 显然也是一个 CUDA 运行时 API 函数，作用是将错误代号转化为错误的文字描述
3. 读者可能会问，宏函数的定义中为什么用了一个 do-while 语句？不用该语句在大部分情况下也是可以的，但在某些情况下不安全（这里不对此展开讨论，感兴趣的读者可自行研究）

作者在之后的代码中，将坚持用这个宏函数包装大部分的 CUDA runtime API 函数

虽然 CUDA runtime API 会返回值，但是核函数是不返回值的（void）,有一个方法可以捕捉调用核函数可能发生的错误，即在调用核函数之后加上如下两个语句

```c++
// kernel function
CHECK(cudaGetLastError());
CHECK(cudaDeviceSynchronize());
```

第一个语句的作用是捕捉第二个语句之前的最后一个错误，第二个语句的作用是同步主机与设备。之所以要同步主机与设备，是因为**核函数的调用是异步的，即主机发出调用核函数的命令后会立即执行后面的语句，不会等待核函数执行完毕**

整个异步代码的执行过程如下：

1. host 发起（launch）一个核函数 `func<<<grid, blcok>>>`
2. 在发起核函数过后，会立即运行 `CHECK(cudaGetLastError());`，而不会等待核函数执行（execute）完毕。如果在发起过程中有任何错误产生，都会被检测。但是该 CHECK 无法检测核函数中**执行**所产生的错误
3. **执行完成** `cudaGetLastError` 之后，会立即运行 `CHECK(cudaDeviceSynchronize());`，这将会等待核函数执行完成，任何发生在核函数执行过程中的错误都将被检测

所以说：

1. 两个 CHECK，一个检查发起错误，一个检查运行错误
2. **核函数是异步运行的，但是 CUDA runtime API 不是异步运行的**。即 cudaDeviceSynchronize 一定是在 cudaGetLastError 之后发起的

CUDA 提供了名为 CUDA-MEMCHECK 的工具集，具体包括 memcheck、 racecheck、initcheck、 synccheck 共 4 个工具。它们可由可执行文件 cuda-memcheck 调用，这里不做整理



## Question

1. 可以向核函数传递非指针变量（如 int N），其内容对每个线程可见，那这个变量将存储在哪儿？
