# C++ 拓展

## 动态内存

下面是使用 new 运算符来为任意的数据类型动态分配内存的通用语.法，通常配合指针使用：

```c++
typeName* pointer = new data-type;
typeName* pointer = new data-type [num_elements];
```

**malloc()** 函数在 C 语言中就出现了，在 C++ 中仍然存在，但建议尽量不要使用 malloc() 函数。new 与 malloc() 函数相比，其主要的优点是，new 不只是分配了内存，它还创建了对象

在任何时候，都可以使用 delete 操作符释放对象占用的内存，但必须要使用指向对象的指针

```c++
delete pointer;
delete [] array_pointer;
```

**`new` 和 `delete` 一定要配对使用，否则会导致程序发生内存泄露（memory leak）**，即分配出去的内存拿不回来，无法再使用

```c++
int *ps = new int;   // 声明指针并分配一个可以存 int 类型的内存给指针
...
delete ps;  // 归还内存
```

## 命名空间

**命名空间**这个概念，作为附加信息来区分不同库中相同名称的函数、类、变量等。本质上，命名空间就是定义了一个范围

命名空间的定义使用关键字 **namespace**，后跟命名空间的名称

```c++
namespace namespace_name {
	// codes
}
```

为了调用带有命名空间的函数或变量，需要在前面加上命名空间的名称

```c++
name::code;  // code is variable or function
```

可以使用 **using namespace** 指令，这样在使用命名空间时就可以不用在前面加上命名空间的名称，但不推荐。using 指令也可以用来指定命名空间中的特定项目

```c++
using namespace std;
using std::cout;
```

命名空间可以定义在几个不同的部分中，因此命名空间是由几个单独定义的部分组成的

命名空间也可以嵌套

### Operator :: and .

这里简要说明两个 oprator 的区别

1. `::` 为 scope resolution operator，用于 static method & static member 和 namespace 解析
2. `.` 为 dot operator，用于访问类成员的操作

## 模板

模板是泛型编程的基础，泛型编程即以一种独立于任何特定类型的方式编写代码

比如 **向量**，我们可以定义许多不同类型的向量，比如 `vector <int>` 或 `vector <string>`

可以使用模板来定义函数和类

模板函数定义的一般形式如下所示

```c++
template <typename type> ret-type func-name(parameter list)
{	
}
```

举个例子

```c++
#include <iostream>
#include <string>
 
using namespace std;
 
template <typename T> T Max (T a, T b) 
{ 
    return a < b ? b:a; 
} 
int main ()
{
 
    int i = 39;
    int j = 20;
    cout << "Max(i, j): " << Max(i, j) << endl; 
 
    double f1 = 13.5; 
    double f2 = 20.7; 
    cout << "Max(f1, f2): " << Max(f1, f2) << endl; 
 
    string s1 = "Hello"; 
    string s2 = "World"; 
    cout << "Max(s1, s2): " << Max(s1, s2) << endl; 
 
    return 0;
}
```

正如我们定义函数模板一样，我们也可以定义类模板

```c++
template <class type> class class-name {
}
```

在这里，**type** 是占位符类型名称，可以在类被实例化的时候进行指定。-可以使用一个逗号分隔的列表来定义多个泛型数据类型

## 头文件

通常为了使得代码模块化，我们会将代码写到多个 cpp 文件里面。当我们想要使用其中的函数或者类时，可以选择进行声明

```txt
- Learn_C++
	- log.cpp
	- main.cpp
```

其中两个 cpp 文件的代码如下

```c++
// log.cpp
#include <iostream>

void log(const char* message)
{
    std::cout << message << std::endl;
}

// main.cpp
#include <iostream>
//  declaration
void log(const char* message);

int main()
{
    log("hello");
    return 0;
}
```

但是如果要引用的函数很多的话，每一次都要声明就非常麻烦，这个时候就可以使用头文件来替我们完成这些声明。创建一个 `log.h` 的头文件

```c++
// log.h
#pragma once

void log(const char* message);
```

其中 `#pragma once` 代表一下内容只会被引入一次，这样就不会重复导入。从此就可以用头文件来完成

```c++
#include "log.h"
#include <iostream>

int main()
{
    log("hello");
    return 0;
}
```

其中你还能注意到，有的时候 `#include` 使用的是 `<>` 但有的时候使用 `""`，这两种形式分别代表相对路径和绝对路径，通常 C++ 自带的标准库可以用 `<>` 表示，而自己项目里的库可以用 `""`，例如使用上一个文件夹中的 cpp 可以用 `"../xxx.cpp"` 来表示

实际上 `"iostream"` 也是能够成功编译的，但为了可读性我们仍然进行区分

在一些项目中我也看到了一些不按照相对路径来进行引用的头文件，例如如下结构

```txt
- include
	- utils.h
- main.cpp
```

我发现代码直接在 `main.cpp` 中使用 `#include "utils.h"` 而没有使用 `#include "include/utils.h"`。这样直接编译 `main.cpp` 是不行的，需要在执行编译时使用 `-I` 命令将其加入到 `IncludePath` 中

```shell
g++ -I./include main.cpp -o out
```

还可以通过 `-H` 参数来列出包含的头文件 `g++ -H main.cpp`

## 字符串

C-style 字符串是有终止符的，这使得在申请字符串空间时要多一个字符

在使用输入流读取字符串有两种方法

```c++
cin.getline(var, size)
cin.get(var)
```

二者遇到回车终止读取

TODO：字符串的规则比较多，后续如果有需要，清晰整理
