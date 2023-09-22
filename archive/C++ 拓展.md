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

可以使用 **using** 指令，`using` 通常用于将其他的命名空间导入到当前的空间中。这样在使用命名空间时就可以不用在前面加上命名空间的名称。using 指令也可以用来指定命名空间中的特定项目

```c++
using namespace std;
using std::cout;

// using can also used for class alias
using test Add	// the same with: typedef Add test
```

命名空间还有3个特点：

1. 命名空间可以定义在几个不同的部分中
2. 命名空间可以嵌套
3. 命名空间可以位于等式左侧 `namespace a = std`

### Operator :: and .

两个操作符中，`::` 的功能要丰富得多！`.` 仅用于访问成员的操作

`::` 为 scope resolution operator，有 6 种使用方式

1. namespace 解析

   最常见使用场景

2. 在 class 之外定义函数 & static variable

   定义不区分公有私有，可通过函数 `()` 后是否为 `;` 来判定函数受否被定义

3. 访问 class 中的 static variable & function

   这里区分公有私有，只能访问 public static variable & function

4. 用于多重继承

   即一个类继承了多个类，而多个类中存在相同名称的变量或者函数

5. 引用嵌套类

   ```c++
   #include <iostream>
   using namespace std;
    
   class outside {
   public:
       int x;
       class inside {
       public:
           int x;
           static int y;
           int foo();
       };
   };
   int outside::inside::y = 5;
    
   int main()
   {
       outside A;
       outside::inside B;
   }
   ```

6. 访问全局变量

   ```c++
   #include<iostream>
   using namespace std;
    
   int x; // Global x
    
   int main()
   {
   int x = 10; // Local x
   cout << "Value of global x is " << ::x;
   cout << "\nValue of local x is " << x;
   return 0;
   }
   ```

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

我们也可以定义类模板

```c++
template <typename type>
class name_of_class {
}
```

在这里，**type** 是占位符类型名称，可以在类被实例化的时候进行指定。**还可以使用一个逗号分隔的列表来定义多个泛型数据类型**

```c++
tempate <typename T1, typename T2,...>
```

实际上你可看待使用 `class` 来替代 `typename` 的情况

```c++
template<class T>
    void func()
```

二者其实是相同的，不过 `typename` 还有更广泛的用途，可以用在模板函数/类里面，其作用是表明某命名为一个类型，而不是一个变量或者其他

```c++
template <class T>
void foo() {
    typename T::iterator* iter;
    // ...
}
```

上述代码就表明 `T::iterator` 为一个类，`T::iterator* iter` 新建了一个类型指针

关于模板实例化 [模板：显式具体化和显式实例化](https://zhuanlan.zhihu.com/p/152211160)，目前的结论是：如果利用头文件使用模板，则在模板所在的源文件需要实例化

## 头文件

通常为了使得代码模块化，我们会将代码写到多个 cpp 文件里面。当我们想要使用其中的函数或者类时，可以通过引入头文件来调用它们。下面以某文件结构举例：

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

其中 `#pragma once` 代表以下内容只会被引入一次，这样就不会重复导入。从此就可以用头文件来完成

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

**头文件是不需要进行编译的，**但是必须要在编译器找得到的地方，换句话说要么头文件在同一个目录下，或者使用 `-I` 来指明头文件的路径。或许用“声明文件”来描述头文件是一个不错的选择？

问题：在头文件里通常要干的事情是什么？[理解头文件](https://www.runoob.com/w3cnote/cpp-header.html)，有时候头文件里的东西让人摸不到头脑

在头文件中可以定义类，参考 [Classes and header files](https://www.learncpp.com/cpp-tutorial/classes-and-header-files/)。主要有几点：

1. 类的定义，和函数的定义要区分开。类的定义不一定要把类的方法进行定义。当我们对类的方法进行定义，有时也成为方法的实现
2. 当直接在类定义体中，对某方法进行定义/实现时，该方法将被默认为内联函数
3. 通常在头文件中实现对类的定义，但不对类的方法进行实现，方法的实现在其他源文件。如果你在头文件中，在类定义体外对方法进行了实现，会有 **重复定义** 的报错

## Lambda 匿名函数

匿名函数的语法如下

```c++
[ capture clause ] (parameters) -> return-type  
{   
   definition of method   
} 
```

其中 `parameters & return-type` 是可以省略的，因为 lambda 可以自动推断返回类型，通常搭配 `auto` 关键字使用。 `capture` 的功能，就是给 lambda 函数在 `parameters` 以外的 local variables，看下面的例子就明白了

```c++
#include <iostream>

int main()
{
    {
        int a = 4; // can't be captured, because it's in another scope
    }
    int a = 3;
    int b = 5;
    
    // capture a, b by value
    auto func1 = [a, b] { std::cout << a << std::endl; };
    func1();

    // capture all local variables by value
    auto func2 = [=] { std::cout << a << " " << b << std::endl; };
    func2();

    // capture a by reference, b by value
    auto func3 = [&a, b] { std::cout << a << std::endl; a = -1;};
    func3();

    // capture all local variables by reference
    auto func4 = [&] { std::cout << a << " " << b << std::endl; };
    func4();
}
```

在 C++ 中是无法在函数中定义函数的，但是可以使用匿名函数，这是匿名函数的用途之一

## Vector



## 字符串

C-style 字符串是有终止符的，这使得在申请字符串空间时要多一个字符

在使用输入流读取字符串有两种方法

```c++
cin.getline(var, size)
cin.get(var)
```

二者遇到回车终止读取

TODO：字符串的规则比较多，后续如果有需要，清晰整理

## {} 的用法

在 Torch C++ 中看到使用 {1, 2, 3} 来初始化，这也可以作为函数参数？

## 零碎关键字

1. `__inline__` or `inline`

   中文翻译为 **内联**，通常用于修饰函数 `inline func`。在我看来就是在编译时将函数的 code 直接展开到其调用的位置。这省略了调用函数的流程，可能加速程序运行。其功能应该与 Macro 类似，但是比 Macro 更贴近于 C++ 代码，方便 debug

2. `noexcept override`

   这两个关键字我在类方法中看到

   ```C++
   class AddScalarPluginCreator : public IPluginCreator
   {
       const char *getPluginName() const noexcept override;
       const char *getPluginVersion() const noexcept override;
   ```

   简单解释 `const noexcept override`

   1. const 代表这个函数不会修改任何变量

   2. noexcept 代表这个函数不会使用 `throw` 来抛出 exception，该操作时的在运行时不会进行 exception 处理从而加速

      ```c++
      // throw an exception
      #include <stdexcept>
      
      int divide_e(int a, int b) {
         if (b == 0) {throw std::runtime_error("Division by zero");}
         return a / b;
      }
      
      // without exception
      int divide_no_e(int a, int b) noexcept {
          return a / b;
      }
      
      // this will cause error, because we use throw in a noexcpet function
      int divide_e(int a, int b) noexcept {
         if (b == 0) {throw std::runtime_error("Division by zero");   }
         return a / b;
      }
      
      // this will not cause error, and return 0 (false), as we might be throwing exception in this function, whether the throw is actual thrown or not
      std::cout << std::boolalpha << noexcept(divide_e(1, 0)) << std::endl;
      std::cout << std::boolalpha << noexcept(divide_e(3, 3)) << std::endl;
      ```

   3. override 代表该函数是重载了所继承类中的同名函数

3. `reinterpret_cast & static_cast & const_cast & (T)`

   分别介绍下这四种 cast，参考 [C++强制类型转换运算符（static_cast、reinterpret_cast、const_cast和dynamic_cast）](http://c.biancheng.net/view/410.html)

   1. `static_cast` 用于进行比较“自然”和低风险的转换，如整型和浮点型、字符型之间的互相转换。但是 `static_cast` 不能用于在不同类型的指针之间互相转换，当然也不能用于不同类型的引用之间的转换。因为这些属于风险比较高的转换

   2. `reinterpret_cast` 用于进行各种不同类型的指针之间、不同类型的引用之间以及指针和能容纳指针的整数类型之间的转换。转换时，执行的是逐个比特复制的操作

   3. `const_cast` 运算符仅用于进行去除 const 属性的转换，它也是四个强制类型转换运算符中唯一能够去除 const 属性的运算符

   4. C-style 转换 `(Type)`，参考 [What is the difference between static_cast and reinterpret_cast?](https://stackoverflow.com/questions/6855686/what-is-the-difference-between-static-cast-and-reinterpret-cast)

      > A C-style cast of the form `(T)` is defined as trying to do a `static_cast` if possible, falling back on a `reinterpret_cast` if that doesn't work. It also will apply a `const_cast` if it absolutely must.
