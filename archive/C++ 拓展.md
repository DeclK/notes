# C++ 拓展

## 动态内存

下面是使用 new 运算符来为任意的数据类型动态分配内存的通用语法：

```c++
new data-type;
```

**malloc()** 函数在 C 语言中就出现了，在 C++ 中仍然存在，但建议尽量不要使用 malloc() 函数。new 与 malloc() 函数相比，其主要的优点是，new 不只是分配了内存，它还创建了对象

在任何时候，都可以使用 delete 操作符释放对象占用的内存，但必须要使用指向对象的指针

```c++
delete pointer;
delete [] array_pointer;
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