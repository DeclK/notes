# C++ 基础

[zhihu](https://zhuanlan.zhihu.com/p/114543112)

[菜鸟教程](https://www.runoob.com/cplusplus/cpp-tutorial.html)

## Install

参考 [bilibili](https://www.bilibili.com/video/BV1Cu411y7vT)

### MinGW

前往链接下载 [MinGW-w64](https://sourceforge.net/projects/mingw-w64/files/mingw-w64/mingw-w64-release/)，我选择了下方版本

<img src="C:\Data\Projects\notes\dev_notes\C++\image-20221227171144324.png" alt="image-20221227171144324" style="zoom:50%;" />

并把解压过后的 `bin` 其添加到环境变量中，完成后在 cmd 中输入

```cmd
gcc -v
```

验证是否安装完成

安装 C/C++ VSCode 插件后就可以愉快编程了

## 语法

基本结构：

1. 头文件，`<iostream>` 就是用于输入输出的，要用 `cin, cout, endl` 就必须要该头文件
2. using namespace std
3. 一定有一个 main

语法：

1. 分号为结束
2. 用大括号分隔语句块
3. 双斜杠表示注释

## 输入输出

使用 `cin & cout & << >>` 来完成

```c++
cin >> name >> age;
// equal
cin >> name;
cin >> age;

cout >> name >> age >> endl;
```

## 数据类型

bool	char	int	float	double	void	string

上面就是 C++ 的基本类型，还可以加入一些修饰：signed, unsigned, short, long

typedef 声名，可以为一个已有的类型取一个新的名字

```c++
typedef type newname;
```

枚举变量的值只能取枚举常量表中所列的值，就是整型数的一个子集

枚举变量只能参与赋值和关系运算以及输出操作，参与运算时用其本身的整数值

```c++
enum color { red, green, blue } c;
c = blue;
```

## 变量

变量是类型的实例，变量名称可由字母、数字、下划线组成

定义一个变量

```c++
type varialble_list;
type variable_name = value;

// examples
int    i, j=1, k;
char   c, ch;
float  f, s;
double d;
```

声明变量，区别于定义变量，仅用于向程序表明变量的类型和名字，声明不给变量分配空间。声明使用 extern 关键字完成

```c++
extern int a, b;
```

一个变量可声明多次，只能被定义一次

### 变量作用域

作用域是程序的一个区域，一般来说有三个地方可以定义变量：

- 在函数或一个代码块内部声明的变量，称为**局部变量**。它们只能被函数内部或者代码块内部的语句使用
- 在所有函数外部声明的变量（通常是在程序的头部），称为**全局变量。**全局变量的值在程序的整个生命周期内都是有效的
- 在函数参数的定义中声明的变量，称为**形式参数**

在程序中，局部变量和全局变量的名称可以相同，但是在函数内，局部变量的值会覆盖全局变量的值

### 常量

定义常量有两种方式

- 使用 **#define** 预处理器
- 使用 **const** 关键字

```c++
#define LENGTH 10   
#define WIDTH  5
#define NEWLINE '\n'
 
const int  LENGTH = 10;
const int  WIDTH  = 5;	
const char NEWLINE = '\n';
```

字符串常量是用双引号 `""` 表示，单个字符用单引号 `''`

### 存储类, static

存储类定义 C++ 程序中变量/函数的范围（可见性）和生命周期。这些说明符放置在它们所修饰的类型之前。常用的几个

- static
- extern
- mutable

重点介绍一下 static 类，其作用如下：

- （1）在修饰变量的时候，static 修饰的静态局部变量只执行初始化一次，而且延长了局部变量的生命周期，直到程序运行结束以后才释放。
- （2）static 修饰全局变量的时候，这个全局变量只能在本文件中访问，不能在其它文件中访问，即便是 extern 外部声明也不可以。
- （3）static 修饰一个函数，则这个函数的只能在本文件中调用，不能被其他文件调用。static 修饰的变量存放在全局数据区的静态变量区，包括全局静态变量和局部静态变量，都在全局数据区分配内存。初始化的时候自动初始化为 0。
- （4）不想被释放的时候，可以使用static修饰。比如修饰函数中存放在栈空间的数组。如果不想让这个数组在函数调用结束释放可以使用 static 修饰。
- （5）考虑到数据安全性（当程序想要使用全局变量的时候应该先考虑使用 static）

## 运算符

算数运算符基本是通用的：`+ - * / % ++ --`

关系运算符基本是通用的：`== != > < >= <=`

逻辑运算符：`&& || !`	分别表示与，或，非

位运算符基本是通用的：`& | ^` 分别表示与，或，异或

赋值运算符基本是通用的：`== += -= *= ...`

杂项运算符：

1. `sizeof` 返回变量大小
2. `condition ? X: Y` 如果 Condition 为真 ? 则值为 X : 否则值为 Y
3. `. & ->` 为成员运算符，用于引用类、结构和共用体的成员
4. `Cast` 强制转换运算符，把一种数据类型转换为另一种
5. `&, *` **指针运算符**，前者返回变量的地址，后者指向一个变量（访问变量）

## 循环与判断

循环有几种语法：

1. while & do while 循环 

   ```c++
   while(condition)
   {
      statement(s);
   }
   
   do
   {
      statement(s);
   }while( condition );
   ```

2. for 循环

   ```c++
   for ( init; condition; increment )
   {
      statement(s);
   }
   ```

使用 break & continue 可打破循环

判断语句有如下几种方式：

1. if & if else 语句

   ```c++
   if(boolean_expression 1)
   {	statement(s)
   }
   else if( boolean_expression 2)
   {	statement(s)
   }
   else 
   {	statement(s)
   }
   ```

## 函数

每个 C++ 程序都至少有一个函数，即主函数 **main()** 

C++ 中的函数定义的一般形式如下

```c++
return_type function_name( parameter list )
{
   body of the function
}
```

参数列表包括函数参数的类型、名称，参数的名称并不重要，只有参数的类型是必需的（但没有名称怎么调用呢🤣

这些参数称为函数的**形式参数**，形式参数就像函数内的其他局部变量，在进入函数时被创建，退出函数时被销毁

### 函数参数

三种传递参数：

1. 传值调用。不改变实际参数值
2. 指针调用和引用调用。会改变实际参数值

参数也可以指定默认值，成为可选参数

## 数组

在 C++ 中要声明一个数组，需要指定元素的类型和元素的数量

```c++
type arrayName [ arraySize ];

double balance[5] = {1000.0, 2.0, 3.4, 7.0, 50.0};
double balance[] = {1000.0, 2.0, 3.4, 7.0, 50.0};
```

C++ 支持多维数组。多维数组声明的一般形式如下

```c++
type name[size1][size2]...[sizeN];
```

数组名是指向数组中第一个元素的常量指针

C++ 传数组给一个函数，数组类型自动转换为指针类型，因而传的实际是地址，一下三种方法都可以

```c++
void myFunction(int *param)
void myFunction(int param[10])
void myFunction(int param[])
```

## 指针

每一个变量都有一个内存位置，可使用连字号（&）运算符访问

**指针**是一个变量，其值为另一个变量的地址，指针变量声明的一般形式为

```c++
type *var-name;

int    *ip;
double *dp;
float  *fp;
char   *ch;
```

所有指针的值的实际数据类型，不管是整型、浮点型、字符型，还是其他的数据类型，都是一样的，**都是一个代表内存地址的长的十六进制数**

关于指针的几点：

1. NULL 表示空指针

2. 可对指针进行四种算数运算，`++, --, +, -`

3. 可以定义一个指针数组，即数组中的每一个值是一个指针

4. 可以定义多个 `**`，表示嵌套的指针，可用于存储指针的地址，即地址的地址

   ```c++
   #include <iostream>
    
   using namespace std;
    
   int main ()
   {
       int  var;
       int  *ptr;
       int  **pptr;
    
       var = 3000;
    
       ptr = &var;
    
       pptr = &ptr;
    }
   ```

5. 可定义返回指针的函数

   ```c++
   int * myFunction()
   ```

   也可定义指向函数的指针

   ```c++
   int (* p)(int, int) = & max; // & is not must
   int a;
   a = p(1, 2);
   ```

## 引用

这是和指针非常相似的概念，引用的本质是在 C++ 内部实现的一个指针常量

```c++
Type& ref = val;
// equal
Type* const ref = &val;
```

C++ 编译器在编译过程中使用常量指针作为引用的内部实现，因此引用占用的空间大小与指针相同，只是这个过程是编译内部实现，用户不可见

所以引用与指针的区别可总结如下：

1. 一旦引用被初始化为一个对象，就不能被指向到另一个对象。指针可以在任何时候指向到另一个对象。
2. 引用必须在创建时被初始化。指针可以在任何时间被初始化。

引用的常用用法如下：

1. 把引用作为参数。这比传一般的参数更安全，可更方便地对参数实现修改
2. 把引用作为返回值。当函数返回一个引用时，则返回一个指向返回值的隐式指针。这样，函数就可以放在赋值语句的左边

## 数据结构

### struct

struct 语句定义了一个包含多个成员的新的数据类型

```c++
struct type_name {
member_type1 member_name1;
member_type2 member_name2;
member_type3 member_name3;
.
.
} object_names;
```

为了访问结构的成员，我们使用**成员访问运算符（.）**

结构指针访问成员则必须使用箭头 `->`，这也是 `->` 唯一用途

```c++
struct Books
{
   char  title[50];
   char  author[50];
};

void printBook( struct Books *book )
{
   cout << "title: " << book->title <<endl;
   cout << "author: " << book->author <<endl;
}
```

### typedef

可以使用它来为类型取一个新的名字。**typedef** 仅限于为类型定义符号名称，**#define** 不仅可以为类型定义别名，也能为数值定义别名，比如您可以定义 1 为 ONE

```c++
typedef long int *pint32;
 
pint32 x, y, z;
```

x, y 和 z 都是指向长整型 long int 的指针
