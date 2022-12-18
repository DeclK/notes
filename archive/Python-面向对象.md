---
title: Python 3 面向对象
tags:
  - Python
  - 面向对象
  - 教程
categories:
  - 编程
  - Python
  - Basic
abbrlink: 59bf225c
date: 2021-08-28 21:53:07
---

# python 面向对象

## 面向对象编程

什么是面向对象编程？用一段话描述面向对象的思想：

> 把一组数据结构和处理它们的方法组成对象（object），把相同行为的对象归纳为类（class），通过类的封装（encapsulation）隐藏内部细节，通过继承（inheritance）实现类的特化（specialization）和泛化（generalization），通过多态（polymorphism）实现基于对象类型的动态分派

这一段话我在之前是看得云里雾里，这样高度的概括过于抽象。现在看过了一些代码过后，再回头看这段话才没那么迷惑

## 类和对象

类是对象的模板，对象是类的实例。在面向对象的世界中，一切皆为对象。下面来创建一个类

```python
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        print("My name is %s. I'm %d." % (self.name, self.age))
```

之前提到过 `__init__()` 方法在类中属于特殊方法，会在创建对象的时候自动调用。通过这个方法可以给对象绑定 `name` & `age` 两个属性，这样在其他方法里就能够随意调用这些属性，而不需要再次传入这些参数。下面来创建 `Student` 类的对象，并使用其内部函数

```python
student_1 = Student('Declan', 23)
student_1.introduce()
# My name is Declan. I'm 23.
```

创建对象，就是将类实例化，具体一点来说：给类传入一些参数，让其从模板称为了一个具体的对象。可以看到在创建的对象的代码里，并没有显式使用，也不能显式使用特殊方法 `__init__()` 

```python
# try this
student_2 = Student.__init__('Declan', 23)
```

这样做一定是会报错的。所以正确的理解是，`Student(name, age)` 中的参数 `name` & `age` 就是给 `__init__()` 的参数，并在创建对象的时候自动运行 `__init__()` 方法

### 访问可见性问题

在很多面向对象编程语言中，我们通常会将对象的属性设置为私有的（private）或受保护的（protected），简单的说就是不允许外界访问，而对象的方法通常都是公开的（public），在 Python 中，属性和方法的访问权限只有两种，也就是公开的和私有的，如果希望属性是私有的，在给属性命名时可以用两个下划线作为开头

```python
class Test:
    def __init__(self):
        self.__private = 'private'
        self.__func()
        
        
    def __func(self):
        print(self.__private)
        print('this function is private')

test = Test()
# private
# this function is private
print(test.__private)
# AttributeError: 'Test' object has no attribute '__private'
# 直接访问被阻拦，但拐个弯还是有方法能够访问
print(test._Test__private)
# private
```

可以看在类的内部调用这些私有变量和函数是没有问题的，但是想要在函数之外直接访问这些变量和函数是被禁止的。在实际开发中，我们并不建议将属性设置为私有的，因为这会导致子类无法访问（后面会讲到）。所以大多数 Python 程序员会遵循一种命名惯例就是让属性名以单下划线开头如 `_name`，来表示属性是受保护的，本类之外的代码在访问这样的属性时应该要保持慎重

## 类的方法

### self

类的方法与普通的函数只有一个特别的区别——它们必须有一个额外的**第一个参数名称**，按照惯例它的名称是 self，当然也可以叫其它名字，下面看看这个 self 到底代表了什么

```python
class Test:
	def __init__(SELF):
		print(SELF.__class__)
	
	def pointer_self(self):
		print(self)
		
test = Test()
test.pointer_self()
# <class '__main__.Test'>
# <__main__.Test object at 0x000001B64F611040>
```

从执行结果可以很明显的看出，self 代表的是类的实例，代表当前对象的地址，而 self.class 则指向类

## 继承

如果一种语言不支持继承，类就没有什么意义。子类会继承父类（也叫基类）的属性和方法，而且还可以定义自己的属性和方法，所以子类比父类拥有更多的能力，下面看一个简短的代码，理解继承机制的逻辑

```python
# 这是之前写的学生类
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.log = 'this is a student'
    
    def introduce(self):
        print("My name is %s. I'm %d." % (self.name, self.age))
   
# 现在建立一个本科生类
class Undergraduate(Student):
    def __init__(self, name, age, grade):
        # 在子类中调用父类的方法
        super().__init__(name, age)
        self.grade = grade

    def which_grade(self):
        print('My grade is %d.' % self.grade)
    
    # 在多态的时候取消注释
    # def introduce(self):
    #     print("Hello! My name is %s. I'm %d." % (self.name, self.age))

student_1 = Undergraduate('Declan', 18, 1)
student_1.which_grade()
# My grade is 1.

# 子类实例直接使用父类的属性和方法
print(student_1.log)
student_1.introduce()
# this is a student
# My name is Declan. I'm 18.
```

重点的几个逻辑：

1. 声明父类，在定义的类名后传入父类名即可，如 `class Undergraduate(Student)`
2. 调用父类，通过 `super()` 函数调用父类中的属性、方法，如果是 python 2.x 的话需要使用 `super(子类名, self)` 调用
3. 实例使用父类，子类实例可以直接使用父类中的属性、方法

## 多态

子类在继承了父类的方法后，可以对父类已有的方法给出新的实现版本，这个动作称之为方法重写（override），父类的方法可以被多个子类进行重写得到多个不同的实现版本，这个就是多态（poly-morphism）

把上一节中关于多态的代码取消注释，再调用 `introduce()` 方法

```python
student_1.introduce()
# Hello! My name is Declan. I'm 18.
```

发现比之前父类的 `introduce()` 多打印了 `Hello!` 重写成功！ 

## 补充：类的特殊方法

这些类的特殊方法也叫魔术方法，在特殊条件下被调用，下面列举一些

```python
- **__init__ :** 构造函数，在生成对象时调用
- **__del__ :** 析构函数，释放对象时使用
- **__repr__ :** 打印对象时调用
- **__setitem__ :** 按照索引赋值
- **__getitem__:** 按照索引获取值
- **__len__:** 获得长度
- **__cmp__:** 比较运算
- **__call__:** 让对象变为可调用对象
# 下面的魔术方法可以进行算符重载
- **__add__:** 加运算
- **__sub__:** 减运算
- **__mul__:** 乘运算
- **__truediv__:** 除运算
- **__mod__:** 求余运算
- **__pow__:** 乘方
```

实验一下 `__repr__, __call__`

```python
class Student:
    def __init__(self, name, age) -> None:
        self.name = name
        self.age = age

    def __repr__(self) -> str:
        return "My name is {}. I'm {}.".format(self.name, self.age)

    def __call__(self, grade) -> str:
        return 'grade {}'.format(grade)


name = 'Declan'
age = 23
stu = Student(name, age)
print(stu)
# My name is Declan. I'm 23.
print(stu(2))
# grade 2
```

## 补充：装饰器

参考 [博客](https://www.cnblogs.com/auguse/articles/9922257.html) 进行整理。装饰器本质上是一个 Python 函数或类，它可以让其他函数或类在不需要做任何代码修改的前提下增加额外功能。由于在学习 mmdetection 中发现 `registry` 类的实现就需要装饰器的帮忙，所以接下来将“花费”一些篇幅来了解装饰器的内部逻辑

### 装饰器基本原理

下面先看看不用 python 装饰器，怎样实现其类似的功能

```python
# 原函数
def test(name, age):
    print("My name is {}, and I'm {}.".format(name, age))

# 定义一个装饰器来包装原函数
def decorate(func):
    def wrap(*args, **kwargs):
        print('This is my introduction.')
        func(*args, **kwargs)
    # 返回新的函数名
    return wrap

# 将原函数包装，并将原函数指向装饰后的函数
test  = decorate(test)
name = 'Declan'
age = 23
test(name, age)
# This is my introduction.
# My name is Declan, and I'm 23.
```

这样就实现了一个装饰器，该装饰器的功能就是在原函数之前打印一句话 `This is my introduction`

python 使用 `@` 语法糖来实现装饰器，具体来说 `@` 实现的是这句话**“将原函数名作为参数，输入到装饰函数中，其返回值指向原函数名”**。将之前的装饰器，用 `@` 重新实现

```python
# 定义一个装饰器来包装原函数
def decorate(func):
    def wrap(*args, **kwargs):
        print('This is my introduction.')
        func(*args, **kwargs)
    # 返回新的函数名
    return wrap

# 原函数+装饰器
@decorate
def test(name, age):
    print("My name is {}, and I'm {}.".format(name, age))

# test  = decortate(test)
name = 'Declan'
age = 23
test(name, age)
# This is my introduction.
# My name is Declan, and I'm 23

print(type(test))
# <class 'function'>
```

可以看到，就是在原函数之前加上 `@decorate` 就实现了对原函数的装饰，替代了 `test  = decorate(test) ` 这一步，**这就是装饰器的本质**。换句话说 `@` 关键字将下一行的函数/类，作为参数传给了 `decorate` 函数，而 `decorate(test)` 通常将返回一个函数，此函数将能用 `test` 进行调用

之前的装饰器只接受了原函数 `test` 作为参数，那如果想要实现传入多个参数，例如 `@decorate(*args)` 应该怎么办呢？接下来实现一个带参数的装饰器，也就是装饰器的装饰器🤣，虽然这么说很绕哈哈

```python
name = 'Declan'
age = 23

def param_decorate(addtional):
    print(f"It's running before test(), and addtional is {addtional}")
    def decorate(func):
        def wrap(*args, **kwargs):
            print('This is a introduction.')
            func(*args, **kwargs)
            return None
        return wrap
    return decorate

@param_decorate(3.14)
def test(name, age):
    print("My name is {}, and I'm {}.".format(name, age))


# It's running before test(), and addtional is 3.14
test(name, age)
# This is a introduction.
# My name is Declan, and I'm 23
```

使用以上代码会发现，即使没有运行 `test` 函数也输出了内容。这是因为给装饰器添加参数过后，那么 `@` 后跟随的就是一个执行函数，python 就会真实地执行该函数的内容。执行 `param_decorate` 函数返回的是一个函数名 `decorate`，那么此时 `@param_decorate()` 相当于 `@decorate`

**还有一个方法来学习装饰器的内部逻辑，就是直接对代码 debug，一步步看程序是如何运行的**

### python 内置装饰器

内置的装饰器和普通的装饰器原理是一样的，只不过一般用于类的方法当中，让类变得更灵活

#### @property

参考 [菜鸟教程](https://www.runoob.com/python/python-func-property.html) [廖雪峰教程 ](https://www.liaoxuefeng.com/wiki/897692888725344/923030547069856)进行整理。`@property` 能够用于管理类的**私有属性**，方便读取属性、修改属性。下面看看如何使用该装饰器

```python
class Student:
    @property
    def get_the_score(self):
        """I'm the 'score' property."""
        return self._score
 
    @get_the_score.setter
    def score(self, value):
        self._score = value
 
    @get_the_score.deleter
    def score(self):
        del self._score


student = Student()
# setter不仅可以更改属性值，也可以创建该属性
student.get_the_score = 60

# 查看属性
print(student.get_the_score)
# 60

# 也使用原属性名称更改和访问
student.score = 100
print(student.score)
# 100

# 删除属性
# del student.score
```

`@property` 的实现比较复杂，我感觉我是真没理解，这里直接引用一下廖雪峰教程中的话：

> 把一个 `get_the_score` 方法变成属性，只需要加上 `@property` 就可以了。此时，`@property`本身又创建了另一个装饰器 `@get_the_score.setter`，负责把一个 setter 方法变成属性赋值，于是，我们就拥有一个可控的属性操作

我自己理解，如果需要只读属性，则只使用 `@property` 装饰器就可以了，如果还需要对属性进行进一步操控则加上其他 `setter, deleter` 装饰器

而且，如果使用了 `property` 装饰器，必须要使用单下划线变量，不然会报错，具体原因参考 [知乎链接](https://zhuanlan.zhihu.com/p/53469919)

#### @classmethod

参考 [知乎链接](https://zhuanlan.zhihu.com/p/35643573) 进行整理，`classmethod` 又被叫做类方法。`__init__()` 作为类的构造函数，能够在生成对象时调用，但如果想要使用其他构造函数时，可以使用`@classmethod` 实现，下面看如何使用类方法创建一个对象（为方便，把静态方法 `staticmethod` 的代码也写在这儿）

```python
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
        print('{}年{}月{}日'.format(self.year, self.month, self.day))

    @classmethod
    def create_from_string(cls, string):
        year, month, day = map(int,string.split('-'))
        return cls(year, month, day)

    @staticmethod
    def is_leap(year):
        if year % 4 == 0 and year % 100 != 0:
            return True
        elif year % 400 == 0:
            return True
        else: return False

# 普通构造
date = Date(2020, 7, 31)
# 2020年7月31日

# 通过类方法构造
date = Date.create_from_string('2020-7-31')
# 2020年7月31日

# 使用静态方法
print(Date.is_leap(1900))
# True
```

重点需要注意的是类方法的第一个参数，其表示调用当前的类名，默认名称为 `cls`，有点类似于 `self` 表示类的实例。那么类方法最后的返回值 `return cls(year, month, day)` 也就不难理解了，相当于 `return Data(year, month, day)` 重新创建了对象

类方法能够在不改变原本构造函数的情况下，给类的构造方法增加一些额外功能，例如对于传入参数做一些不同的处理等等

#### @staticmethod

参考 [博客](https://www.cnblogs.com/Meanwey/p/9788713.html) 的说法：`@staticmethod` 静态方法只是名义上归属类管理，但是不能使用类变量和实例变量，是类的工具包。因为该函数不传入self或者cls，所以不能访问类属性和实例属性。静态方法的一个好处是，不用创建类的实例也能够调用该方法，具体调用方法参考上面代码
