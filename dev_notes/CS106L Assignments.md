## CS106L Assignments

在网上看过了不少 CS106L Assignments，要不有的需要 starter code，要不有的需要 Qt 三方库，为了让整个体验更加丝滑，我选择有 starter code + 不需要 Qt 三方库的练习题作为训练

[CS106L Self Learning](https://github.com/wengwz/CS106L-Self-Learning) 要不还是选择这个 assignments 比较好，不需要 Qt creator

[CS106L](https://github.com/JiNanPiWang/CS106L) 这里面也有五道题

## Assignment1 GapBuffer

-  第一次看 assignments 代码，核心代码都在 `gap_buffer.h` 里面，把所有的代码都写在头文件里是正常的吗？🤣然后使用一个 `gap_buffer_test.cc` 来测试所写的代码

  在 `gap_buffer.h` 里定义 (declare) 了 class 需要实现的类，留下了很多 `TODO` 代码

- `friend` keyword

  allows function for class to have access to private and protected members of the class

  `friend` keyword 不具有传递性：即 A 是 B 的友元，B 是 C 的友元，但是 A 不是 C 的友元

- `explicit` keyword

  The `explicit` keyword is used to p**revent implicit conversions** from happening during function calls or object construction. When applied to constructors, it prevents the constructor from being used for implicit type conversions.

  经常使用在 constructors

- GapBuffer algorithm

  这个方法通常用在 short text editor 里面，对于长文本则优化很差

  该算法有一个好处：在 cursur 附近且 gap buffer 没有用光的情况下，进行删除和添加文字是非常方便的，不用每一次都去对 vector array 进行 copy & move

  但是当遇到以下情况 gap buffer 会非常慢：

  1. 当操作不再 cursor/gap 附近时会非常慢
  2. gap buffer 被消耗光后，重新创建 buffer 会非常慢

  因为他们都要移动对大部分文本进行移动来创建新的 gap，时间复杂度都是 O(N)

- `std::move`

  在代码里出现了两种 `std::move` 的使用方式，一种是课上讲的方法，表示不去 copy 对象资源；另一种是 `algorithm` 头文件的重载，用于移动容器中的元素

  1. `std::move(obj)`

     该操作本质上就是将一个 l-value 转成一个 r-value，通过 assignment 操作可以将该 object 的资源转移到其他 object 上，从而达到更换 ownership 的效果。可以发现资源本身的内存地址是不变的

     ```c++
     #include <iostream>
     #include <vector>
     
     int main() {
         std::vector<int> v1 = {1, 2, 3, 4, 5};
     
         // Before move, print address of v1[0]
         std::cout << "v1[0] address: " << &v1[0] << "\n";
     
         // Move v1 to temp
         std::vector<int> temp = std::move(v1);
         // try to print v1[0] after move
         std::cout << "v1[0] address: " << &v1[0] << "\n";
         // std::cout << "v1[0] value: " << v1[0] << "\n";   // This will cause segmentation fault, because v1 is now empty
     
         // After move, print address of temp[0] and v1[0]
         std::cout << "temp[0] address: " << &temp[0] << "\n";
     
         if (!v1.empty()) {
             std::cout << "v1[0] address: " << &v1[0] << "\n";
         } else {
             std::cout << "v1 is now empty.\n";
         }
     
         return 0;
     }
     ```

     输出看下 move 前后 v1 和 temp 中元素的地址

     ```txt
     v1[0] address: 0x559d187cdeb0
     v1[0] address: 0
     temp[0] address: 0x559d187cdeb0
     v1 is now empty.
     ```

  2. `std::move(start_pos, end_pos, tgt_pos)`

     该方法可以将一个容器中一个范围的元素，移动到另一个容器的制定位置， It performs the actual "move operation" by iterating through the source container, moving each element to the destination.

     根据 [link](https://en.cppreference.com/w/cpp/algorithm/move) 可以认为等效下面代码

     ```c++
     template<class InputIt, class OutputIt>
     OutputIt move(InputIt first, InputIt last, OutputIt d_first)
     {
         for (; first != last; ++d_first, ++first)
             *d_first = std::move(*first);
      
         return d_first;
     }
     ```

     但是我第一次接触这个用法的时候 move 的对象是 int，而 `std::move` 在面对 int 时，是直接进行 copy 的，导致我产生了很大的疑惑，GPT 解释如下

     > For **primitive types** (such as `int`, `double`, etc.), there is no difference between copying and moving because they don’t manage any dynamic resources. They are simple values stored directly in memory, so copying them is as efficient as moving them.

     用一段代码可以看到 move 对象是 int 和 string 会有区别：string 的资源真的被转移了，而 int 资源实际上是被 copy 了

     ```c++
     #include <iostream>
     #include <algorithm>
     #include <vector>
     #include <string>
     
     void move_string(std::vector<std::string>& source, std::vector<std::string>& destination) {
         std::move(source.begin(), source.end(), destination.begin());
     }
     
     void move_int(std::vector<int>& source, std::vector<int>& destination) {
         std::move(source.begin(), source.end(), destination.begin());
     }
     
     int main() {
         std::vector<std::string> source_str = {"hello", "world", "foo", "bar"};
         std::vector<std::string> destination_str(4);  // Destination with same size
         std::vector<int> source_int = {1, 2, 3, 4, 5};
         std::vector<int> destination_int(5);  // Destination with same size
     
         // Moving elements from source to destination
         move_string(source_str, destination_str);
         move_int(source_int, destination_int);
     
         // Print destination contents
         std::cout << "===== Moving strings =====\n";
         for (const auto& s : destination_str) {
             std::cout << s << " ";
         }
         std::cout << "\n";
         // Print source contents (source is in a valid but unspecified state)
         for (const auto& s : source_str) {
             std::cout << (s.empty() ? "moved" : s) << " ";
         }
         std::cout << "\n";
         
     
         // Print destination contents
         std::cout << "===== Moving ints =====\n";
         for (const auto& i : destination_int) {
             std::cout << i << " ";
         }
         std::cout << "\n";
         std::cout << "Source vector int still exist:\n";
         for (const auto& i : source_int) {
             std::cout << i << " ";
         }
         
         return 0;
     }
     ```

     输出结果

     ```txt
     ===== Moving strings =====
     hello world foo bar 
     moved moved moved moved 
     ===== Moving ints =====
     1 2 3 4 5 
     Source vector int still exist:
     1 2 3 4 5 #
     ```

     注意，只有 move primitive types 的时候才会进行 copy，当 move vector of primitive types 的时候仍然会以转移资源的方式进行，也就是说 `std::move(vector<int>)` 仍然会将资源转移，而不是进行 copy

     ```c++
     #include <iostream>
     #include <vector>
     
     int main() {
         std::vector<int> v1 = {1, 2, 3, 4, 5};
     
         // Before move, print the address of v1's data
         std::cout << "v1 data address before move: " << v1.data() << "\n";
     
         // Move v1 to v2
         std::vector<int> v2 = std::move(v1);
     
         // After move, print the address of v2's data
         std::cout << "v2 data address after move: " << v2.data() << "\n";
         
         // Check the size of v1 after move
         std::cout << "v1 size after move: " << v1.size() << "\n";
     
         return 0;
     }
     
     ```

     输出结果

     ```c++
     v1 data address before move: 0x5613e7ac7eb0
     v2 data address after move: 0x5613e7ac7eb0
     v1 size after move: 0
     ```
  
- `typename`

  > The `typename` keyword is only required when you're accessing this dependent type **outside of its defining class or function** using the `::` scope operator. This is because, outside the template class, the compiler cannot infer whether `T::value_type` is a type or something else (like a static member or a value). The `typename` keyword helps disambiguate that it's a type.

  也就是说当我们在使用 `::` scope resolution operator 访问模板类的资源时，它是分不清所获得的东西是一个 type 还是一个 static member，如果是类且定义在一个模板类里面，必须要用 typename 来标明

  ```c++
  #include <iostream>
  
  template <typename T>
  class Example {
  public:
      using value_type = int;  // Define a type alias
      static const int value = 42;  // Define a static constant
  };
  
  template <typename T>
  void func() {
      typename Example<T>::value_type var;  // Is this a type or a value?
      std::cout << Example<T>::value;
  }
  
  
  int main() {
      func<int> ();
  }
  ```

  这也是 typename 除了在模板关键字外的另一个用途

- const correctness

  什么是 const correctness？直接上 GPT 的回答

  If a class method returns a reference to one of its members (for example, an element in a container class), it's a good practice to provide two versions of the method:

  1. **Non-const version**: This method returns a non-const reference, allowing the caller to modify the referenced member.
  2. **Const version**: This method returns a const reference, ensuring the caller can only read the referenced member, not modify it.

  也就是说当类返回了一个自己的成员引用时，最好实现两个版本：一个 const 和一个 non-const，其中 const 版本就保证了所返回的引用不被外部所修改，或者在生成一个 const object 的时候，只会去调用（也只能够调用） const version methods

  下面就时一个例子，我们可以通过这个漏洞去改变 Test 当中的 private member😎

  ```c++
  #include <iostream>
  class Test {
      private:
          int start = 1;
      public:
          int& reference() {
              return this->start;
          }
          // const int& reference() const {
          //     return this->start;
          // }
  };
  
  
  int main() {
      Test test;
      std::cout << test.reference() << std::endl;
      test.reference() = -1;
      std::cout << test.reference() << std::endl;
  }
  ```

- `static_cast` & `const_cast`

  在 assignment 中为了轻松实现上述 const & non-const method，使用了 `const_cast` 来去除掉 const 属性，使得在 const method 中能够调用非 const method（本来在 const method 中只能调用 const method）

  ```c++
  template <typename T>
  typename GapBuffer<T>::const_reference GapBuffer<T>::get_at_cursor() const {
      // Be sure to use the static_cast/const_cast trick in the non-const version.
      return const_cast<GapBuffer<T>*>(this)->get_at_cursor();
      // return static_cast<const_reference>(const_cast<GapBuffer<T>*>(this)->get_at_cursor());
  }
  ```

  将 `this` cast 为 non-const 过后就能够调用之前写好的 `get_at_cursor` function，并且这里不需要再做一次 `static_cast` 来将其转换为 const reference，这里会自动转换的

- iterator operator

  - **如何定义自己的 iterator**

    在 assignment 的代码中 `GapBufferIterator` 继承了 `std::iterator`

    ```c++
    // Class declaration of the GapBufferIterator class
    template <typename T>
    class GapBufferIterator : public std::iterator<std::random_access_iterator_tag, T> 
    ```

    我对这个继承感到疑惑，还好有 GPT 给我回答疑问：

    1. 这个继承最重要的部分就是 `<std::random_access_iterator_tag`，这标志了这个类是一个 random access iterator 类，在使用像 `std::sort` 的方法时，遇到了这个 tag 就会使用更快速的算法。而 random access iterator 需要这个类实现特定的 operators:

       - Use operations like `++` and `--` to move forward and backward.

       - Access elements by an offset (`it + n` or `it - n`).

       - Compare iterators (`<`, `>`, `<=`, `>=`).

       - Get the difference between two iterators (`it2 - it1`).

       最常用也最强大的 iterator 都是 random access iterator，例如 `std::vector & std::array`

    2. Deprecation of `std::iterator`

       在 C++20 中直接废除了这样的继承方式，因为 C++20 认为 iterator 的性质应该直接定义在实现当中，而不需要去继承

       > Iterators should define their traits explicitly instead. Instead of inheriting from `std::iterator`, modern C++ code directly specifies the following five typedefs (or using declarations) in the iterator class:
       >
       > 1. `difference_type`: Used to represent the difference between two iterators.
       > 2. `value_type`: The type of elements the iterator points to (in this case, `T`).
       > 3. `pointer`: The type of a pointer to `value_type`.
       > 4. `reference`: The type of a reference to `value_type`.
       > 5. `iterator_category`: The iterator category, e.g., `std::random_access_iterator_tag`.

       上面这些类也是直接实现在了 `GapBufferIterator` 当中

       ```c++
       template <typename T>
       class GapBufferIterator : public std::iterator<std::random_access_iterator_tag, T> {
       public:
           friend class GapBuffer<T>;
           using value_type = T;
           using size_type = size_t;
           using difference_type = ptrdiff_t;
           using reference = value_type&;
           using iterator = GapBufferIterator<T>;
       ```

  - **operator++(int)** 

    iterator 有两个 increment operator，一个是 pre-increment，一个是 post-increment，这里 int 其实就是一个语法要求，代表这个是 post-increment，不能是其他的类

    ```c++
    Iterator& operator++();  // Pre-increment
    Iterator operator++(int); // Post-increment (optional)
    ```

    并且可以注意到二者返回的类型不一样，pre-increment 返回的是 reference，而 post-increment 返回的却是一个 copy，这是因为 post-increment 本身会改变 iterator 的状态，我们返回的 iterator 应该是改变状态之前的 iterator，这样就需要将原始状态进行 copy

    ```c++
    template <typename T>
    GapBufferIterator<T>& GapBufferIterator<T>::operator++() {
        // TODO: implement this prefix operator (~2 lines long)
        _index++;
        return *this;
    }
    
    template <typename T>
    GapBufferIterator<T> GapBufferIterator<T>::operator++(int) {
        // TODO: implement this postfix operator (~3 lines long)
        iterator temp = *this;
        _index++;
        return temp;
    }
    ```

    同时这里再次复习一下 ++i 和 i++：

    1. ++i pre-increment，因为 ++ 先出来，所以会先 i+1 然后再赋值
    2. i++ post-increment，因为 i 先出来，所以会先赋值再 i+1，并且由于 ++ 靠后，所以不能作为左值 (l-value)
  
- Deconstructor

  > In C++, when you’re implementing a destructor, you typically need to focus on managing dynamic memory to prevent memory leaks. The other members that are not dynamically allocated, they are automatically cleaned up when the instance goes out of scope.

  根据 GPT 的回答，我们的 deconstructor 写作

  ```c++
  template <typename T>
  GapBuffer<T>::~GapBuffer() {
      // TODO: implement this destructor (~1 line long)
      delete [] _elems;
  }
  ```

- `initializer_list`

  通常 `initializer_list` 会在两个概念中提到，但二者是不一样的概念

  1. **Constructor Initialization List**

     这代表的是构造函数中初始化变量成员的一种方式，其语法如下

     ```c++
     ClassName() : member1(value), member2(value) {...}
     ```

     作用：在构造函数中直接初始化成员变量

  2. **`std::initializer_list`**

     这代表的是统一初始话的底层机制，其语法如下

     ```c++
     ClassName(std::initializer_list<int> list, ...) {}
     ClassName obj{value1, value2, value3};
     ```

     作用：支持花括号初始化语法。通常和 uniform initialization 一起出现，`{}` 可以替代原始的 `()` 初始化方法（反过来不行），并且会优先匹配带有 `std::initializer_list` 参数的初始化方法

- `std::move` 带来的麻烦，可以用 `std::unique_ptr` or `std::vector` 进行解决

  C++ 一直都需要注意 dynamic resources 的管理，即使是使用 `std::vector` 也需要注意。对于自身带有 dynamic resources 的类，vector 本身也是没办法进行管理的。所以总而言之 RAII (smart pointers) is all you need，不使用 `new & delete` 就是智能指针诞生的重要原因之一，能更好地管理这种动态内存

- Variadic Templates

  可变参数模板，用于处理变长模板参数，通常和折叠表达式一起出现

  语法形式

  ```c++
  template <typename... Args>  // Args 是模板参数包（Template Parameter Pack）
  void func(Args... args);      // args 是函数参数包（Function Parameter Pack）
  ```

  常用方式：

  1. 递归展开

     注意需要定义递归终止时的处理方式

     ```c++
     // 递归终止函数（处理 0 个参数的情况）
     void print() {
         std::cout << "\n";
     }
     
     // 递归展开参数包
     template <typename T, typename... Args>
     void print(T first, Args... args) {
         std::cout << first << " ";  // 处理第一个参数
         print(args...);             // 递归处理剩余参数
     }
     
     // 调用
     print(1, 3.14, "Hello");  // 输出: 1 3.14 Hello
     ```

     另外，使用 `sizeof...(Args)` 可以获得参数包中的参数数量

  2. 折叠表达式

     ```c++
     // 使用折叠表达式计算所有参数的和（C++17）
     template <typename... Args>
     auto sum(Args... args) {
         return (args + ...);  // 展开为 (arg1 + (arg2 + ...))
     }
     
     // 调用
     auto total = sum(1, 2, 3, 4);  // 返回 10
     ```

     上述展示了一元右折叠，还有四种不同的折叠方式

     1. 一元右折叠：`(pack op ...)` → `(arg1 op (arg2 op (... op argN)))`

        依据括号顺序，将从最后一个元素开始进行 op 计算

     2. 一元左折叠：`(... op pack)` → `(((arg1 op arg2) op ... argN)`

        和右折叠相反，将从第一个元素开始进行 op 计算

     3. 二元右折叠：`(pack op ... op init)` → `(arg1 op (arg2 op (... (argN op init))))`

        似乎很少使用

     4. 二元左折叠：`(init op ... op pack)` → `((((init op arg1) op arg2) op ...) op argN)`

        可以在 `std::cout << ... << args` 连续打印元素

  3. 完美转发参数

     ```c++
     // 将任意参数完美转发给构造函数
     template <typename T, typename... Args>
     std::unique_ptr<T> create(Args&&... args) {
         return std::make_unique<T>(std::forward<Args>(args)...);
     }
     
     // 调用：构造一个 std::vector
     auto vec = create<std::vector<int>>(5, 10);  // vector(5, 10)
     ```

     这个用法比较固定，具体什么是完美转发 perfect forwarding，我在下一节具体整理

- Perfect forwarding

  是和可变参数模板独立的概念，**其功能是保持对象的左值/右值的性质不变**。通常我们在使用变长参数模板的时候，会希望保持每个参数的原始值类别（左值/右值），所以 perfect forwarding 和 variadic templates 也经常一起使用

  > 当处理可变数量参数时，我们需要：
  >
  > 1. 接受任意数量的参数 → 需要可变参数模板
  > 2. 保持每个参数的原始值类别 → 需要完美转发

  万能引用

  万能引用是 Scott Meyers 提出的术语，用于描述模板函数中 `T&&` 这种特殊形式的参数声明。它之所以"万能"，是因为它**既能绑定左值，也能绑定右值**，这是普通引用无法做到的

  万能引用的推导规则：当传入参数时：

  - 左值 → `T` 推导为 `T&` → `T&&` 折叠为 `T&`
  - 右值 → `T` 推导为 `T` → `T&&` 保持为 `T&&`

  ```c++
  void process(int&)  { std::cout << "左值\n"; }
  void process(int&&) { std::cout << "右值\n"; }
  
  template <typename T>
  void wrapper(T&& arg) {
      process(std::forward<T>(arg));
  }
  
  wrapper(100);  // T 推导为右值 int&&
  int x = 42;
  wrapper(x);  // T 推导为左值 int&
  ```

  以下是一个没有 `std::forward` 的情况

  ```c++
  #include <iostream>
  
  void target(int&) { std::cout << "左值版本\n"; }
  void target(int&&) { std::cout << "右值版本\n"; }
  
  template <typename T>
  void bad_wrapper(T&& arg) {
      // 没有使用std::forward
      target(arg); 
  }
  
  int main() {
      int x = 42;
      
      bad_wrapper(x);          // 期望调用左值版本 → 实际调用左值版本
      bad_wrapper(100);        // 期望调用右值版本 → 实际调用左值版本 ❌
      bad_wrapper(std::move(x)); // 期望调用右值版本 → 实际调用左值版本 ❌
  }
  ```

  **在 `bad_wrapper` 内部，无论传入什么值，`arg` 都是一个具名的左值表达式。没有使用 `std::forward`，导致所有参数都被当作左值处理，右值被错误地当作左值使用，无法调用正确的重载**

## Question

- 在练习中使用了 google test 

  ```c++
  #include "gtest/gtest.h"
  TEST(GapBufferTest, TEST1A_INSERT_AT_CURSOR_BASIC) {
      // code
  }
  ```

  应该学习一下如何使用 google test 来完成测试单元

- 这次 assignment 过后应当也学习如何使用 CMakeLists.txt

  DONE at 2025/06/01, hope it's not too late✅

- `template <typename... Args>` 是什么意思？可变参数模板✅

- 后记

  想不到整个 C++ 的学习拖了这么久(2024/05-2025/06)...光是完成这一个 assignment 已经花了很多时间了，其他的 assignment 就放弃掉了🫤因为主线任务仍然是做 CUDA 开发，而不是复杂的 C++ 开发，希望这些基础知识能够支撑我走得更远，在学习过程中没有那么多的疑惑和畏惧🤗

