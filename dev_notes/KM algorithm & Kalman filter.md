# 3DMOT: Multi-object Tracker

## 匈牙利算法 & KM 算法

这是一种分配算法（常解决指派问题），也叫 KM 算法，[wiki](https://zh.wikipedia.org/wiki/%E5%8C%88%E7%89%99%E5%88%A9%E7%AE%97%E6%B3%95)，更严谨的来说：KM 算法是加权后的匈牙利算法，但一般似乎都将匈牙利算法默指 KM 算法...这里我还是分开描述，匈牙利是匈牙利，KM 是 KM

对于指派问题的通俗的问题描述：n 项任务、对应 n 个人承担，第 i 个人做第 j 项任务的代价为 $c_{ij}≥0$，则应指派哪个人完成哪项任务，使完成效率最高？而抽象的问题描述为：**解决二分图的最大匹配问题**，看来需要多学一点图论的知识啊...

但我发现网上很多讲解不是以计算机的思维来做的，什么画圆画勾啊...完全是应试技巧，就不要继续看了

可以先看看 [知乎](https://zhuanlan.zhihu.com/p/62981901) 有一个对匈牙利和 KM 的直观的感受，算法的证明暂时先不要细究了

先说匈牙利算法总结一下思想就是：先直接匹配，遇到冲突过后递归解决。关于递归的整体思路可以看这两篇 [知乎-匈牙利算法](https://zhuanlan.zhihu.com/p/208596378)，[知乎-KM 算法](https://zhuanlan.zhihu.com/p/214072424) 里这句话说的很好：**KM 算法实际上就是想了个办法（降低要求，扩大连接），将问题转换成了匈牙利算法可以解决的形式**

[zhihu1](https://zhuanlan.zhihu.com/p/307751815)

[zhihu2](https://zhuanlan.zhihu.com/p/459758723)

## 卡尔曼滤波

[徐亦达机器学习：Kalman Filter 卡尔曼滤波](https://www.bilibili.com/video/BV1TW411N7Hg?p=1)

[白板推导-卡曼滤波](https://www.bilibili.com/video/BV1zW411U7fa?p=1)

还需要一个实践篇...

https://zhuanlan.zhihu.com/p/58675854

