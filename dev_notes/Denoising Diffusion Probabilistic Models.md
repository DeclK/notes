# Denoising Diffusion Probabilistic Models

[DDPM arxiv](https://arxiv.org/pdf/2006.11239)

[Understanding Diffusion Models: A Unified Perspective arxiv](https://arxiv.org/pdf/2208.11970)

[bilibili-扩散模型 - Diffusion Model【李宏毅2023】](https://www.bilibili.com/video/BV14c411J7f2) [PPT1](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2023-course-data/DiffusionModel%20(v2).pdf) [PPT2](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2023-course-data/StableDiffusion%20(v2).pdf) [PPT3](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2023-course-data/DDPM%20(v7).pdf) [course page](https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php)

本文档根据李宏毅视频整理得到，希望回答以下问题：

1. How to train and inference a diffusion model
2. What are the basic mathmatics behind it
3. How SOTA models improve the original diffusion model
4. What are the insights of diffusion can give us

## Notaions in Probability

- $x\sim p(·)$ 代表着 $x$ 是一个随机变量服从于概率分布 $p$，在论文中，也常常用该 notation 来表示在 $p$ 分布中采样 $x$
- $E_{x\sim p(x)}$ 代表着求解变量 $x$ 的期望，并且该变量服从概率分布 $p(x)$，有时候也会用 $E_{p(x)}$​ 来简写该过程
- $N(x;\mu, \sigma^2)$ 来表示一个高斯分布，其中随机变量为 $x$，其均值和方差分别为 $\mu, \sigma^2$，有时候也会省略掉 $x$，直接写作 $N(\mu, \sigma^2)$。对于一个多维高斯分布使用如下 notation: $N(x;\mu,\Sigma)$，其中 $x,\mu$ 都是多维向量，而 $\Sigma$ 为协方差矩阵 (covariance matrix)

## An intuitive perspective of DDPM

<img src="Denoising Diffusion Probabilistic Models/image-20241206102950079.png" alt="image-20241206102950079" style="zoom:67%;" />

看图说话：DDPM 其实就是一个从随机噪声开始，不断去噪直到生成目标对象的过程。那到底会生成什么样的图像呢？根据上面的示意图，我们其实并没有给去噪模型任何的信息，只给了一个初始噪声就开始让其进行去噪，所以理论上会生成任意的图像。而现在 diffusion model 在 text-to-image 领域应用非常多，所以如果想要生成指定的图像，我们需要给 diffusion model 加入额外的信息，例如文字：

<img src="Denoising Diffusion Probabilistic Models/image-20241206161019438.png" alt="image-20241206161019438" style="zoom: 67%;" />

## How to inference DDPM

- Algorithm

  <img src="Denoising Diffusion Probabilistic Models/image-20241205155625447.png" alt="image-20241205155625447" style="zoom:80%;" />

  Notation of the algorithm

  - $\alpha_t, \bar{\alpha_t}$​

    二者都是超参数，由我们自己决定。实际上这产生于另一个序列 $\beta_1,\beta_2,...\beta_t$，这个序列代表了我们每一个 step 中噪声所占的比例，是一个递增的序列，并且属于 0~1 之间。简单来说，随着 forward step 的加深，我们向原图中加的噪声会越来越大，即噪声所占的比例会越来越大
    $$
    \alpha_t = 1-\beta_t\\
    \bar{\alpha_t}=\alpha_1\alpha_2...\alpha_t
    $$

  - $\epsilon_\theta$​

    其实就是 neural network，其参数用 $\theta$ 表示，输入为 $(x_t,t)$，该网络的作用就是根据输入图像和 time step t，预测出加入到该图像的噪声 $\epsilon$，然后用输入图像减去该噪声就能够获得去噪图像

  - $\sigma_t$

    仍然也是一个超参数，代表每一个 timestep t 额外加入的 variance

- Algorithm 示意图 (at time step 999)

  <img src="Denoising Diffusion Probabilistic Models/image-20241206102258158.png" alt="image-20241206102258158" style="zoom:80%;" />

## How to train DDPM

- Algorithm

  <img src="Denoising Diffusion Probabilistic Models/image-20241205155602678.png" alt="image-20241205155602678" style="zoom: 80%;" />

  DDPM 的训练算法也非常的简洁，简单来看就是用网络 $\epsilon_\theta$ 去预测加入到样本中的噪声，希望这个预测噪声和真实加入的噪声足够的接近。但其实这个训练过程暗藏玄🐔，里面这些系数到底是什么意思呢？为什么在推理的时候似乎是一步一步地去噪，而这个过程在训练里没有呢？这些都需要在之后的数学推导中回答！

## SOTA diffusion framework

- Overall Framework

  现今 SOTA 的 text-to-image 模型都可以分为3大步骤：

  1. 使用 text encoder 将文字进行 encode，生成 text feature
  2. 使用 generation model 对噪声进行去噪。此时需要将 text feature 作为输入，经过模型后输出一个中间产物，该产物可以是 feature map，或者压缩的图片
  3. 对中间产物进行 decode，生成最终的图像

  <img src="Denoising Diffusion Probabilistic Models/image-20241206163249880.png" alt="image-20241206163249880" style="zoom:67%;" />

  这三个组件：text encoder, generation model, decoder 通常都是分开训练的。但是 generation model 是依赖于 decoder 的，因为 decoder 必须要认识 generation model 所产生的特征图。这实际上在训练 decoder 时，我们还训练了一个 auto encoder 来产生这个 latent representations (feature map)，然后通过向 latent representation 加入 noise 训练 generation model

- Text encoder size 对于图像生成质量非常重要，而 vision encoder size 影响较小

  <img src="Denoising Diffusion Probabilistic Models/image-20241206161343570.png" alt="image-20241206161343570" style="zoom: 50%;" />

  FID Frechet Inception Distance 就是用来评价所生成的图像集与目标图像集之间的距离，FID 应该越小越好。可以看到随着 T5 模型的增加，FID 曲线是向着右下角移动，说明生成图像有显著改善

  <img src="Denoising Diffusion Probabilistic Models/image-20241206161802419.png" alt="image-20241206161802419" style="zoom: 50%;" />

- 课程还介绍了 Stable Diffusion & DALL-E，我就不整理了

## From ELBO to VAE

[ELBO wiki](https://en.wikipedia.org/wiki/Evidence_lower_bound)我能够从 EM algorithm 比较顺利地切入到 VAE 当中

- EM algorithm

- Variational Auto Encoder

  VAE 可以说是在早期的图像生成领域中很常用的方法，如果理解了 VAE 相信理解 diffusion model 也是更简单的

## From VAE to Diffusion



## Fundamental Maths

总结理解 VAE & Diffusion model 所需要的基础数学，主要就是贝叶斯理论（Bayesian Theorem）以及相关的概率论基础

- Bayesian Theorem [wiki](https://en.wikipedia.org/wiki/Bayes%27_theorem)

  > Bayes' theorem (alternatively Bayes' law or Bayes' rule, after [Thomas Bayes](https://en.wikipedia.org/wiki/Thomas_Bayes)) gives a mathematical rule for inverting [conditional probabilities](https://en.wikipedia.org/wiki/Conditional_probability), allowing one to **find the probability of a cause given its effect.**

  上面就是 wiki 的第一句话，是对贝叶斯理论的高度总结：find the probability of a cause given its effect，给定这些现象来寻找原因。这个功能是一个非常强大的功能，但其公式却相当的简单
  $$
  P(A|B) = \frac{P(B|A)P(A)}{P(B)}
  $$
  这里的 A 和 B 都是任意的随机事件，并且 P(B) 不为 0。这里的 A 和 B 看起来非常的抽象，如何对应到实际应用当中？在现实中，A 通常用来表示 Hypothesis，即我们的假设，B 通常用来表示 Evidence，即发生的现象，所以也通常看到下方的字母表示
  $$
  P(H|E) = \frac{P(E|H)P(H)}{P(E)}
  $$
  为了更进一步理解，我们将这几个概率表示的意义写作如下：

  1. $P(H)$，Prior，先验。我们通常称 hypothesis 为先验
  2. $P(E)$，Evidence。名称没有变化
  3. $P(E|H)$，Likelihood，似然。基于 hypothesis 所得出的事件概率即为似然
  4. $P(H|E)$​，Posterior，后验。基于 evidence 所更新的 hypothesis 

  此时贝叶斯公式的功能变得更加具象起来：根据事实来更新我们的假设。那么这个假设到底是什么？事实又是什么？用一个更加具象的例子表示：
  $$
  P(\theta|X) = \frac{P(X|\theta)P(\theta)}{P(X)}
  $$
  这下子符号似乎看起来更熟悉了：$\theta$ 就是模型参数，$X$ 就是数据样本，我们的任务就是用模型来估计样本的分布。其中我们用 $\theta$ 来表示了我们的模型参数，**其实这是一个 over simplification，其中还包含了我们的建模假设**，例如：这个模型是一个高斯分布，且数据的分布符合高斯分布，高斯分布的参数为 $\theta$。所以我们能够非常轻松地根据这个假设计算得到 $P(X|\theta)$​，直接根据高斯分布来算就行了

  理解这个公式最常举的例子就是抛硬币的例子：

  我们将上述的变量都进行具体的定义

  - $\theta$ 是一个随机变量，其决定了硬币为正面的概率
  - $X$ 是实验结果，我们的实验结果为抛10次硬币，有7次为正面，3次为反面

  首先我们需要有一个初始猜测：$\theta$ 到底是个什么分布？由于一开始我们没有任何信息，不如假设为最简单的 uniform distribution (均匀分布) $P(\theta)=1, \theta\in [0,1]$

  我们可以很容易根据我们的假设计算得到我们的 likelihood 
  $$
  P(X|\theta) = C_{10}^7·\theta^7(1-\theta)^3
  $$
  OK，现在比较难的是求得 $P(X)$，这里需要使用全概率公式，不过好在我们也能够求到
  $$
  P(X) = \int_0^1P(X|\theta)P(\theta)d\theta
  $$
  最后求得 $P(X)≈0.1$，是一个常数。所以将所有的结果导入到贝叶斯公式当中，就可以得到
  $$
  P(\theta|X) = \frac{C_{10}^7·\theta^7(1-\theta)^3·1}{0.1}
  $$
  可以看到，现在我们的 $\theta$ 被更新为了一个 [beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)，相比于之前的 uniform distribution 改动不小。如果实验量足够多，那么我们所算出的 $\theta$ 应该趋近于一个 delta 分布，收敛到 $\frac{heads}{trials}$​ 这个值，并且无论你的初始 $P(\theta)$ 是多少，都会收敛到最终这个分布上。所以这给我一个启发：

  无论初始 $P(\theta)$ 分布是怎样的，所收获到的 $P(X|\theta)$ 更新都是一样的，这是由我们的建模所决定的，即我们的模型假设：$\theta$ 决定了硬币为正面的概率。并且如果更新的 likelihood 足够强，那么将完全覆盖之前的先验，以 likelihood 为基准

  另外再提一点：我们在计算 $P(X)$ 的时候能使用这个全概率公式，仍然是在我们的模型假设之下的。可以看到我们的模型假设基本上贯穿了所有的计算过程，一个错误的模型假设，即使计算再多的参数，也无法获得好的后验概率

  这是一个非常非常简单的例子，简单到我想问：为什么不一开始我们就算 $\frac{heads}{trials}$ 这个值作为我们最终的 $\theta$ 分布呢🤔 但是有几个重要的启发

  1. $P(X)$ is really hard to calculate
  2. Binominal distribution is too simple

  Prompt: a very weird shaped dice that you can not easily know what is the outcome when you flip it 

  <img src="Denoising Diffusion Probabilistic Models/image-20250124160132877.png" alt="image-20250124160132877" style="zoom:50%;" />

- Joint distribution and conditional distribution

  

- Marginalize

- How to understand latent variable?

  From the Understanding Diffusion Models: A Unified Perspective gives an intuitive philosophy

- Chain of rules in probability

  https://en.wikipedia.org/wiki/Chain_rule_(probability)

- Markov Chain Monte Carlo (MCMC)

- Reparameterization trick

  https://en.wikipedia.org/wiki/Reparameterization_trick

- [Lil's log on diffusion](https://lilianweng.github.io/posts/2021-07-11-diffusion-models)

- How does variational inference connected with ELBO?

  [Evidence lower bound - Wikipedia](https://en.wikipedia.org/wiki/Evidence_lower_bound#Variational_Bayesian_inference)

  These words are extremely important to anwer the question: what does these parameter is trying to model? and how to compute these values actually

  > This defines a family of joint distributions pθ over (X,Z). It is very easy to sample (x,z)∼pθ: simply sample z∼p, then compute fθ(z), and finally sample x∼pθ(⋅|z) using fθ(z).

  In general, it's impossible to perform the integral pθ(x)=∫pθ(x|z)p(z)dz, forcing us to perform another approximation.

## Question

- 为什么在 inference 采样的时候还要加入随机噪声？

  采样，just like sampling when generating tokens

- In the [material](https://arxiv.org/pdf/2208.11970), there are some p is $p_\theta(·)$, but some wihout $\theta$, juse $p(·)$, how to differenciate them?

  It seems that the $p(·)$ without the $\theta$​ means it is a prior distribution, which means we defined it manually at the very beginning, or let's say it is out hypothesis

- Explaining the square root in the $\sqrt{\alpha_t}$ when doing linear gaussian modeling

  This is to maintain the variance structure of origianl distribution

- How to optimize the first term of VAE $E_{z\sim q_{\phi}(z|x)}[\log{p_{\theta}(x|z)}]$​

  we use the network to produce the mean of of gaussian, what about variance?
  
- Why it is hard to compute the $p_\theta(x)$

  [my-chat](https://chatgpt.com/share/675726ae-8364-800a-b33d-0ed508bc3eaf)
  
- What is variational?

  变分这个概念似乎非常

- 什么是生成模型，什么是判别模型，他们的概念是什么？

- 求解 $P(X)$ 的方式除了上述的全概率公式，是否还存在 MCMC 的方法？
