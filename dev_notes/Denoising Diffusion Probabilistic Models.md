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
- $E_{x\sim p(x)}$ 代表着求解变量 $x$ 的期望，并且该变量服从概率分布 $p(x)$，有时候也会用 $E_{p(x)}$ 来简写该过程

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

## Maths

- VAE (Variational Auto Encoder) & Diffusion Model

  

## Question

- 为什么在 inference 采样的时候还要加入随机噪声？

