# Denoising Diffusion Probabilistic Models

[DDPM arxiv](https://arxiv.org/pdf/2006.11239)

[Understanding Diffusion Models: A Unified Perspective arxiv](https://arxiv.org/pdf/2208.11970)

[bilibili-扩散模型 - Diffusion Model【李宏毅2023】](https://www.bilibili.com/video/BV14c411J7f2) [PPT](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2023-course-data/DDPM%20(v7).pdf)

本文档根据李宏毅视频整理得到，希望回答以下问题：

1. How to train and inference a diffusion model
2. What are the basic mathmatics behind it
3. How SOTA models improve the original diffusion model
4. What are the insights of diffusion can give us

## Notaions in Probability

- $x\sim p(·)$ 代表着 $x$ 是一个随机变量服从于概率分布 $p$，在论文中，也常常用该 notation 来表示在 $p$ 分布中采样 $x$
- $E_{x\sim p(x)}$ 代表着求解变量 $x$ 的期望，并且该变量服从概率分布 $p(x)$，有时候也会用 $E_{p(x)}$ 来简写该过程

## An intuitive perspective of DDPM



## How to inference DDPM

- Algorithm

  <img src="Denoising Diffusion Probabilistic Models/image-20241205155625447.png" alt="image-20241205155625447" style="zoom:80%;" />

  explain the notation of the algorithm

  - $\alpha_t, \bar{\alpha_t}$​

    二者都是超参数，由我们自己决定。实际上这产生于另一个序列 $\beta_1,\beta_2,...\beta_t$，这个序列代表了我们每一个 step 中噪声所占的比例，是一个递增的序列，并且属于 0~1 之间。简单来说，随着 forward step 的加深，我们向原图中加的噪声会越来越大，即噪声所占的比例会越来越大
    $$
    \alpha_t = 1-\beta_t\\
    \bar{\alpha_t}=\alpha_1\alpha_2...\alpha_t
    $$

  - $\epsilon_\theta$​

    其实就是 neural network，其参数用 $\theta$ 表示

- 示意图

## How to train DDPM

- Algorithm

  <img src="Denoising Diffusion Probabilistic Models/image-20241205155602678.png" alt="image-20241205155602678" style="zoom: 80%;" />

## SOTA diffusion framework

text encoder 非常重要 ep2 06:00

FID Frechet Inception Distance ep2 7:30

- Stable Diffusion

- DALL-E

- 需要训练 auto encoder 来产生 latent representations ep2 14:00, 然后通过向 latent representation 加入 Noise 训练 noise predictor ep2 16:00
- 

## Maths

- variational inference

  > 

  