# VAE-MNIST

一个基于 MNIST 数据集的 Variational AutoEncoder（VAE）最小实现，用于理解生成模型中的编码、解码、重参数化以及潜变量空间（latent space）的基本行为。

本项目适合作为生成模型入门实验，通过训练一个简单的 VAE，观察潜变量分布与生成结果之间的关系。


## 项目描述

本项目主要完成以下内容：

- 使用 `vae_mnist.py` 在 MNIST 数据集上训练一个 VAE；
- 学习编码器（Encoder）和解码器（Decoder）的基本结构；
- 理解重参数化技巧（Reparameterization Trick）；
- 观察 latent space 对生成结果的影响。

训练完成后，可以利用训练好的模型进行潜变量分析和生成实验。


## 安装与环境配置

建议使用 Python 3.8 及以上版本。

### 安装依赖

```bash
pip install torch torchvision tqdm numpy matplotlib
````


## 运行训练

执行以下命令开始训练：

```bash
python vae_mnist.py
```

训练过程中将自动下载 MNIST 数据集，并输出训练结果。


## 实验任务

跑通 `vae_mnist.py` 后，从下面六个问题中挑选三个感兴趣的问题，尝试使用训练好的模型做测试并回答。

### 可选实验问题

1. 给两张图片的 latent 做插值，观察生成结果是否连续平滑。

2. 固定一张图片的 latent 表示，每次只改变一个维度，观察这个维度是否对应某种可见变化。

3. 对同一个 latent 向量加入不同强度的小噪声，观察生成结果的稳定性。

4. 随机采样时调整噪声尺度，比较不同尺度下生成样本的质量和多样性。

5. 比较像素空间插值和 latent 空间插值，观察哪种方式生成结果更自然。

6. 将输入图像局部遮挡后送入 VAE，观察模型是否能够补出合理结果。

## 说明

建议训练完成后使用Jupyter Notebook调试模型，并结合实验现象进行分析。


