---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python3
  name: python3
---

#  附录 D： 高斯过程（II）

<style>p{text-indent:2em;2}</style>

高斯过程 [Gaussian Processes](https://en.wikipedia.org/wiki/Gaussian_process) 是概率论和数理统计中随机过程的一种，是多元高斯分布的扩展，被应用于机器学习、信号处理等领域。本文对高斯过程进行公式推导、原理阐述、可视化以及代码实现，介绍了以高斯过程为基础的高斯过程回归 Gaussian Process Regression 基本原理、超参优化、高维输入等问题。

<br>

## C.1 一元高斯分布

我们从最简单最常见的一元高斯分布开始，其概率密度函数为

$$
p(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp (-\frac{(x-\mu)^2}{2\sigma^2}) \tag{1}
$$

其中 $\mu$ 和 $\sigma$ 分别表示均值和方差，这个概率密度函数曲线画出来就是我们熟悉的钟形曲线，均值和方差唯一地决定了曲线的形状。

<br>

## C.2 多元高斯分布

从一元高斯分布推广到多元高斯分布，假设各维度之间相互独立，则有联合分布：

$$
p(\boldsymbol{x})=p(x_1, x_2, ..., x_n) = \prod_{i=1}^{n}p(x_i)=\frac{1}{(2\pi)^{\frac{n}{2}}\sigma_1\sigma_2...\sigma_n}\exp \left(-\frac{1}{2}\left [\frac{(x_1-\mu_1)^2}{\sigma_1^2} + \frac{(x_2-\mu_2)^2}{\sigma_2^2} + ... + \frac{(x_n-\mu_n)^2}{\sigma_n^2}\right]\right) \tag{2}
$$

<br>

其中 $\mu_1, \mu_2, \cdots$ 和 $\sigma_1, \sigma_2, \cdots$ 分别是第 1 维、第 2 维... 的均值和方差。可以将上式表示为向量和矩阵形式，令：
<br>
$$
\boldsymbol{x - \mu}=[x_1-\mu_1, \ x_2-\mu_2,\ … \ x_n-\mu_n]^T \\
K = \begin{bmatrix}
    \sigma_1^2 & 0 & \cdots & 0\\
    0 & \sigma_2^2 & \cdots & 0\\
    \vdots & \vdots & \ddots & 0\\
    0 & 0 & 0 & \sigma_n^2
    \end{bmatrix}
$$

则有：
$$\sigma_1\sigma_2...\sigma_n = |K|^{\frac{1}{2}} $$

$$\frac{(x_1-\mu_1)^2}{\sigma_1^2} + \frac{(x_2-\mu_2)^2}{\sigma_2^2} + ... + \frac{(x_n-\mu_n)^2}{\sigma_n^2}=(\boldsymbol{x-\mu})^TK^{-1}(\boldsymbol{x-\mu}) $$

代入式（2）得到向量和矩阵形式：
<br>
$$
p(\boldsymbol{x}) = (2\pi)^{-\frac{n}{2}}|K|^{-\frac{1}{2}}\exp \left( -\frac{1}{2}(\boldsymbol{x-\mu})^TK^{-1}(\boldsymbol{x-\mu}) \right) \tag{3}
$$

其中 $ \boldsymbol{\mu} \in \mathbb{R}^n $ 是均值向量， $ K \in \mathbb{R}^{n \times n} $ 为协方差矩阵。

注意：式中假设了各维度之间相互独立，因此 $ K $ 是对角矩阵。如果各维度变量存在相关，则上述形式仍然成立，但此时协方差矩阵 $ K $ 不再是对角矩阵，而是半正定的对称矩阵。

式（3）通常也简写为：
$$
x \sim \mathcal{N}(\boldsymbol{\mu}, K)
$$

<br>

## C.3 无限元高斯分布

在多元高斯分布基础上进一步扩展，假设有无限多个维度的随机变量呢？我们用一个例子来展示从一元高斯分布到无限元高斯分布的扩展过程（来源：[MLSS 2012: J. Cunningham - Gaussian Processes for Machine Learning](http://www.columbia.edu/~jwp2128/Teaching/E6892/papers/mlss2012_cunningham_gaussian_processes.pdf)）。

（1）一元高斯分布图解

假设在周一到周四的每天 7:00 测 1 次心率，共测得下图所示 4 个点，其可能的高斯分布如图中高瘦的曲线所示。这是一元高斯分布，只有每天 7: 00 心率这一个维度。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/GP-20211010-14-04-29-efff.webp)

（2）二元以及多元高斯分布

现在不仅考虑在每天 7: 00 测一次心率，也在 8:00 时也做一次测量，这时就问题就由原来一个维度的随机变量，变成两个维度的随机变量。此时如果考察两个时间心率的联合概率分布，就会如下图的二元高斯分布图：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/GP-20211010-13-20-42-413b.webp)

上述过程可以进一步扩展到每天每个整点均测量一次心率，则每天当中的整点测量值一起，构成了一个 24 维的高斯概率分布（此时已经很难利用计算机做可视化）。此时，周一到周四中每天的 24 个测量值，作为一个整体（可以视为一个离散函数），可以视为对该 24 维概率分布的一次采样。

（3）无限元的高斯分布

如果我们在每天的无限个时间点都做一次心率测量，则问题变成了下图所示的情况。注意图中的横轴是测量时间，不同颜色的每个线条均代表了在无限元的高斯分布上做的一次采样（或理解为一次采样产生了无限多个位置不重复的样本点），这种由无限多个不重复的样本点构成的采样结果看起来具有函数的表现形式，因此通常也将无限元的高斯分布称为**函数的分布**（即无限元的高斯分布的一次采样对应了一个函数）。这个无限元的高斯分布就是“高斯过程”。按照字面意思可以理解为：一元、二元和多元高斯分布对应于“点的分布” ，而无限元的高斯过程对应于**函数的分布**。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/GP-20211010-14-09-22-27de.webp)



高斯过程正式地定义为：

对于所有可能的  $ \boldsymbol{x} = [x_1, x_2, \cdots, x_n] $，如果 $ f(\boldsymbol{t})=[f(x_1), f(x_2), \cdots, f(x_n)] $ 都服从多元高斯分布，则称 $f$ 是一个高斯过程，表示为：

$$
f(\boldsymbol{x}) \sim \mathcal{N}(\boldsymbol{\mu}(\boldsymbol{x}), \kappa(\boldsymbol{x},\boldsymbol{x})) \tag{4}
$$

（1） $\boldsymbol{\mu}(\boldsymbol{x}): \mathbb{R^{n}} \rightarrow \mathbb{R^{n}} $ 被称为均值函数（Mean function），返回各维度的均值。 均值函数可以是 $\mathbb{R} \rightarrow  \mathbb{R}$ 或 $\mathbb{R}^D \rightarrow \mathbb{R}$  的任何函数；通常取  $\boldsymbol{\mu}(\boldsymbol{x})=0, \forall x$ 。

（2） $ \kappa(\boldsymbol{x},\boldsymbol{x}) : \mathbb{R^{n}} \times \mathbb{R^{n}} \rightarrow \mathbb{R^{n\times n}} $ 被称为协方差函数（ Covariance Function）或核函数（Kernel Function），返回各维度随机变量两两之间的协方差，多个变量一起构成协方差矩阵。 $\kappa$ 可以是任何有效的 $\mathbb{R}^D \times \mathbb{R}^D  \rightarrow \mathbb{R}$ 的 Mercer 核。

```{note}

Mercer 定理： **任何半正定的函数都可以作为核函数。**

半正定函数的定义：在训练数据集 $(x_1,x_2,...x_n)$ 基础上， 定义一个元素值为 $a_{ij} = f(x_i,x_j)$ 的 $n \times n$ 矩阵，如果该矩阵为半正定，那么函数 $f(x_i,x_j)$ 就被称为半正定函数。

Mercer 定理核函数的充分条件，而非必要条件。也就是说还有不满足 Mercer 定理的函数也可以是核函数。常见的核函数有高斯核、多项式核等，基于这些核函数，可以通过核函数的性质（如对称性等）进一步构造出新的核函数。支持向量机（SVM） 是目前核方法应用的经典模型。

```

根据上述定义，一个高斯过程由一个均值函数和一个协方差函数唯一地定义，并且 **一个高斯过程的有限维度的子集都服从多元高斯分布**  。

<br>

```{note}
一元高斯： 单个实数值变量上的分布
多元高斯： 多个实数值变量上的联合分布
高斯过程： 无限个实数值变量构成的函数上的分布
```

<br>

## C.4 核函数（协方差函数）

核函数是一个高斯过程的核心，核函数决定了一个高斯过程的性质。核函数在高斯过程中主要用于衡量任意两个点（在高斯过程语境中，指任意两个随机变量）之间的相似性程度，其离散化表现形式为协方差矩阵（或相关系数矩阵）。目前最常用的核函数是二次高斯核函数，也称为径向基函数（ [RBF](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)）。其基本形式如下。其中 $ \sigma $ 和 $ l $ 是高斯核的超参数。

$$
K(x_i,x_j)=\sigma^2\exp \left( -\frac{\left \|x_i-x_j\right \|_2^2}{2l^2}\right)
$$

以上高斯核函数的 `Python` 实现如下：

```python
import numpy as np

def gaussian_kernel(x1, x2, l=1.0, sigma_f=1.0):
    """Easy to understand but inefficient."""
    m, n = x1.shape[0], x2.shape[0]
    dist_matrix = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            dist_matrix[i][j] = np.sum((x1[i] - x2[j]) ** 2)
    return sigma_f ** 2 * np.exp(- 0.5 / l ** 2 * dist_matrix)

def gaussian_kernel_vectorization(x1, x2, l=1.0, sigma_f=1.0):
    """More efficient approach."""
    dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)

x = np.array([700, 800, 1029]).reshape(-1, 1)
print(gaussian_kernel_vectorization(x, x, l=500, sigma=10))
```

输出的协方差矩阵为：

```
[[100.    98.02  80.53]
 [ 98.02 100.    90.04]
 [ 80.53  90.04 100.  ]]
```

<br>

## C.5 高斯过程可视化

下图是高斯过程的可视化，其中蓝线是高斯过程的均值，浅蓝色区域 95% 置信区间（即每个随机变量的方差，由协方差矩阵的对角线元素确定），每条虚线（函数）代表高斯过程的一次采样（此处用 100 维模拟连续的无限维）。左上角第一幅图是零均值高斯过程先验的四个样本，分别用四种不同颜色的曲线表示。后面几幅图分别展示了观测到新数据点时，不同高斯过程样本的更新过程（即更新高斯过程后验的均值函数和协方差函数）的过程。从图中可以看到：**随着数据点（随机变量）数量的增加，所有高斯过程先验的样本最终都会收敛至真实值附近**。 可以做出更进一步的设想： **即使设置不同的高斯过程先验，随着数据点（随机变量）数量的增加，所有高斯过程先验的样本最终也都会收敛至真实值附近**。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/GP-20211010-16-15-26-f373.webp)

接下来用公式推导上图的过程。

**（1）高斯过程先验的设置**

将高斯过程先验表示为 $f(\boldsymbol{x}) \sim \mathcal{N}(\boldsymbol{\mu}_{f}, K_{ff}) $，对应左上角第一幅图。

**（2）高斯过程后验的更新**

现在观测到一些数据 $ (\boldsymbol{x^{*}}, \boldsymbol{y^{*}}) $ ，则根据模型假设， $ \boldsymbol{y^{*}} $ 与 $f(\boldsymbol{x})$ 服从联合高斯分布：

$$
\begin{bmatrix}
    f(\boldsymbol{x})\\
    \boldsymbol{y^{*}}\\
    \end{bmatrix} \sim \mathcal{N} \left(
    \begin{bmatrix}
    \boldsymbol{\mu_f}\\
    \boldsymbol{\mu_y}\\
    \end{bmatrix},
    \begin{bmatrix}
    K_{ff} & K_{fy}\\
    K_{fy}^T & K_{yy}\\
    \end{bmatrix}
    \right)
$$

 其中 $K_{ff} = \kappa(\boldsymbol{x}, \boldsymbol{x}) $ ，$ K_{fy}=\kappa(\boldsymbol{x}, \boldsymbol{x^{*}}) $，$ K_{yy} = \kappa(\boldsymbol{x^{*}},\boldsymbol{x^{*}})$ ，则有：

$$
f \sim \mathcal{N}(K_{fy}^{T}K_{ff}^{-1}\boldsymbol{y}+\boldsymbol{\mu_f},K_{yy}-K_{fy}^{T}K_{ff}^{-1}K_{fy}) \tag{5}
$$

公式（5）表明：给定数据 $(\boldsymbol{x^{*}}, \boldsymbol{y^{*}})$ 之后，函数 $ f $ 仍然服从高斯过程分布，具体推导可见 [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)。

从公式（5）可以看出一些有趣的性质： 

- **后验的均值函数 $K_{fy}^{T}K_{ff}^{-1}\boldsymbol{y}+\boldsymbol{\mu_f}$ 实际上是观测变量 $ \boldsymbol{y^*} $ 的一个线性函数** ；
- **后验的协方差函数** $K_{yy}-K_{fy}^{T}K_{ff}^{-1}K_{fy}$ 的第一部分 $ K_{yy} $ 是先验的协方差，**减掉的那一项表示观测到数据后函数分布不确定性的减少**，当第二项接近于 0时，说明观测数据后不确定性几乎不变，当第二项非常大时，说明不确定性降低了很多。

**（3）概率视角的解释**

公式（5）其实就是高斯过程回归的基本公式，从贝叶斯视角来看，首先设置一个高斯过程先验  $f(\boldsymbol{x}) \sim \mathcal{N}(\boldsymbol{\mu}_{f}, K_{ff}) $，然后在观测到一些数据后，基于先验和似然（此处的似然假设为 “**$ \boldsymbol{y^{*}} $ 与 $f(\boldsymbol{x})$ 服从联合高斯分布**”），计算得到高斯过程后验（即更新均值函数和协方差函数）。

<br>

## C.6 简单高斯过程回归实现

本节用代码实现一个简单的高斯过程回归模型。由于高斯过程回归是一种非参数化的方法，每次推断（inference）需要利用所有训练数据进行计算，因此没有显式的训练模型参数的过程，所以拟合（ fit）过程只需要将训练数据记录下来，实际的推断过程在预测（ predict）过程中进行。Python 代码如下

```{note}
注：参数模型的贝叶斯过程通常先为模型参数设置先验（或超先验），然后通过拟合学习（推断）出模型参数的后验分布，而后利用最大后验或贝叶斯模型平均得到预测值。而非参数模型没有确定数量的模型参数，其模型参数大多是根据训练数据动态生成的，因此非参数模型的学习过程和预测过程很多时候是纠缠在一起的。
```


```python
from scipy.optimize import minimize


class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)
        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)

        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)
```

```python
def y(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.cos(x) + np.random.normal(0, noise_sigma, size=x.shape)
    return y.tolist()

train_X = np.array([3, 1, 4, 5, 9]).reshape(-1, 1)
train_y = y(train_X, noise_sigma=1e-4)
test_X = np.arange(0, 10, 0.1).reshape(-1, 1)

gpr = GPR()
gpr.fit(train_X, train_y)
mu, cov = gpr.predict(test_X)
test_y = mu.ravel()
uncertainty = 1.96 * np.sqrt(np.diag(cov))
plt.figure()
plt.title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
plt.plot(test_X, test_y, label="predict")
plt.scatter(train_X, train_y, label="train", c="red", marker="x")
plt.legend()
```

结果如下图，红点是训练数据，蓝线是预测值，浅蓝色区域是 95% 置信区间。真实的函数是一个 cosine 函数，可以看到在训练数据点较为密集的地方，模型预测的不确定性较低，而在训练数据点比较稀疏的区域，模型预测不确定性较高。

![https://pic4.zhimg.com/80/v2-5237ebe25306a4a8a241be23893c96a7_720w.jpg](https://gitee.com/XiShanSnow/imagebed/raw/master/images/GP-20211010-13-21-09-42dc.webp)

<br>

## C.7 超参数优化

上文提到高斯过程是一种非参数模型，没有训练模型参数的过程，一旦核函数、训练数据给定，模型就被唯一地确定下来。但注意到核函数本身是有参数的（比如高斯核的参数  $ \sigma $ 和 $ l $  ），我们称为这种参数为模型的超参数（类似于 k-NN 模型中 k 的取值）。

核函数本质上决定了样本点相似性的度量方法，进而影响到整个函数的概率分布的形状。上面的高斯过程回归的例子中使用了 $ \sigma=0.2 $ 和 $ l=0.5 $ 的超参数，现在可以选取不同的超参数看看回归出来的效果有何不同。

![https://pic2.zhimg.com/80/v2-209ee2a5fefd5f1efdc91878328966c1_720w.jpg](https://gitee.com/XiShanSnow/imagebed/raw/master/images/GP-20211010-13-21-35-0d2d.webp)

从上图可以看出，$ l $ 越大函数更加平滑，同时训练数据点之间的预测方差更小，反之 $ l $ 越小则函数倾向于更加“曲折”，训练数据点之间的预测方差更大；$ \sigma $ 则直接控制方差大小，$ \sigma $ 越大方差越大，反之亦然。

如何选择最优的核函数参数 $ \sigma $ 和 $ l $ 呢？答案是：最大化在这两个超参数下 $ \boldsymbol{y} $ 出现的概率，通过 **最大化边缘对数似然（Marginal Log-likelihood）来找到最优的参数**，边缘对数似然表示为：

$$
\mathrm{log}\ p(\boldsymbol{y}|\sigma, l) = \mathrm{log} \ \mathcal{N}(\boldsymbol{0}, K_{yy}(\sigma, l)) = -\frac{1}{2}\boldsymbol{y}^T K_{yy}^{-1}\boldsymbol{y} - \frac{1}{2}\mathrm{log}\ |K_{yy}| - \frac{N}{2}\mathrm{log} \ (2\pi) \tag{6}
$$

具体实现中，在拟合（fit）方法中增加超参数优化这部分代码，最小化负边缘对数似然。

```python
from scipy.optimize import minimize


class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)

         # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            return 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, [self.params["l"], self.params["sigma_f"]],
                   bounds=((1e-4, 1e4), (1e-4, 1e4)),
                   method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]

        self.is_fit = True
```

将训练、优化得到的超参数、预测结果可视化如下图，可以看到最优的 $ l=1.2 $，$ \sigma_f=0.8 $ 。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/GP-20211010-13-21-44-6d38.webp)



这里用 scikit-learn 的 [GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor) 接口进行对比：

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# fit GPR
kernel = ConstantKernel(constant_value=0.2, constant_value_bounds=(1e-4, 1e4)) * RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
gpr.fit(train_X, train_y)
mu, cov = gpr.predict(test_X, return_cov=True)
test_y = mu.ravel()
uncertainty = 1.96 * np.sqrt(np.diag(cov))

# plotting
plt.figure()
plt.title("l=%.1f sigma_f=%.1f" % (gpr.kernel_.k2.length_scale, gpr.kernel_.k1.constant_value))
plt.fill_between(test_X.ravel(), test_y + uncertainty, test_y - uncertainty, alpha=0.1)
plt.plot(test_X, test_y, label="predict")
plt.scatter(train_X, train_y, label="train", c="red", marker="x")
plt.legend()
```

scikit-learn 得到结果为 $ l=1.2, \sigma_f=0.6 $，与我们优化的超参数有些许不同，可能是实现细节有所不同导致。

![https://pic2.zhimg.com/80/v2-5771db0caae19ef0f196f77a2fa33151_720w.jpg](https://gitee.com/XiShanSnow/imagebed/raw/master/images/GP-20211010-13-21-50-6f25.webp)

<br>

## C.8 多维输入

我们上面讨论的训练数据的输入都是一维（即每一个变量 $x_i$ 都是标量，主要用于时间序列分析 ）的，不过高斯过程可以直接扩展到多维输入（ $x_i$ 是 D 维向量，主要用于多变量影响分析）的情况，只需直接将输入维度增加即可。

```python
def y_2d(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.sin(0.5 * np.linalg.norm(x, axis=1))
    y += np.random.normal(0, noise_sigma, size=y.shape)
    return y

train_X = np.random.uniform(-4, 4, (100, 2)).tolist()
train_y = y_2d(train_X, noise_sigma=1e-4)

test_d1 = np.arange(-5, 5, 0.2)
test_d2 = np.arange(-5, 5, 0.2)
test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
test_X = [[d1, d2] for d1, d2 in zip(test_d1.ravel(), test_d2.ravel())]

gpr = GPR(optimize=True)
gpr.fit(train_X, train_y)
mu, cov = gpr.predict(test_X)
z = mu.reshape(test_d1.shape)

fig = plt.figure(figsize=(7, 5))
ax = Axes3D(fig)
ax.plot_surface(test_d1, test_d2, z, cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
ax.scatter(np.asarray(train_X)[:,0], np.asarray(train_X)[:,1], train_y, c=train_y, cmap=cm.coolwarm)
ax.contourf(test_d1, test_d2, z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=0.6)
ax.set_title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))
```


下面是一个二维输入数据的高斯过程回归，左图是没有经过超参优化的拟合效果，右图是经过超参优化的拟合效果。

![https://pic4.zhimg.com/80/v2-ce3895df03430d8b18b9b640243ab25f_720w.jpg](https://gitee.com/XiShanSnow/imagebed/raw/master/images/GP-20211010-13-21-55-bb7a.webp)

以上完整的的代码放在 [toys/GP](https://github.com/borgwang/toys/tree/master/math-GP)。

<br>

## C.9 高斯过程回归的优缺点

- 优点
  - （采用 RBF 作为协方差函数）具有平滑性质，能够拟合非线性数据；
  - 高斯过程回归天然支持获得**模型关于预测的不确定性**，可以直接输出预测点值的概率分布；
  - 通过最大化边缘似然的方式，高斯过程回归可以在不需要交叉验证的情况下给出比较好的正则化效果。

- 缺点
  - 高斯过程是非参数模型，每次的推断都需要对所有的数据点参与计算（矩阵求逆）。对于没有经过任何优化的高斯过程回归，$n$ 个样本点时间复杂度大概是 $ \mathcal{O}(n^3) $，空间复杂度是 $ \mathcal{O}(n^2) $，在数据量大的时候高斯过程变得 intractable；
  - 高斯过程回归中，先验是一个高斯过程分布，似然是高斯分布，根据高斯的共轭特性，其后验仍是一个高斯过程分布。在似然不服从高斯分布的问题中（如分类），需要对得到的后验进行近似使其仍可以是一个高斯过程；
  - 径向基函数（RBF）是最常用的协方差函数，但实际中通常需要根据问题和数据性质选择恰当的协方差函数。

<br>

## 参考资料

- [Carl Edward Rasmussen - Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)
- [MLSS 2012 J. Cunningham - Gaussian Processes for Machine Learning](http://www.columbia.edu/~jwp2128/Teaching/E6892/papers/mlss2012_cunningham_gaussian_processes.pdf)
- [Martin Krasser's blog- Gaussian Processes](http://krasserm.github.io/2018/03/19/gaussian-processes/)
- [scikit-learn GaussianProcessRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)

<br><br>
