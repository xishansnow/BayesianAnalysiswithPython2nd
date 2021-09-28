---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.12.0
kernelspec:
  display_name: Python 3
  language: python3
  name: python3
---

#  附录 C：高斯过程

【原文】Jie Wang (2020). An Intuitive Tutorial to Gaussian Processes Regression. https://arxiv.org/abs/2009.10862

【摘要】本教程旨在提供对高斯过程回归（Gaussian processes regression, GPR）的直观理解。高斯过程回归模型因其表达方式的灵活性和内涵的不确定性预测能力而广泛用于机器学习应用中。本文首先解释了构建高斯过程的基本概念，包括多元正态分布、核、非参数模型、联合和条件概率等。然后，简明描述了 GPR 以及标准 GPR 算法的实现。除了标准 GPR，本文还审查了目前最先进的高斯过程算法软件包。

---

<style>p{text-indent:2em;2}</style>

## 1 引言

高斯过程模型是一种概率监督机器学习框架，已广泛用于回归和分类任务。高斯过程回归 (GPR) 模型可以结合先验知识（核）进行预测，并提供预测的不确定性度量 [11]。高斯过程模型是由计算机科学和统计学界开发的一种监督学习方法。具有工程背景的研究人员经常发现很难清楚地了解它。要理解 GPR，即使只有基础知识也需要了解多元正态分布、核、非参数模型以及联合和条件概率。

在本教程中，我们将提供 GPR 的简洁易懂的解释。我们首先回顾了 GPR 模型所基于的数学概念，以确保读者有足够的基础知识。为了提供对 GPR 的直观理解，图表被积极使用。为生成绘图而开发的代码在 https://github.com/jwangjie/Gaussian-Processes-Regression-Tutorial 处提供。

## 2 数学基础

本节回顾了理解 GPR 所需的基本概念。从高斯（正态）分布开始，然后解释多元正态分布（ MVN ）、核、非参数模型以及联合和条件概率理论。在回归任务中，给定一些观察到的数据点，我们的目的是拟合一个函数来表示这些数据点，然后使用该函数对新数据点进行预测。例如：对于 `图 1(a)` 所示的一组给定观测数据点，理论上能够拟合这些数据点的函数有无数个。在 `图 1（b）` 中， 示例性地展示了其中五个可能的函数。在 GPR 中，高斯过程正是通过定义这些函数（无限个）的分布，来处理回归问题的 [8]。

```{code-cell}
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.format'] = 'pdf'

rom scipy.stats import norm

# Plot 1-D gaussian
n = 1         # n number of independent 1-D gaussian 
m= 1000       # m points in 1-D gaussian 
f_random = np.random.normal(size=(n, m)) 
# more information about 'size': https://www.sharpsightlabs.com/blog/numpy-random-normal/ 
#print(f_random.shape)

for i in range(n):
    #sns.distplot(f_random[i], hist=True, rug=True, vertical=True, color="orange")
    sns.distplot(f_random[i], hist=True, rug=True, fit=norm, kde=False, color="r", vertical=False)

#plt.title('1000 random samples by a 1-D Gaussian')

plt.xlabel(r'$x$', fontsize = 16)
plt.ylabel(r'$P_X(x)$', fontsize = 16)

# plt.show()
# plt.savefig('1d_random.png', bbox_inches='tight', dpi=600)
plt.savefig('1d_random')
```

### 2.1 高斯分布

当某个随机变量 $X$ 的概率密度函数呈下式时，我们称其服从均值为 $μ$ 、方差为例$σ^2$ 的高斯分布：

$$
P_X(x) = \frac{1}{\sqrt{2 \pi} \sigma} exp{\left(-\frac{{\left(x - \mu \right)}^{2}}{2 \sigma^{2}}\right)}
$$

其中， $X$ 表示随机变量， $x$ 表示变量实值。$X$ 的高斯（或正态）分布通常表示为 $ P(x) ~ \sim\mathcal{N}(\mu, \sigma^2)$ 。

下图为 $1-D$ 高斯密度函数。我么从该分布中随机采样了 `1000` 个样本点，并将其绘制在 $x$ 轴上。
