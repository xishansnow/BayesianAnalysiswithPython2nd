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

# 附录 G：模型平均

【摘要】`模型平均`是一种允许估计模型不确定性的方法，它可以提供比`模型选择`更好的估计和更可靠的置信区间。本文通过基于真实数据的示例来说明了 `模型平均` 的使用，讨论了其何时可能有用，并比较了频率主义和贝叶斯方法的区别。

【原文】Fletcher, David. Why Model Averaging? SpringerBriefs in Statistics. 2018.

<style>p{text-indent:2em;2}</style>

## 1 为什么做模型平均？

模型平均是一种在估计中允许模型不确定性的方法，它可以提供比模型选择更好的估计和更可靠的置信区间。其应用领域非常广泛，涉及经济、药理、气象、水文等诸多领域。选择模型平均方法的原因在于：

**（1）模型选择及点估计方法容易产生过于自信的模型。** 

在许多经典统计推断的理论中，参数估计通常面向单个模型，该模型大多是从一组候选模型中选择出来的最佳模型。在解释数据时，选择该最佳模型的过程经常会被忽略，很容易导致点估计偏差和精度高估。这种现象被称为 “无声的丑闻”，而许多研究人员仍然没有意识到这个问题。

**（2）模型平均是一种可以考虑模型不确定性的估计方法。** 

- 在频率主义框架中，模型平均通常来自于对每个候选模型估计值的加权平均，其中权重反映了模型估计的潜力值。模型权重可能基于 `Akaike 信息准则 (AIC)`、`交叉验证` 或`均方误差 (MSE)`。
- 在贝叶斯框架中，模型权重要么是模型为『真』的后验概率，要么是基于预测样本来确定，例如 `Watanabe-Akaike 信息准则 (WAIC)`  或 `交叉验证` 。
- 通常模型权重被限制在单位单纯形上，即非负且总和为 1。

**（3）模型平均也可以被视为一种平衡估计偏差和方差的手段。** 

从频率主义角度来看，就像模型选择一样，较小的模型通常会提供具有较大偏差的估计，而较大的模型会导致估计值具有较高的方差。因此，模型平均在计算置信区间时考虑了模型不确定性，从而产生了比模型选择方法更宽、更可靠的区间。有趣的是，频率主义的一些作者只专注于实现模型平均估计的偏差和方差之间的平衡，而其他人则认为模型平均仅仅是一个允许模型不确定性的手段而已。

> **模型选择、模型集成、模型组合和模型平均的区别：**
>
> - **模型选择**指从若干模型中选择出人们认为最优的那个模型，可视为对模型的点估计。由于评判方法、数据集等原因，模型选择很容易造成对模型过高的自信。
>
> - **模型平均**是为了避免单个模型过于自信的问题，对多个模型计算的结果进行加权平均以得到最终结果的方法，从其意义上来看，模型平均其实可视为下面 `模型集成` 方法之一。
>
> - **模型集成**采用某种方式融合多个已经训练好的模型，进而达到 “取长补短”、提高模型泛化能力的效果。模型集成方法通常在各模型差异性较大、相关性较小的情境下效果更加明显。常用的集成方法有：投票法(voting)、平均法(averaging)、 堆叠法(Stacking)、混合法(Blending)、装袋法（Bagging）、自助法（Boosting）等。
> 
> - **模型组合**是一种极易与模型集成混淆的方法，因为它也会组合多个模型，但与集成方法不同之处在于，模型组合通常在预测变量的不同子空间上选择使用不同模型。也就是说，组合方法中的每一个模型只作用在预测变量的部分空间上，而集成方法中的所有模型都会作用于预测变量的完整空间。

下面分别从贝叶斯框架（ 第 2 节 ）和频率主义框架（第 3 节）角度对模型平均方法做一综述，第 4 节为总结和展望。

## 2 贝叶斯模型平均

### 2.1 概述

原理上，贝叶斯模型平均 ( BMA ) 是非常自然的，因为用于数据分析的贝叶斯框架可以很容易地将参数不确定性和模型不确定性结合起来 [73, 87]。与传统贝叶斯数据分析相比，贝叶斯模型平均不是计算单个模型参数的后验分布，而是计算不同模型后验的加权组合。

> 注解： 作者在此处是想表达三层意思：（1）贝叶斯框架擅长于处理不确定性；（2）这种不确定性既包括参数的不确定性（对于单个模型而言，每个参数都存在不确定性），也包括模型的不确定性（对于一组模型而言，不同的模型有不同的为真概率）；（3）贝叶斯模型平均针对的是模型的不确定性。

在贝叶斯模型平均的经典版本中，一个模型的权重是其为『真』的后验概率，其有一个基础前提假设，即所有模型中仅有一个为『真』。最近，Yao 等人提出了一个显式的、基于预测结果的贝叶斯模型平均版本 `[181]`。该方法中使用交叉验证结果来确定一组最佳权重，是频率主义堆叠方法（见第 3.2.3 节）的的贝叶斯版本。 

经典贝叶斯模型平均关注模型对数据的可解释性（ 合理性，或对『真』模型的识别 ），而基于预测结果的贝叶斯模型平均则具有直接与估计能力相关联的优势。下面的第 2.2 节将重点放在经典贝叶斯模型平均上，因为它在贝叶斯模型平均文献中曾经受到过绝大多数关注； 第 2.3 节将重点讨论根据预测能力实现贝叶斯模型平均的方法。

### 2.2 经典的贝叶斯模型平均

#### 2.2.1 基本框架

一旦为各模型和模型中的参数指定了先验，经典贝叶斯模型平均就像其他贝叶斯方法一样，包含了一个明确定义的、用于获得任意参数`模型平均后验`的过程。与其他贝叶斯方法类似，经典贝叶斯模型平均的简洁性会掩盖其实现上的难度 `[8,32]`，而其中`先验选择`和`后验计算`，依旧是需要面临的最主要的问题。

> 注：
> 
>  经过模型平均后得到的后验概率，其英文 Model-Averaged Posterior 很容易理解，但翻译成中文容易引起歧义，在本文中，统一采用直译 **模型平均后验** 作为其专有名词。

**（1）框架：多模型后验的加权平均**

假设向量 $y=\left(y_{1}, \ldots, y_{n}\right)^{\top}$ 包含一组结果变量的值，我们希望对 $M$ 个嵌套的模型求平均值，其中 $\beta_{m}$ 是第 $m$  个模型中对应的参数向量。正如符号所暗示的那样，$\beta_{m}$ 通常是一组回归系数，但也可能是其他类型的参数，例如在正态线性模型中误差方差。

假设我们现在对估计一个标量类型的参数 $\theta$ 感兴趣，则  $\theta$ 的模型平均后验由下式给出：

$$
\text{单个模型的后验：} p(\theta | y,m) = \frac{p(y|\theta,m)p(\theta,m)}{p(y|m)}
= \frac{p(y|\theta,m)p(\theta,m)}{\int p(y|\theta,m) p(\theta|m) d \theta} \tag{2.1-a}
$$

$$
\text{模型平均的后验：} p(\theta \mid y)=\sum_{m=1}^{M} p(m \mid y) p(\theta \mid y, m) \tag{2.1-b}
$$

其中 $p(m \mid y)$ 是模型  $m$ 为『真』的后验概率，  $p(\theta \mid y, m)$ 我们比较熟悉，是模型 $m$ 为『真』时，参数  $\theta$ 的后验。

依据模型平均的后验表达式可以看出，经典贝叶斯平均在传统单模型贝叶斯分析基础上，将模型不确定性也纳入考虑，模型平均后参数的后验 $p(\theta \mid y)$ 是不同模型后验分布的加权组合，其权值为`各模型的后验概率` $p(m \mid y)$ 。

**(2) 权重：各个模型的后验概率**

由此可见，与单模型方法相比，贝叶斯模型平均需要额外计算各个模型权重，即**模型的后验概率**。按照贝叶斯定理，模型 $m$ 的后验概率定义为：

$$
p(m \mid y) \propto p(m) p(y \mid m) \tag{2.2}
$$

其中， $p(m)$ 为模型 $m$ 的先验概率，而 $p(y \mid m)$ 则是单模型方法中的边缘似然。

而根据单模型贝叶斯定理，见式（2.1-a），模型 $m$ 的边缘似然是在该模型参数空间上的积分：

$$
p(y \mid m)=\int p\left(y \mid \beta_{m}, m\right) p\left(\beta_{m} \mid m\right) d \beta_{m} \tag{2.3}
$$

此处将标量类型的参数 $\theta$ 扩展到了参数向量 $\beta_m$ 。 式中 $p\left(\beta_{m} \mid m\right)$ 为模型 $m$ 中参数的先验分布， $p\left(y \mid \beta_{m}, m\right)$ 为模型 $m$ 的似然函数。当参数 $\beta_{m}$ 为离散型时，式  (2.3) 中的积分改为求和。

> 注解： 在单个模型的贝叶斯定理中，分母项被称为边缘似然，表示模型解释数据的能力，通常边缘似然越大，解释数据的能力越强。但对贝叶斯推断任务了解的读者应该知道，边缘似然的计算通常是难以处理的，需要一些近似计算的方法。本文第 2.2.2 节真是针对此问题的解释。

**（3）贝叶斯因子：模型后验概率的改写**

模型 $m$ 的后验概率也可以写为贝叶斯因子的形式：

$$
p(m \mid y) \propto p(m) B_{m} 
$$

其中

$$
B_{m}=\frac{p(y \mid m)}{p(y \mid 1)}
$$

是比较模型  $m$ 和模型 $1$ 的边缘似然，可以得到的贝叶斯因子。模型 $1$ 可以是任意一个参考模型 `[114,146]`。然而贝叶斯因子对参数的先验分布非常敏感， 即便 $n$ 很大时也一样 `[8,97]`。当一个或多个先验不合适时会导致的极端情况，就是式（2.3）中的边缘似然无法被很好的定义`[73, 161]`。

**（4）参数的模型平均后验：常用统计量**

根据式（ 2.1-b ），参数 $\theta$ 的模型平均后验是一个概率分布，因此有其`均值`和`方差`统计量，由下式定义：

$$
\mathrm{E}(\theta \mid y)=\sum_{m=1}^{M} p(m \mid y) \mathrm{E}(\theta \mid y, m)  \tag{2.4}
$$

$$
\operatorname{var}(\theta \mid y)=\sum_{m=1}^{M} p(m \mid y)\left[\operatorname{var}(\theta \mid y, m)+\{\mathrm{E}(\theta \mid y, m)-\mathrm{E}(\theta \mid y)\}^{2}\right]  \tag{2.5}
$$

其中 $\mathrm{E}(\theta \mid y, m)$ 和 $\operatorname{var}(\theta \mid y, m)$  分别是模型 $m$ 中，参数 $\theta$  的后验分布的均值和方差（注：单模型中，参数的后验也是概率分布，也有其均值和方差统计量）。由此可见，模型平均后，参数的后验方差会受到每个模型的参数不确定性影响，也会受到后验均值在模型间的变化性影响。

我们可以在模型平均后，依据参数的后验分布计算该参数的 $100(1-2 \alpha) \%$ 的中心可信区间，用  $\left[\theta_{L}, \theta_{U}\right]$ 来表示：

$$
\int_{-\infty}^{\theta_{L}} p(\theta \mid y) d \theta=\int_{\theta_{U}}^{\infty} p(\theta \mid y) d \theta=\alpha  \tag{2.6}
$$

当然，也可以使用 $100(1-2 \alpha) \%$ 的最高后验密度可信区间（ HDPI ）。不过，中心可信区间相较最高密度区间计算更容易些，并且其极限可以被解释为后验的中位数。在本文示例中，均采用中心可信区间。

**（5）模型参数的重要性度量：后验包含概率**

上述经典贝叶斯模型平均的一个副产品是可以计算每个预测变量的后验包含概率 （ Posterior Inclusion Probability, PIP ），即所有包含该预测变量的模型后验概率之和  `[7,12,36]` 。这提供了对每个预测变量的相对重要性的有用总结。然而，后验包含概率会受到模型集选择的影响，尤其是预测变量的重要性可能会由于纳入了更多模型而被夸大 [66]。此外，通过比较一组预测变量值对应的结果变量期望值的模型平均后验分布，可以获得更有用的相对重要性总结。在频率主义的模型平均（第 3.7 节）中使用加权求和模型会出现类似的效果。

#### 2.2.2 关键是权重的计算 --- 模型的后验概率

> 注意：
>
> 原文中的 Posterior Model Probability 可以被理解为 Model’s Posterior Probility，即模型后验概率。需要与模型参数的后验概率区分开来。

在依据式（ 2.2 ）计算模型后验概率时，很难确定式（2.3）中定义的边缘似然。尽管在某些设置中（ 例如具有共轭先验参数的 `GLM` ），边缘似然可以有解析表达式 [98]，但更一般的情况是根本无法得到解析表达，只能采用近似值。其中最有名的边缘似然近似是（无需指定参数的先验）：

$$
p(y \mid m) \approx \exp \left(-\mathrm{BIC}_{m} / 2\right) \tag{2.7}
$$

其中

$$
\mathrm{BIC}_{m}=-2 \log p\left(y \mid \widehat{\beta}_{m}, m\right)+p_{m} \log n \tag{2.8}
$$

$\widehat{\beta}_{m}$ 是 $\beta_{m}$ 的最大似然估计。式 (2.8)  就是模型 $m$ 的贝叶斯信息准则（ Bayesian Information Criterion）`[91,98,114,157]` 。其中第一项受模型拟合的影响，第二项可以被视为一个用于惩罚复杂模型的过拟合修正项。

在式 (2.2) 中引入式 (2.7) 可得到如下对模型后验概率的近似：

$$
p(m \mid y) \propto p(m) \exp \left(-\mathrm{BIC}_{m} / 2\right) \tag{2.9}
$$

这个近似有时被称为`广义 BIC 权重 ` `[114]`。当我们计算式（2.9）右边的表达式时，$\mathrm{BIC}_{m}$ 经常被替换为

$$
\mathrm{BIC}_{m}-\min _{k} \mathrm{BIC}_{k},
$$

以避免指数函数中出现过大的参数 `[32]` 。

在式 (2.8) 中指定 $n$ 的值时需要小心。对于正态线性模型，它只是观测的数量；对于二项式模型，它是伯努利试验的总数。当使用对数线性模型来分析联列表时，它是计数总和，而不是表中的单元格数 `[98,145]` 。在生存分析的背景下，[171] 建议将 $n$ 设置为等于未经审查的观测数。对于分层模型，$n$ 的选择取决于分析的重点 `[37,98,138,139,188]`。为了建模调查数据，`[117]` 提出了一个考虑设计效果的 `BIC` 版本。

模型后验概率的高阶近似需要参数的先验及其观测后 Fisher 信息矩阵 `[96,98,116$, 183]`。它在思想上类似于` Takeuchi 信息准则（ TIC ）`，在频率主义设置中，`TIC` 被建议作为 `AIC` 的替代方案[32]（第 3.2.1 节）。

已经提出了逼近模型后验概率的其他方法，包括边缘似然 `[26,28,40,41,121,132,151]` 或马尔可夫链蒙特卡罗 (MCMC) 方法 `[5, 9, 19, 23, 24, 29, 34, 37, 50, 71, 74, 75,79,82,121,142,147,179]`。一种在概念上比较有吸引力的方法是可逆跳跃 MCMC（RJMCMC），在该方法中，我们同时对参数空间和模型空间进行采样 [80]。然而，RJMCMC 可能容易出现性能问题，并且在实现方面具有挑战性 `[8,9,37]` ，以至于 [83] 建议使用式 (2.9) 中的近似值。最近，[8] 开发了一种方法，该方法使用分别拟合每模型获得的 MCMC 输出，并利用来自不同模型的参数之间的关系 [151]。

#### 2.2.3 先验的选择

##### （1）模型的先验

先验模型概率的一个自然而常见的选择是均匀先验，其中 $p(m)=1 / M$。此时，式 ( 2.9 ) 中的模型后验概率就近似简化为众所周知的 `BIC` 权重，由下式给出：

$$
p(m \mid y) \propto \exp \left(-\mathrm{BIC}_{m} / 2\right) \tag{2.10}
$$


> 注： 与模型后验概率一样，给关于模型的先验概率分布定义一个专有名词：**先验模型概率**。

但使用均匀模型先验可能会产生隐含的影响。例如，如果某些预测变量高度相关，则可能会有模型冗余，因为某些模型可能会提供非常相似的参数估计值。使用均匀先验将会稀释`那些不相似模型`应有的先验概率 `[18,57,66,76]` 。 Villa 等提出了一种处理该问题的方法`[170]` ，他建议使用`模型价值`的概念来指定`先验模型概率`，模型价值基于对以下问题的量化：当某个模型是真模型时，如果将其从模型集中剔除，我们预期会失去什么？

另一种选择是为每个预测变量设置独立于其他变量的、被包含概率为 $p$ 的伯努利先验 。这样的话，均匀先验是其中 $p=0.5$ 时的一种特殊情况，对应于包括一半预测变量的先验期望` [37]`。为了有一个关于模型大小的低信息量先验，可以对 $p$ 使用贝塔先验 `[37,105,158]`。

> 注：关于此处稀释的理解。此处指如果某些预测变量高度相关，则与其相关的模型必然相似，因此，均匀的先验模型概率隐含地会提升相似模型的概率（将所有相似的模型视为同一个模型可能更好理解），而相对地压低其他不相似模型的概率。例如，假设共有 10 个模型，其中 3 个模型比较相似，则均匀先验会隐含地将该模型的概率提升到 $3/10$ ，而其他不相似的模型概率依然为 $1/10$ 。

其他指定模型先验的方法还包括：（1）经验贝叶斯法 [37]；（2）允许预测变量相关，例如当某些模型包含交互项时 [30]；（3）对具有相似性的模型使用较低权重[66]等。

此外，为了使`模型先验选择`能够对`模型平均后验（Model-Averaged Posterior）`的形式产生影响，[43] 建议使用`可信模型平均`，其中考虑了多个模型先验。这有效地允许人们进行敏感性分析，以评估`模型平均后验`受`模型先验选择`影响的程度。在` [44-46, 185]` 中可以找到使用可信模型平均的更多示例。

##### （2）模型参数的先验

模型后验概率对模型中参数的先验选择非常敏感，即使该先验在单模型方法中属于非信息先验 `[8,35,93]`。 In particular, as mentioned in Sect. $2.2$, the use of improper priors can lead to the Bayes factors, and hence the posterior model probabilities, not being well defined $[10,82,93,161]$. It is also possible for apparently sensible priors for the parameters to cause the models to have conflicting implicit prior distributions for $\theta$ [85].

In the normal linear model setting, Zellner's g-prior has been used extensively, as it has several desirable properties, including computational convenience $[37,154]$. It involves centring the predictor variables at zero, in order to remove any dependence between the intercept and the regression coefficients, and then specifying a joint prior for the intercept and error variance, plus a joint prior for the regression coefficients given the error variance. For model $m$, this leads to

$$
p\left(\beta_{m 0}, \sigma_{m}^{2} \mid m\right) \propto 1 / \sigma_{m}^{2}
$$
and a multivariate normal prior for the regression coefficients, with mean zero and covariance matrix
$$
g_{m} \sigma_{m}^{2}\left(X_{m}^{\top} X_{m}\right)^{-1},
$$

where $\beta_{m 0}, \sigma_{m}^{2}$, and $X_{m}$ are the intercept, error variance and design matrix for model $m$ respectively, and $g_{m}$ is a hyperparameter $[37,187]$. This prior has a nice interpretation, as it can be thought of as containing $1 / g_{m}$ as much information as that in the data. The resulting posterior model probability is given by

$$
p(m \mid y) \propto p(m) \exp \left(-\mathrm{IC}_{m} / 2\right),
$$

where

$$
\mathrm{IC}_{m}=-2 \log p\left(y \mid \widehat{\beta}_{m}, m\right)+p_{m} \log g_{m} .
$$

$\mathrm{IC}_{m}$ can be thought of as a generalised information criterion in which the correction for overfitting is $p_{m} \log g_{m}$. Setting $g_{m}$ to be arbitrarily large, in order for the prior to be non-informative, can lead to strongly favouring the null model, an example of the Lindley-Jeffreys paradox $[37,91,98,113]$. Using $g_{m}=n$ gives the unit-information prior, which contains the same amount of information as a single observation, and leads to the posterior model probability being the generalised BIC weight in (2.9). Together with a uniform model-prior, this corresponds to using the BIC weight in (2.10), which was found perform well in a simulation study reported by [58]. An empirical Bayes procedure is also possible, in which the choice of $g_{m}$ depends on the data [37]. As with the parameter $p$ in the Bernoulli model-prior, we might also want to put a prior on $g_{m}$, rather than specify its value $[105,109,111,162,186] .$ A version of the g-prior for high-dimensional normal linear models was proposed by [120].

For GLMs, [146] considered several approximations to the Bayes factors, including one that leads to the generalised BIC weight in (2.9). Extensions of Zellner's g-prior to this setting, including use of a prior on $g_{m}$, have been suggested by several authors; see [154] and the references therein. A calibrated information criterion (CIC) prior was proposed by [35]. This is based on the Jeffreys prior used in the single-model setting [91], and for model $m$ it is given by

$$
p\left(\beta_{m} \mid m\right)=(2 \pi)^{-p_{m} / 2}\left|c_{m}^{-1} J\right|^{1 / 2}
$$

where $J$ is the observed Fisher information matrix for $\beta_{m}$ and $c_{m}$ is a hyperparameter. In conjunction with a uniform model-prior, this leads to the model-averaged posterior for $\theta$ being approximated by a multivariate normal distribution with mean $\widehat{\beta}_{m}$ and covariance matrix $J^{-1}$. In addition, the posterior probability for model $m$ is approximated by

$$
p(m \mid y) \propto \exp \left(-\mathrm{CIC}_{m} / 2\right)
$$

where

$$
\mathrm{CIC}_{m}=-2 \log p\left(y \mid \widehat{\beta}_{m}, m\right)+p_{m} \log c_{m},
$$

which has the same form as (2.11). The right-hand side of (2.12) is known as the CIC weight; the BIC weight in $(2.10)$ is a special case, corresponding to $c_{m}=n$.

### 2.3 基于预测的贝叶斯模型平均

As mentioned in Sect. 2.1, classical BMA focusses attention on identification of the true model. Recently, several authors have considered use of prediction-based BMA $[39,102,181]$. In addition to being a more natural approach to model averaging, this has the distinct advantages of not requiring a prior for the models, being less sensitive to the priors for the parameters, and only requiring the usual MCMC output for each individual model.

There are currently two types of prediction-based BMA. The first involves a criterion based on a measure of the within-sample prediction error plus a correction term which allows for overfitting. The second uses cross validation and is therefore based on the error associated with prediction of observation $i$ having fitted the model to all the data except that observation $(i=1, \ldots, n)$. The only difference between these approaches and classical BMA is that we combine posterior distributions using model weights that are not posterior model probabilities.

#### 2.3.1 `DIC` 与 `WAIC`

In Bayesian model selection, the deviance information criterion (DIC) has long been used as an alternative to $\mathrm{BIC}[159,160]$. For model averaging, $[15]$ suggested use of DIC weights, with $\mathrm{BIC}_{m}$ in (2.10) being replaced by
$$
\mathrm{DIC}_{m}=-2 \log p\left(y \mid \widehat{\beta}_{m}, m\right)+2 p_{m}^{D I C},
$$
where $\widehat{\beta}_{m}$ is a point estimate of $\beta_{m}$ and $p_{m}^{D I C}$ is a correction for overfitting, often referred to as the effective number of parameters [159]. Common choices are
$$
\widehat{\beta}_{m}=\mathrm{E}\left(\beta_{m} \mid y, m\right)
$$
and
$$
p_{m}^{D I C}=2 \operatorname{var}\left\{\log p\left(y \mid \beta_{m}\right)\right\} .
$$
The posterior mean in (2.14) and posterior variance in (2.15) are estimated by the mean of the posterior MCMC sample for $\beta_{m}$ and the variance of the posterior MCMC sample for $\log p\left(y \mid \beta_{m}\right)$ respectively. An alternative choice for $p_{m}^{D I C}$ is possible $[22,$, $73,159]$, but this has the disadvantage of sometimes being negative.

DIC has much in common with AIC, which is also a prediction-based criterion (Sect. $3.2 .1$ and [32]). DIC model weights have been used in a range of applications, including ecology $[60,62,115,127,167]$, fisheries $[92,177]$, medicine $[143]$ and physics [112].

The other prediction-based measure we consider is the Watanabe-Akaike Information Criterion (WAIC) $[72,73,89,169,174] .{ }^{7}$ This is more Bayesian than DIC (and BIC), in that it replaces $\widehat{\beta}_{m}$ by the posterior distribution for $\beta_{m}$, and can work well in situations where DIC has problems [22]. The point estimate $\widehat{\beta}_{m}$ in DIC leads to underestimation of the prediction uncertainty, and hence to the possibility that use of DIC will lead to overfitting. ${ }^{8}$ WAIC is also specified in terms of the pointwise predictive densities $p\left(y_{i} \mid \beta_{m}\right)$, rather than the joint predictive density $p\left(y \mid \beta_{m}\right)$, as the former has a close connection with cross validation [72] (Sect. 2.3.2). If the $y_{i}$ are independent given the parameters, use of the joint density is equivalent to the pointwise-approach.

The value of WAIC for model $m$ is given by
$$
\mathrm{WAIC}_{m}=-2 \sum_{i=1}^{n} \log p\left(y_{i} \mid y, m\right)+2 p_{m}^{\text {WAIC }},
$$
where $p_{m}^{\text {WAIC }}$ is again a correction for overfitting. The posterior predictive density in (2.16) is given by
$$
p\left(y_{i} \mid y, m\right)=\int p\left(y_{i} \mid \beta_{m}, y, m\right) p\left(\beta_{m} \mid y, m\right) d \beta_{m}=\mathrm{E}\left\{p\left(y_{i} \mid \beta_{m}, y, m\right)\right\},
$$
One choice for the correction term is
$$
p_{m}^{\text {WAIC }}=\sum_{i=1}^{n} \operatorname{var}\left\{\log p\left(y_{i} \mid \beta_{m}, y, m\right)\right\}
$$
As with DIC, the posterior mean in $(2.17)$ and the posterior variance in (2.18) can be estimated by the mean of the posterior MCMC sample for $p\left(y_{i} \mid \beta_{m}, y, m\right)$ and the variance of the posterior MCMC sample for $\log p\left(y_{i} \mid \beta_{m}, y, m\right)$. As with DIC, an alternative choice for $p_{m}^{\text {WAIC }}$ is possible $[72,73]$; we consider that in (2.18) as it is closely related to leave-one-out cross validation (Sect. 2.3.2).

WAIC weights can be calculated using (2.10), with $\mathrm{BIC}_{m}$ replaced by $\mathrm{WAIC}_{m} .$ As DIC and WAIC are focussed on prediction, we would expect weights based on these criteria to be preferable to BIC weights, which are more focussed on identification of a true model [72]. As WAIC is more Bayesian than DIC, WAIC weights are based on a more reliable assessment of the prediction-uncertainty associated with each model. WAIC is also invariant to transformation of the parameters, whereas DIC will not be if we use (2.13), as the posterior mean is not transformation-invariant [159]. ${ }^{9}$ In addition, use of a pointwise-approach means that $p_{m}^{W A I C}$ will be more stable than $p_{m}^{D I C}[73] .$

As with BIC, when assessing the fit of a hierarchical model the exact form of DIC and WAIC will depend upon the focus of the analysis, as this will determine what we mean by prediction of a new observation [72, 122]; a similar issues arises when using AIC in the frequentist setting (Sect. 3.6.4).

#### 2.3.2 贝叶斯堆叠

Stacking is a cross-validation-based approach to model averaging that has a long history in the frequentist setting [164] (Sect.3.2.3). Like the frequentist version, Bayesian stacking [181] uses a measure of out-of-sample prediction error, which does not require a correction for overfitting. If a logarithmic scoring rule is used to summarise the prediction performance $[78,137]$, the model weights are chosen to be those that maximise the function
$$
\sum_{i=1}^{n} \log \sum_{m=1}^{M} w_{m} p\left(y_{i} \mid y_{-i}, m\right)
$$
where $w_{m}$ is the weight associated with model $m$ and $y_{-i}$ is the response vector $y$ with $y_{i}$ removed. ${ }^{10}$ In order to maximise $(2.19)$ using weights that lie on the unit simplex, we can use a constrained-optimisation method, such as quadratic programming [83]. Following [181], we refer to this approach as Bayesian stacking of predictive distributions (BSP). ${ }^{11}$ Analogous to the form of the posterior predictive density used to calculate $\mathrm{WAIC}_{m}$ in (2.16) (Sect. 2.3.1), we have
$$
p\left(y_{i} \mid y_{-i}, m\right)=\int p\left(y_{i} \mid \beta_{m}, y_{-i}, m\right) p\left(\beta_{m} \mid y_{-i}, m\right) d \beta_{m}=\mathrm{E}\left\{p\left(y_{i} \mid \beta_{m}, y_{-i}, m\right)\right\},
$$
where the posterior mean on the right-hand side is now with respect to $p\left(\beta_{m} \mid y_{-i}, m\right)$, and can be estimated by the mean of the corresponding posterior MCMC sample for $p\left(y_{i} \mid \beta_{m}, y_{-i}, m\right) .$

As computational effort will often be an important consideration in the Bayesian setting, [181] proposed use of Pareto-smoothed importance sampling [168], which only requires a single fit to the data for each model. On the other hand, if the sample size is small estimation of the weights may be unstable [181], an example of which arises in the toxicity example (Sect. 2.4.2).

Determining posterior model weights by minimising an objective function has also been suggested by $[81,172] .$ Likewise, in the context of forecasting in economic time series, $[59,63]$ have proposed using an estimate of out-of-sample prediction error to determine model weights. A decision-theoretic approach to BMA, also based on prediction error, was used by [16] in the context of high-dimensional multivariate regression models.

When $n$ is large, BSP might be expected to produce weights that are similar to those based on WAIC, as the latter is asymptotically equivalent to use of Bayesian leave-one-out cross validation for model selection [174]. A discussion of the relative merits of DIC, WAIC and Bayesian cross validation can be found in [72].

In related work, interpretation of a model-averaged posterior as a mixture distribution has been advocated by [94]; see also [181]. ${ }^{12}$ As with the approach of [172], this leads to improper priors for the model parameters being acceptable.

Use of BSP can be motivated by the fact that classical BMA has been shown to have poorer prediction performance than frequentist stacking, particularly when the true model is not in the model set [33, 53, 178, 181]. Use of classical BMA leads to an asymptotic weight of one for the model closest to the true data-generating mechanism (in terms of Kullback-Leibler divergence). In contrast, BSP finds the optimal combination of predictive distributions that is closest to the data-generating mechanism (in terms of the scoring rule), and the asymptotic BSP weights can all be less than one [181]. A similar motivation led to the idea of Bayesian model combination in the machine-learning literature [100, 124, 126]. If one of the models is a good approximation to the data-generating mechanism, BSP may not perform as well as classical BMA when n is small.



## 3 频率主义模型平均

### 3.1 简介

### 3.2 点估计中的模型平均

#### 3.2.1 信息准则法

#### 3.2.2 装袋法（Bagging）

#### 3.2.3 最优权重法

### 3.3 区间估计中的模型平均

#### 3.3.1 `Wald` 区间

#### 3.3.2 分位数自助区间

#### 3.3.3 模型平均的尾部区间

### 3.4 讨论

#### 3.4.1 尺度的选择

#### 3.4.2 模型集的选择

#### 3.4.3 置信区间

#### 3.4.4 混合模型

#### 3.4.5 缺失数据

#### 3.4.6 模型权重求和




## 4 总结和未来方向

### 4.1 要点总结

**（1）估计而不是识别**

 模型平均是一种估计工具，因此与识别真实模型没有直接关系。在许多设置中，评估该估计对模型选择的敏感性也将很有用。
 
 **（2）感兴趣的参数**
 
 这应该在所有模型中具有相同的解释，因此回归系数的平均不太可能相关。
 
 **（3）模型冗余**
 
 如果在经典 BMA 中使用统一的模型先验，模型冗余会导致一些先验模型概率的稀释。这个问题在基于预测的 BMA 中不会出现，因为它不需要模型先验。同样，在 FMA 模型中，冗余会导致 AIC 权重被稀释。 AIC(w) 和堆叠中使用的单纯形约束缓解了这个问题。
 
 **（4）区间估计** 
 
 目前没有区间能保证良好的覆盖，除非它等于最大模型的区间。 MATA 区间的优点是它可以基于任何计算单模型置信区间的方法。
 
**（5）尺度的选择**
 
 BMA 是变换不变的，而 FMA 不是。然而，通常有一个自然尺度来执行 FMA，例如 GLM 中的线性预测尺度

**（6）对模型权重求和**
 
 对模型权重求和不能提供预测变量重要性的有用度量；对预测变量的特定值的模型平均估计值进行比较是可取的。类似的评论适用于 BMA 中的后验包含概率。
 
 **（7）混合模型**
 
 AIC(w) 和堆叠的混合模型版本是可能的，WAIC 和 BSP 的分层版本也是如此。
 
 **（8）模型集的选择**
 
 使用一组单例模型，每个模型都涉及一个预测器，当有许多预测器时，这似乎是一种很有前途的方法。
 
 **（9）贝叶斯学派的选择**
 
 使用 WAIC 或 BSP 权重比经典 BMA 更可取，因为重点是预测而不是识别真实模型。使用这些权重还避免了模型后验概率的计算以及这些概率对参数先验的敏感性的问题。
 
 **（10）频率学派的选择**
 
 如果计算量不是问题，堆叠是一个不错的选择。 AIC(w) 是大 n 的一个很好的替代方案，但鲁棒性较差。这两种方法在元模型方面也有很好的解释


### 4.2 未来方向

**（1）置信区间** 

需要研究计算模型平均置信区间的最佳方法。这可能涉及在区间覆盖和宽度方面最佳的权重，如贝叶斯设置中的 [2] 所建议的那样。还需要评估何时仅使用最大模型的置信区间最佳。

**（2）置信分布** 

BMA 的一个明显优势是使用后验分布来总结结果。在 FMA 中，拥有置信分布的模型平均版本会很有用，它提供了所有可能置信区间的摘要 [1]。 

**（3）WAIC 和 BSP 的替代版本** 

WAIC 的加权版本，类似于 FMA 中的 AIC(w)，出于同样的原因，AIC(w) 权重似乎比基于 AIC 的权重更可取。同样，对于 GLM，一个 BSP 版本可能有用，它涉及对每个模型的线性预测器进行加权，类似于 FMA 中的堆叠。

**（4）与收缩的比较** 

最好进行模拟研究，将模型平均方法（例如 AIC(w) 和堆叠）与收缩方法（例如 lasso）进行比较。
