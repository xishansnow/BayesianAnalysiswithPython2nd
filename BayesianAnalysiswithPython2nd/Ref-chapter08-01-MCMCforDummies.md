---
jupytext:
 formats: ipynb,.myst.md:myst,md
 text_representation:
  extension: '.md'
  format_name: myst
kernelspec:
 display_name: Python 3
 language: python
 name: python3
---

# MCMC 采样的傻瓜书

<style>p{text-indent:2em;2}</style>

【原文】 [MCMC sampling for dummies — While My MCMC Gently Samples (twiecki.io)](https://twiecki.io/blog/2015/11/10/mcmc-sampling/)

当谈论概率编程和贝叶斯统计时，通常会掩饰统计推断实际执行的细节，本质上将其视为黑匣子。概率编程好处在于“不必为构建模型而理解推断的工作原理”，但了解其原理肯定会有所帮助。

当我向新手介绍一个贝叶斯模型时，他虽然没有接受过贝叶斯统计方面的培训，但通常渴望理解推断原理。而我之前的回答往往是：“ MCMC 通过构造一个以目标后验分布为平衡分布的可逆马尔可夫链，通过从后验分布中产生样本来做预测等后续任务。” 这句话没错，但似乎没有用。这很恼火，因为从来没有人告诉你概念背后的直观感觉或者动机，通常只是给你一些可怕的数学知识。我不得不花无数小时用头撞墙，直到顿悟时刻的到来。通常情况下，一旦我理解了其意思，事情就看起来不那么复杂了。[这篇博客文章](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm) 试图解释 `MCMC` 采样背后的动机。

下面使用代码示例，而不是公式或数学语言，开始建立直观感觉。

## 1 问题及其非直观的解释

首先看贝叶斯公式：

```{math}
P(\theta|x) = \frac{P(x|\theta) P(\theta)} {P(x)}
```

在给定数据情况下，能够得到模型参数 $\theta$ 的概率分布。为计算它，将先验 $P(\theta)$ 和似然 $P(x|\theta)$  相乘得到分子项。通常分子项非常简单，但仔细观察分母项 $P(x)$ （也称为证据或边缘似然），会发现它是对所有可能的参数值进行积分计算而来。

```{math}
P ( x ) = \int _ { \theta } P ( x , \theta ) d \theta
```

这就是贝叶斯公式的关键难点：尽管公式看起来足够简单，但很难以封闭方式计算后验结果。

如果我们解决不了什么问题，可以试着去近似它。例如，如果能从后验分布中抽取样本，就可以用蒙特卡罗方法进行后续任务的近似计算。但新的问题出现了，要直接从后验分布中抽取样本，不仅要求解贝叶斯公式，而且还要求出它的逆函数，这就更难了。

能否构建一个可遍历的可逆马尔可夫链，使其均衡分布与后验分布相匹配呢？ 这听起来很疯狂，因为如果你无法计算后验，不能从中取样，那么构建这样的马尔可夫链肯定会更加困难。但令人惊讶是，这个想法非常容易实现，并存在一类通用算法来支撑，被称为[马尔可夫链蒙特卡罗](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo)，即构造马尔可夫链进行蒙特卡罗逼近。

## 2 问题设置

首先，让导入 python 模块：

```{code-cell} ipython3
%matplotlib inline 
import numpy as np 
import scipy as sp 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy.stats import norm 

sns.set_style('white') 
sns.set_context('talk') 
np.random.seed(123) 
```

先生成一些实验数据，以零为中心的正态分布上的 20 个点。我们的目标是估计平均值 $\mu$ 的后验。

```{code-cell} ipython3
data = np.random.randn(20) 
ax = plt.subplot() 
sns.distplot(data, kde=False, ax=ax) 
_ = ax.set(title='Histogram of observed data', xlabel='x', ylabel='# observations'); 
```

接下来定义模型。在这个简单例子中，假设总体和样本呈正态分布（即模型的似然是正态分布）。正态分布有两个参数：均值（ $\mu$ ）和标准差（ $\sigma$ ）。为简单起见，假设已知 $\sigma=1$，想要推断 $\mu$ 的后验。根据贝叶斯原理，对于每个想要推断的参数，必须选择一个先验。为简单起见，仍然假设参数 $\mu$ 呈正态分布，且均值 $\mu_\mu = 0$ ，标准差 $\mu_\sigma = 1$ ，即将标准正态分布作为 $\mu$ 的先验分布。从统计学角度，模型是：

```{math}
\begin{aligned}
先验：\mu & \sim \operatorname{Normal}(0,1) \\
似然：x \mid \mu & \sim \operatorname{Normal}(x ; \mu, 1) 
\end{aligned}
```

该模型较为简单，实际上可以获得后验的解析解。因为对于已知标准差的正态似然分布，$\mu$ 的先验与后验是共轭的（高斯分布是高斯似然的共轭先验），可以较为容易地计算后验。有关这一点的数学推导，请参见（[此处](https://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnxiYXllc2VjdHxneDplNGY0MDljNDA5MGYxYTM)）。

```{code-cell} ipython3
def calc_posterior_analytical(data, x, mu_0, sigma_0): 
   sigma = 1. 
   n = len(data) 
   mu_post = (mu_0 / sigma_0**2 + data.sum() / sigma**2) / (1. / sigma_0**2 + n / sigma**2) 
   sigma_post = (1. / sigma_0**2 + n / sigma**2)**-1 
   return norm(mu_post, np.sqrt(sigma_post)).pdf(x) 
   
ax = plt.subplot() 
x = np.linspace(-1, 1, 500) 
posterior_analytical = calc_posterior_analytical(data, x, 0., 1.) 
ax.plot(x, posterior_analytical) 
ax.set(xlabel='mu', ylabel='belief', title='Analytical posterior'); 
sns.despine() 
```

这显示了我们的兴趣量，即在考虑到先验信息并看到数据后，参数值 $\mu$ 的概率。但需要清楚，当先验假设并非共轭时，很难获得如此简单的解析解。

## 3 MCMC 采样的代码说明

现在来理解采样逻辑。首先找到起始参数位置（可以随机选择），任意固定为：

```{code-cell} ipython3
mu_current = 1
proposal_width = 1
```

### （1）位置提议

先建议从该位置移动到其他位置（建议的方法可以简单也可以复杂，可以跳跃也可以平稳，当这种移动是平稳时，通常正是 MCMC 的马尔可夫部分）。著名的`Metropolis 采样器`采用的就是比较简单的办法，它从以当前 `mu_current` 为中心的正态分布中抽取样本（注意：此处是 `Metropolis 准则`的设计要求，并非源于模型的高斯假设），该值具有一定标准差（`proposal_width`），该标准差将决定建议移动的距离（这里使用了 `scipy.stats.norm` 计算距离）：

```{code-cell} ipython3
mu_proposal = norm(mu_current, proposal_width).rvs()
```

### （2）位置评估

接下来，将评估这是否是一个好位置。如果所建议的 `mu` 得到的正态分布比原来的 `mu` 更能够解释数据，则肯定想去那里。“更好地解释数据”是什么意思？ 我们按照建议参数值 `mu` 和标准差 `sigma =1`  计算似然并进行量化拟合。通过使用 `scipy.stats.Normal(µ，sigma).pdf(Data)` 来计算每个数据点的概率，然后将各概率相乘（即计算似然，通常可使用对数概率，但此处略过）：

```{code-cell} ipython3
likelihood_current = norm(mu_current, 1).pdf(data).prod()
likelihood_proposal = norm(mu_proposal, 1).pdf(data).prod()
# Compute prior probability of current and proposed mu        
prior_current = norm(mu_prior_mu, mu_prior_sd).pdf(mu_current)
prior_proposal = norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal)
# Nominator of Bayes formula
p_current = likelihood_current * prior_current
p_proposal = likelihood_proposal * prior_proposal
```

到目前为止，我们基本上有一个爬山算法，它只会向建议的随机方向移动，并且只有在 `mu_proposal` 的可能性高于`mu_current` 的情况下才接受跳跃。最终，我们将到达 $µ=0$ （或接近它），从那里不可能再有任何移动。 然而，我们想要的是后验，所以我们有时也不得不接受向另一个方向移动。关键的诀窍是将这两个概率分开

```{code-cell} ipython3
p_accept = p_proposal / p_current
```

我们得到了一个接受率 `p_accept`。如果 `p_proposal` 更大，概率将大于 1 ，我们肯定会接受。然而，如果`p_current` 更大，比方说两倍，就有 50% 的机会搬到那里去：

```{code-cell} ipython3
accept = np.random.rand() < p_accept
if accept:
    # Update position
    cur_pos = proposal
```

这个简单程序给我们提供了后验的样本。

## 4 为什么会起作用？

请注意，接受率 `p_accept` 是整个事情得以解决的主要原因。下式为 `p_accept` 的直观解释，即接受率是建议位置的后验与原来位置后验的比值：

```{math}
\frac{\frac{P(x \mid \mu) P(\mu)}{P(x)}}{\frac{P(x \mid \mu 0) P(\mu 0}{P(x)}}=\frac{P(x \mid \mu) P(\mu)}{P\left(x \mid \mu_{0}\right) P\left(\mu_{0}\right)}
```

将建议参数的后验除以当前参数的后验，$P(x)$ 就被抵消了。所以你可以直觉地认为，实际上是在用一个位置的全部后验除以另一个位置的全部后验。这样，我们访问后验概率较高的区域比后验概率较低的区域就要频繁得多。

将上述过程放在一起：

```{code-cell} ipython3
def sampler(data, samples=4, mu_init=.5, proposal_width=.5, plot=False, mu_prior_mu=0, mu_prio
r_sd=1.): 
   mu_current = mu_init 
   posterior = [mu_current] 
   for i in range(samples): 
       # suggest new position 
       mu_proposal = norm(mu_current, proposal_width).rvs() 
       # Compute likelihood by multiplying probabilities of each data point 
       likelihood_current = norm(mu_current, 1).pdf(data).prod() 
       likelihood_proposal = norm(mu_proposal, 1).pdf(data).prod() 
        
       # Compute prior probability of current and proposed mu         
       prior_current = norm(mu_prior_mu, mu_prior_sd).pdf(mu_current) 
       prior_proposal = norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal) 
        
       p_current = likelihood_current * prior_current 
       p_proposal = likelihood_proposal * prior_proposal 
        
       # Accept proposal? 
       p_accept = p_proposal / p_current 
        
       # Usually would include prior probability, which we neglect here for simplicity 
       accept = np.random.rand() < p_accept 
        
       if plot: 
           plot_proposal(mu_current, mu_proposal, mu_prior_mu, mu_prior_sd, data, accept, posterior, i) 
        
       if accept: 
           # Update position 
           mu_current = mu_proposal 
        
       posterior.append(mu_current) 
        
   return np.array(posterior) 
# Function to display 
def plot_proposal(mu_current, mu_proposal, mu_prior_mu, mu_prior_sd, data, accepted, trace, i): 
   from copy import copy 
   trace = copy(trace) 
   fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(16, 4)) 
   fig.suptitle('Iteration %i' % (i + 1)) 
   x = np.linspace(-3, 3, 5000) 
   color = 'g' if accepted else 'r' 
        
   # Plot prior 
   prior_current = norm(mu_prior_mu, mu_prior_sd).pdf(mu_current) 
   prior_proposal = norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal) 
   prior = norm(mu_prior_mu, mu_prior_sd).pdf(x) 
   ax1.plot(x, prior) 
   ax1.plot([mu_current] * 2, [0, prior_current], marker='o', color='b') 
   ax1.plot([mu_proposal] * 2, [0, prior_proposal], marker='o', color=color) 
   ax1.annotate("", xy=(mu_proposal, 0.2), xytext=(mu_current, 0.2), 
                arrowprops=dict(arrowstyle="->", lw=2.)) 
   ax1.set(ylabel='Probability Density', title='current: prior(mu=%.2f) = %.2f\nproposal: prior(mu=%.2f) = %.2f' % (mu_current, prior_current, mu_proposal, prior_proposal)) 
    
   # Likelihood 
   likelihood_current = norm(mu_current, 1).pdf(data).prod() 
   likelihood_proposal = norm(mu_proposal, 1).pdf(data).prod() 
   y = norm(loc=mu_proposal, scale=1).pdf(x) 
   sns.distplot(data, kde=False, norm_hist=True, ax=ax2) 
   ax2.plot(x, y, color=color) 
   ax2.axvline(mu_current, color= b , linestyle= -- , label= mu_current ) 
   ax2.axvline(mu_proposal, color=color, linestyle='--', label='mu_proposal') 
   #ax2.title('Proposal {}'.format('accepted' if accepted else 'rejected')) 
   ax2.annotate("", xy=(mu_proposal, 0.2), xytext=(mu_current, 0.2), 
                arrowprops=dict(arrowstyle="->", lw=2.)) 
   ax2.set(title='likelihood(mu=%.2f) = %.2f\nlikelihood(mu=%.2f) = %.2f' % (mu_current, 1e14*likelihood_current, mu_proposal, 1e14*likelihood_proposal)) 
    
   # Posterior 
   posterior_analytical = calc_posterior_analytical(data, x, mu_prior_mu, mu_prior_sd) 
   ax3.plot(x, posterior_analytical) 
   posterior_current = calc_posterior_analytical(data, mu_current, mu_prior_mu, mu_prior_sd) 
   posterior_proposal = calc_posterior_analytical(data, mu_proposal, mu_prior_mu, mu_prior_sd) 
   ax3.plot([mu_current] * 2, [0, posterior_current], marker='o', color='b') 
   ax3.plot([mu_proposal] * 2, [0, posterior_proposal], marker='o', color=color) 
   ax3.annotate("", xy=(mu_proposal, 0.2), xytext=(mu_current, 0.2), 
                arrowprops=dict(arrowstyle="->", lw=2.)) 
   #x3.set(title=r'prior x likelihood $\propto$ posterior') 
   ax3.set(title='posterior(mu=%.2f) = %.5f\nposterior(mu=%.2f) = %.5f' % (mu_current, posterior_current, mu_proposal, posterior_proposal)) 
    
   if accepted: 
       trace.append(mu_proposal) 
   else: 
       trace.append(mu_current) 
   ax4.plot(trace) 
   ax4.set(xlabel='iteration', ylabel='mu', title='trace') 
   plt.tight_layout() 
   #plt.legend() 
```

## 5 可视化 MCMC

为了使采样可视化，我们将为计算出的一些量创建曲线图。下面的每一行都是我们的 Metropolis 采样器的一次迭代。

第一列是先验分布-- 即看到数据之前对于 $\mu$ 的信念。可以看到分布是静态的，我们只是插入了 $\mu$ 的建议值 。竖线用蓝色表示我们的当前的 $\mu$ ，而用红色或绿色表示我们建议的 $\mu$ （分别被拒绝或接受）。

第二列是可能性，以及用来评估我们的模型对数据的解释有多好。您可以看到，似然函数随建议的变化而变化。蓝色直方图是我们的数据。绿色或红色的实线是当前建议的 $\mu$ 的可能性。直观地说，可能性和数据之间的重叠越多，模型对数据的解释就越好，由此产生的概率也就越高。相同颜色的虚线是建议的 $\mu$ ，而蓝色虚线是当前的 $\mu$ 。

第三列是后验分布。这里显示的是归一化后验，但正如上面所发现的，可以将当前 $\mu$ 的先验值乘以建议 $\mu$ 的似然值得到非归一化的后验值；然后两者相除得到接受率 `p_accept` 。

第四列是迹（即生成的后验样本）。其中存储每个样本，而不管它是被接受还是被拒绝。

我们经常根据后验密度移动到相对更可能的 $\mu$ 值，只是有时移动到相对不太可能的值，就像在第 14 次迭代中看到的那样。

```{code-cell} ipython3
np.random.seed(123) 
sampler(data, samples=8, mu_init=-1., plot=True)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429134308_87.webp)

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429134320_30.webp)

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429134353_96.webp)

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429134404_fb.webp)

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429134418_6d.webp)

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429134438_3d.webp)

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429134450_87.webp)

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429134506_b7.webp)

MCMC 的神奇之处在于，你只需要做足够长的时间，就会产生来自模型后验分布的样本。有一个严格的数学证明可以保证这一点，我在这里不会详细说明。为了了解这会产生什么，让我们绘制大量样本并绘制它们的曲线图。

```{code-cell} ipython3
posterior = sampler(data, samples=15000, mu_init=1.) 
fig, ax = plt.subplots() 
ax.plot(posterior) 
_ = ax.set(xlabel='sample', ylabel='mu'); 
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429134700_6c.webp)

这通常称为迹。**现在要得到后验的近似值，我们只需取此迹的直方图即可**。重要的是要记住，尽管这看起来与我们上面为拟合模型而采样的数据相似，但两者是完全分开的。下面的情节代表了我们对 $\mu$ 的信念。在本例中，它碰巧也是正态分布的，但对于不同的模型，它可能具有与似然或先验完全不同的形状。

```{code-cell} ipython3
ax = plt.subplot() 
sns.distplot(posterior[500:], ax=ax, label='estimated posterior') 
x = np.linspace(-.5, .5, 500) 
post = calc_posterior_analytical(data, x, 0, 1) 
ax.plot(x, post, 'g', label='analytic posterior') 
_ = ax.set(xlabel='mu', ylabel='belief'); 
ax.legend();
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/bayesian_stat_2021060810153465.webp)

正如您所看到的，通过遵循上面的过程，我们从与解析得出相同分布中获得了样本。

## 6 建议宽度

上面我们将建议宽度设置为 0.5。事实证明，这是一个相当不错的值。一般来说，不希望宽度太窄，因为采样效率会很低，需要很长时间来探索整个参数空间，并且会显示典型的随机游走行为：

```{code-cell} ipython3
posterior_small = sampler(data, samples=5000, mu_init=1., proposal_width=.01) 
fig, ax = plt.subplots() 
ax.plot(posterior_small); 
_ = ax.set(xlabel='sample', ylabel='mu'); 
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429135140_25.webp)

但你也不希望它太大，以至于你永远不会接受跳跃：

```{code-cell} ipython3
posterior_large = sampler(data, samples=5000, mu_init=1., proposal_width=3.) 
fig, ax = plt.subplots() 
ax.plot(posterior_large); plt.xlabel('sample'); plt.ylabel('mu'); 
_ = ax.set(xlabel='sample', ylabel='mu');
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429135222_2a.webp)

然而，请注意，我们仍在从这里的目标后验分布进行采样，正如数学证明所保证的那样，只是效率较低：

```{code-cell} ipython3
sns.distplot(posterior_small[1000:], label='Small step size') 
sns.distplot(posterior_large[1000:], label='Large step size'); 
_ = plt.legend();
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429135307_7d.webp)

有了更多的样本，这最终会看起来像真正的后验。关键是我们希望样本彼此独立，但显然在本例中不是这样的。因此，可以采用自相关性来量化评估采样器的效果--即样本 $i$ 与样本 $i-1$ 、$i-2$ 等的相关性如何：

```{code-cell} ipython3
from pymc3.stats import autocorr 
lags = np.arange(1, 100) 
fig, ax = plt.subplots() 
ax.plot(lags, [autocorr(posterior_large, l) for l in lags], label='large step size') 
ax.plot(lags, [autocorr(posterior_small, l) for l in lags], label='small step size') 
ax.plot(lags, [autocorr(posterior, l) for l in lags], label='medium step size') 
ax.legend(loc=0) 
_ = ax.set(xlabel='lag', ylabel='autocorrelation', ylim=(-.1, 1))
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/bayesian_stat_2021060810144518.webp)

显然，我们希望有一种智能的方法来自动计算出正确的步宽。一种常见的方法是不断调整建议宽度，以便大约 50%的建议被拒绝。

## 7 扩展到更复杂的模型

现在你可以很容易地想象，我们还可以为标准差添加一个 $\sigma$ 参数，然后对第二个参数执行相同的步骤。在这种情况下，我们将为 $µ$ 和 $\sigma$ 生成建议，但算法逻辑将几乎相同。或者，我们可以从非常不同的分布（如二项分布）获得数据，但仍然使用相同的算法并得到正确的后验结果。这非常酷，概率编程的一个巨大好处是：只需定义您想要的模型，让 MCMC 负责推断。

例如，下面的模型可以很容易地用 PyMC3 编写。我们继续使用 Metropolis 采样器（它会自动调整建议的宽度），并得到了相同的结果。您可以随意尝试这一点并更改发行版。有关更多信息以及更复杂的示例，请参阅 PyMC3 文档 （http://pymc-devs.github.io/pymc3/getting_started/）。

```{code-cell} ipython3
import pymc3 as pm 
with pm.Model(): 
   mu = pm.Normal('mu', 0, 1) 
   sigma = 1. 
   returns = pm.Normal('returns', mu=mu, sd=sigma, observed=data) 
    
   step = pm.Metropolis() 
   trace = pm.sample(15000, step) 
    
sns.distplot(trace[2000:]['mu'], label='PyMC3 sampler'); 
sns.distplot(posterior[500:], label='Hand-written sampler'); 
plt.legend();
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429135912_b1.webp)

## 8 总结

有关于 MCMC 的细节当然很重要，但还有很多其他帖子涉及这一点。因此，本文重点在于直观地介绍 MCMC 和 Metropolis 采样器的核心思想。希望您已经收集了一些直观感觉。其他更奇特的 MCMC 算法，如：哈密尔顿蒙特卡罗，实际上与此非常相似，它们只是在提出下一步跳到哪里时要聪明得多。

本文有 Jupyter Notebook 版本，可以从 [此处](https://github.com/twiecki/WhileMyMCMCGentlySamples/blob/master/content/downloads/notebooks/MCMC-sampling-for-dummies.ipynb)  下载。
