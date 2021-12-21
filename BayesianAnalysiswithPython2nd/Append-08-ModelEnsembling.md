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

# 附录 H：模型集成

【摘要】贝叶斯模型平均(贝叶斯模型平均) 提供了一种连贯且系统的机制来解释模型的不确定性。它可以看作是贝叶斯推理在模型选择、组合估计和预测问题上的直接应用。贝叶斯模型平均产生了一个简单的模型选择标准和风险较低的预测。然而，贝叶斯模型平均的应用并不总是直截了当的，导致对其不同方面的不同假设和情境选择。

【原文】Fletcher, David. Why Model Averaging? SpringerBriefs in Statistics. 2018.

<style>p{text-indent:2em;2}</style>

## 1 为什么做模型集成？

模型平均是一种在估计中允许模型不确定性的方法，它可以提供比模型选择更好的估计和更可靠的置信区间。其应用领域非常广泛，涉及经济、药理、气象、水文等诸多领域。选择模型平均方法的原因在于：

**（1）模型选择及点估计方法容易产生过于自信的模型。** 

在许多经典统计推断的理论中，参数估计通常面向单个模型，该模型大多是从一组候选模型中选择出来的最佳模型。在解释数据时，选择该最佳模型的过程经常会被忽略，很容易导致点估计偏差和精度高估。这种现象被称为 “无声的丑闻”，而许多研究人员仍然没有意识到这个问题。

**（2）模型平均是一种可以考虑模型不确定性的估计方法。** 

- 在频率主义框架中，模型平均通常来自于每候选模型估计值的加权平均，其中权重反映了模型估计的潜力值。模型权重可能基于 `Akaike 信息准则 (AIC)`、`交叉验证`或感兴趣参数估计的`均方误差 (MSE)`。
- 在贝叶斯框架中，模型权重要么是模型为真的后验概率，要么是基于预测样本来确定，例如 `Watanabe-Akaike 信息准则 (WAIC)` 或`交叉验证`。通常模型权重被限制在单位单纯形上，即非负且总和为 1。

**（3）模型平均也可以被视为一种平衡估计偏差和方差的手段。** 

从频率主义角度来看，就像模型选择一样，较小的模型通常会提供具有较大偏差的估计，而较大的模型会导致估计值具有较高的方差。因此，模型平均在计算置信区间时考虑了模型不确定性，从而产生了比模型选择方法更宽、更可靠的区间。有趣的是，频率主义的一些作者只专注于实现模型平均估计的偏差和方差之间的平衡，而其他人则认为模型平均仅仅是一个允许模型不确定性的手段而已。


> **模型选择、模型集成、模型组合和模型平均的区别：**
>
> - **模型选择**指从若干模型中选择出人们认为最优的那个模型，由于评判方法、数据集等原因，模型选择很容易造成对模型过高的自信。
> - **模型集成**采用某种方式融合多个已经训练好的模型，进而达到 “取长补短”、提高模型泛化能力的效果。模型集成方法通常在几个模型差异性较大、相关性较小的情境下效果更加明显。常用的集成方法有：投票法(voting)、平均法(averaging)、 堆叠法(Stacking)、混合法(Blending)、状态法（Bagging）、自助法（Boosting）等。
> - **模型组合**是一种极易与模型集成混淆的方法，因为它也会组合多个模型，但与集成方法不同之处在于，模型组合指在预测变量的不同子空间上选择使用不同模型。也就是说，组合方法中的每一个模型只作用在预测变量的部分空间上，而集成方法中的所有模型都会作用于整个预测变量空间。
> - **模型平均**是最基础的模型集成方法之一，其基本原理是对多个模型计算的结果进行加权平均，以得到最终结果，在回归问题及含阈值调节的场景中使用更多一些。

## 2 贝叶斯模型平均

### 2.1 概述

原则上，贝叶斯模型平均 (BMA) 是非常自然的，因为用于数据分析的贝叶斯框架可以很容易地结合参数不确定性和模型不确定性 [73, 87]。我们不是基于单个模型计算参数的后验分布，而是计算来自不同模型的后验分布的加权组合。

在 BMA 的经典版本中，模型的权重是它为真的后验概率，因为我们假设其中一个模型为真。最近，[181] 提出了一个明确的基于预测的 BMA 版本。这使用交叉验证来确定一组最佳权重，并且是称为堆叠的频率论方法的贝叶斯版本（第 3.2.3 节）。 1 

经典 BMA 关注模型一致性（真实模型的识别） ，而基于预测的 BMA 具有直接与估计相关的优势。我们最初关注经典 BMA，因为它在 BMA 文献中受到了绝大多数关注。

### 2.2 经典 BMA

Once the priors are specified for both the models and the parameters in each model, classical BMA involves a well-defined procedure for obtaining the model-averaged posterior for any parameter of interest. As with other Bayesian methods, the apparent simplicity of classical BMA can bely difficulties in its implementation $[8,32]$. The choice of priors can be problematic, as can the computations required to obtain the model-averaged posterior.

Suppose $y=\left(y_{1}, \ldots, y_{n}\right)^{\top}$ contains the values of the response variable, and we wish to average over a set of $M$ nested models, with $\beta_{m}$ being the vector of $p_{m}$ parameters in model $m$. As the notation implies, $\beta_{m}$ will often be a set of regression coefficients, but it may include other types of parameter, such as the error variance in a normal linear model. ${ }^{2}$ Suppose we are interested in estimating a scalar parameter $\theta$. The model-averaged posterior for $\theta$ is given by

$$
p(\theta \mid y)=\sum_{m=1}^{M} p(m \mid y) p(\theta \mid y, m),
$$

where $p(m \mid y)$ is the posterior probability that model $m$ is true, and $p(\theta \mid y, m)$ is the posterior for $\theta$ when we assume model $m$ is true. ${ }^{3}$ Thus $p(\theta \mid y)$ is a weighted combination of the posterior distributions obtained from the different models, the weights being the posterior model probabilities. Using Bayes' theorem, the posterior probability for model $m$ is given by

$$
p(m \mid y) \propto p(m) p(y \mid m),
$$

where $p(m)$ and $p(y \mid m)$ are the prior probability, and the marginal (integrated) likelihood, for model $m$, with

$$
p(y \mid m)=\int p\left(y \mid \beta_{m}, m\right) p\left(\beta_{m} \mid m\right) d \beta_{m},
$$

where $p\left(\beta_{m} \mid m\right)$ is the prior for $\beta_{m}$ and $p\left(y \mid \beta_{m}, m\right)$ is the likelihood under model $m$. When the support of $\beta_{m}$ is discrete, the integral in (2.3) is replaced by a summation.

The posterior probability for model $m$ can also be written as

$$
p(m \mid y) \propto p(m) B_{m},
$$

where

$$
B_{m}=\frac{p(y \mid m)}{p(y \mid 1)}
$$

is the Bayes factor for comparing model $m$ and model 1 , the latter being an arbitrary reference model $[114,146]$. It is well known that Bayes factors can be sensitive to the prior distributions for the parameters, even when $n$ is large $[8,97]$. An extreme case arises when one or more of the priors is improper, as this can lead to the marginal likelihood in (2.3) not being well defined [73, 161].

Two natural summaries of the model-averaged posterior for $\theta$ are the mean and variance, given by

$$
\mathrm{E}(\theta \mid y)=\sum_{m=1}^{M} p(m \mid y) \mathrm{E}(\theta \mid y, m)
$$

and

$$
\operatorname{var}(\theta \mid y)=\sum_{m=1}^{M} p(m \mid y)\left[\operatorname{var}(\theta \mid y, m)+\{\mathrm{E}(\theta \mid y, m)-\mathrm{E}(\theta \mid y)\}^{2}\right]
$$

where $\mathrm{E}(\theta \mid y, m)$ and $\operatorname{var}(\theta \mid y, m)$ are the posterior mean and variance for $\theta$ under model $m$. Thus the model-averaged posterior variance is influenced by both the parameter uncertainty associated with each model and the between-model variability in the posterior mean.

We can also use the model-averaged posterior to calculate a central $100(1-2 \alpha) \%$ credible interval for $\theta$, given by $\left[\theta_{L}, \theta_{U}\right]$, where

$$
\int_{-\infty}^{\theta_{L}} p(\theta \mid y) d \theta=\int_{\theta_{U}}^{\infty} p(\theta \mid y) d \theta=\alpha .
$$

An alternative choice would be to use the highest posterior density $100(1-2 \alpha) \%$ credible region, i.e. the region of values for $\theta$ that contain $100(1-2 \alpha) \%$ of the posterior probability and for which the posterior density is never lower than outside the region. However, the central credible interval is easier to compute than the highest posterior density region, and has the advantage that the limits can be interpreted as quantiles of the posterior. In the examples we therefore use central credible intervals.

A by-product of classical BMA is the ability to calculate a posterior inclusionprobability (PIP) for each predictor variable, i.e. the sum of the posterior probabilities for all the models that include that variable $[7,12,36]$. Some authors have suggested that these provide a useful summary of the relative importance of each predictor variable. However, as they are influenced by the choice of model set, the importance of a predictor variable can be exaggerated by including many models containing that variable [66]. In addition, a more useful summary of relative importance can be obtained by comparing model-averaged posterior distributions for the expected value of the response variable for a suitable set of values of the predictor variables (Sect. 1.4). An analogous issue arises with the use of summed model weights in frequentist model averaging (Sect. 3.7). ${ }^{4}$

#### 2.2.1 后验模型概率

When calculating the posterior model probabilities, it can be difficult to determine the marginal likelihood in (2.3). In some settings, such as GLMs with conjugate priors for the parameters, the marginal likelihood can be expressed analytically [98], but in general we need to use an approximation. The most well-known approximation, which does not require specification of the priors for the parameters, is

$$
p(y \mid m) \approx \exp \left(-\mathrm{BIC}_{m} / 2\right)
$$

where

$$
\mathrm{BIC}_{m}=-2 \log p\left(y \mid \widehat{\beta}_{m}, m\right)+p_{m} \log n,
$$

and $\widehat{\beta}_{m}$ is the maximum likelihood estimate of $\beta_{m}$. The expression in (2.8) is the Bayesian information criterion for model $m[91,98,114,157] .{ }^{5}$ The first term in (2.8) is influenced by the fit of the model, while the second can be thought of as a correction for overfitting which penalises more complex models.

Use of (2.7) in (2.2) leads to the approximation

$$
p(m \mid y) \propto p(m) \exp \left(-\mathrm{BIC}_{m} / 2\right),
$$

which is sometimes referred to as the generalised BIC weight [114]. When we calculate the expression on the right-hand side of (2.9), $\mathrm{BIC}_{m}$ is often replaced by

$$
\mathrm{BIC}_{m}-\min _{k} \mathrm{BIC}_{k},
$$

in order to avoid large arguments in the exponential function [32]. ${ }^{6}$

Care is needed in specifying the value of $n$ in (2.8). For a normal linear model, it is simply the number of observations. For a binomial model it is the total number of Bernoulli trials. When using a log-linear model to analyse a contingency table it is the sum of the counts rather than the number of cells in the table $[98,145]$. In the context of survival analysis, [171] suggested setting $n$ equal to the number of uncensored observations. For a hierarchical model, the choice of $n$ depends on the focus of the analysis $[37,98,138,139,188]$. For modelling survey data, [117] proposed a version of BIC that takes into account the design effect.

A higher-order approximation to the posterior model probabilities requires the priors for the parameters and their observed Fisher information matrix $[96,98,116$, 183]. It is similar in spirit to TIC, Takeuchi's information criterion, which has been suggested as an alternative to $\mathrm{AIC}$ in the frequentist setting [32] (Sect.3.2.1).

Other approaches to approximating the posterior model probabilities have been proposed, involving marginal likelihoods $[26,28,40,41,121,132,151]$ or Markov chain Monte Carlo (MCMC) methods [5, 9, 19, 23, 24, 29, 34, 37, 50, 71, 74, $75,79,82,121,142,147,179]$. One conceptually-appealing method is reversiblejump MCMC (RJMCMC), in which we sample the parameter-space and model-space simultaneously [80]. However, RJMCMC can be prone to performance issues and be challenging to implement $[8,9,37]$, to the extent that $[83]$ recommend use of the approximation in (2.9). Recently, [8] have developed an approach which has the advantage of using the MCMC output obtained from fitting each model separately, and which exploits the relationships between parameters from different models [151].

#### 2.2.2 先验的选择

##### （1）模型的先验

A natural and common choice for the prior model probabilities is the uniform prior, in which $p(m)=1 / M$. The approximation to the posterior model probability in (2.9) then simplifies to the well-known BIC weight, given by

$$
p(m \mid y) \propto \exp \left(-\mathrm{BIC}_{m} / 2\right) .
$$

However, use of the uniform model-prior can have hidden implications. For example, if some of the possible predictor variables are highly correlated, we may have modelredundancy, in that some models will provide very similar estimates of $\theta$. Use of a uniform prior will then dilute the prior probability allocated to any model which is not similar to the others $[18,57,66,76]$. A method for dealing with this problem was proposed by [170], who suggested specifying prior model probabilities using the concept of the worth of a model, which is based on quantifying what we would expect to lose if we removed it from the model set when it is the true model. Another alternative is use of a Bernoulli prior in which each predictor variable has the same probability $p$ of being included, independently of the others. The uniform prior is a special case, with $p=0.5$, and therefore corresponds to a prior expectation that half the predictor variables will be included [37]. In order to have a less informative prior on model size, we might use a beta-prior for $p[37,105,158]$.

Other approaches to specifying the model-prior involve empirical Bayes [37]; allowance for predictor variables being related, such as when some of the models include interaction terms [30]; and use of lower weights for models that are similar to others [66].

In order to allow for the possibility that the choice of model-prior may affect the form of the model-averaged posterior, [43] proposed use of credal model averaging, in which more than one model-prior is considered. This effectively allows one to preform a sensitivity analysis, in order to assess the extent to which the modelaveraged posterior is influenced by the choice of model-prior. Further examples of the use of credal model averaging can be found in [44-46, 185].

##### （2）模型参数的先验

The posterior model probabilities can be sensitive to the choice of prior distribution for the parameters in a model, even if this prior would be regarded as non-informative in the single-model setting $[8,35,93]$. In particular, as mentioned in Sect. $2.2$, the use of improper priors can lead to the Bayes factors, and hence the posterior model probabilities, not being well defined $[10,82,93,161]$. It is also possible for apparently sensible priors for the parameters to cause the models to have conflicting implicit prior distributions for $\theta$ [85].

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

### 2.3 基于预测的 BMA
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
 
 使用 WAIC 或 BSP 权重比经典 BMA 更可取，因为重点是预测而不是识别真实模型。使用这些权重还避免了后验模型概率的计算以及这些概率对参数先验的敏感性的问题。
 
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