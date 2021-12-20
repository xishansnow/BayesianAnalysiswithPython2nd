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

# 附录 B： 变分法推断

<style>p{text-indent:2em;2}</style>

## 1 问题提出

概率模型中的推断通常难以处理，根据之前的介绍，通过对随机变量进行采样的方法，能够为推断问题（例如，边缘似然推断）提供近似解。大多数基于采样的推断算法是属于马尔可夫链蒙特卡罗 (MCMC) 方法。

不幸的是，上述基于采样的方法有几个重要缺点。

- 尽管随机方法可以保证找到全局最优解，但前提是时间充足，而这在实践中通常是受限的
- 目前尚没有好的方法能够判断采样结果与真实解到底有多接近，收敛状况仍然需要人工判断
- 为了找到一个效率足够高的解决方案，MCMC 方法需要选择合适的采样技术（例如，Metropolis-Hastings 、 HMC 、 NUTS 等），而选择本身就是一门艺术。

为此，人们一直在寻找在保证适当精度条件下，比 MCMC 效率更高的后验推断方法，这就是变分推断方法。

## 2 变分推断

### 2.1 核心思想

根据贝叶斯概率框架，我们希望根据已有数据，来推断参数或隐变量的分布 $p$ ，进而能够做后续的预测任务；但是，当 $p$ 不容易表达，无法得到封闭形式解时，可以考虑寻找一个容易表达和求解的分布 $q$ 来近似 $p$ ，当 $q$ 和 $p$ 之间差距足够小时， $q$ 就可以代替 $p$ 作为输出结果，并执行后续任务。 当 $q$ 由若干变分参数 $\nu$ 定义时，该问题就变成了一个寻找 $\nu$ 的最优化问题，优化目标是最小化 $q$ 和 $p$ 两个概率分布之间的差距。其中， $q$ 被成为变分分布， $\nu$ 被成为变分参数。

> 变分推断的精髓是将 “求分布” 的推断问题，变成了 “缩小差距” 的优化问题。

以下图为例进行说明，黄色分布为目标分布 $p$ ，很难求解，但它看上去有些像高斯，我们可以尝试从高斯分布族 $\mathcal{Q}$ 中（此处以一个红色高斯和一个绿色高斯为例，其变分参数为均值 $\mu$ 和 方差 $\sigma^2$ ），寻找和 $p$ 最像的（可以用“重叠面积比率最大” 或 “差异最小” 来度量 ） 那个 $q$ ， 作为 $p$ 的近似分布。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211108085400-e452.webp)

> 图 1：单变量的真实分布与近似分布。 此处简化了假设，更为复杂的分布应当具有更多变量，且涉及多维空间中的多峰分布。

变分方法与采样方法的主要区别是：

- 变分方法几乎永远找不到全局最优解

- 可以知道自身收敛情况，甚至可以对其准确性有限制

- 更适合于随机梯度优化、多处理器并行处理以及使用 GPU 加速

虽然采样方法的历史非常悠久，但随着大数据的发展，变分方法一直在稳步普及，并成为目前使用更为广泛的推断技术。
 
 ###  2.2 进一步加深理解

变分推断的核心要点包括以下四个：

- **模型**：变分推断需要设计一个模型  $p(\mathbf{z, x})$ ，其中 $\mathbf{x}$ 为观测数据， $\mathbf{z}$ 为模型参数或/和隐变量（后面统一用隐变量来表示，但需要理解模型参数和隐变量有时具有不同的含义）。模型来自专家个人的业务知识和归纳偏好，通常可以用概率图模型表达，而数据则来自实际观测。
- **目的**：基于观测数据 $\mathbf{x}$ 确定模型中隐变量 $\mathbf{z}$ 的值，但各种不确定性因素导致隐变量无法得到一个确切值，通常只能给出其概率分布 $p(\mathbf{z \mid x})$ ，而最大的问题在于这个分布可能很复杂，无法直接给出封闭形式的解。
- **原理**：变分推断将人为构造一个由参数 $\boldsymbol{\nu}$ 索引（或定义）的概率分布 $q(\mathbf{z};\boldsymbol{\nu})$ ，并通过一些方法或技巧来优化 $\boldsymbol{\nu}$，使 $q(\mathbf{z};\boldsymbol{\nu})$ 能够近似并且代替复杂的真实分布  $p(\mathbf{z \mid x})$ 。
- **途径**：通过一些优化算法，调整 $\boldsymbol{\nu}$ 以缩小 $q(\mathbf{z};\boldsymbol{\nu})$ 和   $p(\mathbf{z \mid x})$ 之间的差异直至收敛。

#### 2.2.1 变分推断基于贝叶斯模型

简单说，专家利用其领域知识和归纳偏好，给出一个模型假设 $p(\mathbf{z, x})$ ，其中包括隐变量  $\mathbf{z}$  和观测变量 $\mathbf{x}$ ，还有相互之间可能存在依赖关系。变分推断是基于该模型实施的。

> 注： 此处隐变量和观测变量采用黑体符号，表示其为向量形式，即代表了不止一个随机变量。

为了理解隐变量和观测变量之间的关系，一种比较易于理解的表达方式是建立 “观测变量的生成过程”  。可以认为，观测变量是从某个已知的、由隐变量组成的结构中生成出来的。

以高斯混合模型为例（见下图）。图中为我们观测到一个数据集，它实际上是从由 5 个相互独立的高斯组合而成的一个混合分布中采样的结果。如果从单个数据点出发，考虑其生成过程，可以分为两步：首先从 5 个类别组成的离散型分布中抽取一个类别样本（比如粉红色类别），然后从类别对应的高斯分布中抽取一个样本生成相应数据点。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211108085412-fdef.webp)

> 图 2 ： 5 个组份构成的高斯混合模型示意图

在上例中，可以发现隐变量可能有多个：

-  5 个高斯分布的均值参数 $\boldsymbol{\mu}$ 和方差参数 $\boldsymbol{\sigma^2}$ （均为长度为 5 的向量）
-  数据点所属类别 $\mathbf{c}$ （独热向量形式）

上述变量 $\boldsymbol{\mu}$、$\boldsymbol{\sigma}^2$ 和 $\mathbf{c}$ 一起构成了隐变量 $\mathbf{z}$ ，而且 $\boldsymbol{\mu}$、$\boldsymbol{\sigma}^2$  和 $\mathbf{c}$ 之间可能存在一定的依赖关系。

#### 2.2.2 问题本质是基于观测数据推断隐变量的分布

后验概率 $p(\mathbf{z|x})$ 的物理含义是：基于现有观测数据 $\mathbf{x}$ ，推断隐变量 $\mathbf{z}$ 的分布。 

对于上面的高斯混合模型来说，变分推断的目的就是求得隐变量 $\mathbf{z} = \{\boldsymbol{\mu},\boldsymbol{\sigma^2},  \mathbf{c} \}$  的后验分布  $p(\mathbf{z} \mid \mathbf{x})$ 。根据贝叶斯公式，$p(\mathbf{z} \mid \mathbf{x}) = p(\mathbf{z,x}) / p(\mathbf{x})$ 。 根据专家提供的生成过程，能够写出联合分布 $p(\mathbf{z,x})$ 的表达式，但边缘似然 $p(\mathbf{x})$ 是一个很难处理的分母项。当 $\mathbf{z}$ 为连续型时，边缘似然 $p(\mathbf{x})$ 需要求 $p(\mathbf{z,x})$ 关于 $\mathbf{z}$ 所在空间的积分（即边缘化）；当 $\mathbf{z}$ 为离散型时，需要对所有可能的 $\mathbf{z}$ 求和；而积分（和求和）的计算复杂性随着样本数量的增加会呈指数增长。

#### 2.2.3 变分推断的关键是构造变分分布 

变分推断需要构造变分分布 $q(\mathbf{z}; \boldsymbol{\nu})$，并通过优化调整 $\boldsymbol{\nu}$，使 $q(\mathbf{z}; \boldsymbol{\nu})$ 更接近真实后验 $p(\mathbf{z \mid x})$ 。

在变分分布 $q(\mathbf{z}; \boldsymbol{\nu})$ 中， $\mathbf{z}$ 为隐变量，$ \boldsymbol{\nu}$ 是控制 $q$ 形态的参数（例如：假设 $q$ 为高斯分布，则 $\boldsymbol{\nu}$ 为均值和方差 ， 而对于泊松分布，则 $\boldsymbol{\nu}$ 为二值概率）。因此构造变分分布 $q$ 分为两步：

- 首先是概率分布类型的选择。通常依据 $p$ 的形态，由专家给出，例如在上述高斯混合模型中，假设目标分布 $p$ 服从多元高斯分布，则构造 $q$ 时通常依然会考虑高斯分布。

- 其次是概率分布参数的确定。该确定过程是一个逐步优化迭代的过程，通常经过渐进调整 $\boldsymbol{\nu}$ 的值 ，使 $q$ 逐渐逼近 $p$ 。

####  2.2.4 变分推断是一个优化问题

变分推断是一个优化问题，其最直观的优化目标是最小化 $KL$ 散度，但实际可操作的优化目标是最大化证据下界 $ELBO$ 。

直观地理解，变分推断的优化目标很明确，就是最小化 $q$ 和 $p$ 之间的差距，而信息论为我们提供了一个量化两个分布之间差距的工具 $KL$ 散度。但不幸的是，$KL$ 散度的计算并不简单，其计算表达式中同样存在难以处理的边缘似然积分项（边缘似然也称证据，在一般化的数学形式中也被成为配分函数）。为此，有人提出了可操作的优化目标 --- 证据下界 $ELBO$ （见 2.4 节）。

在证据下界 $ELBO$ 的计算表达式中，只包括联合分布  $p(\mathbf{z, x})$  和变分分布 $q(\mathbf{z}; \boldsymbol{\nu})$ ，摆脱了难以处理的边缘似然积分项。并且在给定观测数据后，最大化 $ELBO$ 等价于最小化 $KL$ 。 也就是说，$ELBO$ 的最大化过程结束时，获得的输出 $q(\mathbf{z}; \boldsymbol{\nu^\star})$就是我们寻求的最终变分分布。

#### 2.2.5 变分推断的形象化解释

下图为变分推断的示意图。图中大圈表示了一个分布族，由参数 $\boldsymbol{\nu}$ 索引。我们也可以理解为，大圈是分布 $q$ 的参数 $\boldsymbol{\nu}$ 的取值空间，圈中每个点均表示不同 $\boldsymbol{\nu}$ 对应的分布 $q$ 。例如，在高斯混合模型的例子中， $q$ 属于高斯分布族，则不同的均值和方差，代表了不同的分布 $q$ 。从 $\boldsymbol{\nu}^{init}$ 到 $\boldsymbol{\nu}^*$ 的路径表示变分推断的优化迭代过程，通过优化更新 $\boldsymbol{\nu}$ 值来缩小 $q$ 与 $p$ 之间的差距，而这个差距通常采用 $KL$ 散度来衡量。路径终点 $\boldsymbol{\nu}^*$ 对应的分布 $p$ 与真实后验 $p(z \mid x)$ 之间应当是某种距离测度上的最小值。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211108085528-fa88.webp)

> 图 3 ：变分推断示意图。图中隐变量为 $\mathbf{z}$，包含局部变量和全局变量。$\nu$ 对应了所有变分参数。

### 2.3 概率图表示

#### 2.3.1 贝叶斯概率框架

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211108085443-66c1.webp)

> 图 4 ：贝叶斯概率框架示意图

上图是贝叶斯统计问题的一般求解流程图。领域专家拥有知识可以用来建模，并且存在需要解答的问题。他们依据拥有的知识，给出带有归纳偏好的合理假设，构建出数据的生成过程模型 (Generative  Processing Model)。模型中主要包括隐变量、观测变量以及它们之间的依赖关系。

利用该模型，我们希望通过对数据的处理，挖掘出有价值的模式，然后实现各式各样的应用。

还是以高斯混合模型为例，可以将其生成过程形式化为：

$$
\begin{aligned}
 \boldsymbol{\mu}_{k} & \sim \mathcal{N}\left(0, \sigma^{2}\right), & & k=1, \ldots, K, &（先验）\\
\mathbf{c}_{i} & \sim \text { Categorical }(1 / K, \ldots, 1 / K), & & i=1, \ldots, n, &（先验）\\
x_{i} \mid \mathbf{c}_{i}, \boldsymbol{\mu} & \sim \mathscr{N}\left(\mathbf{c}_{i}^{\top}  \boldsymbol{\mu}, 1\right) & & i=1, \ldots, n &（似然）
\end{aligned}
$$

该模型混合了 $K$ 个相互独立的高斯组份（ $K$ 是超参数），模型中的隐变量包括数据点所在的类别 $ \mathbf{c}_i$，各类别组份的均值 $\boldsymbol{\mu}_k$ 和方差 $ \boldsymbol{\sigma}_k^2$ （为简化问题，此处假设 $ \boldsymbol{\sigma}_k^2$ 为常数 1 ）。

设所有高斯组份的均值参数 $\boldsymbol{\mu}_k$ 的先验为 0 均值的同方差（$\sigma^2$）高斯，如第一行所示；类别参数 $ \mathbf{c}_i$ 的先验为均匀的类别分布（因为缺乏关于类别的先验知识），如第二行所示。假设所有样本数据都是从某个方差为 $1$ 的高斯混合模型中随机生成的，则任一数据点 $x_i$ 的生成过程可以通过似然表达出来（如第三行所示），分为两步 ：

- 第一步，从类别分布中随机抽取一个离散型的类标签 $ \mathbf{c}_i$ （此处采用独热变量形式）；

- 第二步，从类标签 $ \mathbf{c}_i$ 对应的均值为 $\mu =  \mathbf{c}_{i}^{\top} \boldsymbol{\mu}$ 、方差为 $1$ 的高斯分布中，随机抽取一个数据点 $x_i$。

更细致的举例说明见图 2，五个分布用不同颜色表示，代表五个类别。每次从中选择一个类，如第三类粉红色，$\mathbf{c}_i=\{0，0，1，0，0\}$ ，然后从第三类对应的高斯分布中抽取 $x_i$ ，其均值为 $\mathbf{c}_i \cdot \boldsymbol{\mu}=\mu_3$。该点大概率出现在粉红色类覆盖的区域内。

#### 2.3.2 利用概率图表示和理解

概率图模型是表达随机变量之间结构和关系的有力工具（参见概率图理论文献）。上述高斯混合模型的形式化生成过程，在概率图模型中体现为三个随机变量及它们之间的依赖关系（见下图），其中高斯组份的均值 $\boldsymbol{\mu}$ 和数据点所属类别 $\mathbf{z}$ 为隐变量，$\mathbb{x}_i$ 为观测变量。此外，可以做更细致的划分，将 $\boldsymbol{\mu}$ 视为全局随机变量，其作用发挥在所有数据上，而 $\mathbb{z}_i$ 为局部随机变量，只和数据点 $x_i$ 相关，与其他点无关。局部变量用矩形框与全局变量分隔开来。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211108085458-b4a9.webp)

>  图 5：高斯混合模型的概率图（1）白圈为隐变量，灰圈为观测变量；（2）矩形框内为局部变量，外部为全局变量；（3）矩形框表示其内部的变量 $z_i$ 和 $x_i$ 独立重复了 n 次；（4）箭头表示随机变量之间的条件依赖关系。

基于上述概率图，可以解析地写出所有随机变量的联合概率分布（详情参见概率图原理）：

$$
p(\boldsymbol{\mu}, \mathbf{z}, \mathbf{x})=p(\boldsymbol{\mu}) \prod_{i=1}^{n} p\left(z_{i}\right) p\left(x_{i} \mid z_{i}, \boldsymbol{\mu}\right)
$$

那么在上述高斯混合模型中，到底需要变分方法推断什么？

根据统计机器学习框架，此例中观测数据 $\mathbf{x}$ 由无标签的数据点构成，我们需要根据这些数据点来推断 $\mathbf{z}$ 和 $\boldsymbol{\mu}$ 的分布。即给定观测数据 $\mathbf{x}$ 时，推断隐变量的后验概率。根据基础概率公式，其形式化描述为：

$$
p(\boldsymbol{\mu}, \mathbf{c} \mid \mathbf{x})=\frac{p(\boldsymbol{\mu},\mathbf{c} , \mathbf{x})}{p(\mathbf{x})}
$$

式中，分子项为 $p(\boldsymbol{\mu},\mathbf{c} , \mathbf{x})$ ， 可根据概率图模型理论写出表达式，但分母 $p(\mathbf{x})$ 涉及所有数据点的积分问题，是难以处理的，因此，需要用近似的、可处理的近似分布 $q(\boldsymbol{\mu},\mathbf{c})$ 来代替后验分布 $p(\boldsymbol{\mu}, \mathbf{c} \mid \mathbf{x})$ 。

#### 2.3.3 更一般的形式

我们把上述问题梳理一下，能够得到一个更一般的形式，可以用来描述各种各样的贝叶斯统计模型：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211108085516-aec3.webp)

>  图 6：包含全局变量和局部变量的一般性概率图表示

在上面的概率模型中，后验概率的分子项为联合概率分布，可通过概率图得出表达式（图 6 下）；但分母中的边缘似然项是难以处理的（见下式）。即便隐变量是离散型变量，存在 $K$ 个可能值，其计算复杂度也为 $O(K^n)$ ，意味着其计算复杂度会随着数据量 $n$ 的增长呈指数增长。

$$
p(\beta, z \mid x)=\frac{p(\beta, z, x)}{p(x)}=\frac{p(\beta, z, x)}{\iint_{\beta z} p(x) d \beta d z}
$$

### 2.4 目标函数的选择

要将推断转化为优化问题，需要选择或者构造一个变分分布族 $\mathcal{Q}$ ，并设定合适的优化目标 $J(q)$ 。 关于分布族的讨论放在下一节，此处先讨论优化目标 $J(q)$ 。该目标需要捕获 $q$ 和 $p$ 之间的差距（或反之，相似性），而信息论为我们提供了一个直观并且好理解的工具，被称为 **KL（ Kullback-Leibler ）散度**。

从形式上理解，*KL 散度* 指两个分布之间的差异。 $q$ 和 $p$ 之间离散形式的 **KL 散度** 被定义为：

$$
KL(q \| p) = \sum_x q(x) \log \frac{q(x)}{p(x)}
$$

在信息论中，此函数用于测量两个分布中所包含信息的差异大小。$KL$ 散度具有以下性质，使其在近似推断任务中特别有用：

-  对所有 $q$ 和 $p$ ，都有 $KL(q\|p) \geq 0$ 。 
-  当且仅当 $q = p$ 时，$KL(q\|p) = 0$ 。 

> 相关证明过程可以作为练习，本文略过。

但请注意： $KL$ 散度不具备可逆性（或对称性），即 $KL(q\|p) \neq KL(p\|q)$ 。这也是称之为`散度`，而不是`距离`的原因，因为距离是双向对称的。

$KL$ 散度虽然很好理解，但是是否能够直接使用 $KL$ 散度进行变分推断呢？ 我们有必要针对一般化形式的目标分布 $p$ 做一下分析。 

假设 $p$ 为一个泛化（离散、简单）的、具有如下形式的无方向概率模型（其实该形式是统计机器学习框架中后验分布的一般化形式，也是变分推断面临的主要场景）：

$$
p(x_1,\ldots,x_n; \theta) = \frac{\tilde p(x_1,\ldots,x_n ; \theta)}{Z(\theta)} =\frac{1}{Z(\theta)} \prod_{k} \phi_k(x_k; \theta)
$$

其中 $k$ 为模型中因子的数量（有关因子的概念，参见无向概率图或马尔科夫随机场相关资料）， $\phi_k(\cdot)$ 为因子， $Z(\theta)$ 为归一化常数或配分函数。假设该形式能够涵盖所有可能的目标分布， 则在该一般化形式下，归一化常数 $Z(\theta)$ 面临着难以处理的问题。由于 $KL$ 散度的计算需要用到 $p(x)$ ，因此 $Z(\theta)$ 的存在导致无法直接使用 $KL(q\|p)$ 做为优化目标。

考虑到归一化常数虽然难以处理，但在观测数据确定的情况下是常数，因此，可以考虑仅处理形式上与 $KL$ 散度相似的未归一化分子项 $\tilde p(x) = \prod_{k} \phi_k(x_k; \theta)$ ，即将优化目标调整为：
$$
J(q) = \sum_x q(x) \log \frac{q(x)}{\tilde p(x)}
$$

上述目标函数中， $q(x)$ 来自人工构造和 $\tilde p (x)$ 来自于数据概率生成过程，各项均可处理。 更重要的是，其还具有以下重要性质：

$$
\begin{align*} 
J(q) &= \sum_x q(x) \log \frac{q(x)} {\tilde p(x)} \\ 
&= \sum_x q(x) \log \frac{q(x)}{p(x)} - \log Z(\theta) \\ 
&= KL(q\|p) - \log Z(\theta) 
\end{align*}
$$

也就是说，新的目标函数是 $KL$ 散度与对数配分函数之差，而根据 $KL$ 散度的性质， $KL(q\|p) \geq 0$ ，则重排公式可得：

$$
\log Z(\theta) = KL(q\|p) - J(q) \geq -J(q)
$$

该式表明，$-J(q)$ 是对数配分函数 $\log Z(\theta)$ 的下界。

在大多数应用场景中， 对配分函数 $Z(\theta)$ 都有特别的解释。例如，在给定观测数据 $D$ 的情况下，我们可能想计算随机变量 $x$ 的边缘概率 $p(x \mid D) = p(x,D) / p(D)$ 。在这种情况下，最小化 $J(q)$ 等价于最大化对数边缘似然 $\log p(D)$  的下界。正是因为如此， $-J(q)$ 被称为变分下界或证据下界（ $ELBO$ ），这种下界关系经常以如下不公式形式表示：

$$
\log Z(\theta) \geq \mathbb{E}_{q(x)} [ \log \tilde p(x) - \log q(x) ]
$$

最关键的是，对数边缘似然 $\log Z(\theta)$ 和变分下界 $-J(q)$ 的差，正好是 $KL$ 散度。 因此，最大化变分下界等效于通过 "挤压" $-J(q)$ 和  $\log Z(\theta)$ 之间的差，实现 $KL(q\|p)$ 的最小化。 

--- 

**讨论：**

刚刚为变分推断重新定义了优化目标，并且表明，最大化下界等效于最小化散度 $KL(q\|p)$ 。 

回想之前说过的 $KL(q\|p) \neq KL(p\|q)$ ; 当 $q = p$ 时，两种散度都等于零，但当 $q \neq p$ 时，两者的值不同。 这就提出了一个问题：为什么选择其中一个而非另一个，它们有何不同？

也许最重要的区别是计算效率：优化 $KL(q\|p)$ 时，涉及关于 $q$ 求期望；而优化 $KL(p\|q)$ 则需要关于 $p$ 求期望，而这又是难以处理甚至无法评估的。

但当近似分布族 $\mathcal{Q}$ 不包含真实分布 $p$ 时，选择  $KL(p\|q)$ 会影响返回的解。 通过观察会发现， 如果 $p(x) = 0$ 并且 $q(x) > 0$ 时， $KL(q\|p)$ 是无限的 （ 该散度有时被称为 I 投影或信息投影 ） :

$$
KL(q\|p) = \sum_x q(x) \log \frac{q(x)}{p(x)}
$$

这意味着，如果 $p(x) = 0$ ，则必须有 $q(x) = 0$ 。 我们称： $KL(q\|p)$ 对于 $q$ 是零强制的，并且它通常会低估 $p$ 的支持。 

另一方面，当 $q(x) = 0$ 并且 $p(x) > 0$ 时， $KL(p\|q)$ 是无限的 （ 该散度被称为 M 投影或矩投影）。 因此，如果 $p(x) > 0$ ，则必须有 $q(x) > 0$ 。 此时，我们称： $KL(p\|q)$ 对于 $q$ 是零避免的，并且它通常会高估 $p$ 的支持。 

下图以图形方式说明了该现象。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211108081010-6d1d.webp)

> 图 7：将单峰近似分布 $q$（红色）拟合到多峰 $p$（蓝色）。 (a) 使用 $KL(p||q)$ 会导致 $q$ 试图覆盖两个峰。 (b）(c) 使用 $KL(q||p)$ 会迫使 $q$ 选择 $p$ 的其中一个峰。

鉴于两种散度的上述性质，我们经常称 $KL(p\|q)$ 为包容性 $KL$ 散度， 而 $KL(q\|p)$ 是独占性的 $KL$ 散度。

---

小结：

- 由于配分函数 $Z$ 的存在，使得直接计算 $KL$ 散度难以处理
- 采用等价的最大化 $ELBO$ 来代替最小化 $KL$ 散度作为新的优化目标
- $ELBO$ 中只包含变分分布 $q$ 和目标分布的分子项 $\tilde p$ ，并且两者都可计算，进而可以作为真正的优化目标
- $KL(q\|p)$ 和 $KL(p\|q)$ 虽然都是散度，也都可以作为优化目标，但作为目标时两者达到的效果截然不同。对于多峰后验，前者倾向于选择能够覆盖其中某个峰的变分分布，而后者倾向于选择能够覆盖多个峰的变分分布。

## 3 经典方法 --- 平均场、指数族与坐标上升算法

### 3.1 平均场近似的基本原理

在构造变分分布时，需要在 $q(\mathbf{z};\boldsymbol{\lambda}) $ 的表达能力和简单易处理性之间做权衡。其中一种常见的选择是使用完全因子分解的方法来构造分布，也称为平均场分布。

平均场近似假设所有隐变量相互独立，进而简化了推导。但这种独立性假设也会导致不太准确的近似，当后验中的随机变量存在高度依赖时尤其如此。因此，有人研究更为准确的近似方法，本文第 6 节会讨论此类更具表现力的变分分布。

平均场变分推断 ( MFVI ) 起源于统计物理学的[平均场理论](https://ieeexplore.ieee.org/book/6267422)。在平均场近似中，基于独立性假设，变分分布被分解为各因子分布的乘积，而每个因子由其自身变分参数控制：

$$
q(\mathbf{z};\boldsymbol{\lambda}) = \prod_{i=1}^N q(z_i ; \lambda_i)
$$


为了符号简单，我们在本节的其余部分省略了变分参数 $λ$ 。我们现在回顾如何在平均场假设下最大化公式（ 3 ）中定义的 $ELBO$ 。

完全分解的变分分布允许通过简单的迭代更新来优化。为了看到这一点，我们专注于更新与隐变量 $z_j$ 相关的变分参数 $λ_j$ 。将平均场分布插入公式（ 3 ）允许我们表达 $ELBO$ 如下：

$$
\begin{aligned}
\mathscr{L}= \int q\left(z_{j}\right) \mathbb{E}_{q\left(z_{\neg j}\right)}\left[\log p\left(z_{j}, \boldsymbol{x} \mid \boldsymbol{z}_{\neg j}\right)\right] d z_{j} \\
-\int q\left(z_{j}\right) \log q\left(z_{j}\right) d z_{j}+c_{j}
\end{aligned}
$$

$\boldsymbol{z}_{\neg j}$ 表示集合 $\boldsymbol{z}$ 中除不包括 $z_{j}$ 常数 $c_{j}$ 包含所有关于常数的项$z_{j}$，例如与 $z_{\neg j}$ 相关的熵项。因此，我们将完全期望分为对 $\boldsymbol{z}_{\neg j}$ 的内部期望和对 $z_{j}$ 的外部期望。


公式（ 6 ）采用负 $\mathrm{KL}$ 散度的形式，对于变量 $j$ 最大化

$$
\log q^{*}\left(z_{j}\right)=\mathbb{E}_{q\left(z_{\neg j}\right)}\left[\log p\left(z_{j} \mid \boldsymbol{z}_{\neg j}, \boldsymbol{x}\right)\right]+\text { const }
$$

对这个结果求幂和归一化产生：

$$
\begin{aligned}
q^{*}\left(z_{j}\right) & \propto \exp \left(\mathbb{E}_{q\left(z_{\neg j}\right)}\left[\log p\left(z_{j} \mid z_{\neg j}, \boldsymbol{x}\right)\right]\right) \\
& \propto \exp \left(\mathbb{E}_{q\left(z_{\neg j}\right)}[\log p(z, \boldsymbol{x})]\right)
\end{aligned}
$$

使用公式。如图8所示，可以针对每个隐变量迭代地更新变分分布直到收敛。类似的更新也构成了变分消息传递算法 [216]（附录 A.3）的基础。

有关平均场近似及其几何解释的更多详细信息，请读者参阅 [14] 和 [205]。


### 3.2 指数族分布的适用性

变分推断的另一个重要的技术点在于变分分布族 $\mathcal{Q}$ 的选择。在高斯混合模型中，我们知道目标分布 $p$ 是高斯的，那么将变分分布 $q$ 构造为高斯的，会更容易逼近 $p$ 。 

但对于更为复杂的目标分布怎么办？答案是：**可以用更广泛的指数族分布来构造！！！**

数学上已经证明，$p$ 不一定必须是高斯的，只要它属于指数族分布，那么我们都可以将变分分布 $q$ 构造为同一指数族的分布。因为指数族分布有一些很好的性质，允许我们很巧妙地简化自然梯度的推导。事实上指数族分布非常宽泛，基本涵盖了常见的分布形态，如：高斯分布、$\chi ^2$ 分布、伯努利分布、指数分布、贝塔分布、伽马分布、泊松分布等。

但是，当 $p$ 本身确实并不属于指数族时， $q$ 可能永远无法近似 $p$ ，而这也是变分推断的一个缺陷，即最终计算得出的结果是一个难以提前估计的近似。

### 3.3 坐标上升优化算法

在构造 $q$ 时，除了对分布族做选择之外，还需要考虑不同随机变量之间的依赖关系问题。其中一种最简单的近似，就是假设隐变量可以被分解为多个因子，而因子之间相互独立，根据概率图模型，此时隐变量的联合分布可以被分解为多个因子分布的乘积（见下图）。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211108085537-3530.webp)

> 图 8：独立因子假设及其概率图表示

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211108085543-3d28.webp)

> 图 9：完全因子分解后的平均场变分族

在这种因子分解形式的近似中，完全因子分解是使用最广泛的一种构造形式，即每个隐变量有自己独立的分布，与其他隐变量无关，数学形式为： $q(z) = q_1(z_1) q_2(z_2) \cdots q_n(z_n) $，其中所有的 $q_i(z_i)$ 都是一元离散变量上的类别分布（Catalogical Distribution），可描述为一维表形式。

事实表明，对于 $\mathcal{Q}$ 的这种选择很容易优化，并且工作地出奇得好。在优化变分下界时，这种因子分解的方法也许是目前最流行的一种选择。

这种采用完全因子分解方法的变分推断，被称为**平均场近似推断**。此方法包含以下需要解决的优化问题：

$$
\min_{q_1, \ldots, q_n} J(q) 
$$

解决此优化问题的标准方法是在 $q_j$ 上实施坐标下降做最小化（或坐标上升做最大化）： 我们在 $j=1,2,\ldots,n$ 上做迭代，对于每一个 $j$ ，在保持其他"坐标" $q_{-j} = \prod_{i \neq j} q_i$ 固定的情况下，仅关于 $q_j$ 来优化 $KL(q\|p)$ 。

最有趣的是：**单一坐标的优化问题具有一个简单的闭式解**：

$$
 \log q_j(z_j) \gets \mathbb{E}_{q_{-j}} \left[ \log \tilde p(z) \right] + \textrm{const}
$$

注意上述公式两侧都包含一元隐变量 $z_j$ 的函数，我们只是用右侧的另一个函数取代了 $q(z_j)$ 。右式中的常数项为新分布的归一化常数，此外还包含一个因子总和的期望：

$$
\log \tilde p(z) = \sum_k \log \phi(z_k)
$$

根据无向概率图模型（即马尔科夫随机场理论），只有在变量 $z_j$ 的马尔科夫毯内的因子，才是 $z_j$ 的函数 ，马尔科夫毯之外的其余因子对于 $z_j$ 都可视为常数，可以纳入常数项 $\textrm{constant}$。

这样能够大量减少期望公式中的因子数量；当 $z_j$ 的马尔科夫毯很小时，可以比较容易给出 $q(z_j)$ 的解析公式。 例如，假设变量 $z_j$ 是具有 $K$ 个可能值的离散型变量，并且在 $z_j$ 的马尔科夫毯中有 $F$ 个因子和 $N$ 个变量，则计算期望值的时间复杂度为 $O(K F K^N)$：即对于每个 $z_j$ 的 $K$ 种可能的取值，需要对 $N$ 个变量的所有 $K^N$ 次赋值以及 $F$ 个因子求和。

坐标下降过程的结果是：

依据 $KL(q\|p)$ 迭代地拟合了用于近似 $p$ 的完全因子式 $q(z) = q_1(z_1) q_2(z_2) \cdots q_n(z_n) $ ； 而坐标下降的每一步，都增加了变分下界，使其向  $\log Z(\theta)$ 进一步收紧。 最终的因子 $q_j(z_j)$ 不会完全等于真正的边缘分布 $p(z_j)$ ，但对于许多实际目的，它们通常表现足够好，如确定 $z_j$ 的峰值： $\max_{z_j} p(z_j)$ 。 

还是用高斯混合模型的例子，根据概率图模型有：

$$
p(\boldsymbol{\mu},\mathbf{c},\mathbf{x})=p(\boldsymbol{\mu})\prod_{i=1}^{n} p(c_i)p(x_i \mid c_i,\boldsymbol{\mu})
$$

该模型中涉及的隐变量为 $\mathbf{z} = \{\boldsymbol{\mu},\mathbf{c} \}$ ，其中 $\boldsymbol{\mu} =\{ \mu_k\}, k=1...5$  为 5 个高斯组份的均值；$\mathbf{c}$ 为所有数据点的类别向量。$\boldsymbol{\mu}$ 作为全局变量，其变分分布被构造为高斯分布  $q(u_k;m_k,S_k)$ ， $\nu_\mu = \{m_k,S_k\},k=1...5$  为需要优化求解的变分参数。 $c_i$ 为局部变量，其变分分布的构造形式为多项分布  $q(c_i ;\psi_i)$ ，其中 $\nu_c=\{ \psi_i \}, i=1...n$ 为控制每个数据点类别的变分参数（注意 $\psi_i$ 为向量）。

按照平均场近似方法，各因变量之间相互独立，则变分分布 $q(\boldsymbol{\mu},c)$ 可被构造为：

$$
p(\boldsymbol{\mu},\mathbf{c} \mid \mathbf{x}) \approx q(\boldsymbol{\mu},c)=\prod_{k=1}^{K} q(\mu_k;m_k,S_k^2) \prod_{i=1}^{n}q(c_i ;\psi_i)
$$

## 4 提升可扩展性 --- 随机变分推断

上节中讲解了平均场变分分布的构造原理，并介绍了采用坐标上升法对其做优化的算法。但该方法存在两个方面的问题：

- 坐标上升过程需要所有样本参与计算，不适合大样本

- 梯度下降并非最优方向，收敛速度较慢

随机变分推断方法针对上述问题，分别进行了处理。下面分别介绍。

### 4.1 如何适应大数据 ？

在最原始的坐标上升方法中，所有的隐变量同等对待，逐一更新一遍效率较低。为此，有学者对其做了改进，将隐变量分为两种类型，即仅关系单个数据点的局部参数，和关系到所有数据点的全局参数（ $\lambda$ ） 。算法 1 对其进行了高层次的总结：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211108160249-c296.webp)

如算法中所示，即便是这样，在更新全局参数之前，也需要先循环遍历所有文档一遍（单批次梯度下降）。当文档太多时，这就会成为了一个新问题，模型更新的频率太低。不过，该算法将参数区分为全局参数和局部参数的作法，为随机梯度下降提供了一个思路。

为了适应随机梯度下降，可将借鉴算法 1 的思路，将下界分解为两部分（采用对数形式）：**由局部参数 $\phi_i$ 参数化的逐数据点项** 和 **由全局参数 $\lambda$ 参数化的全局项**：

$$
\mathcal{L}\left(\lambda, \phi_{1: n}\right)=\underbrace{\mathbb{E}_{q}[\log p(\beta)-\log q(\beta \mid \lambda)]}_{\text {global contribution }}+\sum_{i=1}^{n} \underbrace{\left\{\mathbb{E}_{q}\left[\log p\left(w_{i}, z_{i} \mid \beta\right]-\log q\left(z_{i} \mid \phi_{i}\right)\right\}\right.}_{\text {per-data point contribution }}
$$

令下式为全局分布：

$$
f(\lambda):=\mathbb{E}_{q}[\log p(\beta)-\log q(\beta \mid \lambda)]
$$

且令下式为第 $i$ 个数据点的逐数据点分布。

$$
g_{i}\left(\lambda, \psi_{i}\right):=\mathbb{E}_{q}\left[\log p\left(w_{i}, z_{i} \mid \beta\right]-\log q\left(z_{i} \mid \phi_{i}\right)\right.
$$

则下界可简写为：

$$
\mathcal{L}\left(\lambda, \phi_{1: n}\right)=f(\lambda)+\sum_{i=1}^{n} \mathrm{g}_{i}\left(\lambda, \phi_{i}\right)
$$

为了优化目标，可以首先关于参数 $\phi_{1: n}$ 最大化，这将产生单参数的下界：

$$
\mathcal{L}(\lambda)=f(\lambda)+\sum_{i=1}^{n} \max _{\phi_{i}} \mathrm{g}_{i}\left(\lambda, \phi_{i}\right)
$$

令每个数据点的优化为：

$$
\phi_{i}^{*}=\arg \max _{\phi} \mathrm{g}_{i}\left(\lambda, \phi_{i}\right)
$$

则单参数下界 $\mathcal{L}(λ)$ 的梯度具有以下形式 ：

$$
\frac{\partial \mathcal{L}(\lambda)}{\partial \lambda}=\frac{\partial f(\lambda)}{\partial \lambda}+\sum_{i=1}^{n} \frac{\partial \mathrm{g}_{i}\left(\lambda, \phi_{i}^*\right)}{\partial \lambda} \tag{7}
$$

由于逐数据点项是每个数据点贡献的总和，每个数据点的导数也可以被累加起来计算每个数据点项的导数，如公式（ 7 ）的第二项所示。这使我们可以使用随机梯度算法来频繁地更新模型，以获得更好的收敛性。 一旦全局参数 $\lambda$ 被估计出来，每个 $\phi_i$ 也都可以在线估计出来。

### 4.2 参数梯度与自然梯度

坐标上升法以及上述面向随机梯度方法的改造，都解决不了另外一个问题：下界的优化是在变分参数空间内进行的，并非基于概率分布，而这会导致更新和收敛速度变慢。这是由于变分参数仅仅是为了描述变分分布的，下界在变分参数空间中的梯度，通常并不是概率分布空间中上升（或下降）最快的那个方向。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211114112004-358c.webp)

> 图 10：变分参数空间中的梯度与概率分布空间中的梯度

产生上述原因的根本原因在于：**变分参数空间中的欧式距离无法准确度量分布之间的距离。**

举例说明这种现象：

两组具有相同均值和方差的高斯分布对（可以想象为变分分布 $q$ 和 真实分布 $p$ ），虽然两者在变分参数（此例指均值）空间中具有相同的“距离”，但其 KL 散度（即下图中同颜色的两个高斯分布之间的重叠区域，下界会有与其相对应的反应）却截然不同。如果固定均值为 0 ，仅考虑方差作为变分参数时，会有类似现象产生。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211108161120-f068.webp)

> 图 11：具有相同变分参数（此处为均值）的两对高斯分布，虽然在变分参数空间中，两者距离相等，但是在概率分布的相似性度量上（可直观理解为两个分布之间的重叠区域），两者并不相等。同时表明，在参数空间的欧式距离上求梯度，不能代表最速下降（或上升）方向。

那能否能够直接基于概率分布空间做优化求解呢？答案是肯定的：那就是**将自然梯度方向作为优化的梯度方向**。

>  自然梯度是 Amari 于 1998 年提出的，主要用于衡量基于统计模型（如 KL 散度）的目标函数。此处知识点可参见 [文献综述](https://arxiv.org/pdf/1412.1193)

无论是从 KL 散度的角度，还是从变分下界的角度，我们期望的目标函数 $\mathcal{L}(\lambda,\phi_{1..n})$ 都是基于概率分布的。 [文献](https://arxiv.org/pdf/1412.1193) 证明： Fisher 信息矩阵（Fisher Information Matrix, FIM）是两个概率分布之间 KL 散度的二阶近似，它表示了统计流形（即概率分布空间）上的局部曲率。进而推导得出，自然梯度可以定义为：

$$
\delta \theta^*==\frac{1}{\lambda} F^{-1}\nabla_\theta \mathcal{L}(\lambda,\phi_{1..n}) 
$$

式中的 $F$ 为 Fisher 信息矩阵。根据公式也可以看出，自然梯度考虑了参数空间上的曲率信息 $ \nabla_\theta \mathcal{L}(\lambda,\phi_{1..n})$。

### 4.3 随机变分推断（ SVI ）：自然梯度与 SGD 的结合

既然给出了目标函数的最速梯度方向，那么与 4.1 节的随机梯度下降相结合就成为一种非常自然的想法。大家都知道， SGD 是小批量梯度下降的特例，由于每次仅随机地使用一个样本（这也是取名为随机梯度下降的原因），因此会引入较大的方差，但总体趋向于最优解。以下算法 2 为**随机变分变分推断算法**， 其基本思想是：在每一轮迭代中，随机抽取一个样本数据点并计算最优局部参数，然后根据自然梯度公式更新全局参数，直至收敛。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211108163233-7872.webp)


## 5 提升通用性 --- 黑盒变分推断

### 5.1 为何要做黑盒变分推断（BBVI）？

上面章节中，我们针对特定模型做出变分推断，其中大家应该已经注意到了，在 $ELBO$ 的计算表达式中，需要人为设定 $q$ ，并给出其数学期望的解析表达式（事实上，[文献](http://www.nowpublishers.com/article/Details/MAL-001) 表明，该方法只适用于**条件共轭指数族分布**）。考虑到现实世界中可能存在无数种模型，而且大部分可能是非共轭的，即便符合条件共轭假设，为每一个模型设计一种变分方案显然也是不可接受的。

因此，人们自然而然在思考：是否存在一个不需特定于某种模型的通用解决方案 ？这个解决方案最好将像黑匣子一样，只需输入模型和海量数据，然后就自动输出变分分布（或变分参数）。事实表明，这是有可能的，此类推断方法被称为**黑盒变分推断（BBVI）**。

> 黑盒变分推断的概念最早出现在文献 [Ranganath et al., 2014](https://arxiv.org/pdf/1401.0118) 中和 [Sal-imans 和 Knowles，2014](https://arxiv.org/pdf/1401.1022); [Kingma and Welling, 2014](https://arxiv.org/abs/1312.6114v10) 和 [Rezende et al., 2014](https://arxiv.org/abs/1401.4082) 提出了利用重参数化技巧实现反向传播和优化的方法；[Rezende and Mohamed, 2015](https://arxiv.org/abs/1505.05770) 提出了归一化流的 BBVI 方案、[Tran et al.,2016](https://arxiv.org/abs/158.06499) 提出了变分高斯过程的 BBVI 方案，均提升了变分推断的精度；[Alp Kucukelbir et al, 2016](https://arxiv.org/abs/1603.00788) 提出自动微分变分推断方法（ ADVI ）；[Yuri Burda et al., 2016](https://arxiv.org/abs/1505.00519) 在 VAE 基础上，提出了重要性加权变分自编码器；[J Domke and D Sheldon, 2018](https://arxiv.org/abs/1807.09034) 对其进行了泛化，提出了重要性加权变分推断。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211115171851-9244.webp)

> > 图 12： 变分推断概念图，愿景是：（1）可以轻松对任何模型进行变分推断；（2）可以利用海量数据进行推断；（3）用户只需指定模型而不需要做其他数学工作。

BBVI 大致分为两种类型：

- **基于打分梯度**的黑盒变分推断（ BBVI ）

- **基于重参数化梯度**的黑盒变分推断（ BBVI ）

后者是变分自编码器 (VAE) 的基础。

### 5.2 使用打分梯度的 BBVI

考虑如下概率模型，其中  $\mathbf{x}$  是观测变量， $\mathbf{z}$ 是隐变量，其变分分布为 $q(\mathbf{z} \mid \lambda)$ 。变分下界 (ELBO) 为：

$$
\mathcal{L}(\lambda) \triangleq \mathbb{E}_{q_{\lambda}(\mathbf{z})}[\log p(\mathbf{x}, \mathbf{z})-\log q(\mathbf{z} \mid \lambda)] \tag{9}
$$

ELBO 关于 $\lambda$ 的梯度为：

$$
\nabla_{\lambda} \mathcal{L}=\mathbb{E}_{q}\left[\nabla_{\lambda} \log q(\mathbf{z} \mid \lambda)(\log p(\mathbf{x}, \mathbf{z})-\log q(\mathbf{z} \mid \lambda))\right] \tag{10}
$$

其中 $\nabla_{\lambda} \log q(\mathbf{z} \mid \lambda)$ 被称为打分函数。

使用式（ 10 ） 中的打分梯度，就可以利用蒙特卡罗的优势，用变分分布的样本来计算 $ELBO$ 的含噪声无偏梯度：

$$
\nabla_\lambda \approx \frac{1}{S} \sum_{S=1}^{S} \nabla_\lambda \log q(z_S \mid \lambda)(\log p(\mathbf{x},z_S) - \log q(z_S \mid \lambda))
$$

其中 $z_S \sim q(\mathbf{z} \mid \lambda) $ 。

<br>

---
**式 (10) 的证明**

> 此处见 [参考文献](http://www.cs.columbia.edu/~blei/fogm/2018F/materials/RanganathGerrishBlei2014.pdf)
与推导出式（10）需要两个基本事实：
- $\nabla_{\lambda} q_{\lambda}(\mathbf{z})=\frac{1}{q_{\lambda}(\mathbf{z})} \nabla_{\lambda} q_{\lambda}(\mathbf{z})=q_{\lambda}(\mathbf{z}) \nabla_{\lambda} q_{\lambda}(\mathbf{z})$

- $\mathbb{E}_{q}\left[\nabla_{\lambda} \log q_{\lambda}(\mathbf{z})\right]=0$ ， 即对数似然梯度（打分函数）的期望为零。

基于这两个事实，可以推导出 $ELBO$ 的打分梯度：

$$
\begin{aligned}
\nabla_{\lambda} \mathcal{L} &=\nabla_{\lambda} \int_{z}\left[q_{\lambda}(z) \log p(x, z)-q_{\lambda}(z) \log q_{\lambda}(z)\right] \mathrm{d} z \\
&=\int_{z}\left\{\log p(x, z) \nabla_{\lambda} q_{\lambda}(z)-\left[\nabla_{\lambda} q_{\lambda}(z) \log q_{\lambda}(z)+q_{\lambda}(z) \frac{1}{q_{\lambda}(z)} \nabla_{\lambda} q_{\lambda}(z)\right]\right\} \mathrm{d} z \\
&=\int_{z} \nabla_{\lambda} q_{\lambda}(z)\left[\log p(x, z)-\log q_{\lambda}(z)-1\right] \mathrm{d} z \\
&=\int_{z} q_{\lambda}(z) \nabla_{\lambda} q_{\lambda}(z)\left[\log p(x, z)-\log q_{\lambda}(z)-1\right] \mathrm{d} z \\
&=\mathbb{E}_{q_{\lambda}}\left[\nabla_{\lambda} q_{\lambda}(z)\left(\log p(x, z)-\log q_{\lambda}(z)\right)\right]-\mathbb{E}_{q_{\lambda}}\left[\nabla_{\lambda} q_{\lambda}(z)\right] \\
&=\mathbb{E}_{q_{\lambda}}\left[\nabla_{\lambda} q_{\lambda}(z)\left(\log p(x, z)-\log q_{\lambda}(z)\right)\right]
\end{aligned}
$$

---

### 5.3 使用重参数化梯度的 BBVI 

依然采用上节中的模型，变分下界为（为方便重复式 9）：

$$
\mathcal{L}(\lambda) \triangleq \mathbb{E}_{q_{\lambda}(\mathbf{z})}[\log p(\mathbf{x}, \mathbf{z})-\log q(\mathbf{z} \mid \lambda)]
$$

假设变分分布可以表示成如下变换：

$$
\begin{aligned}
&\epsilon \sim S(\epsilon)\\
&\mathrm{z}= t(\epsilon, \lambda) 
\end{aligned} \Leftrightarrow \quad z \sim q(\mathbf{z} \mid \lambda)
$$

例如：

$$
\begin{aligned}
&\epsilon \sim \mathcal{N}(0,1) \\
&\mathrm{z}=\mu + \epsilon \cdot \sigma
\end{aligned} \Leftrightarrow \quad \mathrm{z} \sim \mathcal{N}\left(\mu, \sigma^{2}\right)
$$

另外假设 $\log p(\mathbf{x},\mathbf{z})$ 和 $\log q(\mathbf{z})$ 关于 $\mathbf{z}$ 可微，则可以得到 $ELBO$ 关于 $\lambda$ 的重参数化的梯度：

$$
\nabla_{\lambda} \mathcal{L}=\mathbb{E}_{S(\epsilon)} \left[\nabla_{\mathbf{z}} \left [ (\log p(\mathbf{x}, \mathbf{z})-\log q(\mathbf{z} )\right ] \nabla_\lambda t(\epsilon,\lambda)\right] \tag{10}
$$

进而可以使用自动微分来获取梯度，但并不是所有的分布都可以重参数化。

> 此部分最经典案例见 [变分自编码器](http://www.cs.columbia.edu/~blei/fogm/2018F/materials/KingmaWelling2013.pdf)

## 6 提升准确性 ---  新的目标函数和结构化变分近似

## 7 摊销式变分推断与深度学习

### 7.1 摊销变分推断（Amortized Variational Families） 

### 7.2 变分自编码器 （VAE）

### 7.3 更灵活的 VAE 

#### 7.3.1 标准化流 VAE 

#### 7.3.2 重要性加权 VAE

### 7.4  结构化 VAE

####  7.4.1 结构化 VAE

### 7.5 解决僵尸单元的问题

####  7.5.1 有损 VAE

## 8 讨论

下面是一些活跃的研究方向和开放性问题：

### 8.1 变分推断理论方面

尽管在建模和推断方面取得了进展，但很少有作者讨论 VI  的理论方面 [95]、[133]、[213]。一个重要方向是量化变分分布替换真实后验时的近似误差 [133]。与此相关的一个问题是预测误差，例如，使用 VI 近似来做贝叶斯预测分布的边缘化计算。我们还推测 VI 理论可以从与信息论的联系中受益。这已经在 [186]、[187] 中举例说明。信息论还激发了新模型和推断方案的发展 [2]、[13]、[193]。例如，信息瓶颈 [193] 最近推动了深度变分信息瓶颈 [2]。我们期望融合这两条研究线会产生更多有趣的结果。

### 8.2 变分推断和深度学习

尽管最近在各领域取得了成功，但深度学习仍然缺乏原则性的不确定性估计、缺乏其特征表示的可解释性，并且难以包含先验知识。贝叶斯方法（例如贝叶斯神经网络 [137] 和变分自动编码器）正在改进这些方面。最近的工作旨在使用可解释性概率模型作为 VAE 的先验 [38]、[77]、[91]、[168]。在此类模型中，变分推断是必不可少的组成部分。在贝叶斯深度架构中，如何使变分推断计算更为高效且易于实现，正在成为一个重要研究方向 [48]

### 8.3 变分推断和策略梯度

策略梯度估计对于强化学习 (RL)[183]​​ 和随机控制很重要。这些应用中的技术挑战与变分推断非常相似 [98]、[99]、[110]、[173]、[211] 。例如，SVGD 已作为 Steinpolicy 梯度被应用于 RL 设置 [110]。 变分推断在强化学习中的应用目前是一个活跃的研究领域。

### 8.4 自动变分推断

概率编程允许从业者快速实现和修改模型，而不必担心推断问题。用户只需要指定模型，推断引擎就会自动进行推断。流行的概率编程工具包括但不限于：Stan[28]，涵盖了大量的高级 VI 和 MCMC 推断方法； Net[126] 基于变分消息传递和 EP；Automatic Statistician[52] 和 Anglican[198] 主要依靠采样方法；Ed-ward[200] 支持 BBVI 和 MonteCarlo 采样 ； Zhusuan[176]的特点是用于贝叶斯深度学习的 VI 。这些工具的长期目标是改变概率建模的研究方法，使用户能够快速修改和改进模型，并使其他受众可以访问它们。

尽管目前努力使从业者更容易使用 VI，但对于非专家来说，其使用仍然不简单。例如，人工识别后验的对称性并打破这些对称性是 Infer.Net 所必需的。此外，诸如控制变量等减少方差的方法可以极大地加速收敛，但需要模型进行特定设计才能获得最佳性能。在撰写本文时，当前的概率编程工具箱尚未解决此类问题。我们相信这些方向对于推进概率建模在科学和技术中的影响非常重要。

## 9 总结

- 概率机器学习是将领域知识与数据联系起来的机器学习方法
- 它提供了一种用于分析数据的计算方法
- 概率机器学习的最终目标是形成一种具有表现力、可扩展、易于开发的方法
- 后验推断的关键是算法问题
- 变分推断为后验推断提供了可扩展和通用方法
- 平均长近似和坐标上升方法是最为基础的变分推断方法
- 随机变分推断将 VI 扩展到海量数据
- 黑盒变分推断将 VI 泛化到多种模型