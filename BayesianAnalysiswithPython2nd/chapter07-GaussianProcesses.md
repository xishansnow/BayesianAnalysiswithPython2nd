 

#  第7章 高斯过程



在上一章中，我们学习了Dirichlet过程，它是Dirichlet分布的无限维推广，可用于设置未知连续分布的先验。在本章中，我们将学习高斯过程，这是高斯分布的无限维推广，可用于设置未知函数的先验。在贝叶斯统计中，Dirichlet过程和Gauss过程都被用来建立灵活的模型，其中允许参数的数量随着数据的大小而增加。

本章我们将学习：

-  函数作为概率对象
- 核
- 具有高斯似然的高斯过程
- 具有非高斯似然的高斯过程



## 7.1 线性模型和非线性数据

7.1 非参统计

非参统计通常用来描述一类不依赖于参数化概率分布的统计工具/模型。根据这个定义，贝叶斯统计似乎不可能是非参的，因为前面我们学到过，贝叶斯统计的第一步就是在概率模型中准确地将概率 分布组合在一起。第1章中说过，概率分布是构建概率模型的基石。 在贝叶斯框架中，非参模型是指包含有无限多参数的模型，因此，我 们将参数可以随着数据大小而变化的模型称作非参数化模型。对于非 参数化模型而言，理论上其参数个数是无限的，实际使用中会根据数 据将其收缩到一个有限的值，从而让数据本身来决定参数的个数。

 

## 7.2 对函数建模

 我们将首先描述一种将函数表示为概率对象的方法，以开始对高斯过程的讨论。可以把函数 $f$ 看作是从一组输入 $x$ 到一组输出 $y$ 的映射。因此，可以这样写：

$$ y=f(x) $$

表示函数的一种方式是为每个值 $x_i$ 列出其相应值 $y_i$ 。事实上，你可能还记得小学时函数的这种表示方式：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505201816_b3.webp)

一般情况下，$x$ 和 $y$ 的值将位于实数行上；因此，可将函数视为成对$( x_i,y_i)$ 值的(可能)无限有序列表。顺序很重要，因为如果打乱这些值，会得到不同的函数。

函数也可以表示为一个由 $x$ 的值索引的(潜在)无限数组，但重要区别在于，$x$ 的值不限于整数，可以取实数。

使用这些描述，我们可以用数字表示任何想要的特定函数。但如果想要以概率表示函数呢？可以通过让映射具有概率性质来做到这一点。让我来解释一下：可以让每个 $y_i$ 值都是一个高斯分布的随机变量，具有给定的均值和方差。通过这种方式，我们不再有对单个特定功能的描述，而是对一系列分布族的描述。

为了使讨论具体化，让我们使用一些Python代码来构建和绘制这类函数的两个示例：

```
np.random.seed(42)
x = np.linspace(0, 1, 10)
y = np.random.normal(0, 1, len(x))
plt.plot(x, y, 'o-', label='the first one')
y = np.zeros_like(x)
for i in range(len(x)):
    y[i] = np.random.normal(y[i-1], 1)
plt.plot(x, y, 'o-', label='the second one')
plt.legend()
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505202243_46.webp)

图7.1显示，使用高斯分布的样本对函数进行编码并不是那么疯狂或愚蠢，因此我们可能是在正确的轨道上。然而，用于生成图7.1的方法是有限的，并且不够灵活。虽然我们期望实函数具有某种结构或模式，但我们表达第一个函数的方式并不允许我们编码数据点之间的任何关系。事实上，每个点都是完全独立的，因为我们只从一个常见的一维高斯分布中获得了10个独立的样本。对于第二个函数，我们引入了一些依赖项。该点的平均值就是值。然而，接下来我们将看到有一种更通用的方法来捕获依赖项(并且不仅仅是在连续点之间)。

在继续之前，让我停一会儿问你们，为什么我们使用的是高斯分布，而不是任何其他的概率分布？首先，因为通过限制我们与高斯人合作，我们不会失去任何指定不同形状函数的灵活性，因为每个点都可能有其自己的均值和方差；其次，因为从数学的角度来看，与高斯人合作是很好的。

 

### 7.2.1 多元高斯与函数

在图7.1中，我们使用高斯表示一个函数来获取样本。一种替代方法是使用多变量高斯分布来获得长度的样本向量。实际上，您可能希望尝试通过使用 np.random.multivariate_normal(np.zeros_like(x), np.eye(len(x))) 替换np.random.normal(0, 1, len(x)) 来生成如图7.1所示的图形。

您将看到第一条语句等同于第二条语句，但是现在我们可以使用协方差矩阵来编码有关数据点如何相互关联的信息。通过允许协方差矩阵为np.ye(len(X))，我们基本上表示这10个点中的每个点的方差为1，并且它们之间的方差(即，它们的协方差)为0(因此，它们是独立的)。如果我们用其他(正)数字替换这些零，我们可能得到的协方差告诉我们一个不同的故事。因此，要以概率方式表示函数，我们只需要一个具有合适协方差矩阵的多变量高斯函数，我们将在下一节中看到。

### 7.2.2 协方差函数与核

实际上，协方差矩阵是使用称为核的函数指定的。您可能会在统计文献中找到多个核的定义，它们的数学属性略有不同。出于我们讨论的目的，我们将说内核基本上是一个对称函数，它接受两个输入，并在输入中返回零值，否则相同或为正。如果满足这些条件，我们可以将核函数的输出解释为两个输入之间的相似性度量。

在众多可用的有用内核中，常用的一种是指数二次核：
$$
K\left(x, x^{\prime}\right)=\exp \left(-\frac{\left\|x-x^{\prime}\right\|^{2}}{2 \ell^{2}}\right)
$$
此处， $ \left\|x-x^{\prime}\right\|^{2} $ 为平方欧氏距离：
$$
\left\|x-x^{\prime}\right\|^{2}=\left(x_{1}-x_{1}^{\prime}\right)^{2}+\left(x_{2}-x_{2}^{\prime}\right)^{2}+\cdots+\left(x_{n}-x_{n}^{\prime}\right)^{2}
$$
乍一看可能不明显，但幂二次核具有与高斯分布类似的公式(见表达式1.3)。因此，您可能会发现有人将此内核称为高斯内核。该术语称为长度标度(或带宽或方差)，用于控制内核的宽度。

为了更好地理解内核的作用，让我们定义一个Python函数来计算指数二次内核：

```
def exp_quad_kernel(x, knots, ℓ=1):
    """exponentiated quadratic kernel"""
    return np.array([np.exp(-(x-k)**2 / (2*ℓ**2)) for k in knots])
```

以下代码和图7.2旨在显示协方差矩阵如何查找不同的输入。我选择的输入相当简单，由值[-1，0，1，2]组成。理解此示例后，您应该尝试使用其他输入(请参见练习1)：

```
data = np.array([-1, 0, 1, 2])
cov = exp_quad_kernel(data, data, 1)
_, ax = plt.subplots(1, 2, figsize=(12, 5))
ax = np.ravel(ax)
ax[0].plot(data, np.zeros_like(data), 'ko')
ax[0].set_yticks([])
for idx, i in enumerate(data):
    ax[0].text(i, 0+0.005, idx)
ax[0].set_xticks(data)
ax[0].set_xticklabels(np.round(data, 2))
#ax[0].set_xticklabels(np.round(data, 2), rotation=70)
ax[1].grid(False)
im = ax[1].imshow(cov)
colors = ['w', 'k']
for i in range(len(cov)):
    for j in range(len(cov)):
        ax[1].text(j, i, round(cov[i, j], 2),
                   color=colors[int(im.norm(cov[i, j]) > 0.5)],
                   ha='center', va='center', fontdict={'size': 16})
ax[1].set_xticks(range(len(data)))
ax[1].set_yticks(range(len(data)))
ax[1].xaxis.tick_top()
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505203138_6f.webp)

图7.2左侧的面板显示输入，x轴上的值表示每个数据点的值，文本注释显示数据点的顺序(从零开始)。在右边的面板上，我们有一个热图，表示使用指数二次核获得的协方差矩阵。颜色越浅，协方差越大。如您所见，热图是对称的，对角线取较大的值。协方差矩阵中每个元素的值与点之间的距离成反比，因为对角线是每个数据点与其自身进行比较的结果。对于这个核，我们得到最接近的距离0和更高的协方差值1。其他值对于其他内核也是可能的。

内核将数据点沿x轴的距离转换为预期函数值(在y轴上)的协方差值。因此，x轴上的两个点越近，我们预计它们在y轴上的值就越相似。

总而言之，到目前为止，我们已经看到，我们可以使用具有给定协方差的多元正态分布来建模函数。我们可以使用核函数来建立协方差。在下面的示例中，我们使用exp_quad_kernel函数来定义多变量正态分布的协方差矩阵，然后使用该分布中的样本来表示函数：

```
np.random.seed(24)
test_points = np.linspace(0, 10, 200)
fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex=True,
                       sharey=True, constrained_layout=True)
ax = np.ravel(ax)
for idx, ℓ in enumerate((0.2, 1, 2, 10)):
    cov = exp_quad_kernel(test_points, test_points, ℓ)
    ax[idx].plot(test_points, stats.multivariate_normal.rvs(cov=cov,
size=2).T)
    ax[idx].set_title(f'ℓ ={ℓ}')
fig.text(0.51, -0.03, 'x', fontsize=16)
fig.text(-0.03, 0.5, 'f(x)', fontsize=16)
```

 ![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505203247_f6.webp)

如图7.3所示，高斯核包含多种函数，其参数控制函数的平滑度。的值越大，函数越平滑。



### 7.2.3 高斯过程

现在我们已经准备好了解什么是高斯过程(GP)以及它们是如何在实践中使用的。

从维基百科(Wikipedia)上摘取的对高斯过程的一个比较正式的定义如下：

按时间或空间索引的随机变量的集合，使得这些随机变量的每个有限集合都具有多元正态分布，即它们的每个有限线性组合都是正态分布的。

理解高斯过程的诀窍是认识到GP的概念是一个心理(和数学)脚手架，因为在实践中，我们不需要直接处理这个无限的数学对象。相反，我们只在我们有数据的地方评估高斯过程。通过这样做，我们将无限维GP压缩成一个有限的多元高斯分布，其维数和数据点一样多。从数学上讲，这种崩溃是由无限不可观测的维度上的边际化造成的。理论向我们保证，省略(实际上边缘化)所有的点是可以的，除了我们正在观察的那些点。它还保证我们将始终得到多变量的高斯分布。因此，我们可以严格地将图7.3解释为来自高斯过程的实际样本！

请注意，我们将多变量高斯函数的平均值设置为零，并通过指数二次核仅使用协方差矩阵对函数进行建模。在使用高斯过程时，将多变量高斯的平均值设置为零是常见的做法。

高斯过程对于构建贝叶斯非参数模型很有用，因为我们可以将它们用作函数上的先验分布。



##  7.3 高斯过程回归

让我们假设我们可以将一个值建模为加一些噪声的函数：

$$
y \sim \mathcal{N}(\mu=f(x), \sigma=\epsilon) 
$$
此处 $ \epsilon \sim \mathcal{N}\left(0, \sigma_{\epsilon}\right) $

这类似于我们在第3章“线性回归建模”中对线性回归模型所做的假设。主要区别在于，现在我们将优先分配。高斯过程可以作为这样的先验，因此我们可以这样写：

$$
f(x) \sim \mathcal{G} \mathcal{P}\left(\mu_{x}, K\left(x, x^{\prime}\right)\right)
$$

这里，$\mathcal{GP}$ 表示高斯过程分布， $\mu_x$ 为均值函数， $K(x,x')$ 为核函数或协方差函数。在这里，我们用函数这个词来表示，在数学上，均值和协方差是无限的对象，即使在实践中，我们总是处理有限的对象。

如果先验分布是GP，并且似然是正态分布，那么后验分布也是GP，我们可以解析地计算它：
$$
\begin{array}{c}
p\left(f\left(X_{*}\right) \mid X_{*}, X, y\right) \sim \mathcal{N}(\mu, \Sigma) \\
\mu=K_{*}^{T} K^{-1} y \\
\Sigma=K_{* *}-K_{*}^{T} K^{-1} K_{*}
\end{array}
$$
这里：
$$
\begin{array}{l}
\text { - } K=K(X, X) \\
\text { - } K_{*}=K\left(X_{*}, X\right) \\
\text { - } K_{* *}=K\left(X_{*}, X_{*}\right)
\end{array}
$$


$X$ 是观察到的数据点，$X_*$ 表示测试点；也就是我们希望知道推断函数值的新点。

像往常一样，PyMC3允许我们通过为我们处理几乎所有的数学细节来执行推理，高斯过程也不例外。因此，让我们继续创建一些数据，然后创建一个PyMC3模型：

```
np.random.seed(42)
x = np.random.uniform(0, 10, size=15)
y = np.random.normal(np.sin(x), 0.1)
plt.plot(x, y, 'o')
true_x = np.linspace(0, 10, 200)
plt.plot(true_x, np.sin(true_x), 'k--')
plt.xlabel('x')
plt.ylabel('f(x)', rotation=0)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505204212_c1.webp)

在图7.4中，我们看到真正的未知函数是一条黑虚线，而点表示未知函数的样本(带有噪声)。

请注意，为了将公式7.7和7.8编码到PyMC3模型中，我们只需要找出指数二次核的参数、正态似然方差和长度-尺度参数。

GPS在PyMC3中实现为一系列Python类，与我们在以前的模型中看到的略有不同；然而，代码仍然非常PyMC3onic。我在以下代码中添加了一些注释，以指导您完成使用PyMC3定义GP的关键步骤：

```
# A one dimensional column vector of inputs.
X = x[:, None]
with pm.Model() as model_reg:
    # hyperprior for lengthscale kernel parameter
    ℓ = pm.Gamma('ℓ', 2, 0.5)
    # instantiate a covariance function
    cov = pm.gp.cov.ExpQuad(1, ls=ℓ)
    # instantiate a GP prior
    gp = pm.gp.Marginal(cov_func=cov)
    # prior
    ϵ = pm.HalfNormal('ϵ', 25)
    # likelihood
    y_pred = gp.marginal_likelihood('y_pred', X=X, y=y, noise=ϵ)
```

请注意，我们使用的不是表达式7.7中预期的正态似然，而是gp.edge_lisilience方法。你可能还记得，在第一章“概率思维”(公式1.1)和第五章“模型比较”(公式5.13)中，边际似然是似然和先验的积分：
$$
p(y \mid X, \theta) \sim \int p(y \mid f, X, \theta) p(f \mid X, \theta) d f
$$
与往常一样，表示所有未知参数，是自变量，也是因变量。请注意，我们正在对函数的值进行边际化。对于GP先验和正常可能性，可以解析地执行边际化。

根据PyMC3的核心开发者、GP模块的主要贡献者Bill Engels的说法，对于长度尺度参数，优先避免零通常效果更好。在此之前，一个有用的默认设置是pm.Gamma(2，0.5)。你可以从Stan team阅读更多关于默认有用历史的建议：https://github.com/stan-dev/stan/wiki/Prior-ChoiceRecommendations：

```
az.plot_trace(trace_reg)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505204431_42.webp)



现在我们已经找到和的值，我们可能想要从GP后验中获取样本；即，拟合数据的函数的样本。我们可以通过使用gp.条件函数计算对新输入位置求值的条件分布来实现这一点：

```
X_new = np.linspace(np.floor(x.min()), np.ceil(x.max()), 100)[:,None]
with model_reg:
    f_pred = gp.conditional('f_pred', X_new)
```

结果，我们得到了一个新的PyMC3随机变量f_pred，我们可以使用它从后验预测分布中获取样本(以X_NEW值计算)：

```
with model_reg:
    pred_samples = pm.sample_posterior_predictive(trace_reg, vars=[f_pred],samples=82)
```

现在我们可以在原始数据上绘制拟合函数图，以直观地检查它们与数据的拟合程度以及预测中的相关不确定性：

```
_, ax = plt.subplots(figsize=(12,5))
ax.plot(X_new, pred_samples['f_pred'].T, 'C1-', alpha=0.3)
ax.plot(X, y, 'ko')
ax.set_xlabel('X')
```

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505204540_3f.webp" style="zoom:67%;" />

或者，我们可以使用pm.gp.util.lot_gp_dist函数来获得一些不错的绘图。每个绘图代表一个百分位数，范围从51(浅色)到99(深色)：

```
_, ax = plt.subplots(figsize=(12,5))
pm.gp.util.plot_gp_dist(ax, pred_samples['f_pred'], X_new,
palette='viridis', plot_samples=False);
ax.plot(X, y, 'ko')
ax.set_xlabel('x')
ax.set_ylabel('f(x)', rotation=0, labelpad=15)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505204617_c7.webp)

另一种选择是计算在参数空间中给定点评估的条件分布的平均向量和标准差。在下面的示例中，我们对AND使用平均值(跟踪中的样本)。我们可以使用gp.recast函数计算平均值和方差。我们之所以能做到这一点，是因为PyMC3已经解析地计算了后验结果：

```
_, ax = plt.subplots(figsize=(12,5))
point = {'ℓ': trace_reg['ℓ'].mean(), 'ϵ': trace_reg['ϵ'].mean()}
mu, var = gp.predict(X_new, point=point, diag=True)
sd = var**0.5
ax.plot(X_new, mu, 'C1')
ax.fill_between(X_new.flatten(),
                 mu - sd, mu + sd,
                 color="C1",
                 alpha=0.3)
ax.fill_between(X_new.flatten(),
                 mu - 2*sd, mu + 2*sd,
                 color="C1",
                 alpha=0.3)
ax.plot(X, y, 'ko')
ax.set_xlabel('X')
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505204701_d1.webp)

正如我们在第四章“推广线性模型”中所看到的，我们可以使用具有非高斯似然的线性模型和适当的逆链接函数来扩展线性模型的范围。我们可以为全科医生做同样的事情。例如，我们可以使用具有指数逆链接函数的泊松似然。对于这样的模型，后验不再是可分析处理的，但是，尽管如此，我们可以用数值方法来逼近它。在接下来的几节中，我们将讨论这种类型的模型。

## 7.4 空间自相关回归

下面的例子取自理查德·麦克雷思(Richard McElreath)的“统计反思”(Statistics Reink)一书。作者好心地允许我在这里重复使用他的例子。我强烈推荐你读他的书，因为你会发现很多像这样的好例子和非常好的解释。唯一需要注意的是，书中的示例是R/stan格式的，但请不要担心，请继续采样；您可以在https://github.com/pymc-devs/Resources中找到这些示例的Python/PyMC3版本。

好的，回到这个例子，我们有10个不同的岛屿社会；对于每一个，我们都有他们使用的工具的数量。一些理论预测，较大的人口会比较小的人口开发和维持更多的工具。另一个重要因素是人群间的接触率。

由于我们有许多工具作为因变量，我们可以使用泊松回归与总体作为自变量。事实上，我们可以使用人口的对数，因为(根据理论)真正重要的是人口的数量，而不是绝对的大小。将接触率包括在我们的模型中的一种方法是收集有关这些社会在历史上接触的频率的信息，并创建一个分类变量，如低/高接触率(请参阅岛屿数据帧中的Contact列)。另一种方式是使用社会之间的距离作为接触率的代理，因为可以合理地假设距离最近的社会比距离较远的社会更频繁地接触。

我们可以通过阅读本书附带的ians_Dist.csv文件来访问以千公里为单位表示的值的距离矩阵：

```
islands_dist = pd.read_csv('../data/islands_dist.csv',
                           sep=',', index_col=0)
islands_dist.round(1)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505204845_7c.webp)

如你所见，主对角线上填满了零。每个岛国社会都处于零公里的状态。矩阵也是对称的；上三角形和下三角形都有相同的信息。这是从A点到B点的距离与B点到A点的距离相同的直接结果。

工具数量和人口规模存储在另一个文件islands.csv中，该文件也随书一起分发：

```
islands = pd.read_csv('../data/islands.csv', sep=',')
islands.head().round(1)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505204920_fd.webp)

在此DataFrame中，我们将只使用列culture, total_tools, lat, lon2,and logpop:

```
islands_dist_sqr = islands_dist.values**2
culture_labels = islands.culture.values
index = islands.index.values
log_pop = islands.logpop
total_tools = islands.total_tools
x_data = [islands.lat.values[:, None], islands.lon.values[:, None]]
```

我们要构建的模型是：

$$
f \sim \mathcal{G P}\left([0, \cdots, 0], K\left(x, x^{\prime}\right)\right) \\
 \mu \sim \exp (\alpha+\beta x+f) \\
 y \sim \operatorname{Poisson}(\mu)
$$
这里，我们省略了和的前缀，以及内核的超级前缀。是日志填充，是工具总数。

与第四章“推广线性模型”中的模型相比，该模型基本上是一个泊松回归模型，与第四章中的模型相比，线性模型中的一项来自GP。为了计算GP的内核，我们将使用距离矩阵ILANES_DIST。通过这种方式，我们将有效地纳入技术暴露的相似性度量(从距离矩阵估计)。因此，我们将把每个社会的工具数量建模为它们的地理相似性的函数，而不是假设总数量仅仅是人口的结果，并且从一个社会到另一个社会是独立的。

此模型(包括之前的模型)类似于PyMC3中的以下代码：

```
with pm.Model() as model_islands:
    η = pm.HalfCauchy('η', 1)
    ℓ = pm.HalfCauchy('ℓ', 1)
    cov = η * pm.gp.cov.ExpQuad(1, ls=ℓ)
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior('f', X=islands_dist_sqr)
    α = pm.Normal('α', 0, 10)
    β = pm.Normal('β', 0, 1)
    μ = pm.math.exp(α + f[index] + β * log_pop)
    tt_pred = pm.Poisson('tt_pred', μ, observed=total_tools)
    trace_islands = pm.sample(1000, tune=1000)
```

为了了解协方差函数关于距离的后验分布，我们可以根据后验分布绘制一些样本：

```
trace_η = trace_islands['η']
trace_ℓ = trace_islands['ℓ']
_, ax = plt.subplots(1, 1, figsize=(8, 5))
xrange = np.linspace(0, islands_dist.values.max(), 100)
ax.plot(xrange, np.median(trace_η) *
        np.exp(-np.median(trace_ℓ) * xrange**2), lw=3)
ax.plot(xrange, (trace_η[::20][:, None] * np.exp(- trace_ℓ[::20][:, None] *
xrange**2)).T,
        'C0', alpha=.1)
ax.set_ylim(0, 1)
ax.set_xlabel('distance (thousand kilometers)')
ax.set_ylabel('covariance')
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505205145_a6.webp)

图7.9中的粗线是成对社会之间协方差作为距离函数的后验中位数。我们使用中位数是因为和的分布非常不对称。我们可以看到，协方差平均没有那么高，在大约2000公里处也降到了几乎为0。细线代表不确定性，我们可以看到有很大的不确定性

您可能会发现，将MODEL_IILAS及其计算出的后部与 https://github.com/pymc-devs/ 参考资料中的模型m_10_10进行比较会很有趣。您可能希望使用ArviZ函数，例如az.Summary或az.lot_Forest。模型M1010类似于MODEL_ILANES，但不包含高斯过程项。

根据我们的模型，我们现在要探索这些岛屿之间的社会联系有多强。为此，我们必须将协方差矩阵转换为相关矩阵：

```
# compute posterior median covariance among societies
Σ = np.median(trace_η) * (np.exp(-np.median(trace_ℓ) * islands_dist_sqr))
# convert to correlation matrix
Σ_post = np.diag(np.diag(Σ)**-0.5)
ρ = Σ_post @  Σ @ Σ_post
ρ = pd.DataFrame(ρ, index=islands_dist.columns,
columns=islands_dist.columns)
ρ.round(2)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505205246_9e.webp)

从其他地方跳出的两个观察是，夏威夷非常孤独。这是有道理的，因为夏威夷离其他岛屿社会很远。此外，我们还可以看到Malekula(Ml)、Tikopia(Ti)和Santa Cruz(Sc)之间高度相关。这也是有道理的，因为这些社会非常接近，而且他们也有类似数量的工具。

现在我们将使用纬度和经度信息来绘制岛屿-社会的相对位置：

```
# scale point size to logpop
logpop = np.copy(log_pop)
logpop /= logpop.max()
psize = np.exp(logpop*5.5)
log_pop_seq = np.linspace(6, 14, 100)
lambda_post = np.exp(trace_islands['α'][:, None] +
                     trace_islands['β'][:, None] * log_pop_seq)
_, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].scatter(islands.lon2, islands.lat, psize, zorder=3)
ax[1].scatter(islands.logpop, islands.total_tools, psize, zorder=3)
for i, itext in enumerate(culture_labels):
    ax[0].text(islands.lon2[i]+1, islands.lat[i]+1, itext)
    ax[1].text(islands.logpop[i]+.1, islands.total_tools[i]-2.5, itext)
ax[1].plot(log_pop_seq, np.median(lambda_post, axis=0), 'k--')
az.plot_hpd(log_pop_seq, lambda_post, fill_kwargs={'alpha':0},
plot_kwargs={'color':'k', 'ls':'--', 'alpha':1})
for i in range(10):
    for j in np.arange(i+1, 10):
        ax[0].plot((islands.lon2[i], islands.lon2[j]),
                   (islands.lat[i], islands.lat[j]), 'C1-',
                   alpha=ρ.iloc[i, j]**2, lw=4)
        ax[1].plot((islands.logpop[i], islands.logpop[j]),
                   (islands.total_tools[i], islands.total_tools[j]), 'C1-',
                   alpha=ρ.iloc[i, j]**2, lw=4)
ax[0].set_xlabel('longitude')
ax[0].set_ylabel('latitude')
ax[1].set_xlabel('log-population')
ax[1].set_ylabel('total tools')
ax[1].set_xlim(6.8, 12.8)
ax[1].set_ylim(10, 73)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505205338_5f.webp)

图7.10的左侧面板显示了我们之前在相对地理位置上下文中计算的社会之间的后验中值相关性的线条。有些线条是不可见的，因为我们已经使用相关性来设置线条的不透明度(使用matplotlib的alpha参数)。在右侧面板上，我们再次显示了后验中值相关性，但这一次是根据对数总体与工具总数绘制的。虚线表示工具的中位数和HPD 94%的间隔作为对数填充的函数。在这两幅图中，圆点的大小与每个岛屿社会的人口成正比。

请注意Malekula、Tikopia和Santa Cruz之间的相关性如何描述这样一个事实，即它们拥有的工具数量相当少，接近或低于其人口的预期工具数量。类似的事情正发生在特罗布里兰岛和马努斯；它们地理位置相近，拥有的工具比预期的人口规模要少。汤加为其人口提供的工具比预期的要多得多，而且与斐济的相关性相对较高。在某种程度上，这个模型告诉我们，汤加对Lua斐济有积极的影响，增加了工具的总数，抵消了其近邻Malekula、Tikopia和Santa Cruz的影响。



## 7.5 高斯过程分类

高斯过程不限于回归。我们也可以用它们来分类。正如我们在第4章“推广线性模型”中所看到的，我们通过使用带有Logistic逆链接函数的Bernoulli似然(然后应用边界判决规则来分离类)，将线性模型转化为适合分类的模型。对于虹膜数据集，我们将尝试概括第4章“推广线性模型”中的MODEL_0，这次使用的是GP而不是线性模型。

让我们再次邀请虹膜数据集上台：

```
iris = pd.read_csv('../data/iris.csv')
iris.head()
```

![image-20210505205534881](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505205541_6c.webp)

我们将从最简单的分类问题开始：两类，刚毛和杂色，只有一个自变量，萼片长度。像往常一样，我们将使用数字0和1对分类变量setosa和versicolor进行编码：

```
df = iris.query("species == ('setosa', 'versicolor')")
y = pd.Categorical(df['species']).codes
x_1 = df['sepal_length'].values
X_1 = x_1[:, None]
```

对于此模型，我们将使用pm.gp.Latent类，而不是使用pm.gp.Marginal类来实例化GP之前。虽然后者更一般，可以与任何可能性一起使用，但前者仅限于高斯可能性，并且通过利用GP先验与高斯可能性的组合在数学上的易操纵性而具有更高效率的优势：

```
with pm.Model() as model_iris:
    ℓ = pm.Gamma('ℓ', 2, 0.5)
    cov = pm.gp.cov.ExpQuad(1, ℓ)
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior("f", X=X_1)
    # logistic inverse link function and Bernoulli likelihood
    y_ = pm.Bernoulli("y", p=pm.math.sigmoid(f), observed=y)
    trace_iris = pm.sample(1000, chains=1,
compute_convergence_checks=False)
```

现在我们已经找到了的值，我们可能想要从GP后验获取样本。与我们对MARGING_GP_MODEL所做的操作一样，我们还可以借助gp.Conditional函数计算一组新输入位置上计算的条件分布，如以下代码所示：

```
X_new = np.linspace(np.floor(x_1.min()), np.ceil(x_1.max()), 200)[:, None]
with model_iris:
    f_pred = gp.conditional('f_pred', X_new)
    pred_samples = pm.sample_posterior_predictive(
        trace_iris, vars=[f_pred], samples=1000)
```

为了显示此模型的结果，我们将创建一个类似于图4.4的图。我们将使用以下便利函数直接从f_pred计算边界决策，而不是解析地获得边界决策：

```
def find_midpoint(array1, array2, value):
    """
    This should be a proper docstring :-)
    """
    array1 = np.asarray(array1)
    idx0 = np.argsort(np.abs(array1 - value))[0]
    idx1 = idx0 - 1 if array1[idx0] > value else idx0 + 1
    if idx1 == len(array1):
        idx1 -= 1
    return (array2[idx0] + array2[idx1]) / 2
```

以下代码与第4章(泛化线性模型)中用于生成图4.4的代码非常相似：

```
_, ax = plt.subplots(figsize=(10, 6))
fp = logistic(pred_samples['f_pred'])
fp_mean = np.mean(fp, 0)
ax.plot(X_new[:, 0], fp_mean)
# plot the data (with some jitter) and the true latent function
ax.scatter(x_1, np.random.normal(y, 0.02),
           marker='.', color=[f'C{x}' for x in y])
az.plot_hpd(X_new[:, 0], fp, color='C2')
db = np.array([find_midpoint(f, X_new[:, 0], 0.5) for f in fp])
db_mean = db.mean()
db_hpd = az.hpd(db)
ax.vlines(db_mean, 0, 1, color='k')
ax.fill_betweenx([0, 1], db_hpd[0], db_hpd[1], color='k', alpha=0.5)
ax.set_xlabel('sepal_length')
ax.set_ylabel('θ', rotation=0)
plt.savefig('B11197_07_11.png')
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505205739_e4.webp)

如我们所见，图7.11与图4.4非常相似。F_pred曲线看起来像是一条S型曲线，除了尾部在较低的x_1值时上升，在较高的x_1值下降。这是当没有数据(或数据很少)时预测函数向前移动的结果。如果我们只关心边界决策，这应该不是一个真正的问题，但如果我们想要为不同的萼片长度值建立属于setosa或versicolor的概率模型，那么我们应该改进我们的模型，并做一些事情来获得更好的尾部模型。实现这一目标的一种方法是给高斯过程增加更多的结构。获得更好的高斯过程模型的一般方法是组合协方差函数，以便更好地捕捉我们试图建模的函数的细节。

以下模型(Model_Iris2)与model_iris相同，不同之处在于协方差矩阵，我们将其建模为三个内核的组合：

```
cov = K_{ExpQuad} + K_{Linear} + K_{whitenoise}(1E-5)
```

通过添加线性内核，我们修复了尾部问题，如图7.12所示。白噪声核只是一个稳定协方差矩阵计算的计算技巧。对高斯过程的核进行了限制，以保证得到的协方差矩阵是正定的。然而，数字错误可能会导致违反此条件。这个问题的一个表现是，我们在计算拟合函数的后验预测样本时会得到NAN。减轻此错误的一种方法是通过添加一点噪声来稳定计算。事实上，PyMC3已经在幕后做到了这一点，但有时需要更多一点噪音，如以下代码所示：

```
with pm.Model() as model_iris2:
    ℓ = pm.Gamma('ℓ', 2, 0.5)
    c = pm.Normal('c', x_1.min())
    τ = pm.HalfNormal('τ', 5)
    cov = (pm.gp.cov.ExpQuad(1, ℓ) +
           τ * pm.gp.cov.Linear(1, c) +
           pm.gp.cov.WhiteNoise(1E-5))
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior("f", X=X_1)
    # logistic inverse link function and Bernoulli likelihood
    y_ = pm.Bernoulli("y", p=pm.math.sigmoid(f), observed=y)
    trace_iris2 = pm.sample(1000, chains=1,
compute_convergence_checks=False)
```

现在，我们为先前生成的X_new值生成model_iris2的后验预测样本：

```
with model_iris2:
    f_pred = gp.conditional('f_pred', X_new)
    pred_samples = pm.sample_posterior_predictive(trace_iris2,
                                                  vars=[f_pred],
                                                  samples=1000)
_, ax = plt.subplots(figsize=(10,6))
fp = logistic(pred_samples['f_pred'])
fp_mean = np.mean(fp, 0)
ax.scatter(x_1, np.random.normal(y, 0.02), marker='.',
           color=[f'C{ci}' for ci in y])
db = np.array([find_midpoint(f, X_new[:,0], 0.5) for f in fp])
db_mean = db.mean()
db_hpd = az.hpd(db)
ax.vlines(db_mean, 0, 1, color='k')
ax.fill_betweenx([0, 1], db_hpd[0], db_hpd[1], color='k', alpha=0.5)
ax.plot(X_new[:,0], fp_mean, 'C2', lw=3)
az.plot_hpd(X_new[:,0], fp, color='C2')
ax.set_xlabel('sepal_length')
ax.set_ylabel('θ', rotation=0)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505205912_68.webp)



现在，图7.12看起来更类似于图4.4，而不是图7.11。此示例有两个主要目的：

- 展示如何轻松组合内核以获得更具表现力的模型
- 展示了如何使用高斯过程恢复Logistic回归

关于第二点，Logistic回归确实是高斯过程的特例，因为简单的线性回归只是高斯过程的特例。事实上，许多已知的模型可以被视为全科医生的特例，或者至少它们以某种方式与全科医生联系在一起。你可以阅读凯文·墨菲(Kevin Murphy)的“机器学习：概率视角”中的第15章，了解详细信息。

在实践中，使用GP对我们只能用Logistic回归来解决的问题进行建模并没有太大的意义。相反，我们希望使用GP来建模更复杂的数据，而这些数据使用灵活性较低的模型无法很好地捕获。例如，假设我们想要将患病概率建模为年龄的函数。事实证明，非常年轻和非常年长的人比中年人有更高的风险。数据集space_flu.csv是受前面描述启发的假数据集。让我们加载它：

```
df_sf = pd.read_csv('../data/space_flu.csv')
age = df_sf.age.values[:, None]
space_flu = df_sf.space_flu
ax = df_sf.plot.scatter('age', 'space_flu', figsize=(8, 5))
ax.set_yticks([0, 1])
ax.set_yticklabels(['healthy', 'sick'])
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505210018_ba.webp)

以下模型与model_iris基本相同：

```
with pm.Model() as model_space_flu:
    ℓ = pm.HalfCauchy('ℓ', 1)
    cov = pm.gp.cov.ExpQuad(1, ℓ) + pm.gp.cov.WhiteNoise(1E-5)
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior('f', X=age)
    y_ = pm.Bernoulli('y', p=pm.math.sigmoid(f), observed=space_flu)
    trace_space_flu = pm.sample(
        1000, chains=1, compute_convergence_checks=False)
```

现在，我们为model_space_flu生成后验预测样本，然后绘制结果：

```
X_new = np.linspace(0, 80, 200)[:, None]
with model_space_flu:
    f_pred = gp.conditional('f_pred', X_new)
    pred_samples = pm.sample_posterior_predictive(trace_space_flu,
                                                  vars=[f_pred],
                                                  samples=1000)
_, ax = plt.subplots(figsize=(10, 6))
fp = logistic(pred_samples['f_pred'])
fp_mean = np.nanmean(fp, 0)
ax.scatter(age, np.random.normal(space_flu, 0.02),
           marker='.', color=[f'C{ci}' for ci in space_flu])
ax.plot(X_new[:, 0], fp_mean, 'C2', lw=3)
az.plot_hpd(X_new[:, 0], fp, color='C2')
ax.set_yticks([0, 1])
ax.set_yticklabels(['healthy', 'sick'])
ax.set_xlabel('age')
```

![image-20210505210106544](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505210109_bb.webp)

请注意，如图7.14所示，GP能够很好地适应此数据集，即使数据要求函数比逻辑函数更复杂。对于简单的逻辑回归来说，很好地拟合这个数据是不可能的，除非我们引入一些特殊的修改来帮助它(请参见练习6以了解有关此类修改的讨论)。



##  7.6 Cox过程 

现在让我们返回到对计数数据建模的示例。我们将看到两个示例；一个具有时变速率，另一个具有2D空间变化速率。为了做到这一点，我们将使用泊松似然，并且将使用高斯过程对速率进行建模。因为泊松分布的速率被限制为正值，所以我们将使用指数作为逆链接函数，就像我们在第4章-推广线性模型中对零膨胀泊松回归所做的那样。

在文献中，可变速率也以强度的名称出现，因此，这种类型的问题被称为强度估计。此外，这种类型的模型通常被称为Cox模型。考克斯模型是泊松过程的一种，其速率本身就是一个随机过程。正如高斯过程是随机变量的集合，其中这些随机变量的每个有限集合都具有多元正态分布一样，泊松过程是随机变量的集合，其中这些随机变量的每个有限集合都具有泊松分布。我们可以把泊松过程看作是在给定空间中的点的集合上的分布。当泊松过程的速率本身是随机过程时，例如，高斯过程，那么我们就有所谓的Cox过程。

### 7.6.1 煤矿灾害

第一个例子被称为煤矿灾难例子。这个例子包括英国从1851年到1962年的煤矿灾难记录。灾难的数量被认为是在此期间受到安全法规变化的影响。我们希望将灾害率建模为时间的函数。我们的数据集只有一列，每个条目都对应于灾难发生的时间。

让我们加载数据并查看它的一些值：

```
coal_df = pd.read_csv('../data/coal.csv', header=None)
coal_df.head()
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505210251_00.webp)

我们将用来拟合COREE_DF数据框中的数据的模型是：
$$
\begin{aligned} f(x) & \sim \mathcal{G P}\left(\mu_{x}, K\left(x, x^{\prime}\right)\right) \\ 
y & \sim \operatorname{Poisson}(f(x)) \end{aligned}
$$
如你所见，这是一个泊松回归问题。你可能会想，在这一点上，如果我们只有一个列，只有灾难发生的日期，我们将如何执行回归。答案是将数据离散化，就像我们正在构建直方图一样。我们将使用存储箱的中心作为变量，每个存储箱的计数作为变量：

```
# discretize data
years = int(coal_df.max().values - coal_df.min().values)
bins = years // 4
hist, x_edges = np.histogram(coal_df, bins=bins)
# compute the location of the centers of the discretized data
x_centers = x_edges[:-1] + (x_edges[1] - x_edges[0]) / 2
# arrange xdata into proper shape for GP
x_data = x_centers[:, None]
# express data as the rate number of disaster per year
y_data = hist / 4
```

现在我们用PyMC3定义并求解该模型：

```
with pm.Model() as model_coal:
    ℓ = pm.HalfNormal('ℓ', x_data.std())
    cov = pm.gp.cov.ExpQuad(1, ls=ℓ) + pm.gp.cov.WhiteNoise(1E-5)
    gp = pm.gp.Latent(cov_func=cov)
    f = gp.prior('f', X=x_data)
    y_pred = pm.Poisson('y_pred', mu=pm.math.exp(f), observed=y_data)
    trace_coal = pm.sample(1000, chains=1)
```

现在我们绘制结果图：

```
_, ax = plt.subplots(figsize=(10, 6))
f_trace = np.exp(trace_coal['f'])
rate_median = np.median(f_trace, axis=0)
ax.plot(x_centers, rate_median, 'w', lw=3)
az.plot_hpd(x_centers, f_trace)
az.plot_hpd(x_centers, f_trace, credible_interval=0.5,
            plot_kwargs={'alpha': 0})
ax.plot(coal_df, np.zeros_like(coal_df)-0.5, 'k|')
ax.set_xlabel('years')
ax.set_ylabel('rate')
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505210433_70.webp)

图7.15用白线显示了灾害率的中位数与时间的关系。这些条带表示50%HPD间隔(较深)和94%HPD间隔(较浅)。在底部，我们用黑色记号笔标出了每一场灾难(这有时也被称为地毯图)。正如我们所看到的，事故率随着时间的推移而下降，除了最初的短暂上升。PyMC3文档包括煤矿灾难，但从不同的角度建模。我强烈建议您检查该示例，因为它本身非常有用，而且将其与我们刚刚使用model_Coal模型实现的方法进行比较也很有用。

请注意，即使我们将数据入库，我们也会得到一条平滑的曲线。从这个意义上说，我们可以将MODEL_COIL(通常是这种类型的模型)看作是构建一个直方图，然后对其进行平滑。

### 7.6.2 红杉数据集

现在，我们将把注意力集中在使用红木数据对二维空间问题应用我们刚刚做过的相同类型的模型上。这个数据集(使用GPL许可证分发)来自GPStuff包，这是一个用于Matlab、Octave和R的高斯进程包。该数据集由给定区域上红杉树的位置组成。推论的动机是要找出这一地区的树木比率是如何分布的。

像往常一样，我们加载数据并绘制它：

```
rw_df = pd.read_csv('../data/redwood.csv', header=None)
_, ax = plt.subplots(figsize=(8, 8))
ax.plot(rw_df[0], rw_df[1], 'C0.')
ax.set_xlabel('x1 coordinate')
ax.set_ylabel('x2 coordinate')
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505210601_88.webp)

与煤矿灾难示例一样，我们需要对数据进行离散化：

```
# discretize spatial data
bins = 20
hist, x1_edges, x2_edges = np.histogram2d(
    rw_df[1].values, rw_df[0].values, bins=bins)
# compute the location of the centers of the discretized data
x1_centers = x1_edges[:-1] + (x1_edges[1] - x1_edges[0]) / 2
x2_centers = x2_edges[:-1] + (x2_edges[1] - x2_edges[0]) / 2
# arrange xdata into proper shape for GP
x_data = [x1_centers[:, None], x2_centers[:, None]]
# arrange ydata into proper shape for GP
y_data = hist.flatten()
```

请注意，我们将x1和x2数据视为分开的数据，而不是网格网格。这允许我们为每个坐标构建协方差矩阵，从而有效地减小了计算GP所需的矩阵的大小。在使用LatentKron类定义GP时，我们只需小心。重要的是要注意，这不是一个数字技巧，而是这类矩阵结构的数学属性，因此我们不会在我们的模型中引入任何近似或误差。我们只是用一种更快的计算方式来表达它：

```
with pm.Model() as model_rw:
    ℓ = pm.HalfNormal('ℓ',  rw_df.std().values, shape=2)
    cov_func1 = pm.gp.cov.ExpQuad(1, ls=ℓ[0])
    cov_func2 = pm.gp.cov.ExpQuad(1, ls=ℓ[1])
    gp = pm.gp.LatentKron(cov_funcs=[cov_func1, cov_func2])
    f = gp.prior('f', Xs=x_data)
    y = pm.Poisson('y', mu=pm.math.exp(f), observed=y_data)
    trace_rw = pm.sample(1000)
```

最后，我们绘制结果图：

```
rate = np.exp(np.mean(trace_rw['f'], axis=0).reshape((bins, -1)))
fig, ax = plt.subplots(figsize=(6, 6))
ims = ax.imshow(rate, origin='lower')
ax.grid(False)
ticks_loc = np.linspace(0, bins-1, 6)
ticks_lab = np.linspace(0, 1, 6).round(1)
ax.set_xticks(ticks_loc)
ax.set_yticks(ticks_loc)
ax.set_xticklabels(ticks_lab)
ax.set_yticklabels(ticks_lab)
cbar = fig.colorbar(ims, fraction=0.046, pad=0.04)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505210705_b5.webp)



在图7.17中，较浅的颜色意味着比较深的颜色有更高的树木比率。我们可以想象，我们对寻找高增长率地区很感兴趣，因为我们可能对森林是如何从火灾中恢复的感兴趣，或者我们对土壤的性质感兴趣，我们把树木作为代理。



## 7.7 总结

 高斯过程是多元高斯分布到无限多维的推广，由均值函数和协方差函数完全指定。因为我们可以在概念上把函数看作无限长的向量，所以我们可以使用高斯过程作为函数的先验。在实践中，我们处理的不是无限大的对象，而是维数与数据点一样多的多变量高斯分布。为了定义相应的协方差函数，我们使用了适当的参数化核；通过学习这些超参数，我们最终了解了任意复杂函数。

 这一章中，我们简要介绍了高斯过程，还有许多与之相关的主题 需要学习（比如构建一个半参数化模型，将线性模型作为均值函 数），或者是将两个或者多个核函数组合在一起来描述未知函数，或 者是如何将高斯过程用于分类任务，或者是如何将高斯过程与统计学 或者机器学习中的其他模型联系起来。不管怎么说，我希望本章对高 斯过程的介绍以及本书中一些其他主题的介绍能够激励你阅读、使用 和进一步学习贝叶斯统计。

 

##  7.8 练习

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505210819_20.webp)

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505210830_d3.webp)