 # 第6章 混合模型

 <style>p{text-indent:2em;2}</style>

Plate河是地球上最宽的河流，是阿根廷和乌拉圭间的天然边界。19世纪末，这条河沿岸的港口地区是当地人与非洲人和欧洲移民混居的地方。这次相遇的一个结果是混合了欧洲音乐，如华尔兹和马祖卡、非洲坎通贝和阿根廷米隆加，产生了一种舞蹈和音乐，我们现在称之为探戈。

混合已有元素是创造新东西的一种方式，而不仅仅是音乐。在统计学中，混合模型是一种常用的建模方法。这些模型通过混合更简单的分布来获得复杂分布。例如，可以组合两个高斯分布来描述双峰分布，或者组合多个高斯分布来描述任意分布。虽然使用高斯分布很常见，但原则上可以混合任何我们想要的分布族。混合模型用于不同的目的，例如直接建模子总体，或者作为处理不能用更简单的分布描述的复杂分布的有用技巧。

 本章我们将学习以下内容：

- 有限混合模型；
- 无限混合模型；
- 连续混合模型。


## 6.1 混合模型

 当总体是不同的子群体的组合时，混合模型自然会出现。一个非常熟悉的例子是给定成年人口的身高分布，这可以被描述为女性和男性亚群的混合。另一个典型的例子是手写数字的群集。在这种情况下，预期10个子种群是非常合理的，至少在基数为10的系统中是如此！如果我们知道每个观测数据属于哪个子总体，那么使用该信息将每个子总体建模为一个单独的组通常是一个好主意。然而，当我们无法直接访问这些信息时，混合模型就派上了用场。

```{tip}
许多数据集不能用单一的概率分布来正确描述，但可以被描述为这些分布的混合。假设数据来自混合分布的模型称为混合模型。
```

在构建混合模型时，没有必要相信我们数据描述的是真实的子总体。混合模型也可以用作统计技巧，为我们的工具箱增加灵活性。以高斯分布为例，可以用它作为许多单峰分布和对称分布的合理近似。但多模式或不对称分布怎么办呢？我们能用高斯分布来模拟它们吗？是的，如果使用高斯混合模型的话是可以的。

在高斯混合模型中，每个分量将是具有不同平均值和不同标准差（通常）的高斯分布。通过组合多个高斯分布，可以为模型增加灵活性，以便适应复杂的数据分布。事实上，可以通过适当的高斯组合来近似任何我们想要的分布。分布的确切数量将取决于近似的准确性和数据细节。事实上，在整本书的许多情节中，我们一直在运用混合高斯的思想。核密度估计（ `KDE` ）技术是这一思想的非贝叶斯（和非参数）实现。从概念上讲，当我们调用 `az.plot_kde 时` ，该函数将一个高斯（具有固定方差）放在每个数据点的顶部，然后对所有单独的高斯进行求和，以近似数据的经验分布。

下图显示了一个实际示例，说明如何混合八个高斯分布来表示一个复杂分布，就像一条蟒蛇在消化一头大象。图中所有高斯都有相同方差，并且以橙色圆点为中心，这些圆点表示来自可能未知总体的样本点。如果你仔细看图，你可能会注意到，两个高斯基本上是一个叠加在另一个之上：

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505172616_90.webp)

 无论我们是真的相信子总体，还是出于数学上的方便(甚至是中间的原因)而使用它们，混合模型都是一种有用的方式，通过使用混合分布来描述数据，从而为我们的模型增加灵活性。

## 6.2 有限混合模型

建立混合模型的一种方法是考虑两个或多个分布的有限加权混合。这就是所谓的有限混合模型。因此，观测数据的概率密度是数据的子组的概率密度的加权和：
$$
p(y \mid \theta)=\sum_{i=1}^{K} w_{i} p_{i}\left(y \mid \theta_{i}\right) \tag{6.1}
$$
这里 $w_i$ 是每个组分(或类)的权重，可以将 $w_i$ 解释为组分 $i$ 的概率，因此其值被限制在区间 [0，1] ，并且 $ \sum_{i}^{K} w_{i}=1 $ 。这些组分 $p_i(y|\theta_i)$ 实际上可以是我们认为有用的任何东西，从简单分布（如高斯或泊松）到更复杂的对象（如分层模型或神经网络）。对于有限混合模型，$K$ 是一个有限的数字（通常但不一定是一个很小的数字）。为拟合有限混合模型，我们需要提供 $K$ 的值，不管是事先确实已知，还是有根据的猜测。

从概念上讲，求解混合模型需要做的就是将每个数据点适当地分配给其中一个组分。在概率模型中可以通过引入随机变量  $z$ 来实现这一点，该变量的功能是指定特定观测被分配给哪个组分。这个变量通常被称为隐变量 。我们称其为“隐”是因为它无法直接观测。

让我们使用第 2 章概率编程中已经看到的化学漂移数据开始构建混合模型：

```
cs = pd.read_csv('../data/chemical_shifts_theo_exp.csv')
cs_exp = cs['exp']
az.plot_kde(cs_exp)
plt.hist(cs_exp, density=True, bins=30, alpha=0.3)
plt.yticks([])
```

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505173831_56.webp" style="zoom:67%;" />

从图中可以看到这个数据不能用单一的高斯来正确描述，但也许三个或四个可以做到这一点。事实上，有很好的理论原因，我们不会在这里讨论，这表明数据确实来自大约 40 个亚群体的混合，但它们之间有相当大的重叠。

我们可以从抛硬币问题中得到一些启发。对于该模型有两种可能的结果，我们使用伯努利分布来描述它们。因为不知道得到正面或反面的概率，所以使用贝塔分布作为先验。混合模型是相似的，不同的是，现在有了 $K$ 个结果（或分量），而不是像正面或反面只有两个结果。 伯努利分布更一般的形式是类别分布，贝塔分布的一般形式是狄利克雷分布。



### 6.2.1 类别分布

类别分布是最一般的离散分布，并且使用指定每个可能结果的概率的参数来参数化。下图表示类别分布的两个可能实例。点表示类别分布的值，而连续的线条是帮助我们掌握分布形状的视觉辅助工具：

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505174816_08.webp" style="zoom:67%;" />



### 6.2.2 狄利克雷分布

狄利克雷分布初看可能有点奇怪，因为它是一种单纯形，有点像 $n$ 维的三角形：1-单纯形是一个线段；2-单纯形是一个三角形；3-单纯形是正四面体，以此类推。为什么是一个单纯形？直观上看，这是因为该分布是一个长度为 $K$ 的向量，其中每个元素都为正数而且所有元素之和为 1。在理解为何狄利克雷分布是贝塔分布的一般形式前，先回顾下贝塔分布的特性。前面用贝塔分布来描述二分类问题，其中之一的概率为 $p$ ，另一个概率为 $1-p$ ，因而可以认为贝塔分布返回长度为2的向量 $[p, 1- p]$ 。当然，实践中通常忽略 $1-p$ ，因为通过 $p$ 完全可以定义。此外贝塔分布也可以通过两个标量 $α$ 和 $β$ 来定义。

这些参数如何类比到狄利克雷分布中呢？先考虑下最简单的狄利克雷分布，即 3 分类问题建模。此时狄利克雷分布返回一个长度为 3 的向量 $[p, q, r ]$，其中 $r = 1 − (p + q)$ 。也可以用 3 个标量来描述狄利克雷分布，称为 $α$、$β$ 和 $γ$ ，不过不便于推广到高维情形，所以统一使用长度为 $K$ 的向量 $α$ 来描述，其中 $K$ 对应可能出现的结果类型数量。

可将贝塔分布和狄利克雷分布想象成描述概率分布的分布，为更直观地理解，下图将不同参数对应的狄利克雷分布画出来，留意图中每个三角形与参数相近的贝塔分布之间的联系。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505175717_a9.webp)

现在对狄利克雷分布有了了解，就可以进一步掌握学习混合模型的知识。构建混合模型的一种可视化方法是将其看作是基于高斯估计模型的 $K$ 面抛硬币问题，或 $K$ 面掷骰子问题。用 Kruschke 图可以将模型画成如下形式：

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505175913_df.webp" style="zoom:67%;" />

途中圆角矩形框表示我们有 $K$ 个高斯似然（以及对应的先验），类别变量决定具体使用哪个类别来描述数据。 记住，假设已知这些高斯分布的均值和标准差；只需将每个点赋给一个高斯分布即可。

该模型（假设 `clusters=2` ）可以用 `PyMC3` 实现为：

```
with pm.Model() as model_kg:
    p = pm.狄利克雷('p', a=np.ones(clusters))
    z = pm.Categorical('z', p=p, shape=len(cs_exp))
    means = pm.Normal('means', mu=cs_exp.mean(), sd=10, shape=clusters)
    sd = pm.HalfNormal('sd', sd=10)
    y = pm.Normal('y', mu=means[z], sd=sd, observed=cs_exp)
    trace_kg = pm.sample()
```

如果你运行这段代码，会发现非常慢，迹看起来也很糟糕。原因是，在 `model_kg` 中显式地包含了隐变量 $z$ 。此时对离散变量 $z$ 采样通常会导致缓慢的混合，以及对概率分布尾部的大量无效探索。解决该问题的一种方法是对模型重新参数化。

注意，在混合模型中，观测变量 $y$ 是在隐变量 $z$ 基础上的条件建模，也就是 $p(y|z,\theta)$ 。可以认为隐变量 $z$ 是一个可被边缘化的多余变量并获得 $p(y|\theta)$ 。`PyMC3` 包含一个 `NormalMixture` 分布,可用于编写高斯混合模型：

```
clusters = 2
with pm.Model() as model_mg:
    p = pm.狄利克雷('p', a=np.ones(clusters))
    means = pm.Normal('means', mu=cs_exp.mean(), sd=10, shape=clusters)
    sd = pm.HalfNormal('sd', sd=10)
    y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=cs_exp)
    trace_mg = pm.sample(random_seed=123)
```

使用 `ArviZ` 查看迹的效果，我们将在下一节中将其与使用 `model_mgp` 获得的迹进行比较：

```
varnames = ['means', 'p']
az.plot_trace(trace_mg, varnames)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505180821_7d.webp)

同时计算此模型的摘要，我们将在下一节中将其与使用 `model_mgp` 获得的摘要进行比较：

```
az.summary(trace_mg, varnames)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505180929_83.webp)



### 6.2.3 混合模型的不可辨识性

如果仔细检查上图，会发现有一些有趣的事情。图中是两个均值估计为 47 和 57.5 左右的双峰分布，如果检查使用 `az.summary` 获得的摘要，则发现均值几乎相等，在 52 左右。同时可以看到一些与 $p$ 值相似的东西。这是统计学上称为`参数不可辨识性`现象的一个例子。之所以出现该情况，是因为如果组分 1 的均值为 47 ，组分 2 的均值为 57.5 ，则模型是相同的，反之亦然；两种情况完全相同。在混合模型语境中，这也称为标签切换问题。在第三章 “线性回归建模” 中，在讨论线性模型和高相关性变量时已经发现了一个参数不可辨识性的例子。

在可能的情况下，应定义模型以消除不可识别性。对于混合模型，至少有两种方法可以做到这一点：

- 强制对组分进行排序，例如，使用严格递增顺序排列组分的均值
- 使用信息性先验，如果模型参数的多个选择获得相同的似然函数，则模型中的参数将不被识别。

使用`PyMC3`强制对组分排序的简单方法是调用 `pm.potential()` 。`势`（potential）是人为添加到似然中的因素，而非添加到模型中的变量。似然和势之间的主要区别在于：**势不一定取决于数据，而似然取决于数据** 。我们可以使用势来实施约束，如可以这样定义势：如果没有违反约束，在似然上增加一个 0 的因子；否则，增加一个 $-\infty$ 的因子。最终结果是，模型认为违反约束的参数(或参数组合)不可能出现，进而对其他值保证放心：

```
clusters = 2
with pm.Model() as model_mgp:
    p = pm.狄利克雷('p', a=np.ones(clusters))
    means = pm.Normal('means', mu=np.array([.9, 1]) * cs_exp.mean(),
                      sd=10, shape=clusters)
    sd = pm.HalfNormal('sd', sd=10)
    order_means = pm.Potential('order_means',
                               tt.switch(means[1]-means[0] < 0,
                               -np.inf, 0))
    y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=cs_exp)
    trace_mgp = pm.sample(1000, random_seed=123)
varnames = ['means', 'p']
az.plot_trace(trace_mgp, varnames)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505182452_a2.webp)

计算一下此模型的摘要：

```
az.summary(trace_mgp)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505182535_0f.webp)

另一个有用的约束是确保所有成分都有不为 0 的概率，或者换句话说，混合模型中的每个成分都至少有一个观测点。可以使用以下表达式执行该操作：

```
p_min = pm.Potential('p_min', tt.switch(tt.min(p) < min_p, -np.inf, 0))
```

在此可以将 `min_p` 设置为某个任意但合理的值，例如 0.1 或 0.01 。

如图所示，$α$ 值控制狄利克雷分布的密度。与 `model_mgp` 中一样，可获得单纯形上的平坦先验。$α$ 值越大意味着先验信息越丰富。经验证据表明，$\alpha \approx 4或10 $ 通常是比较好的缺省选择，因为这通常导致后验分布中每个分量至少有一个数据点，同时减少高估组分数量的机会。

### 6.2.4 如何选择 $K$ ?

有限混合模型带来的问题之一是如何确定组分数量。一般经验是从相对较少的组分开始尝试，然后增加组分数量，以改进模型适配性的评估。模型适配性的评估可以使用 `后验预测检查`、`WAIC` 或 `LOO测量` 等方法，并基于建模师的专业知识。

让我们来比较一下 $K=\{3,4,5,6\}$ 四个模型。为此对模型进行四次拟合，并保存迹和模型以供后面使用：

```
clusters = [3, 4, 5, 6]
models = []
traces = []
for cluster in clusters:
    with pm.Model() as model:
        p = pm.狄利克雷('p', a=np.ones(cluster))
        means = pm.Normal('means',
                          mu=np.linspace(cs_exp.min(), cs_exp.max(),
                                         cluster),
                          sd=10, shape=cluster,
                          transform=pm.distributions.transforms.ordered)
        sd = pm.HalfNormal('sd', sd=10)
        y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=cs_exp)
        trace = pm.sample(1000, tune=2000, random_seed=123)
        traces.append(trace)
        models.append(model)
```

为更好显示 $K$ 如何影响推断，将把这些模型的拟合结果与使用 `az.plot_kde` 获得的模型进行比较，并且绘制混合模型的高斯分量图：

```
_, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
ax = np.ravel(ax)
x = np.linspace(cs_exp.min(), cs_exp.max(), 200)
for idx, trace_x in enumerate(traces):
    x_ = np.array([x] * clusters[idx]).T
    for i in range(50):
        i_ = np.random.randint(0, len(trace_x))
        means_y = trace_x['means'][i_]
        p_y = trace_x['p'][i_]
        sd = trace_x['sd'][i_]
        dist = stats.norm(means_y, sd)
        ax[idx].plot(x, np.sum(dist.pdf(x_) * p_y, 1), 'C0', alpha=0.1)
    means_y = trace_x['means'].mean(0)
    p_y = trace_x['p'].mean(0)
    sd = trace_x['sd'].mean()
    dist = stats.norm(means_y, sd)
    ax[idx].plot(x, np.sum(dist.pdf(x_) * p_y, 1), 'C0', lw=2)
    ax[idx].plot(x, dist.pdf(x_) * p_y, 'k--', alpha=0.7)
    az.plot_kde(cs_exp, plot_kwargs={'linewidth':2, 'color':'k'},ax=ax[idx])
    ax[idx].set_title('K = {}'.format(clusters[idx]))
    ax[idx].set_yticks([])
    ax[idx].set_xlabel('x')
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505190548_b1.webp)


上图显示了概率密度图，黑色实线为 `KDE` 结果，蓝线为拟合平均线，半透明蓝色线为来自后验的采样。均值参数的高斯分量用黑色虚线表示。在图中，$K=3$ 似乎太低了，4、5 或 6 可能是更好的选择。

注意，高斯混合模型显示了两个明显的峰值（在 55-60 左右），而 `KDE` 预测的峰值不那么明显(更平坦)。这不一定是高斯混合模型拟合的不好，而是因为 `KDE` 通常会自行调整以提供更平滑的曲线。可使用直方图代替 `KDE`，但直方图也是近似密度的方法。正如在第 5 章 “模型比较” 中讨论的，可尝试计算感兴趣检测量的预测后验曲线图，并计算贝叶斯 $p$ 值。下图显示了此类计算和可视化的示例：

```
ppc_mm = [pm.sample_posterior_predictive(traces[i], 1000, models[i])
          for i in range(4)]
fig, ax = plt.subplots(2, 2, figsize=(10, 6), sharex=True,
constrained_layout=True)
ax = np.ravel(ax)
def iqr(x, a=0):
    return np.subtract(*np.percentile(x, [75, 25], axis=a))
T_obs = iqr(cs_exp)
for idx, d_sim in enumerate(ppc_mm):
    T_sim = iqr(d_sim['y'][:100].T, 1)
    p_value = np.mean(T_sim >= T_obs)
    az.plot_kde(T_sim, ax=ax[idx])
    ax[idx].axvline(T_obs, 0, 1, color='k', ls='--')
    ax[idx].set_title(f'K = {clusters[idx]} \n p-value {p_value:.2f}')
    ax[idx].set_yticks([])
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505190939_5f.webp)

从图可以看出，$K=6$ 是一个很好的选择，其贝叶斯 $p$ 值非常接近 0.5。正如在下面 `DataFrame` 和图 6.10 中看到的，`WAIC` 也将 $K=6$ 评为更好的模型：

```
comp = az.compare(dict(zip(clusters, traces)), method='BB-pseudo-BMA')
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505191135_11.webp)

根据 `WAIC` ，大多数情况下看图比看表容易得多，所以让我们画一个图来找出不同模式。如下图所示，虽然 6 分量模型的 `WAIC` 比其他模型低，但当考虑估计标准误差时存在相当大的重叠，特别是对 5 分量模型：

```
az.plot_compare(comp)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505191504_15.webp)


### 6.2.5 混合模型与聚类

聚类是统计学和机器学习中无监督学习的一部分，与分类有点相似，不过更复杂一些，因为聚类问题中没有标签。概括地说，聚类就是在没有标签的情况下将属性相近的数据点归类到一起，使组内数据的距离较近，组间数据的距离较远。聚类有许多应用，比如，在种系遗传学中用来指导解决一些进化问题。商业化应用如根据用户历史消费记录聚类分组，进而估计他们可能对哪些电影、书籍、歌曲感兴趣。在一些无监督学习任务中，可能希望直接做聚类，也可能将其作为探索式数据分析的一种手段。

使用概率模型实现聚类通常称为基于模型的聚类。使用概率模型允许我们计算每个数据点属于每个类的概率。这就是所谓的软聚类，而不是硬聚类（在硬聚类中，数据点都属于概率为 0 或 1 的类）。可以引入一些规则或边界将软聚类变成硬聚类，就像 Logistic 回归分类方法使用 0.5 作为默认边界实施二分类。当然，自然会考虑将数据点分配给概率值最高的类。

混合模型和聚类的主要区别在于：当谈论聚类问题时，通常指的是`将对象分组`；而讨论混合模型时，通常指 `通过简单分布的混合对复杂分布建模` ，或者`识别子总体`，或 `为有一个更灵活的模型来描述数据`。

## 6.3 无限混合模型

对于某些问题（如手写数字的聚类），组分数量可预先知道的；有时可以有较好的猜测，如知道蝴蝶花样本来自一个只有三种蝴蝶花生长的地区，因此使用三组分是合理起点；即使不能明确组分数量时，也可通过模型比较和选择来帮助确定组分数量。然而，对于有些问题，选择先验的组分数量可能并非好事，而我们希望能够从数据中估计出这个数字。此类问题的贝叶斯解与狄拉克雷过程有关。

### 6.3.1 狄利克雷过程

到目前为止，我们看到的所有模型都是`参数化模型`，通常具有固定数量的参数；事实上还可以使用`非参数模型`，准确地说应当是`非固定参数模型`。`非参数模型`是理论上具有无限多个参数的模型，实践中以某种方式让其从理论上无限的参数收缩到某个有限数量，换言之，用数据决定实际参数数量。因此非参数模型非常灵活。在本书中介绍两类非参数模型：高斯过程(下一章介绍)和狄利克雷过程。

由于狄利克雷分布是贝塔分布的 $n$ 维推广，而狄利克雷过程是狄利克雷分布的无限维推广。狄利克雷分布是概率空间上的概率分布，而狄利克雷过程是概率分布空间上的概率分布（即分布的分布），这意味着从狄利克雷过程中采样的结果实际上是一个概率分布。对于有限混合模型，我们可以使用狄利克雷分布为固定数量组分的模型设置先验；而狄利克雷过程则是一种将先验分布分配给非固定数量组分的方式，甚至可以认为狄利克雷过程是从先验分布中采样的一种方式。

在进入实际的非参数混合模型前，先花点时间来讨论一下狄利克雷过程的一些细节。狄利克雷过程的形式定义有些复杂，此处仅介绍一些狄利克雷过程的属性，这些属性与理解狄利克雷过程在混合模型建模中的作用相关：

- 狄利克雷过程是一种实现为概率分布的分布，而不是像高斯分布那样的实数分布。
- 狄利克雷过程由一个基础分布 $\mathcal{H}$ 和称为聚集参数的正实数 $\alpha$ 指定(这类似于狄利克雷分布中的聚集参数)。
- $\mathcal{H}$ 是狄利克雷过程的期望值，这意味着狄利克雷过程将在  $\mathcal{H}$ 周围生成分布，类似于高斯分布的平均值
- 随着 $\alpha$ 增长，实现变得越来越不集中。
- 在实践中，狄利克雷过程总是产生离散分布。
- 在 $\alpha \to \infty$ 情况下，狄利克雷过程的实现等于基本分布。因此，如果基本分布是连续的，则狄利克雷过程将生成连续分布。出于此原因，数学家称，从狄利克雷过程产生的分布几乎肯定是离散的，因为实践中，作为 $\alpha$ 总是一个有限数字，我们总是使用离散分布。

为使这些属性更加具象，让我们再看一看图 6.3 中的类别分布。我们可以通过指示 x 轴上的位置和 y 轴上的高度来完全指定这种分布。对于类别分布，x轴上的位置被限制为整数，高度之和必须为1。保留最后一个限制，但放宽前一个限制。为了生成x轴上的位置，我们将从基本分布中进行采样。原则上可以是我们想要的任何分布，因此如果我们选择高斯分布，位置原则上可以是实线上的任何值，相反，如果我们选择Beta，位置将被限制在区间[0，1]，如果我们选择泊松作为基础分布，位置将被限制在非负整数{0，1，2，...}。

到目前为止还不错，我们如何选择y轴上的值呢？我们遵循一个被称为折断棍棒过程的Gedanken实验。假设我们有一根长度为1的棍子，然后我们把它一分为二(不一定相等)。我们把一部分放在一边，把另一部分一分为二，然后我们就一直这样做，直到永远。在实践中，由于我们不能真正无限地重复这个过程，所以我们将其截断为某个预定义的值，但大意是成立的。为了控制断棒过程，我们使用了一个参数。随着我们增加α的价值，我们将把大棒分成越来越小的部分。因此，在未来，我们不会折断棍子，当我们将它折断成无限的碎片时。图6.11显示了来自狄利克雷过程的四个抽签，对应于四个不同的值。稍后我将解释生成该数字的代码，让我们首先重点了解这些示例告诉我们有关狄利克雷过程的信息：

```
def stick_breaking_truncated(α, H, K):
    """
    Truncated stick-breaking process view of a 狄利克雷过程
    Parameters
    ----------
    α : float
        concentration parameter
    H : scipy distribution
        Base distribution
    K : int
        number of components
    Returns
    -------
    locs : array
        locations
    w : array
    probabilities
    """
    
    βs = stats.beta.rvs(1, α, size=K)
    w = np.empty(K)
    w = βs * np.concatenate(([1.], np.cumprod(1 - βs[:-1])))
    locs = H.rvs(size=K)
    return locs, w
    
# Parameters 狄利克雷过程
K = 500
H = stats.norm
alphas = [1, 10, 100, 1000]
_, ax = plt.subplots(2, 2, sharex=True, figsize=(10, 5))
ax = np.ravel(ax)
for idx, α in enumerate(alphas):
    locs, w = stick_breaking_truncated(α, H, K)
    ax[idx].vlines(locs, 0, w, color='C0')
    ax[idx].set_title('α = {}'.format(α))
plt.tight_layout()
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505193131_69.webp)

从图6.10可以看出，狄利克雷过程是一个离散分布。当α增加时，我们可以得到更分散的分布和更小的棒子，注意y轴比例上的变化，并记住总长度固定为1。基本分布控制位置，因为位置是从基本分布中采样的，我们可以从图6.10中看到，随着狄利克雷过程分布形状的增加，它越来越类似于基本分布，从这一点我们希望可以看到，在这个过程中，我们应该准确地获得基本分布。



我们可以认为狄利克雷过程是随机分布f上的先验分布，其中基分布是我们预期的f，而聚集参数表示我们对先验猜测的置信度。

图6.1显示，如果在每个数据点上放置一个高斯分布，然后将所有高斯分布相加，则可以近似计算数据的分布。我们可以使用狄利克雷过程来做类似的事情，但不是将高斯放在每个数据点的顶部，我们可以在狄利克雷过程实现中的每个子棒的位置放置一个高斯，然后根据该子棒的长度对该高斯进行缩放或加权。此过程提供了无限高斯混合模型的一般配方。或者，我们可以用任何其他分布代替高斯分布，这样我们就有了一个通用的无限混合模型的通用配方。图6.12显示了我们使用混合拉普拉斯分布的情况下这类模型的一个例子。我随意选择拉普拉斯分布只是为了强化这样一种想法，即你绝不会被限制在做高斯混合模型：

```
α = 10
H = stats.norm
K = 5
x = np.linspace(-4, 4, 250)
x_ = np.array([x] * K).T
locs, w = stick_breaking_truncated(α, H, K)
dist = stats.laplace(locs, 0.5)
plt.plot(x, np.sum(dist.pdf(x_) * w, 1), 'C0', lw=2)
plt.plot(x, dist.pdf(x_) * w, 'k--', alpha=0.7)
plt.yticks([])
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505193226_8a.webp)

我希望您在这一点上对狄利克雷过程有一个很好的直觉，仍然缺少的唯一细节是理解函数 `stick_break_truncated` 。数学上狄利克雷过程的断棒过程视图可以用以下方式表示：

$$
 \sum_{k=1}^{\infty} w_{k} \cdot \delta_{\theta_{k}}(\theta)=f(\theta) \sim D P(\alpha, H) 
$$

其中：

$ \delta_{\theta_{k}} $ 是指示函数，它在 $ \delta_{\theta_{k}}\left(\theta_{k}\right)=1 $, 其他均为 0 ，这表示从基础分布 $\mathcal{H}$ 中采样的位置

概率 $ w_{k} $ 定义为:
$$
w_{k}=\beta_{k}^{\prime} \cdot \prod_{i=1}^{k-1}\left(1-\beta_{i}^{\prime}\right)
$$

其中：

 $w_{k}$ 是子棍的长度
 $\prod_{i=1}^{k-1}\left(1-\beta_{i}^{\prime}\right)$  是剩余部分的长度
 $\beta_{k}^{\prime}$ 指示如何打破剩余部分
 $\beta_{k}^{\prime} \sim \operatorname{Beta}(1, \alpha)$, 从公式可看出：当 $\alpha$ 增长时 $\beta_{k}^{\prime}$ 平均将会变小

现在我们已经准备好尝试在`PyMC3`中实现狄利克雷过程了。首先定义一个使用`PyMC3`的 `stick_breaking` 函数：

```
N = cs_exp.shape[0]
K = 20
def stick_breaking(α):
    β = pm.Beta('β', 1., α, shape=K)
    w = β * pm.math.concatenate([[1.],
                                tt.extra_ops.cumprod(1. - β)[:-1]])
    return w
```

必须为 $\alpha$ 定义一个先验。一种常见的选择是伽马分布：

```
with pm.Model() as model:
    α = pm.Gamma('α', 1., 1.)
    w = pm.Deterministic('w', stick_breaking(α))
    means = pm.Normal('means', mu=cs_exp.mean(), sd=10, shape=K)
    sd = pm.HalfNormal('sd', sd=10, shape=K)
    obs = pm.NormalMixture('obs', w, means, sd=sd, observed=cs_exp.values)
    trace = pm.sample(1000, tune=2000, nuts_kwargs={'target_accept':0.9})
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505194120_60.webp)

从图6.13可以看出，$\alpha$ 的值相当低，这表明需要一些组分来描述数据。

因为我们是通过截断折杆过程来近似无穷大的狄利克雷过程，所以检查截断值(在本例中 $K=20$ )没有引入任何偏差是很重要的。要做到这一点，一种简单的方法是绘制每个组分的平均权重，为了安全起见，我们应该有几个权重可以忽略的组分，否则我们必须增加截断值。图6.14就是这类绘图的一个例子。我们可以看到，第一个组分中只有几个是重要的，因此我们可以确信所选的上限值 $K=20$ 对于此模型和数据足够大：

```
plt.figure(figsize=(8, 6))
plot_w = np.arange(K)
plt.plot(plot_w, trace['w'].mean(0), 'o-')
plt.xticks(plot_w, plot_w+1)
plt.xlabel('Component')
plt.ylabel('Average weight')
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505194313_98.webp)

图6.15显示了使用狄利克雷过程模型(黑色)线和后方(灰色)线的样本估计的平均密度，以反映估计中的不确定性。与图6.2和图6.8中的KDE相比，该模型的密度也不那么平滑：

```
x_plot = np.linspace(cs.exp.min()-1, cs.exp.max()+1, 200)
post_pdf_contribs = stats.norm.pdf(np.atleast_3d(x_plot),
                                   trace['means'][:, np.newaxis, :],
                                   trace['sd'][:, np.newaxis, :])
post_pdfs = (trace['w'][:, np.newaxis, :] *
             post_pdf_contribs).sum(axis=-1)
plt.figure(figsize=(8, 6))
plt.hist(cs_exp.values, bins=25, density=True, alpha=0.5)
plt.plot(x_plot, post_pdfs[::100].T, c='0.5')
plt.plot(x_plot, post_pdfs.mean(axis=0), c='k')
plt.xlabel('x')
plt.yticks([])
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505194356_42.webp)


## 6.4 连续混合模型

 本章重点关注了离散的混合模型，不过我们也可以有连续的混合模型。事实上我们已经了解了其中的一部分，连续混合模型之一就是前面介绍过的鲁棒逻辑回归模型，由两部分组成：一个逻辑回归和 一个随机估计。注意此时参数 $π$ 不再是一个开关，而更像是一个球形把手控制着 Logistic 回归和随机变量的比例，只有当 $π$ 为 0 或 1 时，我们才会得到完全随机的结果或者完全的 Logistic 回归结果。

 层次模型也可以看做是连续混合模型，其中每个组的参数来自于上层的连续分布。更具体点，假设我们要对几个不同的组做线性回归，可以假设每个组都有自己的斜率或者所有组都共享相同的斜率。 此外，除了将问题看作这两个极限情况，我们还可以构建一个多层模 型，对这些极限值构建一个连续混合模型，此时这些极限值不过是分层模型中的一些特例罢了。

### 6.3.1 贝塔--二项分布

贝塔-二项分布是一个离散分布，通常用来描述 $n$ 次伯努利实验中成功的次数 $y$ ，其中每次实验成功的概率 $p$ 未知，并且假设其服从参数为 $α$ 和 $β$ 的贝塔分布，对应的数学形式如下：

$$
\operatorname{BetaBinonial}(y \mid n, \alpha, \beta)=\int_{0}^{1} \operatorname{Bin}(y \mid p, n) \operatorname{Beta}(p \mid \alpha, \beta) d p
$$
 也就是说，为了找到观测到结果 $y$ 的概率，我们遍历所有可能的（连续的）$p$ 值然后求平均。因此，贝塔-二项分布也可以看作是连续混合模型。如果你觉得贝塔-二项分布听起来很熟悉，那一定是因为你对本书前两章学得很认真！在抛硬币的问题中，我们用到了该模型，尽管当时显式地使用了一个贝塔分布和一个二项分布，你也可以直接使用贝塔-二项分布。

### 6.3.2 负二项分布

 类似的，还有负二项分布。可以将其看作是一个连续混合的伽马--泊松分布，也就是将一个来自伽马分布的值作为泊松分布的参数，然后对泊松分布连同伽马分布求均值（积分）。该分布常用来解决计数型数据中的一个常见问题：过度离散。假设你用一个泊松分布来对计数型数据建模，然后你意识到数据中的方差超出了模型的方差（使用泊松分布的一个问题是，其均值与方差是有联系的，事实上是用同一个参数描述的），那么解决该问题的一个办法是将数据看作是（连续的）泊松分布的混合，其中的泊松分布的参数来自于一个伽马分布，从而很自然地用到了负二项分布。使用混合分布之后，我们的模型有了更好的灵活性，并且能够更好地适应从数据中观测到的均值和方差。

 贝塔-二项分布和负二项分布都可以用作线性模型的一部分，而且也都有零膨胀的版本，此外二者都已经在 `PyMC3` 中实现了。

### 6.3.3 学生 $t$ 分布

 前面我们介绍了 $t$ 分布是一种更鲁棒的高斯分布。从下面的数学表达式可以看到，$t$ 分布同样可以被看作是连续混合模型：
$$
t_{\nu}(y \mid \mu, \sigma)=\int_{0}^{\infty} N(y \mid \mu, \sigma) \operatorname{Inv} \chi^{2}(\sigma \mid \nu) d \nu
$$
注意，这个表达式与前面的负二项分布的表达式很像，不过这里是参数为 $\mu$ 和 $σ$ 的正态分布以及从参数为 $v$ 的 $Invχ^2$ 分布中采样得到的 $σ$，也就是自由度，通常更倾向于称之为正态参数。这里参数 $v$ 和贝塔-二项分布里的参数 $p$ 概念上相似，等价于有限混合模型中的隐变量 $z$ 。 对于有限混合模型来说，很多时候我们可以在推断之前先对隐变量 $z$  做边缘化处理，从而得到一个更简单的模型，正如前面边缘混合模型中的例子一样。

## 6.5 总结

许多问题可以描述为由不同的子群体组成的总体群体。当我们知道每个观测数据属于哪个子总体时，我们就可以将每个子总体具体建模为一个单独的组。然而，很多时候我们无法直接访问这些信息，因此使用混合模型对该数据进行建模可能更合适。我们可以使用混合模型，试图捕捉数据中真实的子总体，或者作为一种通用的统计技巧，通过组合更简单的分布来模拟复杂的分布。我们甚至可以尝试在中间做些什么。

在这一章中，我们将混合模型分为三类--有限混合模型、无限混合模型和连续混合模型。有限混合模型是两个或多个分布的有限加权混合，每个分布或分量代表数据的一个子组。原则上组分几乎可以是我们认为有用的任何东西，从简单分布（如高斯或泊松）到复杂对象（如分层模型或神经网络）。从概念上讲，要求解混合模型，需要做的就是将每个数据点适当地分配给其中一个分量，我们可以通过引入隐变量 $z$ 来实现这一点。我们使用类别分布（最一般的离散分布）和 狄利克雷 先验（它是贝塔分布的 $n$ 维推广）。对离散变量 $z$ 进行采样可能会遇到问题，因此可以方便将其边际化。`PyMC3` 包括一个正态混合分布和一个混合分布，可以执行这种边际化计算，使得构建混合模型变得更容易。使用混合模型时的一个常见问题是，这些模型可能会导致标签切换，这是一种不可辨识的形式。消除不可辨识性的一种方法是强制对组分进行排序，使用`PyMC3`，可以通过 `pm.potential()` 或有序转换来实现这一点。

有限混合模型的一个挑战是如何确定组分数量。一种解决方案是围绕估计的组分数量对多个模型进行比较，如果可能的话，应根据手头的问题域来指导该估计。另一种选择是尝试从数据中自动估计组分数量。为此引入了 狄利克雷 过程的概念，我们用它来考虑无限混合模型。狄利克雷 过程是 狄利克雷 分布的无限维版本，可以用它来建立非参数混合模型。

作为本章结束语，简要讨论了有多少模型可以解释为连续混合模型，如贝塔-二项式（用于抛硬币问题的模型）、负二项模型、学生 $t$ 分布，甚至分层模型。

## 6.6 习题

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505200439_80.webp)
