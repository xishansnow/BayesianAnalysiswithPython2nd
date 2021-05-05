 # 第4章 利用Logistic 回归 对结果进行分类

在上一章中，我们使用输入变量的线性组合来预测输出变量的平均值，其中假设了后者为高斯分布。在许多情况下都可以使用高斯分布，但对于其他许多情况，选择不同分布可能更明智；当我们用学生 $t$ 分布替换高斯分布时，已经看到了一个鲁棒的例子。在本章中，我们将看到更多使用高斯分布以外分布的例子。正如即将了解到的，有一个通用的主题或模式，可用于将线性模型推广到许多问题中。

本章将会讨论以下内容：

- 广义线性模型；
- Logistic 回归 和逆连结函数；
- 一元Logistic 回归 ；
- 多元Logistic 回归 ；
- softmax函数和多项Logistic 回归 ；
- 泊松回归
- 零膨胀泊松回归



## 4.1 广义线性回归

本章的核心思想相当简单：为预测输出变量的平均值，我们可以对输入变量的线性组合应用任意函数。
$$
\mu = f ( \alpha + X \beta )\tag{4.1}
$$
其中，$f$ 称作逆连结函数。为什么这里把 $f$ 称作逆连结函数而不是连结函数？原因是传统上人们认为此类函数是用来连结输出变量和线性模型的，不过在构建贝叶斯模型时，反过来思考可能更容易一些。因此，为避免疑惑，这里统称逆连结函数。前一章中所有线性模型其实都包含一个逆连结函数，不过书写时省略了，因为它其实是一个恒等函数（函数的返回值和输入值相同）。恒等函数在此没什么用，不过有助于让我们用更一般的形式思考不同模型。

希望使用逆链接函数的一种情况是在处理定类变量时（例如：颜色名称、性别、生物物种或政党/从属关系）。上述所有定类变量都没有被高斯学派很好地建模，因为高斯函数原则上适用于在实数集上取任何值的连续变量，而定类变量是离散的，只取几个值。如果改变数据分布，通常也需要改变对这些分布平均值的建模方式，例如，如果使用二项分布，就像在第1章和第2章中一样，我们将需要一个返回 [0，1] 区间平均值的线性模型。实现的一种方法是保留线性模型，但使用逆链接函数将输出限制在所需区间内。该技巧并不局限于离散变量，例如：我们可能希望对只能取正值的数据建模，因此希望将线性模型限制为返回分布平均值的正值，例如伽马或指数分布。

在继续之前，请注意：一些变量可以被编码为定量的，也可以编码成定性的。这必须根据您的问题上下文做出决定；例如：如果谈论颜色名称时，可以采用“红色”和“绿色”定类变量，或者使用 650 nm 和 510 nm 等连续变量。

## 4.2  用于分类的Logistic 回归

回归问题是关于在给定一个或多个输入变量的值的情况下，预测输出变量的连续值。于此不同，分类是在给定一些输入变量时，将离散值（代表离散类）赋给输出变量。在两种情况中，任务都是获得一个能够正确模拟输出与输入变量之间映射的模型。为做到此点，需要具有正确输出--输入变量对的样本。从机器学习视角看，回归和分类都是监督学习的实例。

我妈妈准备了一道名为 `sopa seca` 的美味菜肴，是以意大利面为基础的食谱，字面意思是干汤。虽然听起来可能是用词不当，甚至是自相矛盾，但当我们了解它是如何烹调时，这道菜的名字就完全有意义了。

 Logistic 回归也是如此，它主要用来解决分类问题。 Logistic 回归模型是线性回归模型的一种扩展，这也是其名称来源。其模型是将式 4. 1中的逆连接函数设定为Logistic函数而形成，Logistic函数为以下形式：
$$
\text{logistic}(z)=\frac{1} { 1 + e ^ { - z } } \tag{4.2}
$$

从分类角度看，Logistic函数最重要的特点是不论参数 $z$ 值为多少，其输出总是介于 0 到 1 之间。因此，该函数将整个实轴压缩到了区间 [0,1] 内。Logistic函数也称作 $S$ 型函数（sigmoid function），因为其形状看起来像 $S$ ，可以运行以下几行代码来看一下。

```python
z = np.linspace(-8, 8)
plt.plot(z, 1 / (1 + np.exp(-z)))
plt.xlabel('z')
plt.ylabel('logistic(z)')
```

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505115639_54.webp" style="zoom:67%;" />

### （1）Logistic模型

我们已经具备了将简单线性回归转化为简单 Logistic 回归的所有要素。先从简单问题开始，假设类别只有两类，比如正常邮件/垃圾邮件、安全/不安全、阴天/晴天、健康/生病等。首先对类别进行编码，假设变量 $y$ 只能有两个值 0 或 1  。这样描述后，就有点像抛硬币问题了。在抛硬币例子中用到了伯努利分布作为似然。此处的区别是：现在 $\theta$ 不再是从 $beta$ 分布中生成，而是由一个线性模型定义。该模型可以返回实轴上任意值，但伯努利分布的值限定在 [0,1] 区间内。因此使用了一个逆连结函数将线性模型的返回值映射到一个适合伯努利分布的区间内，从而将一个线性回归模型转换成了分类模型：

$$
\theta=\operatorname{logistic}(\alpha+x \beta) \\
 y=\operatorname{Bern}(\theta) \tag{4.3}
$$
请注意，与第3章中简单线性回归的主要区别在于：使用了伯努利分布而不是高斯分布，并且使用了Logistic函数而不是恒等函数。下面的 Kruschke 图显示了 Logistic 回归模型。注意与一元简单线性回归模型的区别是：此处用到了伯努利分布而不是高斯分布（或者 $t$ 分布），并使用 Logistic 函数生成区间 [0,1] 范围内的参数 $θ$ ，从而适于输入到伯努利分布中t。

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429224141_27.webp" style="zoom:50%;" />

### （2）鸢尾花数据集

我们将 Logistic 回归应用到鸢尾花数据集上。在构建模型之前先了解下该数据集。鸢尾花数据集是经典数据集，包含有 Setosa 、 Versicolour 和 Virginica  3个种类，这3个类别标签就是要预测的量，即因变量。其中每个种类包含 50 个数据，每个数据包含 4 种变量（或者称为特征），这 4 种变量就是待分析的自变量，分别是：花瓣长度、花瓣宽度、花萼长度和花萼宽度。花萼有点类似小叶，在花还是芽时包围着花，有保护作用。seaborn 中包含鸢尾花数据集，可以用如下代码将其导入成 Pandas 里的 DataFrame 格式：

```python
鸢尾花 = pd.read_csv('../data/鸢尾花.csv')
鸢尾花.head()
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429224345_c1.webp)

现在，可以使用 seaborn 中的 stripplot 函数绘制这三个物种与 stepal_length的关系图：

```python
sns.stripplot(x="species", y="sepal_length", data=鸢尾花, jitter=True)
```

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429224514_92.webp" style="zoom:67%;" />

上图中 $y$ 轴是连续的，而 $x$ 轴是离散的；图中在$x$ 轴上散布的点并没有什么实际意义，只是一个画图技巧而已，通过将 jitter 参数设为True，避免所有点都重叠在一条直线上。你可以尝试将 jitter 参数设为False。这里唯一重要的是 $x$ 轴的含义，分别代表 Setosa 、 Versicolour 和 Virginica  三个类别。还可以用 seaborn 中其他绘图函数（如 violinplot ）来画这些点，只需要一行代码就能完成。

此外可以用 pairplot 画出散点图矩阵，用该函数可以得到一个 4×4 网格（因为有4种特征）。网格是对称的，上三角和下三角表示同样的信息。由于对角线上的散点图其实是变量本身，因此用一个特征的 KDE 图代替了散点图。可以看到，每个子图中分别用 3 种颜色表示 3 种不同的类别标签，这与前面图中的表示一致。

```python
sns.pairplot(鸢尾花, hue='species', diag_kind='kde')
```

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429224656_78.webp" style="zoom:67%;" />

在深入学习之前，花点时间分析下前面这幅图，进一步熟悉这个数据集并了解变量与标签之间的关系。

### （3）将 Logistic 回归模型应用到鸢尾花数据集

我们先从一个简单的问题开始：用花萼长度这一特征（自变量）来区分 Setosa  和 Versicolour 两个种类。和前面一样，这里用 0 和 1 对类别变量进行编码，利用 Pandas 可以这么做：

```python
df = 鸢尾花.query(species == (' Setosa ', 'Versicolour '))
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length'
x_0 = df[x_n].values
x_c = x_0 - x_0.mean()
```

现在数据已经表示成了合适的格式，终于可以用 PyMC3 建模了。留意下面代码中的第一部分与线性回归模型的相似之处。此外留意两个确定变量：$\theta$ 和 `bd` 。  $\theta$ 是对变量 $\mu$ 应用Logistic 函数之后的值，`bd` 是一个有边界的值，用于确定分类结果，稍后会详细讨论。此外，除了像下面明确写出逻辑函数的完整形式外，还可以使用 PyMC3 中的  `pm.math.sigmoid`  函数，该函数是 `Theano` 中 `sigmoid` 函数的别名。

```python
with pm.Model() as model_0:
    α = pm.Normal('α', mu=0, sd=10)
    β = pm.Normal('β', mu=0, sd=10)
    μ = α + pm.math.dot(x_c, β)
    θ = pm.Deterministic('θ', pm.math.sigmoid(μ))
    bd = pm.Deterministic('bd', -α/β)
    yl = pm.Bernoulli('yl', p=θ, observed=y_0)
    trace_0 = pm.sample(1000)
```

为节省页数，同时避免对同一类型图件反复出现感到厌烦，将省略迹图和其他类似的摘要图，但鼓励您制作自己的图和摘要 ，以进一步探索本书中的例子。我们将直接跳到如何生成下图，这是一个数据曲线图，以及拟合的 `sigmoid` 曲线和决策边界：

```python
theta = trace_0['θ'].mean(axis=0)
idx = np.argsort(x_c)
plt.plot(x_c[idx], theta[idx], color='C2', lw=3)
plt.vlines(trace_0['bd'].mean(), 0, 1, color='k')
bd_hpd = az.hpd(trace_0['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='k', alpha=0.5)
plt.scatter(x_c, np.random.normal(y_0, 0.02),
            marker='.', color=[f'C{x}' for x in y_0])
az.plot_hpd(x_c, trace_0['θ'], color='C2')
plt.xlabel(x_n)
plt.ylabel('θ', rotation=0)
# use original scale for xticks
locs, _ = plt.xticks()
plt.xticks(locs, np.round(locs + x_0.mean(), 1))
```

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429225650_74.webp" style="zoom:67%;" />

前面这张图表示了花萼长度与花种类（ Setosa  = 0, Versicolour=1）之间的关系。绿色的 $S$ 型曲线表示 $\theta$ 的均值，这条线可以解释为：在知道花萼长度的情况下花的种类是 Versicolour 的概率，即半透明的绿色区间是 94% HPD 区间。边界判定用一条(黑色)垂直线表示，其94%的HPD为半透明带。根据边界判定，左侧的值(在本例中为萼片长度)对应于 类 0 （ Setosa ），右侧的值对应于类 1 （ Versicolour）。

决策边界由 $y=0.5$ 时的 $x$ 取值定义，可以证明其结果为 $-\frac{\alpha}{\beta}$ ，推导过程如下：

根据模型的定义，我们有如下关系：
$$
\theta=\operatorname{logistic}(\alpha+x \beta)\tag{4.4}
$$
根据逻辑函数的定义，当 $\theta=0.5$ 时，对应的输入为 0，则有：
$$
0.5=\operatorname{logistic}\left(\alpha+x_{i} \beta\right) \Leftrightarrow 0=\alpha+x_{i} \beta\tag{4.5}
$$
移项后可以得出，当 $ \theta=0.5 $ 时，对应有：
$$
x_{i}=-\frac{\alpha}{\beta}\tag{4.6}
$$
值得一提的是：

- 一般来说，$\theta$ 的价值是 $p(y=1|x)$ 。从这点上来说， Logistic 回归是一种真正的回归；关键细节是，在给定特征的线性组合的情况下，回归的是数据点属于类别 1 的概率。
- 我们正在模拟一个二分变量的均值，即 [0-1] 区间内的一个数字。然后引入一条规则，将这种概率转化为二分类赋值。在这种情况下， 如果 $p(y=1)>=0.5$ ，我们分配类 1 ，否则分配类 0 。
- 这里选取的 0.5 并不是什么特殊值，你完全可以选其他 0 到 1 之间的值。只有当我们认为将标签 0（ Setosa ）错误地标为标签1（Versicolour）时的代价，与反过来将标签 1（Versicolour ）错误地标为标签 0（ Setosa ）时的代价相同时，选取 0.5 作为决策边界才是可行的。不过大多数情况下，分类出错的代价并不是对称的。



## 4.3 多元 Logistic 回归用于分类

与多变量线性回归类似，多变量 Logistic 回归 使用多个自变量。这里举例将花萼长度与花萼宽度结合在一起，注意需要对数据做一些预处理。

```
df=鸢尾花.query(species==(\' Setosa \',\'Versicolour \'))\
y_1=pd.Categorical(df\[\'species\'\]).codes
x_n=\[\'sepal_length\',\'sepal_width\'\]
x_1=df\[x_n\].values
```

### 4.3.1 决策边界

如果你对如何推导决策边界不感兴趣的话，可以略过这个部分直接跳到模型实现部分。 根据模型，我们有：

$$
\theta=\operatorname{logistic}\left(\alpha+\beta_{1} x_{1}+\beta_{2} x_{2}\right)\tag{4.7}
$$

从Logistic函数的定义出发，当Logistic回归的自变量为零时，有 $\theta=0.5 $ 。对应：
$$
0.5=\operatorname{logistic}\left(\alpha+\beta_{1} x_{1}+\beta_{2} x_{2}\right) \Leftrightarrow 0=\alpha+\beta_{1} x_{1}+\beta_{2} x_{2}\tag{4.8}
$$
通过移项，我们找到 $\theta=0.5$ 时 $x_2$ 的值：
$$
x_{2}=-\frac{\alpha}{\beta_{2}}+\left(-\frac{\beta_{1}}{\beta_{2}} x_{1}\right)\tag{4.9}
$$
这个决策边界的表达式与直线的表达式在数学形式上是一样的，其中第1项表示截距，第2项表示斜率，这里的括号只是为了表达上更清晰，如果你愿意的话完全可以去掉。为什么决策边界是直线呢？想想看，如果我们有一个特征，还有一维的数据，可以用一个点将数据分成两组；如果有两个特征，也就有一个2维的数据空间，从而我们可以用一条直线来对其分割；对于3维的情况，边界是一个平面，对于更高的维度，我们对应有一个超平面。事实上，从概念上讲，超平面可以大致定义为n维空间中n-1维的子空间，因此我们总是可以将决策边界称为超平面。

### 4.3.2 模型实现

如果要用PyMC3写出多元逻辑回归模型，可以借助其向量化表示的优势，只需要对前面的单参数逻辑回归模型做一些简单的修改即可。

```python
with pm.Model() as model_1:
    α = pm.Normal('α', mu=0, sd=10)
    β = pm.Normal('β', mu=0, sd=2, shape=len(x_n))
    μ = α + pm.math.dot(x_1, β)
    θ = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-μ)))
    bd = pm.Deterministic('bd', -α/β[1] - β[0]/β[1] * x_1[:,0])
    yl = pm.Bernoulli('yl', p=θ, observed=y_1)
    trace_1 = pm.sample(2000)
```

正如对单个预测变量所做的那样，我们将绘制数据和决策边界：

```python
idx = np.argsort(x_1[:,0])
bd = trace_1['bd'].mean(0)[idx]
plt.scatter(x_1[:,0], x_1[:,1], c=[f'C{x}' for x in y_0])
plt.plot(x_1[:,0][idx], bd, color='k');
az.plot_hpd(x_1[:,0], trace_1['bd'], color='k')
plt.xlabel(x_n[0])
plt.ylabel(x_n[1])
```

我们已经看到过的，决策边界现在是一条直线，不要被95%HPD区间的曲线给误导了。图中半透明的曲线是由于在中间部分多条直线绕中心区域旋转的结果（大致围绕 $x$ 的平均值 和 $y$ 的平均值）。

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505135608_4a.webp" alt="image-20210429232402489" style="zoom:67%;" />

### 4.3.3 关于Logistic回归系数的解释

在解释Logistic回归的 $\beta$ 系数时，我们必须小心。解释并不像我们在第3章“线性回归建模”中讨论的线性模型那样简单。使用Logistic逆连接函数引入了我们必须考虑的非线性。如果为 $\beta$ 为正，则增加 $x$ 会增加一些 $p(y=1)$ 的量，但该量值不是的 $x$ 线性函数；相反，它非线性地依赖于的 $x$ 值。我们可以在图中直观地看到这一事实；我们看到的不是具有恒定斜率的直线，而是具有随 $x$ 变化而不断调整斜率的 $S$ 形直线。一点代数知识可以让我们更深入地了解 $p(y=1)$ 是如何随 $\beta$  变化的。基础模型是：

$$
\theta=\operatorname{logistic}(\alpha+X \beta)\tag{4.11}
$$
逻辑斯蒂的逆函数是 logit函数，它是：

$$
\operatorname{logit}(z)=\log \left(\frac{z}{1-z}\right)\tag{4.12}
$$
因此，如果我们取本节中的第一个方程式，并将logit函数应用于这两个项，我们会得到这个方程式：
$$
\text{logit}(\theta)=\alpha+X \beta\tag{4.13}
$$
或等价的：
$$
\log \left(\frac{\theta}{1-\theta}\right)=\alpha+X \beta\tag{4.14}
$$
记住模型中的 $ \theta $ 是 $ p(y=1) $ ：
$$
\log \left(\frac{p(y=1)}{1-p(y=1)}\right)=\alpha+X \beta\tag{4.15}
$$

$ \frac{p(y=1)}{1-p(y=1)} $ 的量值即为赔率。

成功的几率被定义为成功概率与不成功概率之比。虽然掷一个公平骰子得到 2 的概率是 1/6 ，但同一事件的几率是 $\frac{1/6}{5/6}=0.2$ 或1个有利事件对5个不利事件。赌徒经常使用赔率，主要是因为在考虑正确的下注方式时，赔率提供了一种比原始概率更直观的工具。

在Logistic回归中，系数 $\beta$ 编码了对数赔率单位随变量$x$ 单位增加而增加的情况。

从概率到赔率的转换是一个单调的转换，这意味着赔率随着概率的增加而增加，反之亦然。概率被限制在 [0，1] 区间，而赔率则在 [0，∞) 区间内。对数是另一个单调变换，对数赔率在 (-∞，∞) 区间内。下图显示了概率与赔率和对数赔率的关系：

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429233843_e5.webp" style="zoom:67%;" />

```python
probability = np.linspace(0.01, 1, 100)
odds = probability / (1 - probability)
_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(probability, odds, 'C0')
ax2.plot(probability, np.log(odds), 'C1')
ax1.set_xlabel('probability')
ax1.set_ylabel('odds', color='C0')
ax2.set_ylabel('log-odds', color='C1')
ax1.grid(False)
ax2.grid(False)
```

因此，汇总数据提供的系数的值是对数赔率标度的：

```
df = az.summary(trace_1, var_names=varnames)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210429234020_01.webp)

理解模型的一种非常实用的方法是更改参数，看看会发生什么。在下面的代码块中，我们正在计算支持杂色AS的对数赔率 $\text{log_odds_versicolor_i}=\alpha+\beta_1x_1+\beta_2x_2$ ，然后用Logistic函数计算杂色的概率。然后，我们通过固定 $x_2$并 $x_1$ 增加1来重复计算：

```python
x_1 = 4.5  # sepal_length
x_2 = 3   # sepal_width
log_odds_versicolor_i = (df['mean'] * [1, x_1, x_2]).sum()
probability_versicolor_i = logistic(log_odds_versicolor_i)
log_odds_versicolor_f = (df['mean'] * [1, x_1 + 1, x_2]).sum()
probability_versicolor_f = logistic(log_odds_versicolor_f)
log_odds_versicolor_f - log_odds_versicolor_i, probability_versicolor_f -
probability_versicolor_i
```

如果运行代码，您会发现 log-odds 的增加值约为 4.66，这正是 $\beta_0$ (请查看 $trace\_1$ 的摘要)的值。这与我们之前的发现是一致的，即 $\beta$ 编码了对数赔率单位随变量 $x$ 单位增加而增加的情况。概率的增加约为 0.70 。



### 4.3.4 处理相关变量

我们从第3章“线性回归建模”中了解到，当我们处理(高度)相关变量时，有一些棘手的事情等着我们。相关变量转化为能够解释数据的更广泛的系数组合，或者从互补的角度来看，相关数据对模型的约束能力较小。当类变得完全可分离时，也就是说，在我们的模型中给定变量的线性组合的情况下，类之间没有重叠时，也会出现类似的问题。

使用鸢尾花数据集，可以尝试运行model_1，但这一次使用 petal_width 和 petal_length 变量。您会发现 $\beta$ 系数比以前更宽了，而且图中94%的HPD黑带也更宽了：

```
corr = iris[iris['species'] != 'virginica'].corr()
mask = np.tri(*corr.shape).T
sns.heatmap(corr.abs(), mask=mask, annot=True, cmap='viridis')
```

下图是热力图，显示了对于第一个示例中使用的 sepal_length 和 sepal_width 变量，没有第二个示例中使用的 petal_width 和 petal_length 变量之间相关性高：

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505140404_75.webp" style="zoom:67%;" />

为了生成上图，我们使用了一个掩码来删除热图的上三角形和对角线元素，因为在给定下三角形的情况下，这些元素没有任何信息。还要注意，我们已经绘制了相关性的绝对值，因为此时我们不关心变量之间相关性的符号，只关心它的强度。

一个解决办法是删除一个（或者多个）相关变量，不过这个办法可能有时候并不适用。另外一个办法是给先验增加更多的信息，如果我们掌握了一些有用的信息，可以使用一些携带信息的先验，或者更一般地，可以使用一些弱信息的先验。[Andrew Gelman和Stan的开发团队](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations)建议将所有非二进制变量都缩放至均值为 0，而后使用 学生 $t$ 分布作为先验：

$ \beta \sim \operatorname{Student} T(0, \nu, s d)\tag{4.10} $

这里 $sd$ 的取值可以根据期望的尺度引入弱信息，正态参数 $v$ 的值为3到7附近。该先验的含义是：我们期望系数比较小，同时引入了重尾，从而得到一个比高斯分布更鲁棒的模型。如果你忘记了的话，可以回忆一下前两章中的内容。



### 4.3.5 处理类别不平衡数据

鸢尾花数据集的一个优点是：其中不同类别的样本量是均衡的， Setosas、Versicolours 和 Virginicas各有50个。鸢尾花数据集的流行归功于 Fisher，当然，Fisher 还让另一个东西也流行了--- $p$ 值。实际使用中许多数据集中的类别都是不平衡的，也就是说，其中一类数据的数量要远多于其他类别的数据。当这种情况发生时，Logistic 回归会遇到一些问题，相比数据平衡时的情况，Logistic 回归得到的边界没有那么准确了。

现在我们看个实际的例子，这里我们随机从 Setosa 类别中去掉一些数据点。

```
df = iris.query("species == ('setosa', 'versicolor')")
df = df[45:]
y_3 = pd.Categorical(df['species']).codes
x_n = ['sepal_length', 'sepal_width']
x_3 = df[x_n].values
```

和之前一样，你可以在自己电脑上运行多元Logistic 回归 ，这里我直接将结果表示出来。

```
with pm.Model() as model_3:
    α = pm.Normal('α', mu=0, sd=10)
    β = pm.Normal('β', mu=0, sd=2, shape=len(x_n))
    μ = α + pm.math.dot(x_3, β)
    θ = 1 / (1 + pm.math.exp(-μ))
    bd = pm.Deterministic('bd', -α/β[1] - β[0]/β[1] * x_3[:,0])
    yl = pm.Bernoulli('yl', p=θ, observed=y_3)
    trace_3 = pm.sample(1000)
```

可以看到，决策边界向样本量更少的类别偏移了，而且不确定性也比以前更大了。这是Logistic 回归在处理不均衡数据时的常见表现。在一些数据中，类别之间的间隔可能不像这个例子中这么完美，此时用Logistic 回归分类得到的结果中类别重叠的现象更严重。不过你可能觉得不确定性变得更大有可能是因为数据总量变少了，而不只是因为 Setosa 类别的数据相比 Versicolour 更少。这是有可能的，你可以完成练习部分的第2题之后，亲自验证为什么不确定性变大的原因是数据不平衡。

```
idx = np.argsort(x_3[:,0])
bd = trace_3['bd'].mean(0)[idx]
plt.scatter(x_3[:,0], x_3[:,1], c= [f'C{x}' for x in y_3])
plt.plot(x_3[:,0][idx], bd, color='k')
az.plot_hpd(x_3[:,0], trace_3['bd'], color='k')
plt.xlabel(x_n[0])
plt.ylabel(x_n[1])
```

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505141300_ab.webp" style="zoom:67%;" />



一个显而易见的解决方案是，对数据集中的每一类都获取几乎相同数量的样本，如果你自己收集或者生成数据的话一定要记住这点。如果你并不能控制数据集，那么在对类别不平衡的数据进行解释时可要当心了，你可以通过检查模型的不确定性以及运行后验预测检查来确定模型是否对你有用。另外一种做法是给数据加入更多的先验信息，如果可能的话，可以运行本章剩余部分提到的一些其他模型。

### 4.3.6 Softmax 回归

现在我们知道了如何处理二分类问题，接下来将我们所学的内容推广到多分类问题。一种做法是使用多项 Logistic 回归，该模型也被称作 softmax 回归，原因是这里使用的是softmax函数而非 Logistic 函数，softmax函数的形式如下：

$$
\operatorname{softmax}_{i}(\mu)=\frac{\exp \left(\mu_{i}\right)}{\sum \exp \left(\mu_{k}\right)}\tag{4.16}
$$


要计算向量 $μ$ 中第 $i$ 个元素对应的 softmax 输出，需要将该元素的指数除以向量 $μ$ 中每个元素的指数之和。softmax函数保证了输出值为正数而且和为 1。当 $k=2$ 时，softmax函数就变成了逻辑函数。另外，softmax 函数与统计学中的玻尔兹曼分布形式是一样的，玻尔兹曼分布也是物理学中用来描述分子系统中概率分布的一个强大分支。在玻尔兹曼分布中（某些领域中的softmax）有一个称为温度的参数（*T*），在数学形式上前面式子中的 $μ$ 变成了 $μ/T$ ，当 $T→∞$ 时，概率分布变得非常均匀，因而所有状态都是等可能的；当 $T→0$ 时，只有最可能的状态会输出，因而 softmax 表现得就像一个 max函数，这也是其名字来源。

softmax回归模型与Logistic 回归 模型的另一个区别是：伯努利分布换成了类别分布。类别分布其实是伯努利分布推广到两个以上输出时的一般形式。此外，伯努利分布（抛一次硬币）是二项分布（抛多次硬币）的特殊情况，类似地，类别分布（掷一次骰子）是多项分布（掷*N*次骰子）的特殊情况。

这里继续使用鸢尾花数据集，不过这次用到其中的3个类别标签（ Setosa 、Versicolour 及 Virginica ）和4个特征（花萼长度、花萼宽度、花瓣长度及花瓣宽度），同时对数据进行标准化处理（也还可以做中心化处理），这样采样效率更高。

```
iris = sns.load_dataset('iris')
y_s = pd.Categorical(iris['species']).codes
x_n = iris.columns[:-1]
x_s = iris[x_n].values
x_s = (x_s - x_s.mean(axis=0)) / x_s.std(axis=0)
```

从PyMC3的代码可以看出，Logistic 回归模型与softmax模型之间的变化很小，留意 $\alpha$ 系数和 $\beta$ 系数的形状。这段代码中用到了 Theano 中的softmax函数，根据 PyMC3 开发者的惯例，按 import theano.tensor as tt 这种方式导入的Theano。

```
with pm.Model() as model_s:
    α = pm.Normal('α', mu=0, sd=5, shape=3)
    β = pm.Normal('β', mu=0, sd=5, shape=(4,3))
    μ = pm.Deterministic('μ', α + pm.math.dot(x_s, β))
    θ = tt.nnet.softmax(μ)
    yl = pm.Categorical('yl', p=θ, observed=y_s)
    trace_s = pm.sample(2000)
```

那么我们的模型表现如何呢？可以根据准确预测的样本个数来判断。下面的代码中使用了参数的均值来计算每个点分别属于3个类别的概率值，然后使用argmax函数求出概率最大的类别作为结果，最后将结果与观测值进行比较。

```
data_pred = trace_s['μ'].mean(0)
y_pred = [np.exp(point)/np.sum(np.exp(point), axis=0)
 for point in data_pred]
f'{np.sum(y_s == np.argmax(y_pred, axis=1)) / len(y_s):.2f}'
```

分类结果显示准确率约为98%，也就是说，只错分了3个样本。不过，真正要评估模型的效果需要使用模型没有见过的数据，否则，可能会高估了模型对其他数据的泛化能力。下一章我们会详细讨论这个主题，目前暂且把它当做自动一致性检查，证明我们的模型运行正常。

也许你已经注意到了，后验（或者更准确地说，每个参数的边缘分布）看起来分布得很宽；事实上，它们与先验分布得一样宽。尽管我们能做出正确的预测，但这看起来并不令人满意。在前面的线性Logistic 回归问题中，对于有相关性的数据或者可以完美分割的数据，我们也遇到过类似不可识别的问题。在这个例子中，后验分布较广是因为受到了所有概率之和为1的限制。在这种情况下，我们用到的参数个数比实际定义模型所需要的参数个数更多。简单来说就是，假如10个数的和为1，你只需要知道9个数就可以了，剩下的1个数可以用1减去这9个数之和算出来。解决这个问题的办法是将额外的参数固定为某个值（比如0）。下面的代码展示了如何用PyMC3来实现。

```
with pm.Model() as model_sf:
    α = pm.Normal('α', mu=0, sd=2, shape=2)
    β = pm.Normal('β', mu=0, sd=2, shape=(4,2))
    α_f = tt.concatenate([[0] ,α])
    β_f = tt.concatenate([np.zeros((4,1)) , β], axis=1)
    μ = α_f + pm.math.dot(x_s, β_f)
    θ = tt.nnet.softmax(μ)
    yl = pm.Categorical('yl', p=θ, observed=y_s)
    trace_sf = pm.sample(1000)
```

### 4.3.7 判别模型和生成模型

目前为止，我们已经讨论了逻辑回归及其扩展，所有这些情况都是直接计算 $p(y | x)$ ，也就是说，在知道 $x$ 的条件下，计算出类别 $y$ 的概率值。换句话说，我们所做的是直接根据自变量到因变量之间的关系进行建模，然后用一个阈值对得到的（连续的）概率值进行评判，从而得到分类结果。

上面这种方法不是唯一的，另一种方法是先对 $p(x | y)$ 建模，即依据不同类别计算特征的分布，然后再进行分类。这类模型称为生成式分类器，因为我们得到的模型可以从每个类别中生成采样。与此相反，Logistic 回归属于判别式分类器，因为它只能判断一个样本是不是属于某一类别，并不能从每个类别中生成样本。这里我们不打算深入生成式分类器模型，不过可以通过一个例子来说明这类模型用于分类的核心思想。我们只使用两个类别和一个特征，与本章的第一个例子用到的数据一样。

下面的代码用PyMC3实现了一个生成式分类器，从代码中可以看出，现在决策边界变成了高斯分布期望的估计值的均值，当分布是正态分布且标准差相同时，这个决策边界是正确的。这些假设是由一种称作线性判别分析（Linear Discriminant Analysis，LDA）的模型做出的，尽管名字上是判别式分析，不过LDA模型其实是生成式的。

```
with pm.Model() as lda:
    μ = pm.Normal('μ', mu=0, sd=10, shape=2)
    σ = pm.HalfNormal('σ', 10)
    setosa = pm.Normal('setosa', mu=μ[0], sd=σ, observed=x_0[:50])
    versicolor = pm.Normal('versicolor', mu=μ[1], sd=σ,
                           observed=x_0[50:])
    bd = pm.Deterministic('bd', (μ[0] + μ[1]) / 2)
    trace_lda = pm.sample(1000)
```

下面再将 setosa=0 和 versicolor=1 两个类别与花萼长度的关系画出来，一同画出来的还有一条红色的决策边界以及对应的94%HPD区间。

```
plt.axvline(trace_lda['bd'].mean(), ymax=1, color='C1')
bd_hpd = az.hpd(trace_lda['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='C1', alpha=0.5)
plt.plot(x_0, np.random.normal(y_0, 0.02), '.', color='k')
plt.ylabel('θ', rotation=0)
plt.xlabel('sepal_length')
```

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505143106_66.webp" style="zoom:67%;" />

打印出模型的摘要，对决策边界进行检查。

```
az.summary(trace_lda)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505143145_a2.webp)

可以看到LDA模型得到了与逻辑回归类似的结果。线性判别式模型可以使用多元高斯分布对类别建模，从而将其扩
展到超过一个特征的情况。此外，还可以对不同类别的数据共享同一个方差（或者是同一个方差矩阵，对于超过一个特征的情况来说）的假设进行放松。这样便得到了称作二次判别分析（Quadratic Linear Discriminant，QDA）的模型，此时决策边界不再是线性的，而是二次的。

通常，当特征基本符合高斯分布时，LDA或QDA的效果要比逻辑回归更好，如果假设不成立，逻辑回归的效果要更好一些。使用判别式模型分类的一个好处是：在模型中融合先验更容易（或者说更自然）；比如我们可以将数据均值和方差的信息融入到模型中去。

需要注意：LDA和QDA的决策边界是封闭式的，对于两个类别和一个特征的情况，应用LDA的时候只需要分别计算出每个类别分布的均值，然后求二者的均值就得到了决策边界。在前面的模型中，采用了更贝叶斯的一种方式；我们估计出两个高斯分布的参数，然后将这些估计融入公式中，不过公式是怎么来的呢？这里不深入更细节的内容，只需要知道要得到该公式，我们需要假设数据是符合高斯分布的，因此LDA只有在数据分布接近高斯分布的时候更有效。显然，在某些问题中，当我们想对正态性的假设放松的时候（比如t分布或者多元t分布等），就不能再使用LDA（或QDA）了，不过我们仍然可以用PyMC3从数值上计算出决策边界。

## 4.4 泊松回归用于计数变量

另一个非常流行的广义线性模型是泊松回归。此模型假设数据是按泊松分布的。泊松分布对于很多计数场景非常有用，例如：放射性原子核衰变、每对夫妇的孩子数量，推特上关注者的数量等。其共同点是：通常使用离散的非负数 {0，1，2，3，...} 对事件建模。此类变量被称为**计数变量**。

>注：统计学中通常将变量分为四类：连续变量、定类变量、定序变量和计数变量。

### 4.4.1 泊松分布

想象一下，我们正在计算每小时通过一条大道的红色汽车的数量。我们可以用泊松分布来描述这些数据。泊松分布通常用于描述在固定的时间/空间间隔内发生给定数量的事件的概率。因此，泊松分布假定事件以固定的时间和/或空间间隔彼此独立地发生。这种离散分布仅使用一个值(速率，通常也用希腊字母表示)进行参数化。对应于分布的平均值，也对应于分布的方差。泊松分布的概率质量函数如下：
$$
f(x \mid \mu)=\frac{e^{-\mu} \mu^{x}}{x !}\tag{4.17}
$$
方程式描述如下：

- $\mu$  是每单位事件（或空间）事件发生的平均数量
-  $ x $ 是正整数 $ \{0,1,2, \ldots\} 
-  $ x ! $  是 $ x $ 的阶乘， $ x !=x \times(x-1) \times(x-2) \times \cdots \times 2 \times 1 $ 。

在下图中，可以看到针对不同 $\mu$ 值的泊松分布族的示例：



```
mu_params = [0.5, 1.5, 3, 8]
x = np.arange(0, max(mu_params) * 3)
for mu in mu_params:
    y = stats.poisson(mu).pmf(x)
    plt.plot(x, y, 'o-', label=f'μ = {mu:3.1f}')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
```

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505125403_52.webp" style="zoom:67%;" />



注意，$\mu$ 可以是浮点数，但分布的输出始终是整数。在图中，点表示分布的值，而连续的线条是帮助我们掌握分布形状的辅助工具。所以请记住，泊松分布是离散分布。

泊松分布可以看作是当试验次数 $n$ 很多、但成功概率 $p$ 很低时的二项分布。在不涉及太多数学细节情况下，让我们试着澄清一下这一说法。按照汽车的例子，我们可以确认要么看到了红色的汽车，要么没有看到，因此可以使用二项分布。在这种情况下，我们有：
$$
x \sim \operatorname{Bin}(n, p)\tag{4.18}
$$
那么，二项分布的平均值是：
$$
\mathbf{E}[x]=n p\tag{4.19}
$$
方差为：
$$
\mathbf{V}[x]=n p(1-p)\tag{4.20}
$$
但即使在一条非常繁忙的大道上，看到一辆红色汽车的机会与一个城市的汽车总数相比也是非常小的，因此我们必须：
$$
n>>p \Rightarrow n p \simeq n p(1-p)\tag{4.21}
$$
因此，可以做如下近似：

$$
\mathbf{V}[x]=np\tag{4.22}
$$

现在，均值和方差由相同的数字表示，可以很有把握地指出，变量是按泊松分布的：
$$
x \sim \operatorname{Poisson}(\mu=n p) \tag{4.23}
$$


### 4.4.2 零膨胀泊松模型

当数东西的时候，一种选择是不数东西，也就是得零分。数字零通常出现的原因有很多；我们得到零是因为我们数的是红色汽车，而一辆红色汽车没有通过大道，或者是因为我们错过了它。因此，如果使用泊松分布，例如，当执行后验预测检查时，我们会注意到，与数据相比，模型生成的零较少。我们怎么解决这个问题呢？我们可以尝试解决我们的模型预测的零比观察到的少的确切原因，并将这一因素包括在模型中。但是，就像通常的情况一样，只要假设我们有两个进程的混合就足够了，也更容易实现我们的目的：

- 由概率为 $\psi$ 的泊松分布建模
-  概率为 $1-\psi$ 额外 0 值

这就是所谓的零膨胀泊松(ZIP)模型。在一些文本中，你会发现它代表了额外的零和泊松的概率。这没什么大不了的，具体的例子就是注意哪一个是哪一个。

基本上，ZIP分布是：

$$
p\left(y_{j}=0\right)=1-\psi+(\psi) e^{-\mu} \tag{4.24}
$$

$$
p\left(y_{j}=k_{i}\right)=\psi \frac{\mu^{x_{i}} e^{-\mu}}{x_{i} !} \tag{4.25}
$$

其中 $1-\psi$ 是额外零的概率。

为了举例说明ZIP分布的使用，让我们创建几个合成数据点：

```
n = 100
θ_real = 2.5
ψ = 0.1
# Simulate some data
counts = np.array([(np.random.random() > (1-ψ)) *
                   np.random.poisson(θ_real) for i in range(n)])
```

可以很容易地将公式 4.24 和 4.25 实现到 PyMC3 模型中。但可以做一些更简单的事情：可以使用来自 PyMC3 的内置ZIP 分布：

```
with pm.Model() as ZIP:
    ψ = pm.Beta('ψ', 1, 1)
    θ = pm.Gamma('θ', 2, 0.1)
    y = pm.ZeroInflatedPoisson('y', ψ, θ,
                               observed=counts)
    trace = pm.sample(1000)
```

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505130946_25.webp)



### 4.4.3 泊松回归和ZIP回归

ZIP模型可能看起来有点乏味，但有时需要估计简单分布，比如这个分布，或者其他泊松分布或高斯分布等。无论如何，可以使用泊松分布或ZIP分布作为线性模型的一部分。正如在 Logistic 回归和 Softmax 回归中那样，可以使用反向连接函数将线性模型结果转换为一个变量，该变量位于正态分布以外的其他分布范围内。按同样思路，现在可以执行回归分析，其中输出变量是使用泊松分布或ZIP分布的计数变量。可以使用指数函数作为反连接函数，以确保线性模型返回的值始终为正值：
$$
\theta=e^{(\alpha+X \beta)} \tag{4.26}
$$
为了举例说明 ZIP 回归模型的实现，将使用来自数字研究和教育研究所的[数据集](http://www.ats.ucla.edu/stat/data)。我们有250组游客参观公园。以下是每组的部分数据：

- 他们捕到的鱼的数量( count )
- 有几个孩子在一起( child) 
- 是否带了露营车到公园( camper )

使用这些数据，我们将建立一个模型，根据孩子和露营车变量来预测捕捞的鱼的数量。我们可以使用 Pandas 来加载数据：

```
fish_data = pd.read_csv('../data/fish.csv')
```

我把使用绘图和/或 Pandas 函数探索数据集的工作留给您作为练习。目前，我们将继续实现 PyMC3 的 ZIP 回归模型：

```
with pm.Model() as ZIP_reg:
    ψ = pm.Beta('ψ', 1, 1)
    α = pm.Normal('α', 0, 10)
    β = pm.Normal('β', 0, 10, shape=2)
    θ = pm.math.exp(α + β[0] * fish_data['child'] + β[1] * fish_data['camper'])
    yl = pm.ZeroInflatedPoisson('yl', ψ, θ, observed=fish_data['count'])
    trace_ZIP_reg = pm.sample(1000)
```

camper 是一个二进制变量，值为 0 表示非 camper，1 表示 camper。指示属性存在/不存在的变量通常被表示为哑变量或指示变量。请注意，当 camper 取值为 0 时，所涉及的$\beta_1$ 项也将为 0 ，模型将退化为具有单个自变量的回归。

为了更好地理解我们的推断结果，让我们来做一个图：



```
children = [0, 1, 2, 3, 4]
fish_count_pred_0 = []
fish_count_pred_1 = []
for n in children:
    without_camper = trace_ZIP_reg['α'] + trace_ZIP_reg['β'][:,0] * n
    with_camper = without_camper + trace_ZIP_reg['β'][:,1]
    fish_count_pred_0.append(np.exp(without_camper))
    fish_count_pred_1.append(np.exp(with_camper))
plt.plot(children, fish_count_pred_0, 'C0.', alpha=0.01)
plt.plot(children, fish_count_pred_1, 'C1.', alpha=0.01)
plt.xticks(children);
plt.xlabel('Number of children')
plt.ylabel('Fish caught')
plt.plot([], 'C0o', label='without camper')
plt.plot([], 'C1o', label='with camper')
plt.legend()
```

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505132039_73.webp" style="zoom:67%;" />

从图中可以看出，孩子越多，捕捞的鱼越少。此外，使用露营车旅行的人通常会捕到更多鱼。如果你检查孩子和露营者的系数 $\beta$ ，你会发现：

- 每增加一个孩子，预期捕捞到的鱼数量下降约 0.4
- 如果有露营车，则捕获的鱼的预期数量会增加约 2

我们通过分别取 $\beta_1$ 和 $\beta_2$ 系数的指数得出这些结果。



## 4.5 鲁棒Logistic 回归 

我们刚刚了解了如何修复多余的零，而无需直接对生成它们的因素进行建模。Kruschke 建议的类似方法可以用于执行更健壮版本的 Logistic 回归。请记住，在 Logistic 回归中，我们将数据建模为二项式，即 0 和 1 。因此，可能会发现具有不寻常的 0 和/或 1 的数据集。以我们已经看到的 鸢尾花 数据集为例，但添加了一些入侵者：

```
鸢尾花 = sns.load_dataset("鸢尾花")
df = 鸢尾花.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n = 'sepal_length'
x_0 = df[x_n].values
y_0 = np.concatenate((y_0, np.ones(6, dtype=int)))
x_0 = np.concatenate((x_0, [4.2, 4.5, 4.0, 4.3, 4.2, 4.4]))
x_c = x_0 - x_0.mean()
plt.plot(x_c, y_0, 'o', color='k');
```

在这里，有一些萼片长度非常短的 Versicolor(1s)。我们可以用混合模型来解决这个问题。我们会说，输出变量来自随机猜测的概率为 $\pi$ ，或者来自Logistic回归模型的概率为 $1-\pi$ 。从数学上讲，我们有：
$$
p=\pi 0.5+(1-\pi) \operatorname{logistic}(\alpha+X \beta) \tag{4.26}
$$
当 $\pi=1$ 时，我们得到 $p=0.5$ ，并且对于 $\pi=0$ ，我们恢复了 Logistic 回归的表达式。实施此模型是对本章第一个模型的直接修改：

```
with pm.Model() as model_rlg:
    α = pm.Normal('α', mu=0, sd=10)
    β = pm.Normal('β', mu=0, sd=10)
    μ = α + x_c *  β
    θ = pm.Deterministic('θ', pm.math.sigmoid(μ))
    bd = pm.Deterministic('bd', -α/β)
    π = pm.Beta('π', 1., 1.)
    p = π * 0.5 + (1 - π) * θ
    yl = pm.Bernoulli('yl', p=p, observed=y_0)
    trace_rlg = pm.sample(1000)
```

如果将这些结果与 model_0 （本章第一个模型）的结果进行比较，会发现得到的边界大致相同。通过将两图比较，我们可以看到：

<img src="https://gitee.com/XiShanSnow/imagebed/raw/master/images/articles/spatialPresent_20210505133026_b7.webp" style="zoom:50%;" />

您可能还需要计算 model_0 和 model_rlg 的总和，以根据每个模型比较边界的值



## 4.6 GLM模块

正如我们在本章开头所讨论的，线性模型是非常有用的统计工具。我们在本章中看到的扩展使它们成为更通用的工具。因此，PyMC3 包含一个简化线性模型创建的模块：广义线性模型(Generalized Liner Model，GLM)模块。例如，简单的线性回归将如下所示：

```
with pm.Model() as model:
    glm.glm('y ~ x', data)
    trace = sample(2000)
```

前面代码的第二行负责添加截距和斜率的先验。默认情况下，为截距指定平坦的先验，为斜率指定 $N(0,1 \times 10^6)$ 先验。注意默认模型的最大后验概率(MAP)基本上等同于使用普通最小二乘法获得的最大后验概率(MAP)。这些作为默认线性回归是完全可以的；可以使用 priors 参数来调整先验。默认情况下，GLM 模块还会默认添加高斯似然。可以使用 family 参数调整它；可选似然函数包括：Normal(默认)、StudentT、Binomial、Poisson或NegativeBinomial。

为了描述统计模型，GLM 模块使用 [Patsy](https://patsy.readthedocs.
io/en/latest/index.html)。它是提供了一种公式微语言语法的Python库，其灵感来自 R 和 S 中使用的语法。在前面的代码块中，y~x 指定我们有一个输出变量 $y$ ，我们希望将其估计为 $x$ 的线性函数。



## 4.7 总结 

本章讨论的主要思想非常简单：为了预测输出变量的平均值，我们可以对输入变量的线性组合应用任意函数。我知道我在这一章的开头已经说过了，但是重复很重要。我们称这个任意函数为逆连接函数。我们在选择这样一个函数时的唯一限制是，输出必须足够用作抽样分布(通常是平均值)的参数。我们想要使用逆连接函数的一种情况是在处理分类变量时，另一种情况是数据只能取正值时，还有一种情况是当我们需要 [0，1] 区间内的变量时。所有这些不同的变体形成了不同的模型；其中许多模型通常被用作统计工具，并对其应用和统计特性进行了研究。

Logistic回归模型是上一章线性回归模型的推广，可用于分类或一般用于预测二项数据。Logistic回归的显著特点是使用Logistic函数作为逆连接函数，使用伯努利分布作为似然函数。逆连接函数的使用引入了非线性，在解释Logistic回归系数时应该考虑到这一点。该系数编码对数赔率单位随变量 $x$ 单位增加而增加的值。

将Logistic回归推广到两个以上类别的一种方法是使用 Softmax 回归。现在逆连接函数是 Softmax 函数，它是Logistic函数对于多值的推广，并且使用定类的分布作为似然。

Logistic 和 Softmax 都是判别式模型的例子；我们试图在没有显式模型 $p(x)$ 的情况下建立 $p(y|x)$  的模型。而生成式分类模型将首先建模 $p(x|y)$ ，即每个类 $y$ 下 $x$ 的分布建模，然后对类进行分配。这种模型被称为生成式分类器，因为我们正在创建一个模型，可以从该模型生成每个类的样本。我们看到了一个使用高斯分布的生成式分类器的例子。

我们使用 鸢尾花 数据集演示了所有这些模型，并简要讨论了相关变量、完全可分离类和不平衡类。

另一种流行的广义线性模型是泊松回归。该模型假设数据服从泊松分布，逆连接函数为指数函数。泊松分布和回归对计数数据进行建模很有用，即只取非负整数的数据，这些数据来自计数而不是排名。大多数分布是相互联系的，例如高斯分布和学生 $t$ 分布。另一个例子是泊松分布，当试验次数很多但成功概率很低时，它可以被视为二项分布的特例。

扩展泊松分布的一种有用方法是 ZIP 分布；可以将后者看作是两个其他分布的混合：泊松分布和生成额外零的二项分布。另一个可以解释为混合的有用扩展是负二项分布--在本例中是泊松分布的混合，其中rate( $\mu$)是伽马分布的随机变量。当数据过度分散时，负二项分布是泊松分布的一种有用的替代方法，也就是说，它是方差大于平均值的数据。



## 4.8 习题

（1）使用花瓣长度和花瓣宽度作为变量重跑第一个模型。二者的结果有何区别？两种情况下的95%HPD区间分别是多少？

（2）重跑练习（1），这次使用 $t$ 分布作为弱先验信息。尝试使用不同的正态参数 $v$ 。

（3）回到第1个例子中，用Logistic 回归根据花萼长度判断属于 Setosa 还是Versicolour 。尝试用第1章中的单参数回归模型来解决这个问题，线性回归的结果相比Logistic 回归的效果如何？线性回归的结果能解释为概率吗？提示：检查 $y$ 值是否位于 [0,1] 区间内。

（4）假设我们不用 softmax 回归，而是用单参数线性模型，并将类别编码为 Setosa =0,Versicolour =1, Virginica =2。在单参数线性回归模型中，如果我们交换类别的编码方式会发生什么？结果会保持一样还是会有所不同？

（5）在处理不均衡数据的例子中，将df=df\[45:\]改为df\[22:78\]，这样做得到的数据点个数几乎没变，不过现在类别变

得均衡了，试比较这两种情况的结果，哪种情况得到的结果与使用完整数据集得到的结果更相似呢？

（6）比较Logistic 回归 模型与LDA模型的似然，用函数sample_ppc生成预测数据并比较，确保理解其中的不同。
