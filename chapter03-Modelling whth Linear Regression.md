> 第**4**章　利用线性回归模型理解并预测数据
>
> 这一章我们将学习统计学和机器学习中应用最广泛的模型之一： 线性回归模型。这个模型除了本身非常有用之外，还可以看做是许多 其他模型的基石。如果你上过统计学的课程（即使是非贝叶斯的）， 你可能会听说过线性回归、逻辑回归、ANOVA和ANCOVA等。所有 这些方法都是同一个主题的变种：线性回归模型，这也正是本章将要 讨论的核心主题。
>
> 本章将涵盖以下内容：
>
> 线性回归模型\
> 一元线性回归； 鲁棒线性回归； 多层线性回归； 多项式回归；\
> 多元线性回归； 交互作用。
>
> 127
>
> **4.1**　一元线性回归
>
> 在科学界、工业界及商业中，经常会遇到下面这类问题：我们有 一些连续变量，这里连续的意思是变量可以用实数表示（或者说是浮 点数），我们称之为因变量、被预测的变量或者结果变量。我们想
>
> 要对该因变量与其他变量的依赖关系建模，这里其他变量称自变量、 预测变量或者输入变量。自变量可以是连续的，也可以是离散的。\
> 这类问题一般可以通过线性回归建模，如果我们只有一个自变量，那 么称为一元线性回归模型，如果有多个自变量，称为作多元线性回归 模型。使用线性模型的一些典型场景如下。
>
> 对多个因素之间的关系建模，例如雨量、土壤盐渍度以及农作物\
> 生长过程中是否施肥，然后回答一些问题：比如它们之间的关系\
> 是否是线性的？关系有多强？哪个因素的影响最强？
>
> 找出全国平均的巧克力摄入量与诺贝尔奖得主数量之间的关系。\
> 理解为什么这二者之间的关系可能是假的。\
> 根据当地天气预报中的太阳辐射预测家里（用于烧水和做饭）的\
> 燃气账单。该预测的准确性如何？
>
> **4.1.1**　与机器学习的联系
>
> 按照Kevin P. Murphy^\[1\]^ 的说法，机器学习是一个总称，指一系\
> 列从数据中自动学习其中的隐藏规律，并用来预测未知数据，或者是
>
> 在不确定的状态中做决策的方法。机器学习与统计学相互交织，不过 正如Kevin P.Murphy在他书中所说，如果从概率的角度来看，二者之 间的关系就比较清晰了。尽管这两个领域在概念上和数学上都紧密联 系，不过二者之间的术语可能让这种联系显得不那么清晰。因此，在
>
> 128
>
> 这一章中，我会介绍一些机器学习中的术语。用机器学习的行话来\
> 说，回归问题属于典型的监督学习。在机器学习的框架中，如果我们 想学习从*x*到*y*的一个映射，这就是一个回归问题，其中*y*是连续变\
> 量。这里有监督的意思是指，我们已经知道成对的*x*和*y*。某种意思上 来说，我们知道了正确答案，剩下的问题就是如何从这些观测值（或 者数据集）中抽象出一种映射关系来处理未来的观测（也就是只知\
> 道*x*而不知道*y*的
>
> 情形）。
>
> **4.1.2**　线性回归模型的核心
>
> 前面已经讨论了线性回归的一些基本思想，现在我们需要在统计 学和机器学习的术语之间构建一座桥梁，来学习如何构建线性模型。 看下面这个公式：

![](C:/Program Files/Typora/media/image1227.png){width="1.0099136045494312in" height="0.16659448818897638in"}

> 这个等式描述的是变量***x***与变量***y***之间的线性关系。其中，参数*β* 控制的是直线的斜率，这里斜率可以理解为变量***x***的单位变化量所对 应***y***的变化量。另外一个参数*α*可以解释为当*x~i~*=0时*y~i~*的值，在图中表
>
> 示，*α*就是直线与*y*轴交点的坐标。
>
> 计算线性模型参数的方法有很多，最小二乘法是方法之一。每次 使用软件去拟合直线的时候，其底层可能就是用的该方法，这种方法 返回的*α*和*β*能够让观测到的***y***与预测的***y***之间误差平方的均值最小。这 样，估计*α*和*β*就变成了一个最优化问题，最优化问题的目标一般是寻 找函数的最小值（或最大值）。最优化并非求解线性模型的唯一方
>
> 法，同样的问题还可以从概率（贝叶斯）的角度描述。用概率的方式 思考带来的优势是：我们在得到最优*α*和*β*（与最优化方法求解结果相
>
> 129
>
> 同）的同时还知道这些参数的不确定性，而最优化方法需要一些其他 工作来提供这类信息。此外，我们得到的还有贝叶斯方法的灵活性， 这意味着我们可以将模型应用到特定的问题中，比如将正态假设移
>
> 除，或者构建分层线性模型。
>
> 从概率的角度，线性回归模型可以表示成如下形式：

![](C:/Program Files/Typora/media/image1229.png){width="2.155177165354331in" height="0.187419072615923in"}

> 也就是说，这里假设向量*y*是服从均值为*α*+*βx*、标准差为*ε*的正态 分布。由于我们并不知道*α*、*β*或者*ε*，因此需要对其设置先验，一组 合理的先验如下：

![](C:/Program Files/Typora/media/image1230.png){width="1.1973206474190725in" height="0.187419072615923in"}

![](C:/Program Files/Typora/media/image1231.png){width="1.1869094488188976in" height="0.20824365704286965in"}

![](C:/Program Files/Typora/media/image1232.png){width="1.0099136045494312in" height="0.187419072615923in"}

> ![](C:/Program Files/Typora/media/image1233.png){width="0.18740594925634296in" height="0.10412182852143483in"}对于*α*的先验，我们可以使用一个分布很平坦的高斯分布，即\
> 相对数据的值域很大。通常我们并不知道截距是多少，其具体的值根
>
> 据问题不同有很大变化。对于斜率来说同样如此，对很多问题而言， 我们至少根据先验知道它是否大于0。对于*ε*来说，我们可以根据*y*的\
> 值域，设置*h~s~*为一个比较大的值，例如，将其设为*y*的标准差的10
>
> 倍。在这个贝叶斯模型中，对于单参数线性回归问题，只要有效地选 择了相对平坦的先验，我们可以得到和最小二乘方拟合一样的结果。 其中均匀分布还可以换成半正态分布或者半柯西分布。通常，半柯西 分布作为一个不错的正则先验（具体见第6章）很有效。如果我们希
>
> 望在某些值附近给标准差设置很强的先验，那么可以使用伽马分布。 在许多软件库中，伽马分布的默认参数看起来有点奇怪，不过幸运地 是PyMC3可以让我们同时使用形状和比例或者均值和方差来定义。
>
> 130
>
> 状：

继续讨论线性回归之前，先看一下伽马分布在不同参数下的形

> rates = \[1, 2, 5\] scales = \[1, 2, 3\]
>
> x = np.linspace(0, 20, 100)
>
> f, ax = plt.subplots(len(rates), len(scales), sharex=True, sharey=True) for i in range(len(ass)):
>
> for j in range(len(scales)):
>
> rate = rates\[i\]
>
> scale = scales\[j\]
>
> rv = stats.gamma(a=rate, scale=scale)
>
> ax\[i,j\].plot(x, rv.pdf(x))
>
> ax\[i,j\].plot(0, 0,
>
> label=\"\$\\\\alpha\$ = {:3.2f}\\n\$\\\\theta\$ = {:3.2f}\".format(rate,\
> scale), alpha=0)
>
> ax\[i,j\].legend()
>
> ax\[2,1\].set_xlabel(\'\$x\$\')
>
> ax\[1,0\].set_ylabel(\'\$pdf(x)\$\')

![](C:/Program Files/Typora/media/image1239.png){width="5.643024934383202in" height="3.821279527559055in"}

> 再来看线性回归模型，借助Kruschke图，我们有下面这张图：
>
> 131

![](C:/Program Files/Typora/media/image1241.png){width="3.768956692913386in" height="3.2694324146981626in"}

> 现在需要将数据输出给模型，这里再次使用了合成数据，来构建 对模型的直观认识。先假设已经知道参数的真实值，然后构造出数据 集，最后再利用模型将其找出来。
>
> np.random.seed(314)
>
> N = 100
>
> alfa_real = 2.5
>
> beta_real = 0.9
>
> eps_real = np.random.normal(0, 0.5, size=N)
>
> x = np.random.normal(10, 1, N)\
> y_real = alfa_real + beta_real \* x y = y_real + eps_real
>
> plt.figure(figsize=(10,5))\
> plt.subplot(1,2,1)
>
> plt.plot(x, y, \'b.\')
>
> plt.xlabel(\'\$x\$\', fontsize=16)\
> plt.ylabel(\'\$y\$\', fontsize=16, rotation=0) plt.plot(x, y_real, \'k\')\
> plt.subplot(1,2,2)
>
> sns.kdeplot(y)
>
> plt.xlabel(\'\$y\$\', fontsize=16)
>
> 132

![](C:/Program Files/Typora/media/image1247.png){width="5.643024934383202in" height="2.655112642169729in"}

> 现在用PyMC3来拟合数据，代码看起来和前面的模型都差不\
> 多，不过留意这里*μ*是通过pm.Deterministic来定义的，也就是说
>
> 该变量是确定的。目前为止我们见过的所有变量都是随机的，也就是 说，每次求随机变量值的时候，都会得到一个不同的值。而在这里， 确定的变量意味着其值完全由参数决定，尽管在下面的代码中这些参 数是随机的。如果我们声明一个变量是确定的，那么PyMC3会将其
>
> 保存在迹中。
>
> with pm.Model() as model:
>
> alpha = pm.Normal(\'alpha\', mu=0, sd=10) beta = pm.Normal(\'beta\', mu=0, sd=1)\
> epsilon = pm.HalfCauchy(\'epsilon\', 5)
>
> mu = pm.Deterministic(\'mu\', alpha + beta \* x)
>
> y_pred = pm.Normal(\'y_pred\', mu=mu, sd=epsilon, observed=y)
>
> start = pm.find_MAP()
>
> step = pm.Metropolis()
>
> trace = pm.sample(10000, step, start)
>
> 此外，还可以省略固定变量的声明而直接写为如下语句：
>
> y_pred = pm.Normal(\'y_pred\', mu= alpha + beta \* x, sd=epsilon, observed =y)
>
> traceplot函数返回的图像很有趣。注意其中*α*和*β*参数在缓慢地
>
> 133
>
> 上下波动；对比之下，只有*ε*的迹呈现了较好的混合度。显然，出问 题了。
>
> pm.traceplot(chain);

![](C:/Program Files/Typora/media/image1261.png){width="5.643024934383202in" height="3.6859208223972004in"}

> 上面的自相关图表明，*α*和*β*有很强的自相关性，而*ε*没有。接下 来我们会看到，这是线性模型的典型表现。注意这里通过给\
> autocorrplot传递varnames参数，没有画出确定的变量*μ*。
>
> varnames = \[\'alpha\', \'beta\', \'epsilon\'\] pm.autocorrplot(trace, varnames)
>
> 134

![](C:/Program Files/Typora/media/image1267.png){width="5.643024934383202in" height="2.9778904199475065in"}

> **4.1.3**　线性模型与高自相关性
>
> 前面的模型中，*α*和*β*有很糟糕的自相关性，这意味着采样很差， 而且相比实际的采样数，有效的采样很少。为什么呢？其实我们是被 自己的假设误导了。事实上，不论我们用哪条直线去拟合数据，它们 都会穿过一点，***x***的均值和***y***的均值点。因此，拟合直线的过程相当于 将直线固定在数据的中心上进行旋转，斜率越大截距越小。根据模型 的定义，两个参数是相关的，如果将后验画出来的话可以很清楚地看 到这点（这里暂时忽略*ε*）。
>
> sns.kdeplot(trace\[\'alpha\'\], trace\[\'beta\'\])\
> plt.xlabel(r\'\$\\alpha\$\', fontsize=16)\
> plt.ylabel(r\'\$\\beta\$\', fontsize=16, rotation=0)
>
> 135

![](C:/Program Files/Typora/media/image1273.png){width="5.643024934383202in" height="3.7171576990376205in"}

> 可以看到，后验（除了*ε*之外）呈斜对角形状，对于类似\
> Metropolis-Hastings的算法，这可能会存在问题。因为如果给每个独
>
> 立的参数设置一个较大的步长，那么很可能会落在高概率区域以外； 只有步长设得很小的时候，被接受的概率才会较高。不管哪种方式我 们都会得到很高的自相关性和较差的混合度，而且数据的维度越高， 这种情况越严重，因为总的参数空间要比可能的参数空间增长得快得 多，关于这点可以查阅维基百科中维数灾难的内容进行了解。
>
> 在继续深入之前，请允许我澄清一点，前面提到的拟合直线会穿 过数据的均值点只在最小二乘方算法的假设下成立。使用贝叶斯方法 之后，这个限制被放松了。后面的例子中我们可以看到，通常直线会 在***x***和***y***的均值附近而不是正好穿过均值。不过没关系，自相关性与直 线固定在某一点附近的假设仍然是成立的，关于高自相关性问题我们 需要了解的就是这么多了，接下来我们会从两个方面理解和解决高自 相关性问题。
>
> 运行之前先修改数据
>
> 136
>
> ![](C:/Program Files/Typora/media/image1275.png){width="9.370297462817148e-2in" height="0.11453412073490814in"}解决问题的一个简单办法是先将***x***中心化，也就是说，对于每个 点*x~i~*，减去***x***的均值（）。

![](C:/Program Files/Typora/media/image1276.png){width="0.8537412510936133in" height="0.14576990376202975in"}

> 这样做的结果是***x**\'*的中心在0附近，从而修改斜率时旋转点变成 了截距点，参数空间也会变得不那么自相关。
>
> 中心化不仅仅是一种计算技巧，同时有利于解释数据。截距是指 当*x~i~*=0时*y~i~*的值，不过对于许多问题而言，截距并没有什么实际的意
>
> 义。例如，对于身高或者体重这类数值，当值为0时，并没有实际的 意义，因而截距对于理解数据也就没有任何帮助，对于另外一些问\
> 题，估计出截距可能很有用，因为在实验中我们可能无法测量出*x~i~*=0
>
> 的情况，但截距的估计值可以为我们提供有价值的信息。不管怎么\
> 说，推断都有其局限性，应当谨慎使用。
>
> 根据问题和受众不同，我们可能需要汇报中心化之前或者之后估 计到的参数值。如果我们需要汇报的是中心化之前的参数，那么可以 像下面这样将参数转换成原来的尺度：

![](C:/Program Files/Typora/media/image1277.png){width="1.0515594925634295in" height="0.187419072615923in"}

> 上面的公式可以通过以下公式推导出来：

![](C:/Program Files/Typora/media/image1278.png){width="1.8324212598425196in" height="0.8746248906386702in"}

> 然后可以得出：

![](C:/Program Files/Typora/media/image1279.png){width="1.0515594925634295in" height="0.4164873140857393in"}

> 137
>
> 更进一步，在运行模型之前，我们可以对数据进行标准化处理。 标准化在统计学和机器学习中是一种常见的数据处理手段，这是因为 许多算法对于标准化之后的数据效果更好。标准化的过程是在中心化 之后再除以标准差，其数学形式如下：

![](C:/Program Files/Typora/media/image1281.png){width="0.6767465004374453in" height="0.5101979440069991in"}

> 标准化的好处之一是我们可以对数据使用相同的弱先验，而不必 关心数据的具体值域有多大，因为我们已经对数据进行了尺度变换。 对于标准化之后的数据，截距通常在0附近，斜率在-1～1附近。标准 化之后的数据可以使用标准分数（Z-score）来描述参数。如果某人
>
> 声称一个参数的标准分数值为1.3，那么我们就知道该值在标准化之\
> 前位于均值附近1.3倍的标准差处。标准分数每变化一个单位，那么\
> 对应原始数据中变化1倍的标准差。这一点在分析多个变量时很有
>
> 用，因为所有的参数都在同一个尺度，从而简化了对数据的解释。
>
> 更换采样方法
>
> 另外一种解决高自相关性的办法是使用不同的采样方法。NUTS 算法与Metropolis算法相比，在类似受限的对角空间中遇到的困难小 一些。原因是NUTS是根据后验的曲率来移动的，因而更容易沿着对 角空间移动。NUTS算法每走一步要比Metropolis算法更慢，不过要得
>
> 到一个合理的后验近似值所需要的步数更少。
>
> 下面将讨论使用NUTS采样方法得到的结果。
>
> **4.1.4**　对后验进行解释和可视化
>
> 前面我们已经知道了如何使用PyMC3中的traceplot和\
> df_summary函数或者一些自定义的函数探索后验。对于线性回归，
>
> 138
>
> 一种更好的表现方式是将拟合数据的平均直线与*α*和*β*的均值同时表示 在图上：
>
> plt.plot(x, y, \'b.\');
>
> alpha_m = trace_n\[\'alpha\'\].mean()
>
> beta_m = trace_n\[\'beta\'\].mean()
>
> plt.plot(x, alpha_m + beta_m \* x, c=\'k\', label=\'y = {:.2f} + {:.2f} \* x \'.format(alpha_m, beta_m))
>
> plt.xlabel(\'\$x\$\', fontsize=16)
>
> plt.ylabel(\'\$y\$\', fontsize=16, rotation=0)
>
> plt.legend(loc=2, fontsize=14)

![](C:/Program Files/Typora/media/image1287.png){width="5.643024934383202in" height="3.873340988626422in"}

> 或者，我们还可以从后验采样中画出半透明的直线来表示后验的 不确定性。代码如下。
>
> 139

![](C:/Program Files/Typora/media/image1289.png){width="5.643024934383202in" height="3.9358147419072615in"}

> plt.plot(x, y, \'b.\');
>
> idx = range(0, len(trace_n\[\'alpha\'\]), 10)
>
> plt.plot(x, trace_n\[\'alpha\'\]\[idx\] + trace_n\[\'beta\'\]\[idx\] \*x\[:,np.newaxi s\], c=\'gray\', alpha=0.5);
>
> plt.plot(x, alpha_m + beta_m \* x, c=\'k\', label=\'y = {:.2f} + {:.2f} \* x \'.format(alpha_m, beta_m))
>
> plt.xlabel(\'\$x\$\', fontsize=16)\
> plt.ylabel(\'\$y\$\', fontsize=16, rotation=0) plt.legend(loc=2, fontsize=14)
>
> 可以看到，上图中间部分的不确定性较低，不过直线并没有都相 交于一个点（后验并不强制所有的直线都穿过均值点）。
>
> 半透明的直线看起来不错，不过我们可能想给这个图增加点更酷 的东西：用半透明的区间描述*μ*的最大后验密度（HPD）区间。注意 这也是在模型中将变量*μ*定义成一个确定值的主要原因，简化以下代 码：
>
> ![](C:/Program Files/Typora/media/image1294.png){width="3.819444444444445e-2in" height="0.652509842519685in"}plt.plot(x, alpha_m + beta_m \* x, c=\'k\', label=\'y = {:.2f} + {:.2f} \* x \'.format(alpha_m,beta_m))
>
> 140
>
> ![](C:/Program Files/Typora/media/image1298.png){width="2.7777777777777776e-2in" height="2.7777777777777776e-2in"}idx = np.argsort(x)
>
> x_ord = x\[idx\]
>
> sig = pm.hpd(trace_n\[\'mu\'\], alpha=.02)\[idx\]\
> plt.fill_between(x_ord, sig\[:,0\], sig\[:,1\], color=\'gray\')
>
> plt.xlabel(\'\$x\$\', fontsize=16)\
> plt.ylabel(\'\$y\$\', fontsize=16, rotation=0)

![](C:/Program Files/Typora/media/image1302.png){width="5.643024934383202in" height="3.9462259405074365in"}

> ![](C:/Program Files/Typora/media/image1303.png){width="0.10411417322834646in" height="0.16659448818897638in"}另外一种方式是画预测值的HPD（例如95%和50%）区间。也 就是说，我们想要根据模型看到未来95%和50%的数据的分布范围。
>
> 我们在图中将50%HPD区间用深灰色区域表示，将95%HPD区间用浅 灰色表示。利用PyMC3中的sample_ppc函数可以很容易得到预测值
>
> 的采样值。
>
> ppc = pm.sample_ppc(chain_n, samples=1000, model=model_n)
>
> 然后我们可以画出结果。
>
> ![](C:/Program Files/Typora/media/image1308.png){width="3.818678915135608e-2in" height="0.9232272528433946in"}plt.plot(x, y, \'b.\')
>
> plt.plot(x, alpha_m + beta_m \* x, c=\'k\', label=\'y = {:.2f} + {:.2f} \* x \'.format(alpha_m, beta_m))
>
> 141
>
> ![](C:/Program Files/Typora/media/image1312.png){width="2.7777777777777776e-2in" height="2.7777777777777776e-2in"}sig0 = pm.hpd(ppc\[\'y_pred\'\], alpha=0.5)\[idx\]
>
> sig1 = pm.hpd(ppc\[\'y_pred\'\], alpha=0.05)\[idx\]
>
> plt.fill_between(x_ord, sig0\[:,0\], sig0\[:,1\], color=\'gray\', alpha=1)\
> plt.fill_between(x_ord, sig1\[:,0\], sig1\[:,1\], color=\'gray\', alpha=0.5)
>
> plt.xlabel(\'\$x\$\', fontsize=16)\
> plt.ylabel(\'\$y\$\', fontsize=16, rotation=0)

![](C:/Program Files/Typora/media/image1316.png){width="5.01833552055993in" height="3.456852580927384in"}

> 图中HPD区间的边界不是很规则，原因是：为了画出这个图，我 们是从观测值x而不是从连续区间中得到的后验预测采样，此\
> 外，fill_between函数只是简单地对相邻点之间进行线性差值。留 意锯齿的尖锐程度与数据数量之间的关系，可以通过从y_pred中获
>
> 取更多的采样值来减少不规则的程度。
>
> **4.1.5**　皮尔逊相关系数
>
> 许多时候，我们希望衡量两个变量之间的（线性）依赖关系。衡 量两个变量之间线性相关性最常见的指标是皮尔逊相关系数，通常用 小写的*r*表示。如果*r*的值为+1，我们称这两个变量完全正相关，也就 是说一个变量随着另外一个变量的增加而增加；如果*r*的值为-1，那
>
> 么称为完全负相关，也就是说一个变量随着另一个变量的增加而减
>
> 142
>
> 少；当*r*为0时，我们称两个变量之间没有线性相关性。皮尔逊相关系\
> 数并不涉及非线性相关性。人们很容易将皮尔逊相关系数与回归中的\
> 斜率弄混淆，查看以下链接中的图就可以明白二者本质上是两个不同\
> 的量：
>
> [https://en.wikipedia.org/wiki/Correlation_and_dependence\#/media/File:Correlation_ex](https://en.wikipedia.org/wiki/Correlation_and_dependence#/media/File:Correlation_examples2.svg)
>
> 下面的公式可以在某种程度上减轻你的疑惑：

![](C:/Program Files/Typora/media/image1319.png){width="0.8849759405074366in" height="0.4477241907261592in"}

> 也就是说，只有在***x***和***y***的标准差相等时，皮尔逊相关系数才与斜 率相等。当我们对数据标准化时，上式是成立的。需要注意：
>
> 皮尔逊相关系数衡量的是两个变量之间的相关性程度，其值位于 \[-1,1\]区间内，与数据的尺度无关；\
> 斜率表示的是***x***变化一个单位时，***y***的变化量，可以取任意实数。
>
> 皮尔逊相关系数与决定系数（Coefficient of Determination）之间 有联系，对于线性回归模型而言，决定系数就是皮尔逊相关系数的平
>
> 方，即*r*^2^（或者*R*^2^）。决定系数用于度量变量的变异中可以用自变量 解释部分所占的比例。
>
> 现在，我们扩展下线性回归模型，利用PyMC3从两个方面计算*r*
>
> 和*r*^2^。
>
> 一种方式是使用前面看到的皮尔逊相关系数与斜率之间关系的公 式，将其记作变量rb。
>
> 另一种方式与最小二乘方算法相关，这里暂时跳过其来源细节， 将其记作变量rss。仔细阅读代码可以看到，变量ss_reg衡量的
>
> 143
>
> 是拟合的直线与数据均值之间的离差，与模型中的方差成比例， 其公式与方差很像，不过没有除以数据的个数。变量ss_tot与
>
> 预测值的方差成比例。
>
> 完整的模型如下：
>
> with pm.Model() as model_n:
>
> alpha = pm.Normal(\'alpha\', mu=0, sd=10) beta = pm.Normal(\'beta\', mu=0, sd=1)\
> epsilon = pm.HalfCauchy(\'epsilon\', 5)
>
> mu = alpha + beta \* x
>
> y_pred = pm.Normal(\'y_pred\', mu=mu, sd=epsilon, observed=y)
>
> rb = pm.Deterministic(\'rb\', (beta \* x.std() / y.std()) \*\* 2)
>
> y_mean = y.mean()
>
> ss_reg = pm.math.sum((mu - y_mean) \*\* 2)\
> ss_tot = pm.math.sum((y - y_mean) \*\* 2)\
> rss = pm.Deterministic(\'rss\', ss_reg/ss_tot)
>
> start = pm.find_MAP()
>
> step = pm.NUTS()
>
> trace_n = pm.sample(2000, step=step, start=start) pm.traceplot(chain_n)
>
> 144

![](C:/Program Files/Typora/media/image1330.png){width="5.643024934383202in" height="4.6230205599300085in"}

> pm.df_summary(cadena_n, varnames)

+-----------+--------+------+----------+---------+----------+
|           | > mean | sd   | mc_error | hpd_2.5 | hpd_97.5 |
+===========+========+======+==========+=========+==========+
| > alpha   | 2.11   | 0.49 | 1.87e-02 | 1.21    | 3.13     |
+-----------+--------+------+----------+---------+----------+
| > beta    | 0.94   | 0.05 | 1.82e-03 | 0.84    | 1.03     |
+-----------+--------+------+----------+---------+----------+
| > epsilon | 0.45   | 0.03 | 1.30e-03 | 0.39    | 0.52     |
+-----------+--------+------+----------+---------+----------+
| > rb      | 0.80   | 0.08 | 3.09e-03 | 0.64    | 0.96     |
+-----------+--------+------+----------+---------+----------+
| > rss     | 0.80   | 0.08 | 3.22e-03 | 0.64    | 0.95     |
+-----------+--------+------+----------+---------+----------+

> 145
>
> 根据多元高斯分布计算皮尔逊相关系数
>
> 另一种计算皮尔逊相关系数的方法是估计一个多元高斯分布的方 差矩阵。我们暂时只考虑二维的情况，一旦理解了两个变量的情况之 后，理解更高维度就会很容易了。为了充分描述二元高斯分布，我们 需要两个均值（或者一个长度为2的向量），每个均值对应一个边缘
>
> 高斯分布，此外还需要两个标准差，这么说其实不太准确，事实上我 们需要一个像下面这样的2×2的协方差矩阵：

![](C:/Program Files/Typora/media/image1336.png){width="2.196823053368329in" height="0.46854877515310583in"}

> ![](C:/Program Files/Typora/media/image1337.png){width="0.23946412948381451in" height="0.11453412073490814in"}其中，∑是大写的希腊字母希格玛，通常用其表示协方差矩阵。 在主对角线上的两个元素分别是每个变量的方差，用标准差和 的平方表示。剩余的两个元素分别是协方差（变量之间的共变），用 每个变量标准差和*ρ*（变量之间的皮尔逊相关系数）来表示。注意这 里只有一个*ρ*，原因是我们只有两个变量，如果有3个变量的话，对应 的会有3个*ρ*，如果有4个变量的话，对应会有6个*ρ*。计算这些数需要 用到二项系数，回忆一下第1章中的二项
>
> 分布。
>
> ![](C:/Program Files/Typora/media/image1339.png){width="0.23946412948381451in" height="0.11453412073490814in"}下面的代码生成了一些二维高斯分布的等值线图，其中均值都固 定在（0,0），标准差取1，另外一个标准差分别取1或者2，皮尔 逊相关系数取-1到1之间的不同值。
>
> ![](C:/Program Files/Typora/media/image1341.png){width="3.818678915135608e-2in" height="1.9540354330708662in"}sigma_x1 = 1
>
> sigmas_x2 = \[1, 2\]
>
> rhos = \[-0.99, -0.5, 0, 0.5, 0.99\]
>
> x, y = np.mgrid\[-5:5:.1, -5:5:.1\] pos = np.empty(x.shape + (2,))\
> pos\[:, :, 0\] = x; pos\[:, :, 1\] = y
>
> f, ax = plt.subplots(len(sigmas_x2), len(rhos), sharex=True, sharey=Tru
>
> 146
>
> ![](C:/Program Files/Typora/media/image1345.png){width="2.7777777777777776e-2in" height="2.7777777777777776e-2in"}e)
>
> for i in range(2):
>
> for j in range(5):
>
> sigma_x2 = sigmas_x2\[i\]
>
> rho = rhos\[j\]
>
> cov = \[\[sigma_x1\*\*2, sigma_x1\*sigma_x2\*rho\],\
> \[sigma_x1\*sigma_x2\*rho, sigma_x2\*\*2\]\]
>
> rv = stats.multivariate_normal(\[0, 0\], cov)\
> ax\[i,j\].contour(x, y, rv.pdf(pos))
>
> ax\[i,j\].plot(0, 0,
>
> label=\"\$\\\\sigma\_{{x2}}\$ = {:3.2f}\\n\$\\\\rho\$ = {:3.2f}\".format(si gma\_
>
> x2, rho), alpha=0)
>
> ax\[i,j\].legend()
>
> ax\[1,2\].set_xlabel(\'\$x_1\$\')
>
> ax\[1,0\].set_ylabel(\'\$x_2\$\')

![](C:/Program Files/Typora/media/image1349.png){width="5.643024934383202in" height="3.9462259405074365in"}

> 了解了多维高斯分布之后，我们就可以拿它来估计皮尔逊相关系 数了。由于我们并不知道协方差矩阵，可以先为其设置一个先验。一 种做法是使用威沙特分布（Wishart distribution）^\[1\]^ ，威沙特分布是
>
> 多维正态分布逆协方差矩阵的共轭先验，可以看作是前面见过的伽马 分布在高维空间的一般形式，也可以看作卡方分布（chi squared
>
> 147
>
> ![](C:/Program Files/Typora/media/image1339.png){width="0.23946412948381451in" height="0.11453412073490814in"}distribution）的一般形式。另一种做法是使用LKJ先验，该先验用于\
> 相关性矩阵的（不是协方差矩阵），如果考虑相关性的话，使用起来 更方便一些。这里我们讨论第3种做法，直接为、和*ρ*设置先\
> 验，然后用这些值手动构造协方差矩阵。
>
> with pm.Model() as pearson_model:
>
> mu = pm.Normal(\'mu\', mu=data.mean(0), sd=10, shape=2)
>
> sigma_1 = pm.HalfNormal(\'sigma_1\', 10) sigma_2 = pm.HalfNormal(\'sigma_2\', 10) rho = pm.Uniform(\'rho\', -1, 1)
>
> cov = pm.math.stack((\[sigma_1\*\*2, sigma_1\*sigma_2\*rho\], \[sigma_1\* sigma_2\*rho, sigma_2\*\*2\]))
>
> y_pred = pm.MvNormal(\'y_pred\', mu=mu, cov=cov, observed=data)
>
> start = pm.find_MAP()
>
> step = pm.NUTS(scaling=start)
>
> trace_p = pm.sample(1000, step=step, start=start)

![](C:/Program Files/Typora/media/image1357.png){width="5.643024934383202in" height="3.6442727471566054in"}

> 注意这里得到的*ρ*是皮尔逊相关系数，而前面的例子中得到的是 皮尔逊相关系数的平方，考虑这点之后你会发现这里得到的结果与前
>
> 148
>
> 面例子中的结果是一致的。

149

> **4.2**　鲁棒线性回归
>
> 在许多情况下，假设数据服从高斯分布是非常合理的。我们假设 数据符合高斯特性，并不是说数据真的就是符合高斯分布的，而是说 我们认为高斯分布对于我们的问题而言是一个合理的近似。前面我们 知道了，有时候高斯假设并不成立，例如出现异常值的时候，利用t
>
> 分布，可以有效地解决异常值的问题，从而得到更鲁棒的推断。类似 的思想同样可以应用到线性回归问题中。
>
> 为了验证t分布确实能增加线性回归的鲁棒性，这里我们使用一\
> 个非常简单的数据集：安斯库姆四重奏（Anscombe\'s quartet）中的第
>
> 3组数据。如果你不知道安斯库姆四重奏数据集，可以在维基百科上 查看，这里我们从seaborn中加载它。
>
> ans = sns.load_dataset(\'anscombe\')
>
> x_3 = ans\[ans.dataset == \'III\'\]\[\'x\'\].values y_3 = ans\[ans.dataset == \'III\'\]\[\'y\'\].values
>
> 然后来看看这个数据集长什么样。
>
> plt.figure(figsize=(10,5))
>
> plt.subplot(1,2,1)
>
> beta_c, alpha_c = stats.linregress(x_3, y_3)\[:2\]
>
> plt.plot(x_3, (alpha_c + beta_c\* x_3), \'k\', label=\'y ={:.2f} + {:.2f} \* x\'.format(alpha_c, beta_c))
>
> plt.plot(x_3, y_3, \'bo\')
>
> plt.xlabel(\'\$x\$\', fontsize=16)
>
> plt.ylabel(\'\$y\$\', rotation=0, fontsize=16)
>
> plt.legend(loc=0, fontsize=14)
>
> plt.subplot(1,2,2)
>
> sns.kdeplot(y_3);
>
> plt.xlabel(\'\$y\$\', fontsize=16)
>
> 150

![](C:/Program Files/Typora/media/image1370.png){width="5.643024934383202in" height="2.655112642169729in"}

> 现在我们用t分布替换模型中的高斯分布，这个改变需要引入正 态参数*v*，有关该参数的含义，可以参照前一章的内容。
>
> 在下面的模型中，我们使用了移位指数分布来避免nu的值接近\
> 0，因为非移位指数分布对于0附近的值赋予了太高的权重。根据我的
>
> 经验，对于没有异常点或者含有少量异常点的数据集而言，使用非移 位指数分布就够了，不过对于某些包含极限异常值的数据（或者是只 有少量聚集点的数据集）而言，例如我们用到的安斯库姆四重奏数据 集的第3组，移位指数分布更合适。这些建议都只是基于我（或者别
>
> 人）处理某些数据集或者问题的经验。此外，正态参数*v*的一些常见\
> 先验还有gamma(2,0.1)或者gamma(mu=20, sd=15)。
>
> 注意，在PyMC3中以下划线_结尾的变量，如nu\_（译者注：新 版本的PyMC3中写成_nu\_），对于用户是不可见的。
>
> ![](C:/Program Files/Typora/media/image1371.png){width="3.818897637795276e-2in" height="1.9436231408573927in"}with pm.Model() as model_t:
>
> alpha = pm.Normal(\'alpha\', mu=0, sd=100)
>
> beta = pm.Normal(\'beta\', mu=0, sd=1)
>
> epsilon = pm.HalfCauchy(\'epsilon\', 5)
>
> nu = pm.Deterministic(\'nu\', pm.Exponential(\'nu\_\', 1/29) + 1)
>
> y_pred = pm.StudentT(\'y_pred\', mu=alpha + beta \* x_3, sd=epsilon, n u=nu, observed=y_3)
>
> 151 ![](C:/Program Files/Typora/media/image1374.png){width="2.7777777777777776e-2in" height="2.7777777777777776e-2in"}
>
> ![](C:/Program Files/Typora/media/image1376.png){width="2.7777777777777776e-2in" height="2.7777777777777776e-2in"} start = pm.find_MAP()
>
> step = pm.NUTS(scaling=start)
>
> trace_t = pm.sample(2000, step=step, start=start)
>
> 这里为了节省版面，省略一些代码和图（如迹和自相关性的\
> 图），不过你自己尝试的时候可别这么做。我在这里只画出了平均的
>
> 拟合直线，此外还用SciPy中的linregress函数画了一条非鲁棒的直\
> 线。你可能需要尝试根据前面例子中的贝叶斯方法画出对应的直线。
>
> beta_c, alpha_c = stats.linregress(x_3, y_3)\[:2\]
>
> plt.plot(x_3, (alpha_c + beta_c \* x_3), \'k\', label=\'non-robust\', alpha= 0.5)
>
> plt.plot(x_3, y_3, \'bo\')
>
> alpha_m = trace_t\[\'alpha\'\].mean()
>
> beta_m = trace_t\[\'beta\'\].mean()
>
> plt.plot(x_3, alpha_m + beta_m \* x_3, c=\'k\', label=\'robust\')
>
> plt.xlabel(\'\$x\$\', fontsize=16)\
> plt.ylabel(\'\$y\$\', rotation=0, fontsize=16) plt.legend(loc=2, fontsize=12)

![](C:/Program Files/Typora/media/image1384.png){width="5.643024934383202in" height="3.6963331146106735in"}

> 执行下后验预测检查，看看模型是如何捕捉数据的。这里可以用 PyMC3替我们从后验中采样：
>
> 152
>
> ppc = pm.sample_ppc(chain_t, samples=200, model=model_t, random_seed=2) for y_tilde in ppc\[\'y_pred\'\]:
>
> sns.kdeplot(y_tilde, alpha=0.5, c=\'g\')
>
> sns.kdeplot(y_3, linewidth=3)
>
> 然后会看到类似下面的图：

![](C:/Program Files/Typora/media/image1390.png){width="5.643024934383202in" height="4.081585739282589in"}

> 对大多数数据，我们都拟合得相当不错，而且注意，这里不仅对 中心部分的数据拟合得很好，对于异常点部分也拟合得不错。对于当 前的目标来说，这个模型表现得不错，不需要更多的改进了。不过， 在某些问题中，我们可能不希望有负数，这种情况下，我们可能需要 回过头修改下模型，限制***y***为正数。
>
> 153
>
> **4.3**　分层线性回归
>
> 前面，我们学习了分层模型的基础知识，现在我们可以将这些概 念应用到线性回归，在组这一层同时对多个组建模并进行估计。和前 面一样，这里是通过引入超先验实现的。 我们先创建8个相关的数据 组，其中包括一个只有一个点的组。
>
> N = 20
>
> M = 8
>
> idx = np.repeat(range(M-1), N) idx = np.append(idx, 7)
>
> alpha_real = np.random.normal(2.5, 0.5, size=M)\
> beta_real = np.random.beta(60, 10, size=M)\
> eps_real = np.random.normal(0, 0.5, size=len(idx))
>
> y_m = np.zeros(len(idx))
>
> x_m = np.random.normal(10, 1, len(idx))
>
> y_m = alpha_real\[idx\] + beta_real\[idx\] \* x_m + eps_real
>
> 我们的数据如下：
>
> j, k = 0, N
>
> for i in range(M):\
> plt.subplot(2,4,i+1)\
> plt.scatter(x_m\[j:k\], y_m\[j:k\]) plt.xlim(6, 15)
>
> plt.ylim(7, 17)
>
> j += N
>
> k += N
>
> plt.tight_layout()
>
> 154

![](C:/Program Files/Typora/media/image1402.png){width="5.643024934383202in" height="2.707173009623797in"}

> 在输入到模型之前，先对其进行中心化处理：
>
> x_centered = x_m - x_m.mean()
>
> 首先，和前面的做法一样，先用非多层的模型拟合，唯一的区别 是需要增加部分代码将*α*转换到原始的尺度。
>
> with pm.Model() as unpooled_model:
>
> alpha_tmp = pm.Normal(\'alpha_tmp\', mu=0, sd=10, shape=M) beta = pm.Normal(\'beta\', mu=0, sd=10, shape=M)\
> epsilon = pm.HalfCauchy(\'epsilon\', 5)
>
> nu = pm.Exponential(\'nu\', 1/30)
>
> y_pred = pm.StudentT(\'y_pred\', mu=alpha_tmp\[idx\] + beta\[idx\]\* x_cen tered, sd=epsilon, nu=nu, observed=y_m)
>
> alpha = pm.Deterministic(\'alpha\', alpha_tmp - beta \*x_m.mean())
>
> start = pm.find_MAP()
>
> step = pm.NUTS(scaling=start)
>
> trace_up = pm.sample(2000, step=step, start=start)
>
> 从结果中可以看到，除了其中一组*α*和*β*参数，大多数情况下结果 都很正常。根据它们的迹来看，似乎这一组参数一直在自由移动而没 有收敛。
>
> ![](C:/Program Files/Typora/media/image1411.png){width="3.818897637795276e-2in" height="0.41302821522309713in"}varnames=\[\'alpha\', \'beta\', \'epsilon\', \'nu\'\]
>
> ![](C:/Program Files/Typora/media/image1414.png){width="2.7777777777777776e-2in" height="2.7777777777777776e-2in"}155
>
> ![](C:/Program Files/Typora/media/image1416.png){width="2.7777777777777776e-2in" height="2.7777777777777776e-2in"}pm.traceplot(trace_up, varnames);

![](C:/Program Files/Typora/media/image1420.png){width="5.643024934383202in" height="3.665096237970254in"}

> 显然，用一条唯一的直线去拟合一个点是不合适的，我们至少需 要两个点，或者参数*α*和*β*是有界的，除非我们能提供一些额外的信息 （比如加入先验），给*α*加入一个很强的先验能够得到一组明确定义
>
> 的先验，即使我们的数据中只有一个点。另一种方式是通过构建多层 模型往模型中加入信息，这主要是因为多层模型中组与组之间的信息 能够共享，这一点对于已经有不同分组的稀疏数据非常有用。这里我 们用到的例子中将数据稀疏性推向了极致（其中一组只有一个数
>
> 据），目的是将问题描述得更清楚一些。
>
> 现在我们实现一个与前面线性回归模型相同的多层模型，不过这 次用的是超先验，你可以从下面的Kruschke图中看到。
>
> 156

![](C:/Program Files/Typora/media/image1422.png){width="5.643024934383202in" height="4.1752952755905515in"}

> 用PyMC3代码实现的模型与之前模型的主要区别如下。
>
> 增加了超先验。\
> 增加了几行代码将参数转换到中心化之前的尺度。记住这并非强 制的，我们完全可以将参数保留在转换后的尺度上，只是对结果 进行解释的时候需要小心。\
> 使用了ADVI而不是find_MAP()函数去初始化，为NUTS算法提 供了一个协方差缩放矩阵。如果你不记得什么是ADVI了，可以 看一下第2章中的相关理论描述以及第2章练习部分更偏实际一些 的讨论。
>
> ![](C:/Program Files/Typora/media/image1426.png){width="3.818897637795276e-2in" height="1.7249671916010498in"}with pm.Model() as hierarchical_model:
>
> alpha_tmp_mu = pm.Normal(\'alpha_tmp_mu\', mu=0, sd=10) alpha_tmp_sd = pm.HalfNormal(\'alpha_tmp_sd\', 10)\
> beta_mu = pm.Normal(\'beta_mu\', mu=0, sd=10)\
> beta_sd = pm.HalfNormal(\'beta_sd\', sd=10)
>
> alpha_tmp = pm.Normal(\'alpha_tmp\', mu=alpha_tmp_mu, sd=alpha_tmp\_
>
> 157
>
> ![](C:/Program Files/Typora/media/image1430.png){width="2.7777777777777776e-2in" height="2.7777777777777776e-2in"}sd, shape=M)
>
> beta = pm.Normal(\'beta\', mu=beta_mu, sd=beta_sd, shape=M) epsilon = pm.HalfCauchy(\'epsilon\', 5)
>
> nu = pm.Exponential(\'nu\', 1/30)
>
> y_pred = pm.StudentT(\'y_pred\', mu=alpha_tmp\[idx\] + beta\[idx\] \* x\_ centered, sd=epsilon, nu=nu, observed=y_m)
>
> alpha = pm.Deterministic(\'alpha\', alpha_tmp - beta \* x_m.mean())\
> alpha_mu = pm.Deterministic(\'alpha_mu\', alpha_tmp_mu - beta_mu \* x\_ m.mean())
>
> alpha_sd = pm.Deterministic(\'alpha_sd\', alpha_tmp_sd - beta_mu \* x\_ m.mean())
>
> mu, sds, elbo = pm.variational.advi(n=100000, verbose=False) cov_scal = np.power(hierarchical_model.dict_to_array(sds), 2) step = pm.NUTS(scaling=cov_scal, is_cov=True)
>
> trace_hm = pm.sample(1000, step=step, start=mu)
>
> 接下来是traceplot部分，包括只有一个点的组。
>
> varnames=\[\'alpha\', \'alpha_mu\', \'alpha_sd\', \'beta\', \'beta_mu\', \'beta_sd\' , \'epsilon\', \'nu\'\]
>
> pm.traceplot(trace_hm, varnames)
>
> 158

![](C:/Program Files/Typora/media/image1439.png){width="5.643024934383202in" height="7.486377952755905in"}

> 将拟合的直线画出来，包括只有一个点的那一组。显然，那条直 线主要受到了其他组数据点的影响。
>
> ![](C:/Program Files/Typora/media/image1440.png){width="3.8190069991251095e-2in" height="1.1106463254593175in"} j, k = 0, N
>
> x_range = np.linspace(x_m.min(), x_m.max(), 10) for i in range(M):
>
> plt.subplot(2,4,i+1)\
> plt.scatter(x_m\[j:k\], y_m\[j:k\])
>
> 159
>
> ![](C:/Program Files/Typora/media/image1444.png){width="2.7777777777777776e-2in" height="2.7777777777777776e-2in"} alfa_m = cadena_a\[\'alpha\'\]\[:,i\].mean()
>
> beta_m = cadena_a\[\'beta\'\]\[:,i\].mean()
>
> plt.plot(x_range, alfa_m + beta_m\*x_range, c=\'k\', label=\'y = {:.2f} + {:.2f} \* x\'.format(alfa_m, beta_m))\
> plt.xlim(x_m.min()-1, x_m.max()+1)
>
> plt.ylim(y_m.min()-1, y_m.max()+1)
>
> j += N
>
> k += N
>
> plt.tight_layout()

![](C:/Program Files/Typora/media/image1448.png){width="5.643024934383202in" height="2.686349518810149in"}

> **4.3.1**　相关性与因果性
>
> 现在假设已经知道了当地的太阳辐射量，我们想要预测冬天家里 的燃气费。在这个问题中，太阳的辐射量是自变量***x***，燃气费是因变 量***y***。当然，我们完全可以将问题反过来，根据燃气费推算太阳辐射 量，一旦我们建立了一种线性关系（或者其他什么关系），我们就可 以根据***x***得出***y***，或者反过来这么做。我们称一个变量为自变量是因为 它的值不是从模型中预测出来的，而是作为模型的输入，相应的因变 量作为模型的输出。当我们说一个变量依赖于另一个变量的时候，这 其中的依赖关系是由模型决定的。
>
> 我们建立的并不是变量之间的因果关系，即并不是说***x***导致了***y***。 永远要记住这句话：相关性并不意味着因果关系。就这个话题多说一 点，我们可能根据家中的燃气费预测出太阳辐射量或者反过来根据太
>
> 160
>
> 阳辐射量预测出家中的燃气费。但是我们显然并不能通过调节燃气阀 门来控制太阳的辐射量。不过，太阳辐射量的高低是与燃气费的高低 相关的。
>
> 因此，需要强调一点，我们构建的统计模型是一回事，变量之间 的物理机制又是另外一回事。想要将相关性解释为因果关系，我们还 需要给问题的描述增加一些可信的物理机制，仅仅相关性还不够。有 一个网页，描述了一些有相关性但并没有因果关系的变
>
> 量：<http://www.tylervigen.com/spurious-correlations>[。](http://www.tylervigen.com/spurious-correlations。)
>
> 那么，相关性是否在确定因果关系时一点用都没有呢？不是。事 实上，如果能够进行一些精心设计的实验，那么相关性是能够用于支 撑因果关系的。举例来说，我们知道全球变暖与大气中二氧化碳的含 量是高度相关的。仅仅根据这个观测，我们无法得出结论是温度升高 导致的二氧化碳含量上升，还是二氧化碳含量的上升导致了温度升
>
> 高。更进一步，可能存在某种我们没考虑到的第3个变量，导致二氧 化碳含量和温度同时上升了。不过，我们可以设计一个实验，将玻璃 箱子中充满不同比例的二氧化碳含量，其中一个是正常空气中的含量 （约0.04%），其余箱子中二氧化碳含量逐渐增加，然后让这些箱子 接受一定时间的阳光照射（比如3个小时）。如果这么做之后能证实 二氧化碳含量较高的箱子温度也更高，那么就能得出二氧化碳的含量 导致温室效应的结论。同样的实验，我们可以反过来让相同二氧化碳 含量的箱子接受不同温度的照射，然后可以看到二氧化碳含量并不会 上升（至少空气中的二氧化碳含量不会上升）。事实上，更高的温度 会导致二氧化碳含量的上升，因为海洋中蕴含着二氧化碳，随着温度 上升，水中蕴含的二氧化碳含量会降低。简言之，全球正在变暖而我 们没有采取足够措施解决该问题。
>
> 161
>
> 这个例子中还有一点需要说明下，尽管太阳辐射量与燃气费相\
> 关，根据太阳辐射量可能预测出燃气费，不过如果考虑到一些其他变
>
> 量，这中间的关系就变得复杂了。我们一起来看一下，更高的太阳辐 射量意味着更多的能量传递到家里，部分能量被反射掉了，还有部分 转化成了热能，其中部分热量被房子吸收，还有部分散失到环境中
>
> 了。热能消失多少取决于许多因素，比如室外的温度、风力等。此\
> 外，我们还知道，燃气费也受到很多因素影响，比如国际上石油和燃 气的价格，燃气公司的成本/利润（及其贪婪程度），国家对燃气公\
> 司的管控等。而我们在尝试用两个变量和一条直线对所有这一切建\
> 模。因此，充分考虑问题的上下文是有必要的，而且有利于得出更合 理的解释，降低得出荒谬结论的风险，从而得到更好的预测，此外还 有可能为我们提供线索改进模型。
>
> 162
>
> **4.4**　多项式回归
>
> 接下来，我们将学习如何用线性回归拟合曲线。使用线性回归模 型去拟合曲线的一种做法是构建如下多项式：

![](C:/Program Files/Typora/media/image1455.png){width="3.4982589676290465in" height="0.20824365704286965in"}

> 如果留心的话，可以看到多项式中其实包含了一元线性回归模 型，只需要将*n*大于1的系数*β~n~*设为0即可，然后得到下式：

![](C:/Program Files/Typora/media/image1456.png){width="1.301434820647419in" height="0.20824365704286965in"}

> 多项式回归仍然是线性回归，这里线性的意思是指模型中的参数 是线性组合的，而不是指变量是线性变化的。现在我们先从一个简单 的多项式（抛物线）开始构建多项式回归模型：

![](C:/Program Files/Typora/media/image1457.png){width="1.894889545056868in" height="0.20824365704286965in"}

> 其中第3项控制的是曲率。我们选用安斯库姆四重奏的第2组作为 数据集，通过seaborn将其载入并画出来。
>
> ans = sns.load_dataset(\'anscombe\')
>
> x_2 = ans\[ans.dataset == \'II\'\]\[\'x\'\].values y_2 = ans\[ans.dataset == \'II\'\]\[\'y\'\].values x_2 = x_2 - x_2.mean()
>
> y_2 = y_2 - y_2.mean()
>
> plt.scatter(x_2, y_2)
>
> plt.xlabel(\'\$x\$\', fontsize=16)\
> plt.ylabel(\'\$y\$\', fontsize=16, rotation=0)
>
> 163

![](C:/Program Files/Typora/media/image1463.png){width="5.01833552055993in" height="3.425615704286964in"}

> with pm.Model() as model_poly:
>
> alpha = pm.Normal(\'alpha\', mu=0, sd=10)
>
> beta1 = pm.Normal(\'beta1\', mu=0, sd=1)
>
> beta2 = pm.Normal(\'beta2\', mu=0, sd=1)
>
> epsilon = pm.Uniform(\'epsilon\', lower=0, upper=10)
>
> mu = alpha + beta1 \* x_2 + beta2 \* x_2\*\*2
>
> y_pred = pm.Normal(\'y_pred\', mu=mu, sd=epsilon, observed=y_2)
>
> start = pm.find_MAP()
>
> step = pm.NUTS(scaling=start)
>
> trace_poly = pm.sample(3000, step=step, start=start) pm.traceplot(trace_poly)
>
> 164

![](C:/Program Files/Typora/media/image1469.png){width="5.643024934383202in" height="3.7588068678915136in"}

> 这里再次省略了一些检查和总结，只将结果画了出来，可以看到 一条非常好看的曲线完美拟合了数据，几乎没有误差。
>
> x_p = np.linspace(-6, 6)
>
> y_p = trace_poly\[\'alpha\'\].mean() + trace_poly\[\'beta1\'\].mean() \* x_p + t race_poly\[\'beta2\'\].mean() \* x_p\*\*2
>
> plt.scatter(x_2, y_2)
>
> plt.xlabel(\'\$x\$\', fontsize=16)
>
> plt.ylabel(\'\$y\$\', fontsize=16, rotation=0)
>
> plt.plot(x_p, y_p, c=\'r\')
>
> 165

![](C:/Program Files/Typora/media/image1475.png){width="5.643024934383202in" height="3.790042650918635in"}

> **4.4.1**　解释多项式回归的系数
>
> 多项式回归的问题之一在于参数的可解释性。如果我们想知道***y***\
> 相对于***x***的变化量，不能只看*β*~1~，因为*β*~2~和更高项的系数对其也有影 响。因此，系数*β*的值不再表示斜率。前面的例子中*β*~1~是正数，因而 曲线是以一个大于0的斜率开始的，不过由于*β*~2~是负数，因而随后曲
>
> 线的斜率开始下降。这看起来就好像有两股力量，一个使直线向上， 另一个使直线向下，二者相互作用的结果取决于***x***，当*x~i~*\<～11时，*β*~1~ 起决定作用，而当*x~i~*\>～11时，*β*~2~起决定作用。
>
> 如何解释参数不仅仅是个数学问题，因为我们可以通过仔细检查 和理解模型来解决这个问题。不过许多情况下，参数并不能根据我们 的领域知识转换成有意义的量，我们无法将其与细胞的新陈代谢速率 或者恒星释放的能量或者房间里的卧室数联系起来。它们只是些没有 物理意义的参数。这样一个模型或许对于预测很有用，不过对于理解 数据在底层是如何生成的并没有多大帮助。而且在实际中，超过2阶
>
> 166
>
> 或者3阶的多项式模型并没有多大用途，我们更倾向于使用一些其他 模型，这部分将在后面的章节中讨论。
>
> **4.4.2**　多项式回归**------**终极模型？\
> 我们知道，直线可以看作是当*β*~2~为0时的抛物线的子模型，还可
>
> 以看作是*β*~2~和*β*~3~都为0时的3次方模型的子模型。显然，抛物线模型也 可以看作是当*β*~3~为0时3次方模型的子模型......这意味着有一种算法可
>
> 以使用线性回归模型去拟合任意复杂的模型，我们先构建一个无限高 阶的多项式，然后将其中的大部分参数置零，直到我们得到对数据的 完美拟合。为了验证这个想法，你可以从简单的例子开始，用刚刚构 建的2次模型去拟合安斯库姆四重奏的第3个数据集。
>
> 完成练习之后，你会发现用2次模型去拟合直线是可能的。这个 例子看起来似乎验证了可以使用无限高阶多项式去拟合数据这一思 想，但是通常用多项式去拟合数据并不是最好的办法。为什么呢？因 为它并不关心具体使用的数据是怎么来的，从原理上讲，我们始终可 以找到一个多项式去完美拟合数据。如果一个模型完美拟合了当前的 数据，那么通常对于没有观测到的数据会表现得很糟糕，原因是现实 中的任意数据集都同时包含一些噪声和一些感兴趣的模式。一个过于 复杂的模型会拟合噪声，从而使得预测的结果变差，这称作过拟合， 一个在统计学和机器学习中常见的现象。越复杂的模型越容易导致过 拟合，因而分析数据时，需要确保模型没有产生过拟合，我们将在第 6章模型比较中详细讨论。
>
> 除了过拟合问题，我们通常更倾向于更容易理解的模型。从物理 意义上讲，线性模型的参数要比3次模型的参数更容易解释，即便3次 模型对数据拟合得更好。
>
> 167
>
> **4.5**　多元线性回归
>
> 前面的所有例子中，我们讨论的都是一个因变量和一个自变量的\
> 情况，不过在许多例子中，我们的模型可能包含多个自变量。例如：
>
> 红酒的口感（因变量）与酒的酸度、比重、酒精含量、甜度以及 硫酸盐含量（自变量）的关系；\
> 学生的平均成绩（因变量）与家庭收入、家到学校的距离、母亲 的受教育程度（自变量）的关系。
>
> 这种情况下，因变量可以这样建模：

![](C:/Program Files/Typora/media/image1481.png){width="3.3941437007874016in" height="0.16659448818897638in"}

> 注意这个式子与前面看到过的多项式回归的式子不一样，现在有 了多个变量而不再是一个变量的多次方。 用线性代数的表示方法可 以写成更简洁的形式：

![](C:/Program Files/Typora/media/image1482.png){width="1.0203248031496064in" height="0.16659448818897638in"}

> 其中，***β***是一个长度为*m*的系数向量，也就是说，自变量的个数\
> 为*m*。变量***X***是一个维度为*n*×*m*的矩阵，其中，*n*表示观测的样本 数，*m*表示自变量个数。有关线性代数，你可以阅读维基百科上关于 向量点乘和矩阵乘法的相关知识，可以用以下这种更简洁的形式描述 模型：

![](C:/Program Files/Typora/media/image1483.png){width="2.9256266404199476in" height="0.2707163167104112in"}

> 在一元线性回归模型中，我们希望找到一条直线来解释数据，而 在多元线性回归模型中，我们希望找到的是一个维度为*m*的超平面。 因此，多元线性回归模型本质上与一元线性回归模型是一样的，唯一
>
> 168
>
> 的区别是：现在***β***是一个向量而***X***是一个矩阵。 现在我们定义如下数 据：
>
> np.random.seed(314)
>
> N = 100
>
> alpha_real = 2.5
>
> beta_real = \[0.9, 1.5\]
>
> eps_real = np.random.normal(0, 0.5, size=N)
>
> X = np.array(\[np.random.normal(i, j, N) for i,j in zip(\[10, 2\], \[1, 1.5 \])\])
>
> X_mean = X.mean(axis=1, keepdims=True)
>
> X_centered = X - X_mean
>
> y = alpha_real + np.dot(beta_real, X) + eps_real
>
> 然后定义一个函数去画3个散点图，前两个表示的是自变量与因 变量的关系，最后一个表示的是两个自变量之间的关系。这只是个普
>
> 通的绘图函数，本章后面将会反复用到。
>
> def scatter_plot(x, y):
>
> plt.figure(figsize=(10, 10))
>
> for idx, x_i in enumerate(x):\
> plt.subplot(2, 2, idx+1)\
> plt.scatter(x_i, y)\
> plt.xlabel(\'\$x\_{}\$\'.format(idx), fontsize=16) plt.ylabel(\'\$y\$\', rotation=0, fontsize=16)
>
> plt.subplot(2, 2, idx+2)
>
> plt.scatter(x\[0\], x\[1\])\
> plt.xlabel(\'\$x\_{}\$\'.format(idx-1), fontsize=16)\
> plt.ylabel(\'\$x\_{}\$\'.format(idx), rotation=0, fontsize=16)
>
> 用前面刚刚定义的scatter_plot可以将我们的合成数据可视化
>
> 地表示出来。
>
> scatter_plot(X_centered, y)
>
> 169

![](C:/Program Files/Typora/media/image1498.png){width="5.643024934383202in" height="5.6642410323709536in"}

> 现在用PyMC3针对多变量线性回归问题定义出一个合适的模\
> 型，代码部分与单变量线性回归的代码基本一致，唯一的区别是：
>
> 参数beta是高斯分布，大小为2，每个参数都有一个斜率；\
> 这里使用pm.math.dot()来定义变量mu，也就是前面提到的线
>
> 性代数中的点乘（或者矩阵相乘）。
>
> 如果你对NumPy比较熟悉，那么你应该知道NumPy包含一个点\
> 乘函数，而且Python 3.5（以及NumPy 1.10）之后增加了一个新的操
>
> 作符@。不过这里我们使用的是PyMC3中的点乘函数（其实是Theano 中矩阵相乘的一个别名），因为变量beta在这里是一个Theano中的张
>
> 170
>
> 量而不是NumPy数组。
>
> with pm.Model() as model_mlr:
>
> alpha_tmp = pm.Normal(\'alpha_tmp\', mu=0, sd=10) beta = pm.Normal(\'beta\', mu=0, sd=1, shape=2)\
> epsilon = pm.HalfCauchy(\'epsilon\', 5)
>
> mu = alpha_tmp + pm.math.dot(beta, X_centered)
>
> alpha = pm.Deterministic(\'alpha\', alpha_tmp -- pm.math.dot(beta, X_m ean))
>
> y_pred = pm.Normal(\'y_pred\', mu=mu, sd=epsilon, observed=y)
>
> start = pm.find_MAP()
>
> step = pm.NUTS(scaling=start)
>
> trace_mlr = pm.sample(5000, step=step, start=start) varnames = \[\'alpha\', \'beta\', \'epsilon\'\]\
> pm.traceplot(trace_mlr, varnames)

![](C:/Program Files/Typora/media/image1506.png){width="5.643024934383202in" height="2.7175853018372704in"}

> 现在看一下推断出来的参数的总结，这样分析结果会更容易一 些。我们的模型表现如何呢？
>
> pm.df_summary(trace_mlr, varnames)

+---+--------+----+----------+---------+----------+
|   | > mean | sd | mc_error | hpd_2.5 | hpd_97.5 |
+---+--------+----+----------+---------+----------+

![](C:/Program Files/Typora/media/image1511.png){width="3.818897637795276e-2in" height="0.6316852580927385in"}

> alpha_0 2.07 0.50 5.96e-03 1.10 3.09

![](C:/Program Files/Typora/media/image1524.png){width="2.7777777777777776e-2in" height="2.7777777777777776e-2in"}

> 171

![](C:/Program Files/Typora/media/image1531.png){width="2.7777777777777776e-2in" height="2.7777777777777776e-2in"}

+-----------+------+------+----------+------+------+
| > beta_0  | 0.95 | 0.05 | 6.04e-04 | 0.85 | 1.05 |
+===========+======+======+==========+======+======+
| > beta_1  | 1.48 | 0.03 | 4.30e-04 | 1.43 | 1.54 |
+-----------+------+------+----------+------+------+
| > epsilon | 0.47 | 0.03 | 4.63e-04 | 0.41 | 0.54 |
+-----------+------+------+----------+------+------+

> 可以看到，模型能够重现正确的值（对比生成数据用的值）。接 下来，我们将重点关注在分析多变量线性回归模型中需要注意的点， 特别是对斜率的解释。这里需要特别提醒的是：每个参数只有在整体 考虑了其他参数的情况下才有意义。
>
> **4.5.1**　混淆变量和多余变量
>
> 设想这样一种情况：有一个变量***z***与预测变量***x***相关，同时还与另 一个被预测的变量***y***相关。假设***z***对***x***和***y***都有影响，例如，***z***可以是工 业革命（一个相当复杂的变量），***x***是海盗的数量，***y***是二氧化碳浓 度。如果在分析中将***z***去掉，我们会得出结论：***x***与***y***之间有着完美的 线性相关性，我们甚至可以通过***x***预测***y***。不过，如果我们关注的重点 是如何缓解全球变暖问题，那么可能完全没搞清到底发生了什么以及 其内在机制是什么。记住，前面已经讨论了相关性并不意味着因果关 系，原因是我们可能在分析过程中忽略了变量***z***。这种情况下，***z***称作 混淆变量，或者是混淆因素。问题是在很多情况下，***z***很容易被忽 略。也许是因为我们压根没有测量***z***，或者是因为没有包含在传给我 们的数据集中，又或者是因为我们压根没想到它可能与我们的问题有 联系。没有考虑到混淆变量可能会导致我们的分析得出奇怪的相关 性，在解释数据和做预测（有时候我们不关心内在的机制）的时候， 这可能是个问题。理解底层的机制有利于将学到的东西迁移到新的场
>
> 172
>
> 景中，相反，盲目的预测很难迁移。例如，帆布鞋产量可以作为一个 国家经济实力的易测指标，不过对于生产链不同或者文化背景不同的 国家而言，这可能是个糟糕的指标。
>
> 我们将使用合成数据来探索混淆变量的问题。下面的代码中模拟 了一个混淆变量*x*~1~，注意这个变量是如何影响*x*~2~和***y***的。
>
> N = 100
>
> x_1 = np.random.normal(size=N)
>
> x_2 = x_1 + np.random.normal(size=N, scale=1)
>
> y = x_1 + np.random.normal(size=N)
>
> X = np.vstack((x_1, x_2))
>
> 根据生成数据的方式，可以看出这些变量已经是中心化了。因\
> 此，不需要再进一步对数据进行中心化处理来加速推断过程了。事实
>
> 上这个例子中的数据已经是标准化的了。
>
> scatter_plot(X, y)
>
> 173

![](C:/Program Files/Typora/media/image1560.png){width="5.643024934383202in" height="5.560119203849519in"}

> 然后用PyMC3创建模型并从中采样，现在，你对这个模型应该 已经相当熟悉了。
>
> with pm.Model() as model_red:
>
> alpha = pm.Normal(\'alpha\', mu=0, sd=10)\
> beta = pm.Normal(\'beta\', mu=0, sd=10, shape=2) epsilon = pm.HalfCauchy(\'epsilon\', 5)
>
> mu = alpha + pm.math.dot(beta, X)
>
> y_pred = pm.Normal(\'y_pred\', mu=mu, sd=epsilon, observed=y)
>
> start = pm.find_MAP()
>
> step = pm.NUTS(scaling=start)
>
> trace_red = pm.sample(5000, step=step, start=start)
>
> 174

![](C:/Program Files/Typora/media/image1566.png){width="5.643024934383202in" height="2.7175853018372704in"}

> 现在用pandas中的dataframe将结果的总结打印出来，重点关注 beta参数的均值。
>
> pm.df_summary(trace_red)

+-----------+----------+------+----------+---------+----------+
|           | > mean   | sd   | mc_error | hpd_2.5 | hpd_97.5 |
+===========+==========+======+==========+=========+==========+
| > alpha   | 0.01     | 0.10 | 1.59e-03 | -0.20   | 0.19     |
+-----------+----------+------+----------+---------+----------+
| > beta_0  | 0.96     | 0.16 | 3.13e-03 | 0.67    | 1.29     |
+-----------+----------+------+----------+---------+----------+
| > beta_1  | -0.05    | 0.10 | 2.06e-03 | -0.24   | 0.14     |
+-----------+----------+------+----------+---------+----------+
| > epsilon | 1.00e+00 | 0.07 | 1.15e-03 | 0.87    | 1.15     |
+-----------+----------+------+----------+---------+----------+

> 可以看到*β*~1~接近0，这意味着*x*~2~对***y***来说几乎没有作用。因为通过 检查合成的数据已经知道了起重要作用的是变量*x*~1~。现在需要先做一 些测试，重跑两次模型，其中一次只用*x*~1~，另一次只用*x*~2~。如果你查
>
> 看下代码的话，可以看到有几行是被注释了的，你或许会用得上。请
>
> 175
>
> 问，这3种情况下beta系数的均值有多大区别？如果你做了实验，就\
> 会注意到，对于*β*~2~的值，多元线性回归模型中得到的要比一元线性回 归模型得到的更低。换句话说，当*x*~1~加入到模型中之后，*x*~2~对模型的
>
> 解释性降低了（甚至降为0）。
>
> **4.5.2**　多重共线性或相关性太高
>
> 前面的例子中，我们看到了多元线性回归模型是如何处理冗余变 量的，同时还了解了混淆变量的重要性。接下来我们沿着前面的例子 继续深入学习当两个变量高度相关时会发生什么。为了研究这个问题 及其对推断的影响，我们使用和前面一样的合成数据和模型，不过通 过减小根据*x*~1~生成*x*~2~时的随机噪声，增加了*x*~1~和*x*~2~之间的相关性：
>
> x_2 = x_1 + np.random.normal(size=N, scale=0.01)
>
> 上面这行代码相当于对*x*~1~增加了一个很小的扰动，因而得到的两
>
> 个变量可以看作是一样的，然后你可以修改下数据的尺度并引入少量 的极限值，不过，当下我们尽量保持简单。生成新的数据后，可以用 散点图查看下数据是什么样的；你应该可以看到*x*~1~与*x*~2~之间的关系是
>
> 一条斜率接近于1的直线。运行模型并对结果进行检查。在本书附带 的代码中，你可以看到有几行代码可以根据beta系数画2D的核密度估 计图。你应该可以看到与下面类似的图。
>
> 176

![](C:/Program Files/Typora/media/image1577.png){width="5.643024934383202in" height="3.883753280839895in"}

> beta参数的HPD区间相当广，与先验几乎一样。

![](C:/Program Files/Typora/media/image1578.png){width="5.643024934383202in" height="3.8316918197725283in"}

> 可以看到，beta的后验是一条很窄的斜对角线。当其中一个beta\
> 参数增加时，另一个一定下降。两个参数非常相关，这是模型和数据
>
> 177
>
> 共同作用的结果。根据我们的模型，均值*μ*有如下形式：

![](C:/Program Files/Typora/media/image1580.png){width="1.6554254155730534in" height="0.16659448818897638in"}

> 假设*x*~1~和*x*~2~不只是近似相同，而是完全一样的，那么可以将模型
>
> 改写成如下形式：

![](C:/Program Files/Typora/media/image1581.png){width="1.60336832895888in" height="0.187419072615923in"}

> 可以看到，对*μ*有影响的是*β*~1~与*β*~2~的和而不是二者单独的值，因\
> 而模型是不确定的（或者说，数据并不能决定*β*~1~和*β*~2~的值）。在我们
>
> 的例子中，beta并不能在区间 \[-∞, ∞\]内自由移动，原因有两个：其\
> 一，两个变量几乎是相同的，不过并非完全一样；其二，更重要的是 beta系数的可能取值受到先验的限制。
>
> 这个例子中有几点需要注意。首先，后验只是根据模型和数据得 出的逻辑上的结果，因而得出一个分布很广的beta分布并没有错，事 实就是这样子；第2点是，我们可以依据该模型做预测，可以尝试做 一些后验预测检查，该模型预测得到的值与数据分布是一致的，也就 是说模型对数据拟合得很好；第3点是，对于理解问题而言这可能不 是一个很好的模型，更好的做法是从模型中去掉一个参数，这样模型 的预测能力与以前一样，但是更容易解释。
>
> 在任何真实的数据集中，相关性在某种程度上是普遍存在的。那 么两个或多个变量之间相关性多高时会导致问题呢？事实上并没有确 切的数值。我们可以在运行贝叶斯模型之前，构建一个相关性矩阵， 对其中相关性较高（比如说高于0.9）的变量进行检查。不过，这种
>
> 做法的问题是：根据相关性矩阵观察到的成对变量之间的相关性并不 太重要，重要的是在某个具体模型中变量的相关性。前面已经看到\
> 了，不同变量在单独情况下的表现与在模型中放一起的表现是不同
>
> 178
>
> 的。在多元回归模型中，两个或多个变量之间的相关性可能会受到其 他变量的影响，从而使得他们之间的相关性降低或者升高。通常，建 议在迭代式构建模型的同时加入一些诊断（比如检查自相关性和后
>
> 验），这有利于发现问题和理解模型与数据。
>
> 如果发现了高度相关的变量应该怎么做呢？
>
> 如果相关性非常高，我们可以从分析中将其中一个变量去掉。如\
> 果两个变量的信息都差不多，具体去掉哪个并不重要，可以视方\
> 便程度（比如去掉最不常见的或者最难解释或测量的变量）。
>
> 另外一种可行的做法是构建一个新的变量对冗余变量求均值。更\
> 高级的做法是使用一些降维算法，如主成分分析法（PCA）。不\
> 过PCA的一个问题是得到的结果变量是原始变量的线性组合，通\
> 常会对结果的可解释性造成模糊。
>
> 还有一种办法是给变量可能的取值设置一个较强的先验。在第6\
> 章中我们会简要讨论如何选择这类先验（通常称作正则先验）。
>
> **4.5.3**　隐藏的有效变量
>
> 有一种情况与前面见过的类似，其中某个变量与预测变量正相关 而另外一个与预测变量负相关。这里先构建一些玩具数据来说明。
>
> N = 100\
> r = 0.8\
> x_0 = np.random.normal(size=N)
>
> x_1 = np.random.normal(loc=x_0 \* r, scale=(1 -- r \*\* 2) \*\* 0.5) y = np.random.normal(loc=x_0 - x_1)
>
> X = np.vstack((x_0, x_1))
>
> scatter_plot(X, y)
>
> 179

![](C:/Program Files/Typora/media/image1591.png){width="5.643024934383202in" height="5.65382874015748in"}

> with pm.Model() as model_ma:
>
> alpha = pm.Normal(\'alpha\', mu=0, sd=10)\
> beta = pm.Normal(\'beta\', mu=0, sd=10, shape=2) epsilon = pm.HalfCauchy(\'epsilon\', 5)
>
> mu = alpha + pm.math.dot(beta, X)
>
> y_pred = pm.Normal(\'y_pred\', mu=mu, sd=epsilon, observed=y)
>
> start = pm.find_MAP()
>
> step = pm.NUTS(scaling=start)
>
> trace_ma = pm.sample(5000, step=step, start=start) pm.traceplot(trace_ma)
>
> 180

![](C:/Program Files/Typora/media/image1597.png){width="5.643024934383202in" height="2.7592344706911636in"}

> pm.forestplot(trace_ma, varnames=\[\'beta\'\])

![](C:/Program Files/Typora/media/image1602.png){width="5.643024934383202in" height="3.98787510936133in"}

> 从后验可以看出，beta的值接近1和-1。也就是说，*x*~1~与***y***正相关\
> 而*x*~2~与***y***负相关。现在重新分析，不过（也许你已经猜到了）这一次
>
> 我们对每个单独的变量进行分析。
>
> 对于单个的变量，可以看到***β***接近0，也就是说每个单独变量***x***都 不足以预测***y***。相反，如果我们将***x***组合在一起后就可以预测***y***。当*x*~1~
>
> 181
>
> 增加时*x*~2~也增加，而当*x*~2~增加时***y***也降低，因此如果单独看变量*x*~1~而忽 略*x*~2~的话，我们会看到当*x*~1~增加的时候，***y***几乎不增加，而当*x*~2~增加
>
> 时，***y***几乎不降低。因变量之间具有相关性，每个因变量都有反作\
> 用，因而忽略其中任何一个都会造成对变量影响力的低估。
>
> **4.5.4**　增加相互作用
>
> 目前为止，所有多元回归模型的定义中，在其他预测变量固定的 条件下，*x*~1~的变化都会（隐式地）带来***y***的稳定变化。不过这显然并 非一定的，有可能改变*x*~2~之后，原来***y***与*x*~1~之间的关系发生了改变。一
>
> 个经典的例子是药物之间的相互作用，例如，在没有使用药物B（或 者药物B的剂量较低）时，增加药物A的剂量有正向影响，而当增加\
> 药物B的剂量时，药物A反而有负向（甚至致命的）影响。
>
> 目前见过的所有例子中，因变量对于预测变量的作用都是叠加\
> 的。我们做的只是增加变量（每个变量乘以一个系数）。如果我们希
>
> 望捕捉到变量的效果，就像前面的药物例子一样，我们需要给模型增 加一项非叠加的量，比如，变量之间的乘积：

![](C:/Program Files/Typora/media/image1604.png){width="2.342584208223972in" height="0.16659448818897638in"}

> 注意这里系数*β*~3~乘的是*x*~1~和*x*~2~的乘积，这个非叠加项只是一个用
>
> 来说明统计学中的变量之间相互作用的例子，因为它衡量了变量（在 前面的例子中是药物）之间的相关性。对相关性建模的表达式有很多 种，相乘只是其中一个比较常用的。
>
> 在多元线性回归模型中，如果没有变量之间的乘积，我们得到的 是一个超平面，也就是说，一个平坦的超曲面，加入乘积之后，该超 曲面会变得弯曲。
>
> 182
>
> **4.6**　**glm**模块
>
> 在统计学和机器学习中，线性模型有着广泛的应用，PyMC3有 一个叫做glm的模块，也就是通用线性模块（generalized linear model）。一个线性模型可以这么表示：
>
> with Model() as model:\
> glm.glm(\'y \~ x\', data) trace = sample(2000)
>
> 代码中的第2行会默认给截距和斜率添加均匀先验，同时给因变 量添加一个高斯似然。如果你只是想跑一个默认的线性回归问题，这
>
> 样就可以了，需要注意这个模型的MAP在本质上跟使用普通的（频\
> 率学派中的）最小二乘法得到的结果是一样的。如果需要，你可以使 用glm模块并修改其中的先验和似然。如果你对R语言的语法不熟，\
> 需要说明一下，这里y～x的意思是用glm模块中的线性函数定义了输 出变量*y*。在glm模块中还包含一个函数用来画后验预测的图。
>
> 183
>
> **4.7**　总结
>
> 线性回归是统计学和机器学习中最常用的模型之一，同时也是构 建其他更复杂模型的基石，其应用相当广泛，不同领域的人对同一个 东西有不同的称呼，因此本章一开始先介绍了统计学和机器学习中的 一些常见词汇。然后深入研究了线性模型的核心，用一个表达式衔接 了输入变量与输出变量。本章用到的是高斯分布和t分布作为因变量
>
> 的似然，后面我们会将该模型扩展到其他分布。我们还处理了一些计 算方面的问题，以及如何通过中心化和（或）标准化数据解决问题， 还有幸见识到了NUTS相比Metropolis的优势。
>
> 在一元线性模型中，我们应用了多层模型，同时探讨了如何使用 多项式回归拟合曲线以及这类模型的一些问题。此外，还讨论了如何 用多个参数构建线性回归模型，解释线性模型中的一些注意事项，下 一章我们将学习如何扩展线性回归模型并用于分类问题。
>
> 184
>
> **4.8**　深入阅读
>
> 《Statistical Rethinking》中的第4章和第5章。
>
> 《Doing Bayesian Data Analysis, Second Edition》中的第17章和第 18章。
>
> 《An Introduction to Statistical Learning》中的第4章。
>
> 《Bayesian Data Analysis, Third Edition》中的第14章～第17章。\
> 《Machine Learning: A probabilistic Probabilistic Perspective》中的 第7章。
>
> 《Data Analysis Using Regression and Multilevel/Hierarchical\
> Models》一书。
>
> 185
>
> **4.9**　练习
>
> （1）选一个你觉得有意思的数据集并用一元线性回归去拟合。 用不同的方法重跑一遍，重新画图并计算出皮尔逊相关系数。如果没
>
> 有合适的数据，可以上网找一下，网址为：<http://data.worldbank.org/>\
> 或者<http://www.stat.ufl.edu/~winner/datasets.html>[。](http://www.stat.ufl.edu/~winner/datasets.html。)
>
> （2）阅读并运行PyMC3官方文档中的例子[https://pymc-](https://pymc-devs.github.io/pymc3/notebooks/LKJ.html。) [devs.github.io/pymc3/notebooks/LKJ.html。](https://pymc-devs.github.io/pymc3/notebooks/LKJ.html。)
>
> （3）对于没有进行池化操作的模型，尝试修改beta先验中sd的\
> 值（比如1和100），观察每组数据中预估斜率值的变化，哪组数据更
>
> 容易受到该变化的影响？
>
> （4）查看本书附带的代码model_t2（以及附带的数据），尝试\
> 更换nu的先验，比如非漂移的指数先验和伽马先验（代码中已经注释
>
> 掉了）。画出先验并确保你理解了（一个简单的做法是将似然注释掉 并运行traceplot函数）。
>
> （5）降低ADVI的迭代次数（目前是100000），例如降到\
> 10000，对NUTS每秒迭代次数有什么影响？观察traceplot的返回结
>
> 果，看看对采样值有什么影响。将ADVI换成前面其他模型用过的\
> find_MAP()，能否观察到这样做的优势？
>
> （6）运行model_mlr的例子，不过这次不对数据进行中心化处 理。比较两种情况下alpha参数的不确定性。你能对结果做出些解释 吗？提示：回忆一下参数alpha的定义（截距）。
>
> （7）阅读并运行以下PyMC3文档中的记事本：
>
> 186
>
> <https://pymc-devs.github.io/pymc3/notebooks/GLM-linear.html>\
> <https://pymc-devs.github.io/pymc3/notebooks/GLM-robust.html>\
> <https://pymc-devs.github.io/pymc3/notebooks/GLM-hierarchical.html>
>
> （8）运行多元线性回归模型部分的练习。
>
> \[1\] 　Machine Learning: a Probabilistic Perspective 一书的作者。------ 译者注
>
> \[2\] 　Wishart distribution虽然是协方差矩阵的共轭先验，在MCMC中 它的混合度很差而且很难从中准确采样，我们推荐使用LKJCorr 或\
> LKJCholeskyCov作为协方差矩阵的先验。------译者注
>
> 187