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

# 附录 J：贝叶斯优化

【摘要】贝叶斯数据分析方法为使用概率论处理所有观测、模型参数和模型结构中的不确定性提供了一种强大的方法。概率编程语言使指定和拟合贝叶斯模型变得更加容易，但这仍然给我们留下了许多关于构建、评估和使用这些模型的选择，以及许多计算方面的挑战。使用贝叶斯推断解决实际问题不仅需要统计技能、学科知识和编程，还需要在数据分析过程中做出决策。所有这些方面都可以理解为应用贝叶斯统计的复杂工作流程的组成部分。除了推断以外，工作流还包括迭代模型构建、模型检查、计算问题的验证及故障排除、模型理解和模型比较。本文在几个示例背景下，回顾了工作流的各个方面。需要提醒读者：在实际工作中，应当为任何给定的问题拟合多个模型，即使最终可能只有非常有限的子集能够与结论相关。
【原文见文末】

<style>p{text-indent:2em;2}</style>

## 1 概述

### 1.1.从贝叶斯推断到贝叶斯工作流 

如果数理统计是『关于如何应用统计学』的理论，那么任何对贝叶斯方法的认真讨论都需要明确其在实践中的使用方式。特别是，我们需要明确地**将贝叶斯推断与贝叶斯数据分析的概念分开，特别是将其与完整的贝叶斯工作流程分开**。

贝叶斯推断只是条件概率或概率密度 $p(θ|y) ∝p(θ)p(y|θ)$ 的形式化和计算，可视为一种计算工具。而贝叶斯工作流则包括**模型构建**、**统计推断**和**模型检查/改进** 三个步骤，以及不同**模型之间的比较**（ 这种比较不仅仅是为了模型选择或模型平均，更是为了深入理解这些模型）。例如：为何某些模型无法预测数据的某些方面？ 或者一些重要参数的不确定性估计为什么会因模型而不同？ 即使我们有一个非常喜欢的模型，将其与更简单和更复杂模型推断进行比较，对于理解模型也是很有用的。图 1 提供了一个贝叶斯工作流程的概要。扩展的贝叶斯工作流应当还包括（在收集数据和测量数据前的）试验设计以及推断完成后的决策制定，不过本文重点在于对数据进行建模。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211210110642-1649.webp)

> **图 1：在贝叶斯工作流中考虑的主要步骤。** 括号中的数字对应本文中讨论相应步骤的章节。该图旨在显示单个贝叶斯分析可能经历的步骤和路径，当然同时应该理解任何特定分析都不太可能涉及所有这些步骤。我们研究工作流程的目标之一是掌握这些想法是如何组合在一起的，以便更系统地应用它们。

在典型贝叶斯工作流程中，我们最终会拟合出一系列模型，其中一些模型可能是糟糕的选择（ 原因包括：数据拟合不佳、缺乏与实际理论或实际目标的联系、先验太弱/太强/不合适、编程错误等）；另外一些模型有用但可能存在缺陷（例如： 无依据地随意选择混杂因素或变量进行回归、只选择了捕获部分而非全部功能关系的参数形式等 ）；另外一些的最终结果可能值得采纳。

在实践中，必须认识到：错误模型、有缺陷模型都会是拟合有用模型无法避免的前驱步骤。理解这一点有助于帮助我们改变应用和配置统计方法的方式。

### 1.2 为什么需要贝叶斯工作流？

我们需要贝叶斯工作流程的原因很多，而不仅仅是贝叶斯推断。

- 该方法本身就是不断试错的过程。计算可能是一项挑战，但是为了获得可信的推断，我们还是不得不经常完成各种步骤，包括：拟合更简单的或替代的模型、做不太准确但速度更快的近似计算、摸索拟合过程等。 

- 数据的动态性需要不断调整模型。在困难问题中，通常无法提前知道想要拟合什么样的模型。即便在极少数情况下提前选择了可接受的模型，也通常会在增加数据时或需要对数据进行更详细解释时调整它。 

- 模型对比和理解的需要。即使数据是静态的，并且我们知道要拟合的模型，拟合也没有问题，我们也仍然希望理解模型及其与数据的关系，而这种理解往往必须通过模型比较才能实现。 

- 模型自身不确定性的需要。有时不同的模型会得出不同的结论，但没有一个是明显有利的。在这种情况下，展示多个模型有助于说明模型选择的不确定性。

### 1.3. 工作流及其与统计理论和实践的关系

工作流在不同上下文中具有不同的含义。就本文而言，工作流可能比 『示例』 更通用一些，但又不如 『方法』 那样足够精确（ 见图 2 ）。实际上，我们受到了计算中有关工作流想法的影响。这些想法包括一些统计上的进展（例如 tidyverse ），有些进展可能不是特别贝叶斯，但具有类似的体验式学习的感觉（Wickham 和 Groelmund，2017 年）。机器学习的许多最新进展都具有类似即插即用的感觉：它们易于使用、易于试验，并且让用户有一种健康的感觉，即拟合模型是一种从数据中学习某些东西的方式，而该方式可能无需表达对某种概率模型或统计假设的承诺。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211210114936-bcfd.webp)

> 图2：统计方法学的元工作流程，代表了新想法首先出现在实例中，然后被形式化为案例研究，编制成工作流程，被赋予一般通用实现算法，并最终形成理论的主题。

图 2 显示了我们对统计方法学发展的看法，即 『 **实例 --> 案例研究 --> 工作流 --> 方法 --> 理论** 』 的编码过程。并非所有方法都能达到最终的数学抽象级别，但纵观统计学历史，我们已经看到很多在特定实例、风格化案例、新问题中的工作流等背景下开发的方法，其中很多在可能的情况下，都做了形式化、编码和理论研究。

理解图 2 的一种方法是沿着这条路径从左向右移动地理解统计史上的一些重要思想。有许多想法最初是作为黑客或与统计相关的工具开始的，最终被形式化为方法并进入统计学的核心。例如：

- 多层建模扩展了模型以便将关于先验的推断纳入到完全贝叶斯框架中，并被形式化为对先验分布的经验贝叶斯估计。

- 探索性数据分析可以理解为一种预测模型检查的形式（Gelman，2003）。

- 诸如 Lasso (Tibshirani, 1996) 和 Horseshoe (Piironen et al., 2020) 等正则化方法已经取代了回归任务中的临时变量选择工具。

- 高斯过程（O’Hagan，1978；Rasumussen and Williams，2006 ）等非参数模型可以被认为是核平滑等程序的贝叶斯替代品。

在上述案例还有许多其他案例中，统计方法论的框架已经通过使方法更加模块化和更有用的方式，被扩展到现有方法。

『工作流』 一词已逐渐在统计和数据科学中使用；参见例如 Liu 等，2005；Lins等，2008； Long，2009；Turner and Lambert，2015。工作流的相关思想在软件开发和其他信息学领域中广为流传；最近针对从业者的讨论包括 Wilson 等人，2014 年和 2017 年。

应用统计（不仅仅是贝叶斯统计）变得越来越计算化和算法化，这将工作流置于统计实践的中心（ 例如，参见 Grolemund and Wickham，2017；Bryan，2017 ； Yu and Kumbier，2020 ），以及应用领域（ 例如，Lee 等 2019 年讨论了心理学研究中的建模工作流程）。

 『贝叶斯工作流』 已被 Savage (2016)、Gabry (2019) 和 Betancourt (2020a) 等人表达为一个一般概念。。 Gelman (2011) 讨论了贝叶斯工作流的几个单独组件，但没有以连贯的方式进行讨论。此外，针对特定问题开发了贝叶斯工作流，如 Shi 和 Stevens (2008) 以及 Chiu 等人。 (2017)。

在本文中，我们介绍了贝叶斯工作流的几个方面，希望这些方面最终能够进入日常实践和自动化软件。我们使用概率编程语言 Stan（Carpenter et al.，2017，Stan Development Team，2020）建立了大部分工作流程，但类似的想法也适用于其他计算环境。

### 1.4 组织贝叶斯工作流的诸多方面

统计工作流的教科书表达通常是线性的，不同的路径对应不同的问题情况。例如：医学临床试验通常从样本量计算和分析计划开始，然后是数据收集、清理和统计分析，最后是报告 $p$ 值和置信区间。经济学中的观测性研究可能从探索性数据分析开始，然后为变量转换的选择提供信息，然后是一组回归分析，然后是一系列替代性声明和稳健性研究。

本文中讨论的统计工作流程比教科书和研究文章中介绍的普通数据分析工作流程更加复杂。额外的复杂性来自以下几个地方，并且在更高级别的工作流中有许多子工作流：

1. 复杂模型的拟合计算本身就很困难，需要进行相当数量的实验来解决从后验分布的计算、逼近或预测模拟问题，同时检查计算算法是否按照预期执行.

2. 对于复杂的问题，我们通常会想到一个更复杂的通用模型（例如，包括相关性、层次结构和随时间变化的参数等特征），因此我们从已知缺少一些重要特征的模型开始，希望它在计算上更容易，并理解我们将逐渐添加特征。

3. 与此相关，我们经常考虑数据不固定的问题，要么是因为数据收集正在进行中，要么是因为我们能从相关数据集中抽取数据，例如民意分析中的新调查或药物试验中其他实验的数据。增加新的数据往往需要模型扩展，以允许参数变化或扩展函数形式，例如线性模型一开始可能很适合，但随着新条件下数据的增加而崩溃。

4. 除了拟合和扩展的所有挑战之外，通常可以通过与替代模型下的推断进行比较来最好地理解模型。因此，我们的工作流程包括用于理解和比较能够拟合相同数据的多个模型的工具。

统计是关于不确定性的。除了数据和模型参数中常见的不确定性之外，我们常常不确定是否正确地拟合了模型，不确定如何最好地建立和扩展模型，以及对它们的解释不确定。一旦我们超越了简单的预先分配的设计和分析，工作流程就会变得混乱，我们的重点在数据探索、实质性理论、计算和结果解释之间来回移动。因此，任何组织工作流程步骤的尝试都会过于简单，而且许多子工作流程非常复杂，足以证明它们自己的文章或书籍章节是合理的。

我们讨论了工作流的许多方面，但实际考虑因素 —— 尤其是可用时间、计算资源和错误惩罚的严重程度 —— 可能会迫使从业者走捷径。这样的捷径会使结果的解释变得更加困难，但我们必须意识到它们会被采用，并且根本不拟合模型可能比使用近似计算拟合它更糟糕（其中近似可以定义为不给出准确的总结）即使在无限计算时间的限制下的后验分布）。因此，我们描述统计工作流程的目的也是明确地将各种捷径理解为完整工作流程的近似值，让从业者就如何投资有限的时间和精力做出更明智的选择。

### 1.5.文章的目的和结构 
应用统计学中有各种各样的隐性知识，并不总能将其纳入已发表的论文和教科书。本文旨在公开其中一些想法，以改进应用贝叶斯分析并为理论、方法和软件的未来发展提出建议。

我们的目标受众是

 (a) 应用贝叶斯统计的从业者，尤其是 Stan 等概率编程语言的用户。
 
 (b) 面向这些用户的方法和软件的开发人员。
 
 我们还针对贝叶斯理论和方法的研究人员，因为我们认为工作流的许多方面都没有得到充分研究。

在本文的其余部分，我们将更缓慢地介绍贝叶斯工作流程的各个方面，如图 1 所示，从拟合模型之前要完成的步骤（第 2 节）开始，通过拟合、调试和评估模型（第 3-6 节)，然后修改模型（第 7 节）并理解和比较一系列模型（第 8 节）。

然后，第 10 节和第 11 节通过两个例子来完成这些步骤，一个是我们将特征逐步添加到高尔夫推杆模型中，另一个是我们通过一系列调查来解决拟合行星运动的简单模型。第一个示例展示了新数据如何推动模型改进，还说明了扩展模型时出现的一些意外挑战。第二个示例展示了计算中的挑战可以指向建模困难的方式。这两个例子并没有说明贝叶斯工作流的所有方面，但至少表明，将贝叶斯模型开发的许多方面系统化可能会有好处。我们在第 12 节中总结了一些一般性讨论以及对工作流程潜在批评的回应。

## 2 在拟合模型之前

### 2.1 选择初始模型

几乎所有分析的出发点都是从适应先前所做的工作开始，使用来自教科书、案例研究或已应用于类似问题的已发表论文的模型（有些类似软件工程中设计模式的概念）。使用从之前的一些分析中获取的模型并对其进行更改可以被视为有效数据分析的捷径。通过分析模型模板的结果，我们可以知道模型向哪个方向风化或简化更有空间。模板可以节省模型构建和计算的时间，同时我们也要考虑到那些需要理解结果的人可能存在认知负担。快捷方式对人类和计算机都很重要，快捷方式有助于解释为什么典型的工作流程是迭代的（更多信息请参见第 12.2 节）。类似地，如果我们要尝试对计算机进行编程以自动执行数据分析，则它必须通过某种算法来构建模型，而这种算法的构建块将代表某种模板。尽管 “食谱分析” 有负面含义，但我们认为模板可以用作更精细分析的起点和比较点。我们应该认识到理论的动态性，科学理论的发展过程与统计模型的发展过程并不相同（Navarro，2020）。

有时，我们的工作流程从一个简单的模型开始，目的是稍后添加特征（对不同的参数进行建模，包括测量误差、相关性等）。其他一些时候，我们从一个大模型开始，并打算在接下来的步骤中将其剥离，努力找到一些简单易懂但仍能捕获数据关键特征的东西。有时我们甚至会考虑多种完全不同的方法来对相同的数据进行建模，因此有多个起点可供选择。

### 2.2 模块化构建

贝叶斯模型是由模块构建的，这些模块通常可以被视为在必要时被替换的占位符。例如，我们用正态分布对数据建模，然后用长尾分布或混合分布替换它；我们将潜在回归函数建模为线性并将其替换为非线性样条或高斯过程；我们可以将一组观测值视为精确的，然后添加一个测量误差模型；我们可以从弱先验开始，然后当发现后验推断包含不切实际的参数值时，将其增强。将模块视为占位符可以减轻模型构建过程的一些压力，因为您可以随时返回并根据需要概括或添加信息。

模块化构造的想法与统计文献中的长期传统背道而驰，在传统中，整个模型都被命名，并且每次对现有模型进行细微更改时都会给出一个新名称。给模型的模块（而不是整个模块）命名，可以更容易地看到不同模型之间的联系，并使它们适应给定项目的特定要求。

### 2.3 缩放和转换参数

我们希望我们的参数能够出于实际和道德原因进行解释。这导致需要它们在自然尺度上并将它们建模为独立的，如果可能的话，或者具有可解释的依赖结构，因为这有助于使用信息先验（Gelman，2004）。它还可以帮助分离标度，以便未知参数无标度。例如，在药理学中的一个问题（Weber 等人，2018 年）中，我们有一个参数，我们预计该参数在测量范围内的值约为 50；遵循缩放的原则，我们可能会在log(θ/50)上建立一个模型，这样0对应一个可解释的值（原始尺度上的50），而0.1的差异，例如，在对数尺度上对应于增加或减少约 10%。这种转换不仅仅是为了便于解释；它还以一种为有效的分层建模做好准备的方式设置参数。当我们构建更大的模型时，例如通过合并来自其他患者组或其他药物的数据，允许参数因组而异（如我们在第 12.5 节中讨论的那样）是有意义的，并且部分池化可以在规模上更有效——自由参数。例如，毒理学模型需要研究中每个人的肝脏体积。我们没有直接将分层模型拟合到这些体积中，而是将每个人的肝脏建模为体重的比例；我们预计这些无标度因素在不同人之间的差异较小，因此与建模绝对体积的结果相比，拟合模型可以进行更多的部分池化。缩放变换是一种促进有效分层建模的分解

In many cases we can put parameters roughly on unit scale by using logarithmic or logit transformations or by standardizing, subtracting a center and dividing by a scale. If the center and scale are themselves computed from the data, as we do for default priors in regression coefficients in rstanarm (Gabry et al., 2020a), we can consider this as an approximation to a hierarchical model in which the center and scale are hyperparameters that are estimated from the data.

在许多情况下，我们可以通过使用对数或 logit 变换或通过标准化、减去中心并除以尺度来粗略地将参数置于单位尺度上。如果中心和尺度本身是根据数据计算出来的，就像我们对 rstanarm 中回归系数的默认先验所做的那样（Gabry 等人，2020a），我们可以将其视为分层模型的近似，其中中心和尺度是从数据中估计的超参数。

More complicated transformations can also serve the purpose of making parameters more interpretable and thus facilitating the use of prior information; Riebler et al. (2016) give an example for a class of spatial correlation models, and Simpson et al. (2017) consider this idea more generally

更复杂的转换也可以用于使参数更易于解释，从而促进先验信息的使用；里布勒等人。 (2016) 给出了一类空间相关模型的例子，Simpson 等人。 (2017) 更普遍地考虑这个想法

### 2.4 先验预测检查

先验预测检查是在生成模型的背景下理解先验分布的含义的有用工具（Box，1980，Gabry 等，2019；有关如何使用先验分布的详细信息，另请参见第 7.3 节）。特别是，由于先前的预测检查使用来自模型的模拟而不是观察到的数据，因此它们提供了一种无需多次使用数据即可优化模型的方法。图 3 显示了逻辑回归模型的简单先验预测检查。模拟表明，随着模型中协变量数量的增加，即使是各个系数的独立先验也具有不同的含义。这是回归模型中的普遍现象，随着预测变量数量的增加，如果我们想让模型远离极端预测，我们需要模型系数（或足够的数据）的更强先验。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211210202053-9d7d.webp)

> 图 3：演示使用先验预测检查来了解模型的非明显特征。上图对应于具有 100 个数据点和 2、4 或 15 个二元协变量的逻辑回归模型。在每种情况下，回归系数都被赋予独立的正态 (0,1) 先验分布。对于每个模型，我们执行了先验预测检查，1000 次模拟来自先验的系数向量 θ，然后模拟来自逻辑模型的数据集 y，然后通过模拟数据的均值 ̄y 对其进行总结。每个图都显示了此汇总统计量的先验预测分布，即 ̄y 的 1000 次模拟。当模型中的协变量数量很少时，这种先验预测分布会展开，表明该模型与范围广泛的数据体系兼容。但是随着协变量数量的增加，后验预测分布变得集中在 ̄y = 0 或 1 附近，这表明模型各个系数的弱先验意味着这个特定预测量的强先验。如果我们想要 ̄y 上更温和的先验预测分布，则系数的先验需要强烈集中在零附近。

一种有用的方法是考虑结果的先验，然后推导出相应的参数联合先验（参见，例如，Piironen 和 Vehtari，2017 年，以及 Zhang 等人，2020 年）。更一般地说，联合先验允许我们控制较大参数集的整体复杂性，这有助于生成更合理的先验预测，而独立先验很难或不可能实现这些预测。

图 4 显示了对高斯过程模型的三个先验分布选择进行先验预测检查的示例（Rasmussen 和 Willams，2006 年）。这种模拟和图形比较在处理任何模型时很有用，在设置不熟悉或复杂的模型时必不可少。

![](https://gitee.com/XiShanSnow/imagebed/raw/master/images/stats-20211210201950-f271.webp)

> 图 4：先验预测从具有平方指数协方差函数和幅度参数 τ 和长度尺度参数 l 的不同值的高斯过程模型得出。从格尔曼等人。 (2013)。

先验预测模拟的另一个好处是，它们可用于引出有关可测量的感兴趣量的专家先验知识，这通常比征求专家对不可观察的模型参数的意见更容易（O'Hagan 等，2006）。

最后，即使我们跳过计算先验预测检查，考虑我们选择的先验如何影响假设的模拟数据集也可能有用。

### 2.5 生成式和部分生成式模型

完全贝叶斯数据分析需要一个生成模型——即所有数据和参数的联合概率分布。这一点很微妙：贝叶斯推断实际上并不需要生成模型；它所需要的只是数据的似然性，不同的生成模型可以具有相同的似然性。但是贝叶斯数据分析要求生成模型能够进行预测模拟和模型检查（第 2.4、4.1、4.2、6.1 和 6.2 节），而贝叶斯工作流会考虑一系列生成模型。

举一个简单的例子，假设我们有数据 y ∼ binomial(n,θ)，其中 n 和 y 是观察到的，我们希望对 θ 进行推断。就贝叶斯推断而言，数据是用固定 n 采样（二项式采样）还是采样直到出现指定数量的成功（负二项式采样）是无关紧要的：这两个似然对于估计 θ 的目的是等效的，因为它们不同仅通过一个取决于 y 和 n 但不取决于 θ 的乘法因子。然而，如果我们想模拟来自预测模型的新数据，这两个模型是不同的，因为二项式模型产生具有固定值 n 的重复项，而负二项式模型产生具有固定 y 值的重复项。在这两种不同的生成模型下，先验和后验预测检查（第 2.4 和 6.1 节）看起来会有所不同。

这并不是说贝叶斯方法一定更好；生成模型的假设可以提高推断效率，但也可能出错，这激发了我们的大部分工作流程。

It is common in Bayesian analysis to use models that are not fully generative. For example, in regression we will typically model an outcome y given predictors x without a generative model for x. Another example is survival data with censoring, where the censoring process is not usually modeled. When performing predictive checks for such models, we either need to condition on the observed predictors or else extend the model to allow new values of the predictors to be sampled. It is also possible that there is no stochastic generative process for some parts of the model, for example if x has been chosen by a deterministic design of experiment.

在贝叶斯分析中使用不完全生成的模型是很常见的。例如，在回归中，我们通常会对给定预测变量 x 的结果 y 建模，而没有 x 的生成模型。另一个例子是带有删失的生存数据，其中删失过程通常不建模。在对此类模型执行预测检查时，我们要么需要以观察到的预测变量为条件，要么扩展模型以允许对预测变量的新值进行采样。模型的某些部分也可能没有随机生成过程，例如，如果 x 是通过确定性的实验设计选择的。

从生成模型的角度思考可以帮助阐明可以从观察中学到的东西的局限性。例如，我们可能想对具有复杂自相关结构的时间过程建模，但如果我们的实际数据在时间上相距很远，我们可能无法将该模型与具有几乎独立误差的更简单过程区分开来。

此外，使用不正确先验的贝叶斯模型不是完全生成的，因为它们没有数据和参数的联合分布，并且不可能从先验预测分布中采样。当我们确实使用不正确的先验时，我们将它们视为占位符或沿着路径的步骤，以建立完整的贝叶斯模型，并在参数和数据上进行适当的联合分布。

在应用工作中，复杂性通常来自于合并不同的数据源。例如，我们使用州和全国民意调查为 2020 年总统选举拟合贝叶斯模型，部分汇总基于政治和经济“基本面”的预测（Morris、Gelman 和 Heidemanns，2020 年）。该模型包括一个针对州和国家舆论中潜在时间趋势的随机过程。使用 Stan 拟合模型会产生后验模拟，用于计算选举结果的概率。基于贝叶斯模型的方法表面上类似于 Katz (2016) 描述的投票聚合，后者也通过随机模拟总结了不确定性。不同之处在于我们的模型可以向前运行以生成轮询数据；它不仅是一个数据分析程序，而且还为国家和州级的舆论提供了一个概率模型。

更一般地思考，我们可以考虑从最少到最多的生成模型的进展。一个极端是完全非生成方法，它们被简单地定义为数据摘要，根本没有数据模型。接下来是经典统计模型，其特征在于给定参数 θ 的数据 y 的概率分布 p(y; θ)，但没有 θ 的概率分布。下一步是我们通常拟合的贝叶斯模型，它们在 y 和 θ 上生成，但包括额外的未建模数据 x，例如样本大小、设计设置和超参数；我们将这样的模型写成 p(y,θ|x)。最后一步将是一个完全生成的模型 p(y,θ,x)，没有“遗漏”数据 x。

在统计工作流程中，我们可以在这个阶梯上上下移动，例如从未建模的数据缩减算法开始，然后将其公式化为概率模型，或者从概率模型的推断开始，将其视为基于数据的估计，并以某种方式对其进行调整以提高性能。在贝叶斯工作流中，我们可以将数据移入和移出模型，例如采用未建模的预测变量 x 并允许其具有测量误差，以便模型包含新级别的潜在数据（Clayton，1992，Richardson 和 Gilks​​， 1993）。

## 3 拟合一个模型

Traditionally, Bayesian computation has been performed using a combination of analytic calculation and normal approximation. Then in the 1990s, it became possible to perform Bayesian inference  for a wide range of models using Gibbs and Metropolis algorithms (Robert and Casella, 2011). The current state of the art algorithms for fitting open-ended Bayesian models include variational inference (Blei and Kucukelbir, 2017), sequential Monte Carlo (Smith, 2013), and Hamiltonian
Monte Carlo (HMC; Neal, 2011, Betancourt, 2017a). Variational inference is a generalization of the expectation-maximization (EM) algorithm and can, in the Bayesian context, be considered as providing a fast but possibly inaccurate approximation to the posterior distribution. Variational inference is the current standard for computationally intensive models such as deep neural networks. Sequential Monte Carlo is a generalization of the Metropolis algorithm that can be applied to any Bayesian computation, and HMC is a different generalization of Metropolis that uses gradient computation to move efficiently through continuous probability spaces.

传统上，贝叶斯计算是使用解析计算和正态近似的组合来执行的。然后在 1990 年代，使用 Gibbs 和 Metropolis 算法对各种模型执行贝叶斯推断成为可能（Robert 和 Casella，2011）。当前用于拟合开放式贝叶斯模型的最先进算法包括变分推断（Blei 和 Kucukelbir，2017）、顺序蒙特卡罗（Smith，2013）和 HamiltonianMonte Carlo（HMC；Neal，2011，Betancourt，2017a）。变分推断是对期望最大化 (EM) 算法的概括，在贝叶斯上下文中，可以被视为提供了对后验分布的快速但可能不准确的近似。变分推断是计算密集型模型（例如深度神经网络）的当前标准。 Sequential Monte Carlo 是 Metropolis 算法的推广，可以应用于任何贝叶斯计算，而 HMC 是 Metropolis 的不同推广，它使用梯度计算在连续概率空间中高效移动。

In the present article we focus on fitting Bayesian models using HMC and its variants, as implemented in Stan and other probabilistic programming languages. While similar principles should apply also to other software and other algorithms, there will be differences in the details.

在本文中，我们专注于使用 HMC 及其变体拟合贝叶斯模型，如在 Stan 和其他概率编程语言中实现的。虽然类似的原则也适用于其他软件和其他算法，但细节上会有差异。

To safely use an inference algorithm in Bayesian workflow, it is vital that the algorithm provides strong diagnostics to determine when the computation is unreliable. In the present paper we discuss such diagnostics for HMC.

要在贝叶斯工作流中安全地使用推断算法，算法必须提供强大的诊断功能来确定计算何时不可靠。在本文中，我们讨论了 HMC 的此类诊断。

### 3.1 初值、适应和预热


### 3.2 运行迭代算法需要多长时间

### 3.3 近似算法和近似模型

### 3.4 快速适应，快速失败




## 4 使用构造的数据发现和理解问题

验证计算的第一步是检查模型是否在可接受的时间范围内实际完成了拟合过程，并且收敛诊断是否合理。在 HMC 的背景下，这主要是由于没有发散转变、̂R 诊断接近 1，以及集中趋势、尾分位数和能量的足够有效样本量（Vehtari 等人，2020 年）。但是，这些诊断无法防止概率程序正确计算但编码的模型与用户预期的不同。

我们用于确保统计计算完成得相当好的主要工具是将模型实际拟合到某些数据并检查拟合是否良好。为此目的，真实数据可能很尴尬，因为建模问题可能会与计算问题发生冲突，并且无法判断问题是计算问题还是模型本身。为了解决这个挑战，我们首先通过将模型拟合到模拟数据来探索模型

### 4.1 假数据模拟 

### 4.2 基于模拟的较正

### 4.3 使用构造的数据进行实验

## 5 解决计算问题

### 5.1 统计计算的民间定理

当您遇到计算问题时，您的模型通常会出现问题（Yao、Vehtari 和 Gelman，2020）。并非总是如此——有时你会有一个难以拟合的模型——但许多收敛不佳的情况对应于参数空间中没有实质性意义的区域，甚至对应于一个无意义的模型。图 6 给出了参数空间不相关区域中的病理示例。 根本问题模型的示例可能是代码中的错误或对高斯或逻辑回归上下文中的每个单独观察使用高斯分布的不同截距，在这些情况下，它们不能由数据告知。面对有问题的模型时，我们的第一直觉不应该是在模型上投入更多的计算资源（例如，通过运行采样器进行更多迭代或减少 HMC 算法的步长），而是检查我们的模型是否包含一些实质性的病理。

### 5.2 从简单和复杂的模型开始并在中间相遇 

图 11 说明了一种常用的调试方法。出发点是模型表现不佳，可能无法收敛或无法在假数据模拟中重现真实参数值，或者无法很好地拟合数据，或者产生不合理的推论。诊断问题的途径是从两个方向移动：逐渐简化

### 5.3 掌握需要很长时间才能适应的模型


### 5.4 监控中间量


### 5.5 堆叠以重新加权混合不良的链条

在实践中，通常我们的 MCMC 算法混合得很好。其他时候，模拟会快速移动到参数空间的不合理区域，表明可能存在模型错误指定、信息不足或信息不足的观察，或者只是几何困难

但在中间情况下也很常见，其中多个链混合起来很慢，但它们在一般合理的范围内。在这种情况下，我们可以使用堆叠来组合模拟，使用交叉验证为不同的链分配权重（Yao、Vehtari 和 Gelman，2020）。这将具有丢弃卡在目标分布的偏僻低概率模式中的链的近似效果。 stacking的结果不一定等价，甚至是渐近地等价于完全贝叶斯推断，但它服务于许多相同的目标，特别适合在模型探索阶段，让我们向前迈进，花更多的时间和精力在其他部分贝叶斯工作流，而不会被完全拟合一个特定模型所困扰。此外，当与跟踪图和其他诊断工具一起使用时，非均匀堆叠权重可以帮助我们以迭代方式了解将精力集中在哪里

### 5.6 具有多峰和其他困难几何形状的后验分布

我们可以粗略地区分四类与多模态和其他困难的后验几何相关的 MCMC 问题： 

- 有效分离后验体积，其中除了一种模式外，所有模式的质量都接近于零。一个例子出现在第 11 节。在这些问题中，可以通过明智地选择模拟初始值、添加先验信息或参数的硬约束来避免次要模式，或者可以通过近似估计每个模式的质量来修剪它们。 

- 有效地分离微对称的高概率质量的后验体积，例如混合模型中的标签切换。这里的标准做法是以某种方式限制模型以识别感兴趣的模式；参见例如 Bafumi 等人。 (2005) 和贝当古 (2017b)。 

- 有效地分离不同的高概率质量的后验体积。例如，在基因调控模型 (Modrák, 2018) 中，一些数据承认两种不同的调控机制具有相反的效应符号，而接近于零的效应具有低得多的后验密度。这个问题更具挑战性。在某些情况下，我们可以使用堆叠（预测模型平均）作为近似解决方案，认识到这并不完全通用，因为它需要定义感兴趣的预测量。一种更完整的贝叶斯替代方法是通过引入强混合先验将模型分成几部分，然后在给定先验的每个组件的情况下分别拟合模型。其他时候，可以使用具有排除某些可能模式的效果的强先验来解决问题。 

- 具有算术不稳定尾部的高概率质量的单个后验体积。如果您在分布的质量附近初始化，则大多数推断应该不会出现问题。如果对极其罕见的事件特别感兴趣，那么无论如何都应该重新参数化问题，因为从几百到几千的通常默认有效样本大小中可以学到的东西是有限的。 

### 5.7 重参数化



### 5.8 边缘化

### 5.9 增加先验信息

### 5.10 增加数据


## 6 评估和使用拟合后的模型

### 6.1 后验预测检查

### 6.2 单个数据点和数据子集的交叉验证和影响

### 6.3 先验信息的影响

### 6.4 总结推断结果并传播不确定性

## 7 修改模型

### 7.1 为数据构建模型

### 7.2 合并额外的数据

### 7.3 使用先验分布

### 7.4 模型拓扑

## 8. 理解和比较多个模型 

### 8.1 可视化彼此相关的模型

### 8.2 交叉验证和模型平均

### 8.3 比较大量模型

## 9 将建模当做软件开发

### 9.1 版本控制使与他人和您过去自己的协作顺利进行 

版本控制软件，例如 Git，应该是项目的第一个基础设施。学习版本控制似乎是一项巨大的投资，但能够键入单个命令以恢复到以前的工作版本或获取当前版本和旧版本之间的差异是非常值得的。当您需要与他人共享工作时甚至是在纸上共享工作时效果更好 - 工作可以独立完成然后自动合并。虽然版本控制会跟踪一个模型中较小的更改，但将明显不同的模型保存在不同的文件中以方便比较模型是很有用的。版本控制还有助于记录迭代模型构建中的发现和决策，从而提高过程的透明度。版本控制不仅仅针对代码。它也适用于报告、图表和数据。

版本控制不仅仅针对代码。它也适用于报告、图表和数据。版本控制是确保所有这些组件同步的关键部分，更重要的是，可以将项目回滚到以前的状态。版本控制特别有用，因为它能够打包和标记与里程碑报告和出版物相对应的模型和数据的“候选发布”版本，并将它们存储在同一目录中，而无需求助于可怕的 _final_final_161020.pdf 样式命名约定。

在处理用于指导政策决策制定的模型时，公共版本控制存储库提高了特定报告使用了哪些模型、数据、推断参数和脚本的透明度。一个很好的例子是帝国理工学院的模型和脚本存储库，用于估计 COVID-19 的死亡和病例数（Flaxman 等人，2020 年）。

### 9.2 边走边测试

理想情况下，软件设计从最终用户的目标到实现它所需的技术机制自上而下地进行。对于贝叶斯统计模型，自上而下的设计至少涉及数据输入格式、数据的概率模型和先验，但也可能涉及模拟器和模型测试，如基于模拟的校准或后验预测检查。理想情况下，软件开发从经过良好测试的基础功能到更大的功能模块自下而上地工作。这样，开发将通过一系列经过充分测试的步骤进行，在每个阶段都只在经过测试的部分上进行构建。与构建大型程序然后调试它相比，以这种方式工作的优势与增量模型开发相同——更容易跟踪开发出错的地方，并且您在使用经过良好测试的基础工作的每一步都更有信心。无论是初始开发还是修改代码，计算开发的关键是模块化。大杂乱的功能难以记录、难以阅读、异常难以调试，并且几乎不可能维护或修改。模块化意味着从较小的可信部分构建更大的部分，例如低级功能。每当代码片段重复时，都应该将它们封装为函数。这导致代码更易于阅读和维护。作为低级函数的示例，可以通过实施标准化 z(v) = (v −mean(v))/sd(v) 为广义线性模型重新调整预测变量。虽然这个函数看起来很简单，但从 sd 函数开始，它有时被定义为 sd(v) =√∑ni=1(vi −mean(v))2/n，有时被定义为 sd(v) = √ ∑ni=1(vi -mean(v))2/(n -1)。如果在标准化函数层面不解决这个问题，在推断过程中会产生神秘的偏差。不依赖于 sd() 函数的简单测试将在函数开发过程中解决这个问题。如果选择是除以 n -1 的估计值，则需要决定当 v 是长度为 1 的向量时要做什么。在存在非法输入的情况下，在输入-输出例程中进行检查会有所帮助让用户知道什么时候出现问题，而不是让错误渗透到以后神秘的除以零错误。微分方程的三次样条或欧拉求解器的实现是高级函数的一个示例，应在使用前对其进行测试。随着函数变得越来越复杂，由于边界条件组合学的问题、更一般的输入（例如要积分的函数）、数值不稳定性或不精确性，其域的区域可能会或可能不会接受，因此它们变得更难测试，这取决于应用程序，需要稳定的衍生品等。

###9.3 使其基本上可重现

任何项目的一个崇高目标是使其完全可重现，因为另一台机器上的另一个人可以重新创建最终报告。这不是科学领域考虑的可重复性类型，科学领域希望确保影响得到未来新数据的证实（现在通常称为“可重复性”，以便更好地区分不同概念）。相反，这是确保始终如一地完成一项特定分析的更有限（但仍然至关重要）的目标。特别是，我们希望能够生成与原始文档基本相同的分析和图表。位级可再现性可能是不可能的，但我们仍然会在实际水平上将等效性进行比较。如果这种类型的复制改变了论文的结果，我们会争辩说原始结果不是特别可靠。

与其在运行模型时在命令行上输入命令（或直接将命令输入到 R 或 Python 等交互式编程语言中），不如编写脚本来通过模型运行数据并生成您需要的任何后验分析。可以为 shell、R、Python 或任何其他编程语言编写脚本。

脚本应该是独立的，因为它应该在一个完全干净的环境中运行，或者最好在不同的计算机上运行。这意味着脚本不得依赖于已设置的全局变量、正在读入的其他数据或脚本中没有的任何其他内容。脚本是很好的文档。如果运行项目只是一行代码，这似乎有点过分，但脚本不仅提供了一种运行代码的方法，而且还提供了一种关于正在运行的内容的具体文档的形式。

对于复杂的项目，我们经常发现构建良好的一系列脚本比一个大的 R Markdown 文档或 Jupyter notebook 更实用。根据长期的再现性需求，为手头的工作选择合适的工具很重要。为了保证位级的可重复性，有时甚至只是为了让程序运行，从硬件到操作系统，再到每一个软件和设置，都必须用它们的版本号来指定。随着脚本的初始编写和复制尝试之间的时间流逝，即使环境随脚本一起提供，也几乎不可能实现位级可复制性，就像在 Docker 容器中一样

### 9.4 使其具有可读性和可维护性

像对待其他形式的写作一样对待程序和脚本为观众提供了如何使用代码的重要视角。不仅其他人可能想要阅读和理解一个程序或模型，开发人员自己以后也会想要阅读和理解它。 Stan 设计的动机之一是让模型在变量使用（例如，数据或参数）、类型（例如，协方差矩阵或无约束矩阵）和大小方面进行自我记录。这使我们能够理解 Stan 代码（或其他静态类型概率编程语言的代码），以便在没有应用它的数据上下文的情况下也能被理解。

可读性的很大一部分是一致性，特别是在命名和布局方面，不仅是程序本身，还有存储它们的目录和文件。编码的另一个关键原则是避免重复，而是将共享代码提取到可以重用的函数中。

代码的可读性不仅仅与注释有关——它还与可读性的命名和组织有关。事实上，注释会使代码的可读性降低。最好的方法是编写可读的代码，而不是带有注释的不透明代码。例如，我们不想这样写：

real x17; // 氧气水平，应该是正的

当我们可以这样写时：

real<lower = 0>oxygen_level;

同样，我们也不想这样做： 

target += -0.5 * (y - mu)^2 / sigma^2; // y 分布 normal(mu, sigma), 

当我们可以写：

target += normal_lpdf(y | mu, sigma);

好的做法是尽量减少内联代码注释，而是编写可读代码。如上述示例所示，编程语言为用户提供了他们需要使用的工具，从而促进了干净的代码。

面向用户的函数应该在函数级别记录其参数类型、返回类型、错误条件和行为——这是用户看到的应用程序编程接口 (API)，而不是代码内部。针对开发人员的内联代码注释的问题在于，它们在开发过程中很快就会变得陈旧，最终弊大于利。相反，与其在内联记录实际代码，不如将函数减少到可管理的大小，并应选择名称以便代码可读。较长的变量名并不总是更好，因为它们会使代码结构更难以扫描。编写代码文档时应该假设读者很好地理解编程语言；因此，只有在代码偏离了语言的惯用用法或涉及复杂算法时才需要文档。当试图对长表达式或代码块进行注释时，请考虑将其替换为一个命名良好的函数。

与可读性相关的是工作流代码的可维护性。在拟合一系列相似的模型时，它们之间会共享很多模块（参见第 2.2 节），因此也会共享相应的代码。如果我们每次编写新模型时都复制所有模型代码，然后发现共享模块中的一个错误，我们将不得不在所有模型中手动修复它。这又是一个容易出错的过程。相反，不仅以模块化方式构建模型而且保持相应的代码模块化并根据需要将其加载到模型中是明智的。这样，修复模块中的错误只需要在一处而不是多处更改代码。当我们在整个工作流程中移动时，将不可避免地发生错误和其他对以后更改的要求，如果我们相应地准备我们的建模代码，它将为我们节省大量时间

## 10. 包含模型构建和扩展的工作流示例：高尔夫推杆

我们使用一组适合高尔夫推杆数据的模型示例演示了贝叶斯建模的基本工作流程 (Gelman, 2019)。图 24 显示了职业高尔夫球手关于成功推杆的比例作为距离球洞（圆形）距离的函数的数据。不出所料，投篮的概率随着距离的增加而下降。

### 10.1 第一个模型：逻辑回归

### 10.2 从第一原则建模

### 10.3 在新数据上测试拟合后的模型

### 10.4 一种解释击球力度的新模型

### 10.5 通过加入一个模糊因子来扩展模型

### 10.6 高尔夫示例的一般教训

## 11 具有不可预期的多峰后验的工作流示例：行星运动

### 11.1 运动的机械模型

### 11.2 拟合一个简化模型

### 11.3 坏马尔可夫链，慢马尔可夫链？

### 11.4 建立模型

### 11.5 行星运动示例的一般经验教训

## 12 讨论

### 12.1 统计建模和预测的不同视角

### 12.2 迭代式模型构建的必要性

### 12.3 模型选择和过拟合

### 12.4 更大的数据集需要更大的模型

### 12.5 预测、泛化和后分层化

### 12.6 继续往前走

## 参考文献

[1]. Afrabandpey, H., Peltola, T., Piironen, J., Vehtari, A., and Kaski, S. (2020). Making Bayesian predictive models interpretable: A decision theoretic approach. Machine Learning 109, 1855–1876.

[2]. Akaike, H. (1973). Information theory and an extension of the maximum likelihood principle. In Proceedings of the Second International Symposium on Information Theory, ed. B. N. Petrov and F. Csaki, 267–281. Budapest: Akademiai Kiado. Reprinted in Breakthroughs in Statistics, ed. S. Kotz, 610–624. New York: Springer (1992).

[3]. Berger, J. O., Bernardo, J. M., and Sun, D. (2009). The formal definition of reference priors. Annals of Statistics 37, 905–938.

[4]. Berk, R., Brown, L., Buja, A., Zhang, K., and Zhao, L. (2013). Valid post-selection inference. Annals of Statistics 41, 802–837.

[5]. Berry, D. (1995). Statistics: A Bayesian Perspective. Duxbury Press.

[6]. Betancourt, M. (2017a). A conceptual introduction to Hamiltonian Monte Carlo. arxiv.org/abs/1701.02434

[7]. Betancourt, M. (2017b). Identifying Bayesian mixture models. Stan Case Studies 4. mc-stan.org/users/documentation/case-studies/identifying_mixture_models.html

[8]. Betancourt, M. (2018). Underdetermined linear regression. betanalpha.github.io/assets/case_studies/underdetermined_linear_regression.html

[9]. Betancourt, M. (2020a). Towards a principled Bayesian workflow. betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html

[10]. Betancourt, M. (2020b). Robust Gaussian Process Modeling. github.com/betanalpha/knitr_case_studies/tree/master/gaussian_processes

[11]. Betancourt, M., and Girolami, M. (2015). Hamiltonian Monte Carlo for hierarchical models. In Current Trends in Bayesian Methodology with Applications, ed. S. K. Upadhyay, U. Singh, D. K. Dey, and A. Loganathan, 79–102.

[12]. Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. (2017). Variational inference: A review for statisticians. Journal of the American Statistical Association 112, 859–877.

[13]. Blitzer, J., Dredze, M., and Pereira, F. (2007). Biographies, Bollywood, boom-boxes and blenders: Domain adaptation for sentiment classification. In Proceedings of the 45th Annual Meeting of the Association of Computational Linguistics, 440–447.

[14]. Box, G. E. P. (1980). Sampling and Bayes inference in scientific modelling and robustness. Journal of the Royal Statistical Society A 143, 383–430.

[15]. Broadie, M. (2018). Two simple putting models in golf. statmodeling.stat.columbia.edu/wp-content/uploads/2019/03/putt_models_20181017.pdf

[16]. Bryan, J. (2017). Project-oriented workflow. www.tidyverse.org/blog/2017/12/workflow-vs-script

[17]. Bürkner, P.-C. (2017). brms: An R Package for Bayesian multilevel models using Stan. Journal of Statistical Software 80, 1–28.

[18]. Carpenter, B. (2017). Typical sets and the curse of dimensionality. Stan Case Studies 4. mc-stan.org/users/documentation/case-studies/curse-dims.html

[19]. Carpenter, B. (2018). Predator-prey population dynamics: The Lotka-Volterra model in Stan. Stan Case Studies 5. mc-stan.org/users/documentation/case-studies/lotka-volterra-predator-prey.html

[20]. Carpenter, B., Gelman, A., Hoffman, M., Lee, D., Goodrich, B., Betancourt, M., Brubaker, M., Guo, J., Li, P., and Riddell, A. (2017). Stan: A probabilistic programming language. Journal of Statistical Software 76 (1).

[21]. Chen, C., Li, O., Barnett, A., Su, J., and Rudin, C. (2019). This looks like that: Deep learning for interpretable image recognition. 33rd Conference on Neural Information Processing Systems. papers.nips.cc/paper/9095-this-looks-like-that-deep-learning-for-interpretable-image-recognition.pdf

[22]. Chiu, W. A., Wright, F. A., and Rusyn, I. (2017). A tiered, Bayesian approach to estimating of population variability for regulatory decision-making. ALTEX 34, 377–388.

[23]. Chung, Y., Rabe-Hesketh, S., Gelman, A., Liu, J. C., and Dorie, A. (2013). A non-degenerate penalized likelihood estimator for hierarchical variance parameters in multilevel models. Psychometrika 78, 685–709.

[24]. Chung, Y., Rabe-Hesketh, S., Gelman, A., Liu, J. C., and Dorie, A. (2014). Nonsingular covariance estimation in linear mixed models through weakly informative priors. Journal of Educational and Behavioral Statistics 40, 136–157.

[25]. Clayton, D. G. (1992). Models for the analysis of cohort and case-control studies with inaccurately measured exposures. In Statistical Models for Longitudinal Studies of Exposure and Health, ed. J. H. Dwyer, M. Feinleib, P. Lippert, and H. Hoffmeister, 301–331. Oxford University Press.

[26]. Cook, S., Gelman, A., and Rubin, D. B. (2006). Validation of software for Bayesian models using posterior quantiles. Journal of Computational and Graphical Statistics 15, 675–692.

[27]. Daumé, H. (2009). Frustratingly easy domain adaptation. arxiv.org/abs/0907.1815 

[28]. Deming, W. E., and Stephan, F. F. (1940). On a least squares adjustment of a sampled frequency table when the expected marginal totals are known. Annals of Mathematical Statistics 11, 427–444.

[29]. Devezer, B., Nardin, L. G., Baumgaertner, B., and Buzbas, E. O. (2019). Scientific discovery in a model-centric framework: Reproducibility, innovation, and epistemic diversity. PLoS One 14,e0216125.

[30]. Devezer, B., Navarro, D. J., Vanderkerckhove, J., and Buzbas, E. O. (2020). The case for formal methodology in scientific reform. doi.org/10.1101/2020.04.26.048306 

[31]. Dragicevic, P., Jansen, Y., Sarma, A., Kay, M., and Chevalier, F. (2019). Increasing the transparency of research papers with explorable multiverse analyses. Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems, paper no. 65.

[32]. Efron, B. (2013). Estimation and accuracy after model selection. Journal of the American Statistical Association 109, 991–1007.

[33]. Finkel, J. R., and Manning, C. D. (2009). Hierarchical Bayesian domain adaptation. In Proceedings of Human Language Technologies: The 2009 Annual Conference of the North American Chapter of the Association for Computational Linguistics, 602–610.

[34]. Fithian, W., Taylor, J., Tibshirani, R., and Tibshirani, R. J. (2015). Selective sequential model selection. arxiv.org/pdf/1512.02565.pdf

[35]. Flaxman, S., Mishra, S., Gandy, A., et al. (2020). Estimating the effects of non-pharmaceutical interventions on COVID-19 in Europe. Nature 584, 257–261. Data and code at github.com/ImperialCollegeLondon/covid19model 

[36]. Fuglstad, G. A., Simpson, D., Lindgren, F., and Rue, H. (2019). Constructing priors that penalize the complexity of Gaussian random fields. Journal of the American Statistical Association 114,445–452.

[37]. Gabry, J., et al. (2020a). rstanarm: Bayesian applied regression modeling via Stan, version 2.19.3. cran.r-project.org/package=rstanarm

[38]. Gabry, J., et al. (2020b). bayesplot: Plotting for Bayesian models, version 1.7.2. cran.r-project.org/package=bayesplot

[39]. Gabry, J., Simpson, D., Vehtari, A., Betancourt, M., and Gelman, A. (2019). Visualization in Bayesian workflow (with discussion and rejoinder). Journal of the Royal Statistical Society A 182, 389–441.

[40]. Gelman, A. (2003). A Bayesian formulation of exploratory data analysis and goodness-of-fit testing. International Statistical Review 71, 369–382.

[41]. Gelman, A. (2004). Parameterization and Bayesian modeling. Journal of the American Statistical Association 99, 537–545.

[42]. Gelman, A. (2011). Expanded graphical models: Inference, model comparison, model checking, fake-data debugging, and model understanding. www.stat.columbia.edu/~gelman/70presentations/ggr2handout.pdf

[43]. Gelman, A. (2014). How do we choose our default methods? In Past, Present, and Future of Statistical Science, ed. X. Lin, C. Genest, D. L. Banks, G. Molenberghs, D. W. Scott, and J. L. Wang. London: CRC Press. 

[44]. Gelman, A. (2019). Model building and expansion for golf putting. Stan Case Studies 6. mc-stan.org/users/documentation/case-studies/golf.html

[45]. Gelman, A., et al. (2020). Prior choice recommendations. github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations

[46]. Gelman, A., and Azari, J. (2017). 19 things we learned from the 2016 election (with discussion). Statistics and Public Policy 4, 1–10.

[47]. Gelman, A., Bois, F. Y., and Jiang, J. (1996). Physiological pharmacokinetic analysis using population modeling and informative prior distributions. Journal of the American Statistical Association 91, 1400–1412.

[48]. Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and Rubin, D. B. (2013). Bayesian Data Analysis, third edition. London: CRC Press.

[49]. Gelman, A., and Hill, J. (2007). Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge University Press.

[50]. Gelman, A., Hill, J., and Vehtari, A. (2020). Regression and Other Stories. Cambridge University Press.

[51]. Gelman, A., Hill, J., and Yajima, M. (2012). Why we (usually) don’t have to worry about multiple comparisons. Journal of Research on Educational Effectiveness 5, 189–211.

[52]. Gelman, A., Hullman, J., Wlezien, C., and Morris, G. E. (2020). Information, incentives, and goals in election forecasts. Judgment and Decision Making 15, 863–880.

[53]. Gelman, A., and Loken, E. (2013). The garden of forking paths: Why multiple comparisons can be a problem, even when there is no “fishing expedition” or “p-hacking” and the research hypothesis was posited ahead of time. www.stat.columbia.edu/~gelman/research/unpublished/forking.pdf

[54]. Gelman, A., Meng, X. L., and Stern, H. S. (1996). Posterior predictive assessment of model fitness via realized discrepancies (with discussion). Statistica Sinica 6, 733–807.

[55]. Gelman, A., Simpson, D., and Betancourt, M. (2017). The prior can often only be understood in the context of the likelihood. Entropy 19, 555.

[56]. Gelman, A., Stevens, M., and Chan, V. (2003). Regression modeling and meta-analysis for decision making: A cost-benefit analysis of a incentives in telephone surveys. Journal of Business and Economic Statistics 21, 213–225.

[57]. Gharamani, Z., Steinruecken, C., Smith, E., Janz, E., and Peharz, R. (2019). The Automatic Statistician: An artificial intelligence for data science. www.automaticstatistician.com/index 

[58]. Ghitza, Y., and Gelman, A. (2020). Voter registration databases and MRP: Toward the use of large scale databases in public opinion research. Political Analysis 28, 507–531.

[59]. Giordano, R. (2018). StanSensitivity. github.com/rgiordan/StanSensitivity 

[60]. Giordano, R., Broderick, T., and Jordan, M. I. (2018). Covariances, robustness, and variational Bayes. Journal of Machine Learning Research 19, 1981–2029.

[61]. Goel, P. K., and DeGroot, M. H. (1981). Information about hyperparameters in hierarchical models.Journal of the American Statistical Association 76, 140–147.

[62]. Grinsztajn, L., Semenova, E., Margossian, C. C., and Riou, J. (2020). Bayesian workflow for disease transmission modeling in Stan. mc-stan.org/users/documentation/case-studies/boarding_school_case_study.html

[63]. Grolemund, G., and Wickham, H. (2017). R for Data Science. Sebastopol, Calif.: O’Reilly Media.

[64]. Gunning, D. (2017). Explainable artificial intelligence (xai). U.S. Defense Advanced Research Projects Agency (DARPA) Program.

[65]. Henderson, C. R. (1950). Estimation of genetic parameters (abstract). Annals of Mathematical Statistics 21, 309–310.

[66]. Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference. Journal of Computational and Graphical Statistics 20, 217–240.

[67]. Hodges, J. S., and Reich, B. J. (2010). Adding spatially-correlated errors can mess up the fixed effect you love. American Statistician 64, 325–334.

[68]. Hoffman, M., and Ma, Y. (2020). Black-box variational inference as a parametric approximation to Langevin dynamics. Proceedings of Machine Learning and Systems, in press.

[69]. Hunt, A., and Thomas, D. (1999). The Pragmatic Programmer. Addison-Wesley.

[70]. Hwang, Y., Tong, A. and Choi, J. (2016). The Automatic Statistician: A relational perspective. ICML 2016: Proceedings of the 33rd International Conference on Machine Learning.

[71]. Jacquez, J. A. (1972). Compartmental Analysis in Biology and Medicine. Elsevier.

[72]. Kale, A., Kay, M., and Hullman, J. (2019). Decision-making under uncertainty in research synthesis: Designing for the garden of forking paths. Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems, paper no. 202.

[73]. Kamary, K., Mengersen, K., Robert, C. P., and Rousseau, J. (2019). Testing hypotheses via a mixture estimation model. arxiv.org/abs/1412.2044

[74]. Katz, J. (2016). Who will be president? www.nytimes.com/interactive/2016/upshot/presidential-polls-forecast.html

[75]. Kay, M. (2020a). ggdist: Visualizations of distributions and uncertainty. R package version 2.2.0. mjskay.github.io/ggdist. doi:10.5281/zenodo.3879620.

[76]. Kay, M. (2020b). tidybayes: Tidy data and geoms for Bayesian models. R package version 2.1.1. mjskay.github.io/tidybayes. doi:10.5281/zenodo.1308151.

[77]. Kennedy, L., Simpson, D., and Gelman, A. (2019). The experiment is just as important as the likelihood in understanding the prior: A cautionary note on robust cognitive modeling. Computational Brain and Behavior 2, 210–217.

[78]. Kerman, J., and Gelman, A. (2004). Fully Bayesian computing. www.stat.columbia.edu/~gelman/research/unpublished/fullybayesiancomputing-nonblinded.pdf

[79]. Kerman, J., and Gelman, A. (2007). Manipulating and summarizing posterior simulations using random variable objects. Statistics and Computing 17, 235–244.

[80]. Kucukelbir, A., Tran, D., Ranganath, R., Gelman, A., and Blei, D. M. (2017). Automatic differentiation variational inference. Journal of Machine Learning Research 18, 1–45.

[81]. Kumar, R., Carroll, C., Hartikainen, A., and Martin, O. A. (2019). ArviZ a unified library for exploratory analysis of Bayesian models in Python. Journal of Open Source Software, doi:10.21105/joss.01143.

[82]. Lambert, B., and Vehtari, A. (2020). R∗: A robust MCMC convergence diagnostic with uncertainty using gradient-boosted machines. arxiv.org/abs/2003.07900

[83]. Lee, M. D., Criss, A. H., Devezer, B., Donkin, C., Etz, A., Leite, F. P., Matzke, D., Rouder, J. N.,Trueblood, J. S., White, C. N., and Vandekerckhove, J. (2019). Robust modeling in cognitive science. Computational Brain and Behavior 2, 141–153.

[84]. Lin, C. Y., Gelman, A., Price, P. N., and Krantz, D. H. (1999). Analysis of local decisions using hierarchical modeling, applied to home radon measurement and remediation (with discussion). Statistical Science 14, 305–337.

[85]. Lindley, D. V. (1956). On a measure of the information provided by an experiment. Annals of Mathematical Statistics 27, 986–1005.

[86]. Lins, L., Koop, D., Anderson, E. W., Callahan, S. P., Santos, E., Scheidegger, C. E., Freire, J., and Silva, C. T. (2008). Examining statistics of workflow evolution provenance: A first study. In Scientific and Statistical Database Management, SSDBM 2008, ed. B. Ludäscher and N. Mamoulis, 573–579. Berlin: Springer.

[87]. Linzer, D. A. (2013). Dynamic Bayesian forecasting of presidential elections in the states. Journal of the American Statistical Association 108, 124–134.

[88]. Liu, Y., Harding, A., Gilbert, R., and Journel, A. G. (2005). A workflow for multiple-point geostatistical simulation. In Geostatistics Banff 2004, ed. O. Leuangthong and C. V. Deutsch. Dordrecht: Springer.

[89]. Loftus, J. (2015). Selective inference after cross-validation. arxiv.org/pdf/1511.08866.pdf

[90]. Long, J. S. (2009). The Workflow of Data Analysis Using Stata. London: CRC Press.

[91]. Mallows, C. L. (1973). Some comments on Cp. Technometrics 15, 661–675.

[92]. Margossian, C. C., and Gelman, A. (2020). Bayesian model of planetary motion: Exploring ideas for a modeling workflow when dealing with ordinary differential equations and multimodality. github.com/stan-dev/example-models/tree/case-study/planet/knitr/planetary_motion

[93]. Margossian, C. C., Vehtari, A., Simpson, D., and Agrawal, R. (2020a). Hamiltonian Monte Carlo using an adjoint-differentiated Laplace approximation: Bayesian inference for latent Gaussian models and beyond. Advances in Neural Information Processing Systems 34. arXiv:2004.12550

[94]. Margossian, C. C., Vehtari, A., Simpson, D., and Agrawal, R. (2020b). Approximate Bayesian inference for latent Gaussian models in Stan. Presented at StanCon2020. researchgate.net/publication/343690329_Approximate_Bayesian_inference_for_latent_Gaussian_models_in_Stan

[95]. Mayo, D. (2018). Statistical Inference as Severe Testing: How to Get Beyond the Statistics Wars. Cambridge University Press.

[96]. McConnell, S. (2004). Code Complete, second edition. Microsoft Press.

[97]. Meng, X. L., and van Dyk, D. A. (2001). The art of data augmentation. Journal of Computational and Graphical Statistics 10, 1–50.

[98]. Merkle, E. C., Furr, D., and Rabe-Hesketh, S. (2019). Bayesian comparison of latent variable models: Conditional versus marginal likelihoods. Psychometrika 84, 802–829.

[99]. Millar, R. B. (2018). Conditional vs marginal estimation of the predictive loss of hierarchical models using WAIC and cross-validation. Statistics and Computing 28, 375–385.

[100]. Modrák, M. (2018). Reparameterizing the sigmoid model of gene regulation for Bayesian inference. 

[101]. Computational Methods in Systems Biology. CMSB 2018. Lecture Notes in Computer Science, vol. 11095, 309–312.

[102]. Montgomery, J. M., and Nyhan, B. (2010). Bayesian model averaging: Theoretical developments and practical applications. Political Analysis 18, 245–270.

[103]. Morgan, S. L., and Winship, C. (2014). Counterfactuals and Causal Inference: Methods and Principles for Social Research, second edition. Cambridge University Press.

[104]. Morris, G. E., Gelman, A., and Heidemanns, M. (2020). How the Economist presidential forecast works. projects.economist.com/us-2020-forecast/president/how-this-works

[105]. Navarro, D. J. (2019). Between the devil and the deep blue sea: Tensions between scientific judgement and statistical model selection. Computational Brain and Behavior 2, 28–34.

[106]. Navarro, D. J. (2020). If mathematical psychology did not exist we might need to invent it: A comment on theory building in psychology. Perspectives on Psychological Science. psyarxiv.com/ygbjp

[107]. Neal, R. M. (1993). Probabilistic inference using Markov chain Monte Carlo methods. Technical Report CRG-TR-93-1, Department of Computer Science, University of Toronto.

[108]. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. In Handbook of Markov Chain Monte Carlo, ed. S. Brooks, A. Gelman, G. L. Jones, and X. L. Meng, 113–162. London: CRC Press.

[109]. Niederlová, V., Modrák, M., Tsyklauri, O., Huranová, M., and Štěpánek, O. (2019). Meta-analysis of genotype-phenotype associations in Bardet-Biedl Syndrome uncovers differences among causative genes. Human Mutation 40, 2068–2087.

[110]. Nott, D. J., Wang, X., Evans, M., and Englert, B. G. (2020). Checking for prior-data conflict using prior-to-posterior divergences. Statistical Science 35, 234–253.

[111]. Novick, M. R., Jackson, P. H., Thayer, D. T., and Cole, N. S. (1972). Estimating multiple regressions in m-groups: a cross validation study. British Journal of Mathematical and Statistical Psychology 25, 33–50.

[112]. O’Hagan, A., Buck, C. E., Daneshkhah, A., Eiser, J. R., Garthwaite, P. H., Jenkinson, D. J., Oakely, J. E., and Rakow, T. (2006). Uncertain Judgements: Eliciting Experts’ Probabilities. Wiley.

[113]. Paananen, T., Piironen, J., Bürkner, P.-C., and Vehtari, A. (2020). Implicitly adaptive importance sampling. Statistics and Computing, in press.

[114]. Pearl, J., and Bareinboim, E. (2011). Transportability of causal and statistical relations: A formal approach. In Data Mining Workshops (ICDMW), 2011 IEEE 11th International Conference, 540–547.

[115]. Pearl, J., and Bareinboim, E. (2014). External validity: From do-calculus to transportability across populations. Statistical Science 29, 579–595. 

[116]. Piironen, J., and Vehtari, A. (2017). Sparsity information and regularization in the horseshoe and other shrinkage priors. Electronic Journal of Statistics 11, 5018–5051.

[117]. Pirš, G., and Štrumbelj, E. (2009). Bayesian combination of probabilistic classifiers using multivariate normal mixtures. Journal of Machine Learning Research 20, 1–18.

[118]. Price, P. N., Nero, A. V., and Gelman, A. (1996). Bayesian prediction of mean indoor radon concentrations for Minnesota counties. Health Physics 71, 922–936.

[119]. Rasmussen, C. E., and Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. MIT Press.

[120]. Raudenbush, S. W., and Bryk, A. S. (2002). Hierarchical Linear Models, second edition. Sage Publications.

[121]. Richardson, S., and Gilks, W. R. (1993). A Bayesian approach to measurement error problems in epidemiology using conditional independence models. American Journal of Epidemiology 138, 430–442.

[122]. Riebler, A., Sørbye, S. H., Simpson, D., and Rue, H. (2018). An intuitive Bayesian spatial model for disease mapping that accounts for scaling. Statistical Methods in Medical Research 25,1145–1165.

[123]. Robert, C., and Casella, G. (2011). A short history of Markov chain Monte Carlo: Subjective recollections from incomplete data. Statistical Science 26, 102–115.

[124]. Rubin, D. B. (1984). Bayesianly justifiable and relevant frequency calculations for the applied statistician. Annals of Statistics 12, 1151–1172.

[125]. Rudin, C. (2018). Please stop explaining black box models for high stakes decisions. NeurIPS 2018 Workshop on Critiquing and Correcting Trends in Machine Learning. arxiv.org/abs/1811.10154

[126]. Rue, H., Martino, S., and Chopin, N. (2009). Approximate Bayesian inference for latent Gaussian models by using integrated nested Laplace approximations. Journal of the Royal Statistical Society B 71, 319–392.

[127]. Sarma, A., and Kay, M. (2020). Prior setting in practice: Strategies and rationales used in choosing prior distributions for Bayesian analysis. Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems.

[128]. Savage, J. (2016). What is modern statistical workflow? khakieconomics.github.io/2016/08/29/What-is-a-modern-statistical-workflow.html

[129]. Shi, X., and Stevens, R. (2008). SWARM: a scientific workflow for supporting bayesian approaches to improve metabolic models. CLADE ’08: Proceedings of the 6th International Workshop on Challenges of Large Applications in Distributed Environments, 25–34.

[130]. Shirani-Mehr, H., Rothschild, D., Goel, S., and Gelman, A. (2018). Disentangling bias and variance in election polls. Journal of the American Statistical Association 118, 607–614.

[131]. Simmons, J., Nelson, L., and Simonsohn, U. (2011). False-positive psychology: Undisclosed flexibility in data collection and analysis allow presenting anything as significant. Psychological Science 22, 1359–1366.

[132]. Simpson, D., Rue, H., Riebler, A., Martins, T. G., and Sørbye, S. H. (2017). Penalising model component complexity: A principled, practical approach to constructing priors. Statistical Science 32, 1–28.

[133]. Singer, E., Van Hoewyk, J., Gebler, N., Raghunathan, T., and McGonagle, K. (1999). The effects of incentives on response rates in interviewer-mediated surveys. Journal of Official Statistics 15, 217–230.

[134]. Sivula, T., Magnusson, M, and Vehtari, A. (2020). Uncertainty in Bayesian leave-one-out cross-validation based model comparison. arxiv.org./abs/2008.10296

[135]. Skrondal, A. and Rabe-Hesketh, S. (2004). Generalized Latent Variable Modeling: Multilevel, Longitudinal and Structural Equation Models. London: CRC Press.

[136]. Smith, A. (2013). Sequential Monte Carlo Methods in Practice. New York: Springer.

[137]. Stan Development Team (2020). Stan User’s Guide. mc-stan.org

[138]. Steegen, S., Tuerlinckx, F., Gelman, A., and Vanpaemel, W. (2016). Increasing transparency through a multiverse analysis. Perspectives on Psychological Science 11, 702–712.

[139]. Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions (with discussion). Journal of the Royal Statistical Society B 36, 111–147.

[140]. Stone, M. (1977). An asymptotic equivalence of choice of model cross-validation and Akaike’s criterion. Journal of the Royal Statistical Society B 36, 44–47.

[141]. Talts, S., Betancourt, M., Simpson, D., Vehtari, A., and Gelman, A. (2020). Validating Bayesian inference algorithms with simulation-based calibration. www.stat.columbia.edu/~gelman/research/unpublished/sbc.pdf

[142]. Taylor, J., and Tibshirani, R. J. (2015). Statistical learning and selective inference. Proceedings of the National Academy of Sciences 112, 7629–7634.

[143]. Taylor, S. J., and Lethem, B. (2018). Forecasting at scale. American Statistician 72, 37–45. 

[144]. Tierney, L., and Kadane, J. B. (1986). Accurate approximations for posterior moments and marginal densities. Journal of the American Statistical Association 81, 82–86.

[145]. Turner, K. J., and Lambert, P. S. (2015). Workflows for quantitative data analysis in the social sciences. International Journal on Software Tools for Technology Transfer 17, 321–338.

[146]. Unwin, A., Volinsky, C., and Winkler, S. (2003). Parallel coordinates for exploratory modelling analysis. Computational Statistics and Data Analysis 43, 553–564.

[147]. Vehtari, A. (2019). Cross-validation for hierarchical models. avehtari.github.io/modelselection/rats_kcv.html

[148]. Vehtari A., Gabry J., Magnusson M., Yao Y., Bürkner P., Paananen T., Gelman A. (2020). loo: Efficient leave-one-out cross-validation and WAIC for Bayesian models. R package version 2.3.1, mc-stan.org/loo.

[149]. Vehtari, A., and Gabry, J. (2020). Bayesian stacking and pseudo-BMA weights using the loo package. mc-stan.org/loo/articles/loo2-weights.html

[150]. Vehtari, A., Gelman, A., and Gabry, J. (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. Statistics and Computing 27, 1413–1432.

[151]. Vehtari, A., Gelman, A., Simpson, D., Carpenter, D., and Bürkner, P.-C. (2020). Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC. Bayesian Analysis.

[152]. Vehtari, A., Gelman, A., Sivula, T., Jylanki, P., Tran, D., Sahai, S., Blomstedt, P., Cunningham,J. P., Schiminovich, D., and Robert, C. P. (2020). Expectation propagation as a way of life: A framework for Bayesian inference on partitioned data. Journal of Machine Learning Research 21, 1–53.

[153]. Vehtari, A., Simpson, D., Gelman, A., Yao, Y., and Gabry, J. (2015). Pareto smoothed importance sampling. arxiv.org/abs/1507.02646

[154]. Wang, W., and Gelman, A. (2015). Difficulty of selecting among multilevel models using predictive accuracy. Statistics and Its Interface 8 (2), 153–160.

[155]. Weber, S., Gelman, A., Lee, D., Betancourt, M., Vehtari, A., and Racine-Poon, A. (2018). Bayesian aggregation of average data: An application in drug development. Annals of Applied Statistics 12, 1583–1604.

[156]. Wickham, H. (2006). Exploratory model analysis with R and GGobi. had.co.nz/model-vis/2007-jsm.pdf

[157]. Wickham, H., Cook, D., and Hofmann, H. (2015). Visualizing statistical models: Removing the blindfold. Statistical Analysis and Data Mining: The ASA Data Science Journal 8, 203–225.

[158]. Wickham, H., and Groelmund, G. (2017). R for Data Science. Sebastopol, Calif.: O’Reilly.

[159]. Wilson, G., Aruliah, D. A., Brown, C. T., Hong, N. P. C., Davis, M., Guy, R. T., Haddock, S. H. D.,Huff, K. D., Mitchell, I. M., Plumbley, M. D., Waugh, B., White, E. P., and Wilson, P. (2014).Best practices for scientific computing. PLoS Biology 12, e1001745.

[160]. Wilson, G., Bryan, J., Cranston, K., Kitzes, J. Nederbragt, L., and Teal, T. K. (2017). Good enough practices in scientific computing. PLoS Computational Biololgy 13, e1005510.

[161]. Yao, Y., Cademartori, C., Vehtari, A., and Gelman, A. (2020). Adaptive path sampling in metastable posterior distributions. arxiv.org/abs/2009.00471

[162]. Yao, Y., Vehtari, A., and Gelman, A. (2020). Stacking for non-mixing Bayesian computations: The curse and blessing of multimodal posteriors. arxiv.org/abs/2006.12335

[163]. Yao, Y., Vehtari, A., Simpson, D., and Gelman, A. (2018a). Yes, but did it work?: Evaluating variational inference. In Proceedings of International Conference on Machine Learning, 5581–5590.

[164]. Yao, Y., Vehtari, A., Simpson, D., and Gelman, A. (2018b). Using stacking to average Bayesian predictive distributions (with discussion). Bayesian Analysis 13, 917–1003.

[165]. Yu, B., and Kumbier, K. (2020). Veridical data science. Proceedings of the National Academy of Sciences 117, 3920–3929.

[166]. Zhang, Y. D., Naughton, B. P., Bondell, H. D., and Reich, B. J. (2020). Bayesian regression using a prior on the model fit: The R2-D2 shrinkage prior. Journal of the American Statistical Association, doi:10.1080/01621459.2020.1825449



## 引文信息

@Article{gelman2020bayesian,
  author  = {Gelman, Andrew and Vehtari, Aki and Simpson, Daniel and Margossian, Charles C and Carpenter, Bob and Yao, Yuling and Kennedy, Lauren and Gabry, Jonah and B{\"u}rkner, Paul-Christian and Modr{\'a}k, Martin},
  journal = {arXiv preprint arXiv:2011.01808},
  title   = {Bayesian workflow},
  year    = {2020}
}