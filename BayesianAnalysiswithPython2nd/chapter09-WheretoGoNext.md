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

#  第 9 章 下一步去哪儿？

<style>p{text-indent:2em;2}</style>

本书面向那些已经熟悉 Python 和 Python 数据栈，但不太熟悉统计分析的读者，介绍了贝叶斯统计的主要概念和实践。读过前八章后，你应该对贝叶斯统计的许多主题有了实际理解。尽管你可能不会成为此方面专家，但应当能够创建概率模型来解决自己的数据分析问题了。

如果你真的对贝叶斯统计学感兴趣，本书显然不够。想要更加熟练地掌握贝叶斯统计，你需要练习、时间、耐心、热情和更多的练习，你需要从不同视角重新审视一些思想和概念。

在 [本书源码的 Github 仓库](https://​/​github.com.​aloctavodia/​BAP) 中，你可以找到书中讨论内容的示例代码。由于空间或时间的原因，有些例子未在书中体现，事实上，有些示例在写本书时还不存在。我会不时地在那里添加一些新的示例。

要掌握额外的资料，你还应该查看 [PyMC3 文档](https:/​/​docs.pymc.io)，尤其是案例部分，其中包含了本书以及本书以外许多模型的示例。另外，[ArviZ](https:/​/github.​com/​arviz-​devs/​arviz_​resources) 虽然是一个非常新的库，但我们已经在编写利用它做探索性分析的教育资源，希望是一个有用的参考，特别是对贝叶斯建模的新手。

如果你在本书中发现错误，无论是文本还是代码，可以在 [Github 仓库](https://​/​github.com.​aloctavodia/​BAP) 上提交问题。如果有关于贝叶斯统计的一般性问题，特别是与 PyMC3 或 arviz 相关的问题，可以在​ [PyMC3](https://discourse.pymc.io/)上提问。

下面列出了一些确实影响了作者贝叶斯思维方式的材料。相信你也会发现其中至少一部分非常有用和令人兴奋。

**（1）从贝叶斯统计角度出发的数据**

- 强烈推荐你读 Richard McElreath 的 《统计反思 (Statistics Rethinking)》。这是有关贝叶斯分析极好的介绍性书籍，不过代码示例是基于 R 和 STAN 的。一些志愿者将该书的示例移植到 Python 和 PyMC3，更多信息请查看 [GitHub 仓库](https://github.com/pymc-devs/resources/blob/master/Rethinking/)。

> 读后感： 不像作者说的那么好，感觉该书中有些例子和语言容易让中文读者费解，此外中文版翻译水平真是不敢沟通，这可能源于文化差异。
  
- 另一本移植到 PyMC3 上的书籍是 John K.Kruschke 的《做贝叶斯分析（Doing Bayesian Analysis）》（也被称为小狗书）。这是一本关于贝叶斯分析的入门好书。该书第一版的大多数示例已移植到 [GitHub 仓库](https://github.com/aloctavodia/Doing_bayesian_data_analysis)中。你可以在 [这里](https:/​/​github.​com/​JWarmenhoven/​DBDApython) 找到第二版的源码。与《统计反思》 不同，该书更侧重于贝叶斯与常用频率主义模型之间的对比。
  
- Allen B. Downey 写了很多好书，[《Think Bayes》](http://greenteapress.com/wp/Think-Bayes/​) 也不例外 。该书有一些有趣的案例和场景肯定帮助你掌握解决问题的贝叶斯方法。不过该书没有使用 PyMC3，而是专门构建的 Python 库。第二版在本书出版时还没有编写，它将使用 PyMC3，甚至可能使用 ArviZ。你可以查看 [第二版的仓库](https:/​/​github.​com/​AllenDowney/​ThinkBayes2)。

- Cameron Davidson-Pilon 等编写的《概率编程和贝叶斯方法 （Probabilistic Programming and Bayesian Methods for Hackers）》一书，最初用 PyMC2 编写，现在已经移植到 [PyMC3](https://github.com/quantopian/Probabilistic-Programming-and-Bayesian-Methods-for-Hacker) 。


- Andrew Gelman 等人撰写的 《贝叶斯数据分析 （Bayesian Data Analysis）》被称为真正的 『贝叶斯』书。这不是一本入门书，可能更适合作为参考书。如果你不熟悉统计学，建议首先选择 Richard McElreath 的《统计反思》，然后再尝试《贝叶斯数据分析》。你可能还想看看 Andrew Gelman 和 Jennifer Hill 合作撰写的 《 使用回归和分层模型进行数据分析（Data Analysis Using Regression and Multilevel/Hierarchical Models）》

- 如果你想继续学习高斯过程，请阅读 Carl Edward Rasmussen 和 Christopher K.I.Williams 合著的《机器学习中的高斯过程（Gaussian Processes for Machine Learning）》。该书被授予 2009 年国际贝叶斯分析学会的德格鲁特奖，其唯一缺点是大家都希望有新的版本！

 **（2） 从机器学习角度出发的书籍**

下面是几本带有贝叶斯特色的机器学习书籍：

- 《机器学习：概率视角（Machine Learning: A probabilistic Probabilistic Perspective）》，Kevin Murphy著。这是很棒的一本书，它试图解释有多少方法和模型是通过概率方法来实现的。该书可能有点枯燥，或者非常简练和切中要害。无论哪种方式，书中都充满了例子，而且写得非常实用。Kevin Murphy 引用了许多其他来源的例子，因此该书也是对许多其他资源的极好总结。作者本人第一次听说深度学习的概念，也是来自该书。

- Christopher Bishop 的 《模式识别与机器学习 （Pattern Recognition And Machine Learning）》是机器学习的经典著作。概述拥有更多贝叶斯视角，不过与《机器学习：概率视角》有相当大重叠。如果作为教科书，它可能比 Murphy 的书更容易阅读。

当我还是孩子的时候，梦想着会飞的汽车、清洁的无限能源、在火星上度假、一个追求全人类福祉的全球政府……。是啊，我曾经是个梦想家！出于许多原因，这些梦想都没有实现。但取而代之，我们有了一些完全不可想象的东西：计算机方法的公平化。计算机革命的副作用之一是，任何对 Python 这样的编程语言有了解的人，现在都可以使用计算方法来进行数据分析、模拟和其他复杂任务。我在大学时学习统计学的方式，以及我必须记住如何使用预制方法的方式令人沮丧，毫无用处，与所有这些变化完全无关。在非常个人的层面上，该书可能是对那段令人沮丧的经历的回应。

我试图写一本统计书籍，强调建模方法和明智的上下文相关分析。我不确定我在这条战线上是否真的成功了。其中一个原因可能是我仍然需要更多地了解这一点（也许我们作为一个社区需要更多地了解这一点）。另一个原因是，正确的统计分析应该以领域知识和上下文为指导，在一本目标受众非常广泛的书中，提供上下文通常是困难的。尽管如此，我希望我对统计模型提供了一个理智的、持怀疑态度的观点，提供了一些有用的例子，并提供了足够的动力让你继续学习。
