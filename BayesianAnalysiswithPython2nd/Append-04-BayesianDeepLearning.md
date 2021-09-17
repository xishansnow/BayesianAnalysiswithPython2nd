---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.12.0
kernelspec:
  display_name: Python 3
  language: ipython3
  name: python3
---


# 附录 D：贝叶斯神经网络的实践 -- 面向深度学习用户的教程

[原文](https://arxiv.org/abs/2007.06823)

[作者]
- LAURENT VALENTIN JOSPIN,University of Western Australia
- WRAY BUNTINE,Monash University
- FARID BOUSSAID,University of Western Australia
- HAMID LAGA,Murdoch university
- MOHAMMED BENNAMOUN,University of Western Australia

[引用]
Laurent Valentin Jospin, Wray Buntine, Farid Boussaid, Hamid Laga, and Mohammed Bennamoun. 2020.Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users.ACM Comput. Surv.1, 1 ( July 2020),35 pages

<style>p{text-indent:2em;2}</style>

现代深度学习方法已经为研究人员和工程师提供了令人难以置信的强大工具，以解决以前似乎不可能解决的问题。然而，由于深度学习方法是作为黑箱操作的，与他们的预测相关的不确定性往往是难以量化的。贝叶斯统计学提供了一个形式化的方法来理解和量化与深度神经网络预测相关的不确定性。本文为正在使用机器学习，特别是深度学习的研究人员和科学家提供了一个相关文献的概述和一个完整的工具集来设计、实现、训练、使用和评估贝叶斯神经网络。

## 1 简介

深度学习导致了机器学习的革命，为解决现实生活中复杂而具有挑战性的问题提供了解决方案。然而，深度学习模型容易过拟合，这对其泛化能力产生了不利影响。深度学习模型也倾向于对其预测结果过于自信（当他们提供一个置信区间时）。所有这些对于诸如自动驾驶汽车[74]、医疗诊断[38]或交易和金融[11]等应用来说都是有问题的，因为无声的失败会导致戏剧性的结果。因此，人们提出了许多方法来减轻风险，特别是通过随机神经网络来估计模型预测的不确定性。贝叶斯范式为分析和训练随机神经网络提供了严格框架，并且更广泛地支持了学习算法的发展。


图 1.  本文所涉及的主题的思维导图。这些可以大致分为贝叶斯深度神经网络的概念、不同的（严格意义上的或近似于贝叶斯的）学习方法、评估方法，以及研究人员可用于实施的工具集。