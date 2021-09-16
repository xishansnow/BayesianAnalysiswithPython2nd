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

# 贝叶斯深度学习

原文：https://twiecki.io/blog/2016/06/01/bayesian-deep-learning/

<style>p{text-indent:2em;2}</style>


## 1 概述

### 1.1 机器学习的当前趋势

目前机器学习有三大趋势：`概率编程`、`深度学习`和`大数据`。在概率编程中，很多创新使用`变分推断`以使事物能够适应规模。在本文中，我将展示如何在 `PyMC3` 中使用`变分推断`来拟合一个简单的贝叶斯神经网络。我还将讨论如何将概率编程和深度学习联系起来，以便在未来研究中开辟非常有趣的途径。

### 1.2 大规模概率编程

概率编程允许灵活地创建自定义概率模型，并且主要关注洞察力和从数据中学习。该方法本质上是贝叶斯的，因此可以指定先验来约束模型，并以后验分布的形式获得不确定性估计。使用 `MCMC` 采样算法，可以从后验中抽取样本来灵活估计这些模型。 `PyMC3` 和 `Stan` 是当前构建和估计此类模型的领先工具。然而，采样法的主要缺点是非常慢，尤其是对于高维模型。这就是为什么最近`变分推断算法`与 `MCMC` 一样灵活，但速度要快得多。

变分算法不从后验中抽取样本，而是将某种分布（例如正态）拟合到后验上，将采样问题转化为优化问题。 自动微分变分推断（ `ADVI` ）方法在 `PyMC3` 、 `Stan` 和 `Edward`（现在改为 `TensorFlow Probability -- TFP`）中被作为变分推断的主要算法实现。

不幸的是：在传统的分类或回归等机器学习问题中，概率编程通常在非参数学习、集成学习（例如随机森林或梯度提升回归树）等算法中扮演着次要角色。

### 1.3 深度学习 

现在深度学习已经成为热点，支配了几乎所有的物体识别基准，在 [`Atari` 游戏](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)中获胜，并且[战胜了世界围棋冠军李世石](http://www.nature.com/nature/journal/v529/n7587/full/nature16961.html)。从统计学角度看，神经网络非常擅长非线性函数逼近和表征学习。深度学习不仅用于分类任务，还可以通过自编码器和其他各种有趣的方法扩展到非监督学习，例如：采用 [循环神经网络 `RNN`](https://en.wikipedia.org/wiki/Recurrent_neural_network)，或[混合密度神经网络 `MDN`](http://cbonnett.github.io/MDN_EDWARD_KERAS_TF.html) 来估计多模态分布。其效果为何如此好？没有人真正知道原因，因为有些统计特性仍不为人完全理解。

深度学习的优势之一是可以训练极其复杂的模型。这依赖于几个基础：

- **速度：** 提高GPU性能获得更快的处理。
- **软件：** 像[Theano](http://deeplearning.net/software/theano/)和[TensorFlow](https://www.tensorflow.org/)这样的框架允许灵活创建抽象模型，然后可以对其优化并编译到 `CPU` 或 `GPU` 上。
- **学习算法：** 在数据子集上作随机梯度下降可以让我们在海量数据上训练这些模型。使用 `drop-out` 等技术可避免过拟合。
- **架构：** 大量创新都是改变输入层，比如卷积神经网络，或改变输出层，比如[MDN](http://cbonnett.github.io/MDN_EDWARD_KERAS_TF.html)。


### 1.4 连接深度学习和概率编程

一方面，概率编程允许我们以非常有原则和易于理解的方式构建相当小但聚焦的模型，以深入了解数据；另一方面，深度学习使用启发式方法来训练巨大且高度复杂的模型，并在预测方面非常出色。

然而，最新的 `变分推断` 方法允许概率编程扩展模型复杂性和数据大小。因此，我们正处于能够将两种方法结合起来以开启机器学习新创新的风口浪尖。如需更多动机，另请参阅 `Dustin Tran` 最近的 [博客文章](http://dustintran.com/blog/a-quick-update-edward-and-some-motivations/)。

虽然这种变化将使概率编程能够应用于更广泛的问题，但我相信这种变化也为深度学习的创新带来了巨大可能性。一些基本的想法包括： 

- **预测的不确定性：** 正如下面将看到的，贝叶斯神经网络能够得到其预测的不确定性。我认为不确定性在机器学习中是一个被低估的概念，因为它不仅对现实世界的应用很重要，而且在训练中也很有用。例如，可以专门针对最不确定的样本训练模型。

- **表征的不确定性：** 通过贝叶斯神经网络能够获得权重的不确定性估计，进而告诉我们表征学习的稳定性。

- **使用先验进行正则化：** 权重通常被 `L2 正则化`以避免过度拟合，这等效于为权重参数设置高斯先验。但我们可以设想其他先验，比如用`尖板分布(spike-and-slab)` 来强化稀疏性（更像使用 L1 正则）。

- **有先验知识的迁移学习：** 如果想在新对象识别数据集上训练网络，我们可以为网络权重设置以某些预训练网络（如 `GoogLeNet`）权重为中心的先验，进而引导新网络的训练和学习。

- **分层神经网络：** 概率编程中一个非常强大的方法是分层模型，它允许将在子组上学到的东西汇集到总体中（参见 `PyMC3` 中的[分层线性回归教程](https://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/)）。而结合神经网络，可以在分层数据集中训练单个神经网络以专注于子组，同时仍然了解整体人口的表征。例如：想象一个经过训练的网络，可以从汽车图片中对汽车模型进行分类。我们可以设置一个分层神经网络，其中训练子神经网络来区分单个制造商的模型。直觉是来自某个制造商的汽车都有某些相似之处，因此训练针对品牌的专门子网络是有意义的。但由于各网络在更高层存在连接，它们仍会与其他专门子网络共享有关所有品牌的有用信息。有趣的是，神经网络的不同层可以通过层次结构的各层来通知，例如：提取线条的网络层在所有子网络中可能是相同的，而高阶表征会有所不同。分层模型将从数据中学习所有这些信息。

- **其他混合架构：** 我们可以更自由地构建各种神经网络。例如，非参数的贝叶斯模型可用于灵活调整隐藏层的大小和形状，以便在训练期间针对手头问题优化网络架构。目前，这需要代价高昂的超参数优化和大量知识。

## 2 PyMC3 中的贝叶斯神经网络

### 2.1 生成数据

首先，生成一些有关玩具的数据，用于分析一个非线性可分的简单二分类问题。

```{code-cell}
%matplotlib inline
import theano 
import pymc3 as pm
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
sns.set_style('white')
from sklearn import datasets
from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons

X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
X = scale(X)
X = X.astype(floatX)
Y = Y.astype(floatX)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[Y==0, 0], X[Y==0, 1], label='Class 0')
ax.scatter(X[Y==1, 0], X[Y==1, 1], color='r', label='Class 1')
sns.despine(); ax.legend()
ax.set(xlabel='X', ylabel='Y', title='Toy binary classification data set');
```

### 2.2 模型定义

该模型对应的神经网络非常简单。基本单元是一个 `logistic 回归` 的感知机。我们并行使用多个感知机，然后将它们堆叠起来以获得隐藏层。此处将使用 2 个隐藏层，每个隐藏层有 5 个神经元，这足以解决本例的简单问题。

```{code-cell}
def construct_nn(ann_input, ann_output):
  n_hidden = 5
  
  # Initialize random weights between each layer
  init_1 = np.random.randn(X.shape[1], n_hidden).astype(floatX)
  init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
  init_out = np.random.randn(n_hidden).astype(floatX)
    
  with pm.Model() as neural_network:
    # Weights from input to hidden layer
    weights_in_1 = pm.Normal('w_in_1', 0, sd=1, 
                 shape=(X.shape[1], n_hidden), 
                 testval=init_1)
    
    # Weights from 1st to 2nd layer
    weights_1_2 = pm.Normal('w_1_2', 0, sd=1, 
                shape=(n_hidden, n_hidden), 
                testval=init_2)
    
    # Weights from hidden layer to output
    weights_2_out = pm.Normal('w_2_out', 0, sd=1, 
                 shape=(n_hidden,), 
                 testval=init_out)
    
    # Build neural-network using tanh activation function
    act_1 = pm.math.tanh(pm.math.dot(ann_input, 
                     weights_in_1))
    act_2 = pm.math.tanh(pm.math.dot(act_1, 
                     weights_1_2))
    act_out = pm.math.sigmoid(pm.math.dot(act_2, 
                       weights_2_out))
    
    # Binary classification -> Bernoulli likelihood
    out = pm.Bernoulli('out', 
              act_out,
              observed=ann_output,
              total_size=Y_train.shape[0] # IMPORTANT for minibatches
             )
  return neural_network

# Trick: Turn inputs and outputs into shared variables. 
# It's still the same thing, but we can later change the values of the shared variable 
# (to switch in the test-data later) and `PyMC3` will just use the new data. 
# Kind-of like a pointer we can redirect.
# For more info, see: http://deeplearning.net/software/ `Theano` /library/compile/shared.html
ann_input = `Theano` .shared(X_train)
ann_output = `Theano` .shared(Y_train)
neural_network = construct_nn(ann_input, ann_output)
```

正态先验有助于正则化权重。通常我们会在输入中添加一个常量 `b`，但在这里省略了它以保持代码清晰。

### 2.3 `变分推断`：提升模型的复杂性

现在可以运行像 `NUTS` 这样的随机采样器，由于模型并不复杂，所以工作得很好，但随着将模型扩展到具有更多层的深度架构，随机方法将变得非常缓慢。此时，使用 OPVI 框架中的ADVI 自动变分推断算法，就会快得多，并可更好地扩展。

```{admonition} 请注意
ADVI 采用平均场近似，因此此处忽略了后验中的相关性。
```

```{code-cell}
from PyMC3.Theanof import set_tt_rng, MRG_RandomStreams
set_tt_rng(MRG_RandomStreams(42))
```

```{code-cell}
%%time

with neural_network:
  inference = pm.ADVI()
  approx = pm.fit(n=50000, method=inference)
```

下面为了让它真正快起来，我们可能希望在 `GPU` 上运行神经网络。

由于样本更方便后续处理，因此可以从变分近似推断结果中采样（此处仅简单从推断得到得正态分布中采样），而这与 `MCMC` 方法完全不同：

```{code-cell}
trace = approx.sample(draws=5000)
```

绘制目标函数（ELBO），可以看到随着时间推移，优化算法在逐渐改善拟合。

```{code-cell}
plt.plot(-inference.hist)
plt.ylabel('ELBO')
plt.xlabel('iteration');
```

现在已经训练了模型，可以使用后验预测检查来预测验证集。

```{code-cell}
# Replace arrays our NN references with the test data
ann_input.set_value(X_test)
ann_output.set_value(Y_test)

with neural_network:
  ppc = pm.sample_ppc(trace, samples=500, progressbar=False)

# Use probability of > 0.5 to assume prediction of class 1
pred = ppc['out'].mean(axis=0) > 0.5
```

看看预测结果：

```{code-cell}
fig, ax = plt.subplots()
ax.scatter(X_test[pred==0, 0], X_test[pred==0, 1])
ax.scatter(X_test[pred==1, 0], X_test[pred==1, 1], color='r')
sns.despine()
ax.set(title='Predicted labels in testing set', xlabel='X', ylabel='Y');
```

```{code-cell}
print('Accuracy = {}%'.format((Y_test == pred).mean() * 100))
```

可以看出，我们的神经网络做得很好！


### 2.4 分类器到底学到了什么？

我们在整个输入空间的网格上评估类概率的预测。

```{code-cell}
grid = pm.floatX(np.mgrid[-3:3:100j,-3:3:100j])
grid_2d = grid.reshape(2, -1).T
dummy_out = np.ones(grid.shape[1], dtype=np.int8)
```

```{code-cell}
ann_input.set_value(grid_2d)
ann_output.set_value(dummy_out)

with neural_network:
  ppc = pm.sample_ppc(trace, samples=500, progressbar=False)
```

### 2.5 概率曲面

```{code-cell}
cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
fig, ax = plt.subplots(figsize=(14, 8))
contour = ax.contourf(grid[0], grid[1], ppc['out'].mean(axis=0).reshape(100, 100), cmap=cmap)
ax.scatter(X_test[pred==0, 0], X_test[pred==0, 1])
ax.scatter(X_test[pred==1, 0], X_test[pred==1, 1], color='r')
cbar = plt.colorbar(contour, ax=ax)
_ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='X', ylabel='Y');
cbar.ax.set_ylabel('Posterior predictive mean probability of class label = 0');
```

### 2.6 预测值的不确定性

目前展示的一切，都可以用非贝叶斯神经网络来完成。每个类别标签的后验预测值的平均值应当与最大似然预测值相同。但，在贝叶斯神经网络中，我们还可以查看后验预测值的标准差，来了解预测的不确定性：

```{code-cell}
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
fig, ax = plt.subplots(figsize=(14, 8))
contour = ax.contourf(grid[0], grid[1], ppc['out'].std(axis=0).reshape(100, 100), cmap=cmap)
ax.scatter(X_test[pred==0, 0], X_test[pred==0, 1])
ax.scatter(X_test[pred==1, 0], X_test[pred==1, 1], color='r')
cbar = plt.colorbar(contour, ax=ax)
_ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel='X', ylabel='Y');
cbar.ax.set_ylabel('Uncertainty (posterior predictive standard deviation)');
```

可以看到，在非常接近决策边界时，预测标签类别的不确定性最高。可以想象，将预测与不确定性关联起来是许多应用（如医疗保健）的关键特性。为进一步提高精度，我们可能希望主要根据高不确定性区域的样本来训练模型。

### 2.7 小批量 ADVI

由于训练数据集规模较小，前面采用的方法直接在所有数据上同时训练模型。但这显然无法扩展到类似 `ImageNet` 这种海量数据集上。此外，小批量数据训练的随机梯度下降方法可有效避免局部极小值，并更快的收敛。因此，贝叶斯神经网络应当具备支持小批量训练的能力。

幸运的是，`ADVI` 支持小批量运行。只需做一些简单设置：

```{code-cell}
minibatch_x = pm.Minibatch(X_train, batch_size=32)
minibatch_y = pm.Minibatch(Y_train, batch_size=32)

neural_network_minibatch = construct_nn(minibatch_x, minibatch_y)
with neural_network_minibatch:
  inference = pm.ADVI()
  approx = pm.fit(40000, method=inference)
```

```{code-cell}
  plt.plot(-inference.hist)
  plt.ylabel('ELBO')
  plt.xlabel('iteration')
```

小批量 `ADVI` 的运行时间短得多，其收敛得也更快，并且可以查看迹图。而最关键的是同时能够得到神经网络权重的不确定性度量。

```{code-cell}
  pm.traceplot(trace)
```

注意：

```{note} 
您在 GPU 上运行上述示例，但需要在 .theanorc 文件中设置 device = GPU 和  floatX = float32 。您可能会说，上述模型并不是真正的深层网络，但实际上我们可以轻松地将其扩展到更多层（包括卷积层），以便在更具挑战性的数据集上进行训练。
```

我还在 `PyData London` 上介绍了一些这方面的工作，请观看以下 [视频](https://www.youtube.com/watch?v=LlzVlqVzeD8&feature=emb_imp_woyt)。此外，你可以从 [此处](https://github.com/twiecki/WhileMyMCMCGentlySamples/blob/master/content/downloads/notebooks/bayesian_neural_network.ipynb) 下载本例对应的 NoteBook。

## 3 使用 PyMC3 和 Lasagne 构建更为复杂的分层神经网络

`Lasagne` 是一个灵活的 `Theano` 库，用于构建各种类型的神经网络。`PyMC3` 也在使用 `Theano` ，因此将两者结合用于构建复杂的贝叶斯神经网络是可能的。可以利用 `Lasagne` 构建人工神经网络（`ANN`），并在参数上设置贝叶斯先验，然后利用 `PyMC3` 的 `ADVI 变分推断` 来估计模型。令人兴奋的是，这个想法不仅可能，而且非常直接。

下面，首先展示如何集成 `PyMC3` 和 `Lasagne` ，以构建一个密集的 2 层人工神经网络；然后使用`小批量 ADVI` 在 `MNIST` 手写数字数据集上拟合模型；最后实现分层贝叶斯神经网络模型。由于 `Lasagne` 的强大能力，我们同样可以轻松构建具有最大池层的分层贝叶斯卷积神经网络，并在 `MNIST` 上实现 98% 的准确率。

此处使用的大部分代码都是从 [`Lasagne` 教程](http:// `Lasagne` .readthedocs.io/en/latest/user/tutorial.html) 中借用而来。

```{code-cell}
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')

import PyMC3 as pm
import Theano.tensor as T
import Theano 

from scipy.stats import mode, chisquare

from sklearn.metrics import confusion_matrix, accuracy_score

import Lasagne 
```

### 3.1 数据集：MNIST

我们将使用手写数字的经典 MNIST 数据集。与我上一篇仅限于玩具数据集的博文相反，MNIST 实际上是一项具有挑战性的 ML 任务（当然不像 ImageNet 那样具有挑战性），具有合理数量的维度和数据点。

```{code-cell}
import sys, os

def load_dataset():
  # We first define a download function, supporting both Python 2 and 3.
  if sys.version_info[0] == 2:
    from urllib import urlretrieve
  else:
    from urllib.request import urlretrieve

  def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)

  # We then define functions for loading MNIST images and labels.
  # For convenience, they also download the requested files if needed.
  import gzip

  def load_mnist_images(filename):
    if not os.path.exists(filename):
      download(filename)
    # Read the inputs in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    data = data.reshape(-1, 1, 28, 28)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)

  def load_mnist_labels(filename):
    if not os.path.exists(filename):
      download(filename)
    # Read the labels in Yann LeCun's binary format.
    with gzip.open(filename, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data

  # We can now download and read the training and test set images and labels.
  X_train = load_mnist_images('train-images-idx3-ubyte.gz')
  y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
  X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
  y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

  # We reserve the last 10000 training examples for validation.
  X_train, X_val = X_train[:-10000], X_train[-10000:]
  y_train, y_val = y_train[:-10000], y_train[-10000:]

  # We just return all the arrays in order, as expected in main().
  # (It doesn't matter how we do this as long as we can read them again.)
  return X_train, y_train, X_val, y_val, X_test, y_test

print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
```

```{code-cell}
# Building a `Theano` .shared variable with a subset of the data to make construction of the model faster.
# We will later switch that out, this is just a placeholder to get the dimensionality right.
input_var = `Theano` .shared(X_train[:500, ...].astype(np.float64))
target_var = `Theano` .shared(y_train[:500, ...].astype(np.float64))
```

### 3.2 模型定义

我认为，仅仅因为意大利宽面条和 `PyMC3` 都依赖于西亚诺，就有可能将它们连接起来。然而，不清楚这到底有多困难。幸运的是，第一个实验效果很好，但有一些潜在的方法可以使这变得更容易。我在 `Lasagne` 的回购协议上发行了一期 GitHub，几天后，PR695 被合并，这使得两者能够更好地融合，如下所示。OSS 万岁。

首先， `Lasagne` 函数用于创建一个 ANN，其中有两个完全连接的隐藏层，每个层有 800 个神经元，这是纯 `Lasagne` 代码，几乎直接取自教程。在用 `Lasagne` 创建图层时会用到技巧。层。DenseLayer，在这里我们可以传入一个函数 init，它必须返回一个 `Theano` 表达式作为权重和偏差矩阵。这就是我们将传递 `PyMC3` 创建的优先级的地方，这些优先级也是无表达式：

```{code-cell}
def build_ann(init):
  l_in = `Lasagne` .layers.InputLayer(shape=(None, 1, 28, 28),
                   input_var=input_var)

  # Add a fully-connected layer of 800 units, using the linear rectifier, and
  # initializing weights with Glorot's scheme (which is the default anyway):
  n_hid1 = 800
  l_hid1 = `Lasagne` .layers.DenseLayer(
    l_in, num_units=n_hid1,
    nonlinearity= `Lasagne` .nonlinearities.tanh,
    b=init,
    W=init
  )

  n_hid2 = 800
  # Another 800-unit layer:
  l_hid2 = `Lasagne` .layers.DenseLayer(
    l_hid1, num_units=n_hid2,
    nonlinearity= `Lasagne` .nonlinearities.tanh,
    b=init,
    W=init
  )

  # Finally, we'll add the fully-connected output layer, of 10 softmax units:
  l_out = `Lasagne` .layers.DenseLayer(
    l_hid2, num_units=10,
    nonlinearity= `Lasagne` .nonlinearities.softmax,
    b=init,
    W=init
  )
  
  prediction = `Lasagne` .layers.get_output(l_out)
  
  # 10 discrete output classes -> `PyMC3` categorical distribution
  out = pm.Categorical('out', 
             prediction,
             observed=target_var)
  
  return out
```

  接下来是为 ANN 创建权重的函数。因为 `PyMC3` 要求每个随机变量都有不同的名称，所以我们将创建一个类来创建唯一命名的优先级。

  先验值在这里充当正则化器，以尝试保持 ANN 的权重较小。这在数学上相当于将惩罚较大权重的 L2 损失项放入目标函数中，就像通常所做的那样。

```{code-cell}
class GaussWeights(object):
def __init__(self):
  self.count = 0
def __call__(self, shape):
  self.count += 1
  return pm.Normal('w%d' % self.count, mu=0, sd=.1, 
           testval=np.random.normal(size=shape).astype(np.float64),
           shape=shape)
```

  如果你把我们到目前为止所做的与之前的博客文章相比较，很明显，使用 `Lasagne` 要舒服得多。我们不必手动跟踪单个矩阵的形状，也不必处理底层矩阵的数学运算以使其全部匹配在一起。

  接下来是一些设置 mini-batch-ADVI 的函数，您可以在 [前面的博文](https://twiecki.github.io/blog/2016/06/01/bayesian-deep-learning/) 中找到更多信息。

```{code-cell}
  from six.moves import zip

# Tensors and RV that will be using mini-batches
minibatch_tensors = [input_var, target_var]

# Generator that returns mini-batches in each iteration
def create_minibatch(data, batchsize=500):
  
  rng = np.random.RandomState(0)
  start_idx = 0
  while True:
    # Return random data samples of set size batchsize each iteration
    ixs = rng.randint(data.shape[0], size=batchsize)
    yield data[ixs]

minibatches = zip(
  create_minibatch(X_train, 500),
  create_minibatch(y_train, 500),
)

total_size = len(y_train)

def run_advi(likelihood, advi_iters=50000):
  # Train on train data
  input_var.set_value(X_train[:500, ...])
  target_var.set_value(y_train[:500, ...])
  
  v_params = pm.variational.advi_minibatch(
    n=advi_iters, minibatch_tensors=minibatch_tensors, 
    minibatch_RVs=[likelihood], minibatches=minibatches, 
    total_size=total_size, learning_rate=1e-2, epsilon=1.0
  )
  trace = pm.variational.sample_vp(v_params, draws=500)
  
  # Predict on test data
  input_var.set_value(X_test)
  target_var.set_value(y_test)
  
  ppc = pm.sample_ppc(trace, samples=100)
  y_pred = mode(ppc['out'], axis=0).mode[0, :]
  
  return v_params, trace, ppc, y_pred
```

###  3.3 模型训练

让我们使用`小批量 ADVI` 运行模型：

```{code-cell}
with pm.Model() as neural_network:
  likelihood = build_ann(GaussWeights())
  v_params, trace, ppc, y_pred = run_advi(likelihood)
```

确保所有内容都收敛：

```{code-cell}
plt.plot(v_params.elbo_vals[10000:])
sns.despine()
```

```{code-cell}
sns.heatmap(confusion_matrix(y_test, y_pred))
```

```{code-cell}
print('Accuracy on test data = {}%'.format(accuracy_score(y_test, y_pred) * 100))
```

性能并不是非常高，但嘿，它似乎真的起作用了。

### 3.4 分层贝叶斯神经网络

`L2 惩罚项`强度之前的重量标准偏差之间的联系引出了一个有趣的想法。上面我们刚刚修正了 sd=0。1 表示所有层，但第一层的值可能与第二层的值不同。也许是 0。1 开始时太小或太大。在贝叶斯建模中，在这样的情况下，通常只需放置超先验，然后从数据中学习要应用的最佳正则化。这使我们不用在代价高昂的超参数优化中调整该参数。有关分层建模的更多信息，请参阅我的 [另一篇博文](https://twiecki.io/blog/2016/07/05/bayesian-deep-learning/)。

```{code-cell}
class GaussWeightsHierarchicalRegularization(object):
  def __init__(self):
    self.count = 0
  def __call__(self, shape):
    self.count += 1
    
    regularization = pm.HalfNormal('reg_hyper%d' % self.count, sd=1)
    
    return pm.Normal('w%d' % self.count, mu=0, sd=regularization, 
             testval=np.random.normal(size=shape),
             shape=shape)
  
minibatches = zip(
  create_minibatch(X_train, 500),
  create_minibatch(y_train, 500),
)
```

```{code-cell}
with pm.Model() as neural_network_hier:
  likelihood = build_ann(GaussWeightsHierarchicalRegularization())
  v_params, trace, ppc, y_pred = run_advi(likelihood)
```

```{code-cell}
print('Accuracy on test data = {}%'.format(accuracy_score(y_test, y_pred) * 100))
```

我们得到了一个小但很好的提高精度。让我们看看超参数的后验值：

```{code-cell}
pm.traceplot(trace, varnames=['reg_hyper1', 'reg_hyper2', 'reg_hyper3', 'reg_hyper4', 'reg_hyper5', 'reg_hyper6']);
```

有趣的是，它们都非常不同，这表明改变在网络的每一层应用的正则化量是有意义的。

### 3.5 卷积神经网络的贝叶斯模型

这很好，但正如我在上一篇文章中所展示的，到目前为止，在 `PyMC3` 中直接实现的一切都非常简单。真正有趣的是，我们现在可以构建更复杂的人工神经网络，比如卷积神经网络：

```{code-cell}
def build_ann_conv(init):
  network = `Lasagne` .layers.InputLayer(shape=(None, 1, 28, 28),
                    input_var=input_var)

  network = `Lasagne` .layers.Conv2DLayer(
      network, num_filters=32, filter_size=(5, 5),
      nonlinearity= `Lasagne` .nonlinearities.tanh,
      W=init)

  # Max-pooling layer of factor 2 in both dimensions:
  network = `Lasagne` .layers.MaxPool2DLayer(network, pool_size=(2, 2))

  # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
  network = `Lasagne` .layers.Conv2DLayer(
    network, num_filters=32, filter_size=(5, 5),
    nonlinearity= `Lasagne` .nonlinearities.tanh,
    W=init)
  
  network = `Lasagne` .layers.MaxPool2DLayer(network, 
                      pool_size=(2, 2))
  
  n_hid2 = 256
  network = `Lasagne` .layers.DenseLayer(
    network, num_units=n_hid2,
    nonlinearity= `Lasagne` .nonlinearities.tanh,
    b=init,
    W=init
  )

  # Finally, we'll add the fully-connected output layer, of 10 softmax units:
  network = `Lasagne` .layers.DenseLayer(
    network, num_units=10,
    nonlinearity= `Lasagne` .nonlinearities.softmax,
    b=init,
    W=init
  )
  
  prediction = `Lasagne` .layers.get_output(network)
  
  return pm.Categorical('out', 
          prediction,
          observed=target_var)
```

```{code-cell}
print('Accuracy on test data = {}%'.format(accuracy_score(y_test, y_pred) * 100))          
```

更高的准确度——很好。我也尝试了分层模型，但它的准确率较低（95%），我认为这是由于过度拟合。

让我们更多地利用我们处于贝叶斯框架中的事实，探索我们预测中的不确定性。由于我们的预测是分类的，我们不能简单地计算后验预测标准差。相反，我们计算卡方统计量，它告诉我们样本是多么均匀。越是统一，我们的不确定性就越高。我不太确定这是否是最好的方法，如果有一个更成熟的方法我不知道，请留下评论。

正如我们所见，当模型出错时，答案的不确定性要大得多（即提供的答案更加一致）。你可能会说，你从一个常规的人工神经网络得到的多项式预测得到了同样的效果，然而，[事实并非如此](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)。

```{code-cell}
miss_class = np.where(y_test != y_pred)[0]
corr_class = np.where(y_test == y_pred)[0]
```

```{code-cell}
preds = pd.DataFrame(ppc['out']).T
```

```{code-cell}
chis = preds.apply(lambda x: chisquare(x).statistic, axis='columns')
```

```{code-cell}
sns.distplot(chis.loc[miss_class].dropna(), label='Error')
sns.distplot(chis.loc[corr_class].dropna(), label='Correct')
plt.legend()
sns.despine()
plt.xlabel('Chi-Square statistic');
```

### 3.6 结论

通过连接 `Lasagne` 和 `PyMC3` ，并使用 mini-batch ADVI 在大小适中的复杂数据集（MNIST）上训练贝叶斯神经网络，我们朝着实际问题的贝叶斯深度学习迈出了一大步。

 `Lasagne` 开发人员设计了 API，使得为这个不常见的应用程序集成变得微不足道，这是他们的荣幸。他们也非常乐于助人，乐于助人。

最后，我还认为这显示了 `PyMC3` 的好处。通过依赖一种常用语言（Python）和抽象计算后端（ `Theano` ），我们能够非常轻松地利用该生态系统的强大功能，并以创建 `PyMC3` 时从未想到的方式使用 `PyMC3` 。我期待着将其扩展到新的领域。

这篇博文写在一本 Jupyter 笔记本上。你可以在这里访问 [笔记本](https://github.com/twiecki/twiecki.github.io/blob/master/downloads/notebooks/bayesian_neural_network_ `Lasagne` .ipynb)，并在 Twitter 上关注 [我](https://twitter.com/twiecki) 以了解最新情况。

## 4 致谢

`Taku Yoshioka` 在 `PyMC3` 中对最初的 ADVI 实现做了大量工作。我还要感谢斯坦兄弟（特别是阿尔普·库库克尔比尔和丹尼尔·李）推导出了 ADVI 并教给我们关于它的知识。还要感谢克里斯·丰内斯贝克、安德鲁·坎贝尔、吉冈拓谷和皮达尔·科伊尔对早期草稿的有用评论
