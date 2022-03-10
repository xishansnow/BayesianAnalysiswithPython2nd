---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
---

# Deep Probabilistic Modelling with with Gaussian Processes
### [Neil D. Lawrence](http://inverseprobability.com), Amazon and University of Sheffield

**Abstract**: Neural network models are algorithmically simple, but mathematically
complex. Gaussian process models are mathematically simple, but
algorithmically complex. In this tutorial we will explore Deep Gaussian
Process models. They bring advantages in their mathematical simplicity
but are challenging in their algorithmic complexity. We will give an
overview of Gaussian processes and highlight the algorithmic
approximations that allow us to stack Gaussian process models: they are
based on variational methods. In the last part of the tutorial will
explore a use case exemplar: uncertainty quantification. We end with
open questions.

$$
\newcommand{\Amatrix}{\mathbf{A}}
\newcommand{\KL}[2]{\text{KL}\left( #1\,\|\,#2 \right)}
\newcommand{\Kaast}{\kernelMatrix_{\mathbf{ \ast}\mathbf{ \ast}}}
\newcommand{\Kastu}{\kernelMatrix_{\mathbf{ \ast} \inducingVector}}
\newcommand{\Kff}{\kernelMatrix_{\mappingFunctionVector \mappingFunctionVector}}
\newcommand{\Kfu}{\kernelMatrix_{\mappingFunctionVector \inducingVector}}
\newcommand{\Kuast}{\kernelMatrix_{\inducingVector \bf\ast}}
\newcommand{\Kuf}{\kernelMatrix_{\inducingVector \mappingFunctionVector}}
\newcommand{\Kuu}{\kernelMatrix_{\inducingVector \inducingVector}}
\newcommand{\Kuui}{\Kuu^{-1}}
\newcommand{\Qaast}{\mathbf{Q}_{\bf \ast \ast}}
\newcommand{\Qastf}{\mathbf{Q}_{\ast \mappingFunction}}
\newcommand{\Qfast}{\mathbf{Q}_{\mappingFunctionVector \bf \ast}}
\newcommand{\Qff}{\mathbf{Q}_{\mappingFunctionVector \mappingFunctionVector}}
\newcommand{\aMatrix}{\mathbf{A}}
\newcommand{\aScalar}{a}
\newcommand{\aVector}{\mathbf{a}}
\newcommand{\acceleration}{a}
\newcommand{\bMatrix}{\mathbf{B}}
\newcommand{\bScalar}{b}
\newcommand{\bVector}{\mathbf{b}}
\newcommand{\basisFunc}{\phi}
\newcommand{\basisFuncVector}{\boldsymbol{ \basisFunc}}
\newcommand{\basisFunction}{\phi}
\newcommand{\basisLocation}{\mu}
\newcommand{\basisMatrix}{\boldsymbol{ \Phi}}
\newcommand{\basisScalar}{\basisFunction}
\newcommand{\basisVector}{\boldsymbol{ \basisFunction}}
\newcommand{\activationFunction}{\phi}
\newcommand{\activationMatrix}{\boldsymbol{ \Phi}}
\newcommand{\activationScalar}{\basisFunction}
\newcommand{\activationVector}{\boldsymbol{ \basisFunction}}
\newcommand{\bigO}{\mathcal{O}}
\newcommand{\binomProb}{\pi}
\newcommand{\cMatrix}{\mathbf{C}}
\newcommand{\cbasisMatrix}{\hat{\boldsymbol{ \Phi}}}
\newcommand{\cdataMatrix}{\hat{\dataMatrix}}
\newcommand{\cdataScalar}{\hat{\dataScalar}}
\newcommand{\cdataVector}{\hat{\dataVector}}
\newcommand{\centeredKernelMatrix}{\mathbf{ \MakeUppercase{\centeredKernelScalar}}}
\newcommand{\centeredKernelScalar}{b}
\newcommand{\centeredKernelVector}{\centeredKernelScalar}
\newcommand{\centeringMatrix}{\mathbf{H}}
\newcommand{\chiSquaredDist}[2]{\chi_{#1}^{2}\left(#2\right)}
\newcommand{\chiSquaredSamp}[1]{\chi_{#1}^{2}}
\newcommand{\conditionalCovariance}{\boldsymbol{ \Sigma}}
\newcommand{\coregionalizationMatrix}{\mathbf{B}}
\newcommand{\coregionalizationScalar}{b}
\newcommand{\coregionalizationVector}{\mathbf{ \coregionalizationScalar}}
\newcommand{\covDist}[2]{\text{cov}_{#2}\left(#1\right)}
\newcommand{\covSamp}[1]{\text{cov}\left(#1\right)}
\newcommand{\covarianceScalar}{c}
\newcommand{\covarianceVector}{\mathbf{ \covarianceScalar}}
\newcommand{\covarianceMatrix}{\mathbf{C}}
\newcommand{\covarianceMatrixTwo}{\boldsymbol{ \Sigma}}
\newcommand{\croupierScalar}{s}
\newcommand{\croupierVector}{\mathbf{ \croupierScalar}}
\newcommand{\croupierMatrix}{\mathbf{ \MakeUppercase{\croupierScalar}}}
\newcommand{\dataDim}{p}
\newcommand{\dataIndex}{i}
\newcommand{\dataIndexTwo}{j}
\newcommand{\dataMatrix}{\mathbf{Y}}
\newcommand{\dataScalar}{y}
\newcommand{\dataSet}{\mathcal{D}}
\newcommand{\dataStd}{\sigma}
\newcommand{\dataVector}{\mathbf{ \dataScalar}}
\newcommand{\decayRate}{d}
\newcommand{\degreeMatrix}{\mathbf{ \MakeUppercase{\degreeScalar}}}
\newcommand{\degreeScalar}{d}
\newcommand{\degreeVector}{\mathbf{ \degreeScalar}}
% Already defined by latex
%\newcommand{\det}[1]{\left|#1\right|}
\newcommand{\diag}[1]{\text{diag}\left(#1\right)}
\newcommand{\diagonalMatrix}{\mathbf{D}}
\newcommand{\diff}[2]{\frac{\text{d}#1}{\text{d}#2}}
\newcommand{\diffTwo}[2]{\frac{\text{d}^2#1}{\text{d}#2^2}}
\newcommand{\displacement}{x}
\newcommand{\displacementVector}{\textbf{\displacement}}
\newcommand{\distanceMatrix}{\mathbf{ \MakeUppercase{\distanceScalar}}}
\newcommand{\distanceScalar}{d}
\newcommand{\distanceVector}{\mathbf{ \distanceScalar}}
\newcommand{\eigenvaltwo}{\ell}
\newcommand{\eigenvaltwoMatrix}{\mathbf{L}}
\newcommand{\eigenvaltwoVector}{\mathbf{l}}
\newcommand{\eigenvalue}{\lambda}
\newcommand{\eigenvalueMatrix}{\boldsymbol{ \Lambda}}
\newcommand{\eigenvalueVector}{\boldsymbol{ \lambda}}
\newcommand{\eigenvector}{\mathbf{ \eigenvectorScalar}}
\newcommand{\eigenvectorMatrix}{\mathbf{U}}
\newcommand{\eigenvectorScalar}{u}
\newcommand{\eigenvectwo}{\mathbf{v}}
\newcommand{\eigenvectwoMatrix}{\mathbf{V}}
\newcommand{\eigenvectwoScalar}{v}
\newcommand{\entropy}[1]{\mathcal{H}\left(#1\right)}
\newcommand{\errorFunction}{E}
\newcommand{\expDist}[2]{\left<#1\right>_{#2}}
\newcommand{\expSamp}[1]{\left<#1\right>}
\newcommand{\expectation}[1]{\left\langle #1 \right\rangle }
\newcommand{\expectationDist}[2]{\left\langle #1 \right\rangle _{#2}}
\newcommand{\expectedDistanceMatrix}{\mathcal{D}}
\newcommand{\eye}{\mathbf{I}}
\newcommand{\fantasyDim}{r}
\newcommand{\fantasyMatrix}{\mathbf{ \MakeUppercase{\fantasyScalar}}}
\newcommand{\fantasyScalar}{z}
\newcommand{\fantasyVector}{\mathbf{ \fantasyScalar}}
\newcommand{\featureStd}{\varsigma}
\newcommand{\gammaCdf}[3]{\mathcal{GAMMA CDF}\left(#1|#2,#3\right)}
\newcommand{\gammaDist}[3]{\mathcal{G}\left(#1|#2,#3\right)}
\newcommand{\gammaSamp}[2]{\mathcal{G}\left(#1,#2\right)}
\newcommand{\gaussianDist}[3]{\mathcal{N}\left(#1|#2,#3\right)}
\newcommand{\gaussianSamp}[2]{\mathcal{N}\left(#1,#2\right)}
\newcommand{\given}{|}
\newcommand{\half}{\frac{1}{2}}
\newcommand{\heaviside}{H}
\newcommand{\hiddenMatrix}{\mathbf{ \MakeUppercase{\hiddenScalar}}}
\newcommand{\hiddenScalar}{h}
\newcommand{\hiddenVector}{\mathbf{ \hiddenScalar}}
\newcommand{\identityMatrix}{\eye}
\newcommand{\inducingInputScalar}{z}
\newcommand{\inducingInputVector}{\mathbf{ \inducingInputScalar}}
\newcommand{\inducingInputMatrix}{\mathbf{Z}}
\newcommand{\inducingScalar}{u}
\newcommand{\inducingVector}{\mathbf{ \inducingScalar}}
\newcommand{\inducingMatrix}{\mathbf{U}}
\newcommand{\inlineDiff}[2]{\text{d}#1/\text{d}#2}
\newcommand{\inputDim}{q}
\newcommand{\inputMatrix}{\mathbf{X}}
\newcommand{\inputScalar}{x}
\newcommand{\inputSpace}{\mathcal{X}}
\newcommand{\inputVals}{\inputVector}
\newcommand{\inputVector}{\mathbf{ \inputScalar}}
\newcommand{\iterNum}{k}
\newcommand{\kernel}{\kernelScalar}
\newcommand{\kernelMatrix}{\mathbf{K}}
\newcommand{\kernelScalar}{k}
\newcommand{\kernelVector}{\mathbf{ \kernelScalar}}
\newcommand{\kff}{\kernelScalar_{\mappingFunction \mappingFunction}}
\newcommand{\kfu}{\kernelVector_{\mappingFunction \inducingScalar}}
\newcommand{\kuf}{\kernelVector_{\inducingScalar \mappingFunction}}
\newcommand{\kuu}{\kernelVector_{\inducingScalar \inducingScalar}}
\newcommand{\lagrangeMultiplier}{\lambda}
\newcommand{\lagrangeMultiplierMatrix}{\boldsymbol{ \Lambda}}
\newcommand{\lagrangian}{L}
\newcommand{\laplacianFactor}{\mathbf{ \MakeUppercase{\laplacianFactorScalar}}}
\newcommand{\laplacianFactorScalar}{m}
\newcommand{\laplacianFactorVector}{\mathbf{ \laplacianFactorScalar}}
\newcommand{\laplacianMatrix}{\mathbf{L}}
\newcommand{\laplacianScalar}{\ell}
\newcommand{\laplacianVector}{\mathbf{ \ell}}
\newcommand{\latentDim}{q}
\newcommand{\latentDistanceMatrix}{\boldsymbol{ \Delta}}
\newcommand{\latentDistanceScalar}{\delta}
\newcommand{\latentDistanceVector}{\boldsymbol{ \delta}}
\newcommand{\latentForce}{f}
\newcommand{\latentFunction}{u}
\newcommand{\latentFunctionVector}{\mathbf{ \latentFunction}}
\newcommand{\latentFunctionMatrix}{\mathbf{ \MakeUppercase{\latentFunction}}}
\newcommand{\latentIndex}{j}
\newcommand{\latentScalar}{z}
\newcommand{\latentVector}{\mathbf{ \latentScalar}}
\newcommand{\latentMatrix}{\mathbf{Z}}
\newcommand{\learnRate}{\eta}
\newcommand{\lengthScale}{\ell}
\newcommand{\rbfWidth}{\ell}
\newcommand{\likelihoodBound}{\mathcal{L}}
\newcommand{\likelihoodFunction}{L}
\newcommand{\locationScalar}{\mu}
\newcommand{\locationVector}{\boldsymbol{ \locationScalar}}
\newcommand{\locationMatrix}{\mathbf{M}}
\newcommand{\variance}[1]{\text{var}\left( #1 \right)}
\newcommand{\mappingFunction}{f}
\newcommand{\mappingFunctionMatrix}{\mathbf{F}}
\newcommand{\mappingFunctionTwo}{g}
\newcommand{\mappingFunctionTwoMatrix}{\mathbf{G}}
\newcommand{\mappingFunctionTwoVector}{\mathbf{ \mappingFunctionTwo}}
\newcommand{\mappingFunctionVector}{\mathbf{ \mappingFunction}}
\newcommand{\scaleScalar}{s}
\newcommand{\mappingScalar}{w}
\newcommand{\mappingVector}{\mathbf{ \mappingScalar}}
\newcommand{\mappingMatrix}{\mathbf{W}}
\newcommand{\mappingScalarTwo}{v}
\newcommand{\mappingVectorTwo}{\mathbf{ \mappingScalarTwo}}
\newcommand{\mappingMatrixTwo}{\mathbf{V}}
\newcommand{\maxIters}{K}
\newcommand{\meanMatrix}{\mathbf{M}}
\newcommand{\meanScalar}{\mu}
\newcommand{\meanTwoMatrix}{\mathbf{M}}
\newcommand{\meanTwoScalar}{m}
\newcommand{\meanTwoVector}{\mathbf{ \meanTwoScalar}}
\newcommand{\meanVector}{\boldsymbol{ \meanScalar}}
\newcommand{\mrnaConcentration}{m}
\newcommand{\naturalFrequency}{\omega}
\newcommand{\neighborhood}[1]{\mathcal{N}\left( #1 \right)}
\newcommand{\neilurl}{http://inverseprobability.com/}
\newcommand{\noiseMatrix}{\boldsymbol{ E}}
\newcommand{\noiseScalar}{\epsilon}
\newcommand{\noiseVector}{\boldsymbol{ \epsilon}}
\newcommand{\norm}[1]{\left\Vert #1 \right\Vert}
\newcommand{\normalizedLaplacianMatrix}{\hat{\mathbf{L}}}
\newcommand{\normalizedLaplacianScalar}{\hat{\ell}}
\newcommand{\normalizedLaplacianVector}{\hat{\mathbf{ \ell}}}
\newcommand{\numActive}{m}
\newcommand{\numBasisFunc}{m}
\newcommand{\numComponents}{m}
\newcommand{\numComps}{K}
\newcommand{\numData}{n}
\newcommand{\numFeatures}{K}
\newcommand{\numHidden}{h}
\newcommand{\numInducing}{m}
\newcommand{\numLayers}{\ell}
\newcommand{\numNeighbors}{K}
\newcommand{\numSequences}{s}
\newcommand{\numSuccess}{s}
\newcommand{\numTasks}{m}
\newcommand{\numTime}{T}
\newcommand{\numTrials}{S}
\newcommand{\outputIndex}{j}
\newcommand{\paramVector}{\boldsymbol{ \theta}}
\newcommand{\parameterMatrix}{\boldsymbol{ \Theta}}
\newcommand{\parameterScalar}{\theta}
\newcommand{\parameterVector}{\boldsymbol{ \parameterScalar}}
\newcommand{\partDiff}[2]{\frac{\partial#1}{\partial#2}}
\newcommand{\precisionScalar}{j}
\newcommand{\precisionVector}{\mathbf{ \precisionScalar}}
\newcommand{\precisionMatrix}{\mathbf{J}}
\newcommand{\pseudotargetScalar}{\widetilde{y}}
\newcommand{\pseudotargetVector}{\mathbf{ \pseudotargetScalar}}
\newcommand{\pseudotargetMatrix}{\mathbf{ \widetilde{Y}}}
\newcommand{\rank}[1]{\text{rank}\left(#1\right)}
\newcommand{\rayleighDist}[2]{\mathcal{R}\left(#1|#2\right)}
\newcommand{\rayleighSamp}[1]{\mathcal{R}\left(#1\right)}
\newcommand{\responsibility}{r}
\newcommand{\rotationScalar}{r}
\newcommand{\rotationVector}{\mathbf{ \rotationScalar}}
\newcommand{\rotationMatrix}{\mathbf{R}}
\newcommand{\sampleCovScalar}{s}
\newcommand{\sampleCovVector}{\mathbf{ \sampleCovScalar}}
\newcommand{\sampleCovMatrix}{\mathbf{s}}
\newcommand{\scalarProduct}[2]{\left\langle{#1},{#2}\right\rangle}
\newcommand{\sign}[1]{\text{sign}\left(#1\right)}
\newcommand{\sigmoid}[1]{\sigma\left(#1\right)}
\newcommand{\singularvalue}{\ell}
\newcommand{\singularvalueMatrix}{\mathbf{L}}
\newcommand{\singularvalueVector}{\mathbf{l}}
\newcommand{\sorth}{\mathbf{u}}
\newcommand{\spar}{\lambda}
\newcommand{\trace}[1]{\text{tr}\left(#1\right)}
\newcommand{\BasalRate}{B}
\newcommand{\DampingCoefficient}{C}
\newcommand{\DecayRate}{D}
\newcommand{\Displacement}{X}
\newcommand{\LatentForce}{F}
\newcommand{\Mass}{M}
\newcommand{\Sensitivity}{S}
\newcommand{\basalRate}{b}
\newcommand{\dampingCoefficient}{c}
\newcommand{\mass}{m}
\newcommand{\sensitivity}{s}
\newcommand{\springScalar}{\kappa}
\newcommand{\springVector}{\boldsymbol{ \kappa}}
\newcommand{\springMatrix}{\boldsymbol{ \mathcal{K}}}
\newcommand{\tfConcentration}{p}
\newcommand{\tfDecayRate}{\delta}
\newcommand{\tfMrnaConcentration}{f}
\newcommand{\tfVector}{\mathbf{ \tfConcentration}}
\newcommand{\velocity}{v}
\newcommand{\sufficientStatsScalar}{g}
\newcommand{\sufficientStatsVector}{\mathbf{ \sufficientStatsScalar}}
\newcommand{\sufficientStatsMatrix}{\mathbf{G}}
\newcommand{\switchScalar}{s}
\newcommand{\switchVector}{\mathbf{ \switchScalar}}
\newcommand{\switchMatrix}{\mathbf{S}}
\newcommand{\tr}[1]{\text{tr}\left(#1\right)}
\newcommand{\loneNorm}[1]{\left\Vert #1 \right\Vert_1}
\newcommand{\ltwoNorm}[1]{\left\Vert #1 \right\Vert_2}
\newcommand{\onenorm}[1]{\left\vert#1\right\vert_1}
\newcommand{\twonorm}[1]{\left\Vert #1 \right\Vert}
\newcommand{\vScalar}{v}
\newcommand{\vVector}{\mathbf{v}}
\newcommand{\vMatrix}{\mathbf{V}}
\newcommand{\varianceDist}[2]{\text{var}_{#2}\left( #1 \right)}
% Already defined by latex
%\newcommand{\vec}{#1:}
\newcommand{\vecb}[1]{\left(#1\right):}
\newcommand{\weightScalar}{w}
\newcommand{\weightVector}{\mathbf{ \weightScalar}}
\newcommand{\weightMatrix}{\mathbf{W}}
\newcommand{\weightedAdjacencyMatrix}{\mathbf{A}}
\newcommand{\weightedAdjacencyScalar}{a}
\newcommand{\weightedAdjacencyVector}{\mathbf{ \weightedAdjacencyScalar}}
\newcommand{\onesVector}{\mathbf{1}}
\newcommand{\zerosVector}{\mathbf{0}}
$$

## What is Machine Learning?

What is machine learning? At its most basic level machine learning is a
combination of

$$ \text{data} + \text{model} \xrightarrow{\text{compute}} \text{prediction}$$

where *data* is our observations. They can be actively or passively
acquired (meta-data). The *model* contains our assumptions, based on
previous experience. That experience can be other data, it can come from
transfer learning, or it can merely be our beliefs about the
regularities of the universe. In humans our models include our inductive
biases. The *prediction* is an action to be taken or a categorization or
a quality score. The reason that machine learning has become a mainstay
of artificial intelligence is the importance of predictions in
artificial intelligence. The data and the model are combined through
computation.

In practice we normally perform machine learning using two functions. To
combine data with a model we typically make use of:

**a prediction function** a function which is used to make the
predictions. It includes our beliefs about the regularities of the
universe, our assumptions about how the world works, e.g. smoothness,
spatial similarities, temporal similarities.

**an objective function** a function which defines the cost of
misprediction. Typically it includes knowledge about the world's
generating processes (probabilistic objectives) or the costs we pay for
mispredictions (empiricial risk minimization).

The combination of data and model through the prediction function and
the objectie function leads to a *learning algorithm*. The class of
prediction functions and objective functions we can make use of is
restricted by the algorithms they lead to. If the prediction function or
the objective function are too complex, then it can be difficult to find
an appropriate learning algorithm. Much of the acdemic field of machine
learning is the quest for new learning algorithms that allow us to bring
different types of models and data together.

A useful reference for state of the art in machine learning is the UK
Royal Society Report, [Machine Learning: Power and Promise of Computers
that Learn by
Example](https://royalsociety.org/~/media/policy/projects/machine-learning/publications/machine-learning-report.pdf).

You can also check my blog post on ["What is Machine
Learning?"](http://inverseprobability.com/2017/07/17/what-is-machine-learning)

### Artificial Intelligence

### Uncertainty

In practice, we normally also have uncertainty associated with these
functions. Uncertainty in the prediction function arises from

1.  scarcity of training data and
2.  mismatch between the set of prediction functions we choose and all
    possible prediction functions.

There are also challenges around specification of the objective
function, but for we will save those for another day. For the moment,
let us focus on the prediction function.

### Neural Networks and Prediction Functions

Neural networks are adaptive non-linear function models. Originally,
they were studied (by McCulloch and Pitts [@McCulloch:neuron43]) as
simple models for neurons, but over the last decade they have become
popular because they are a flexible approach to modelling complex data.
A particular characteristic of neural network models is that they can be
composed to form highly complex functions which encode many of our
expectations of the real world. They allow us to encode our assumptions
about how the world works.

We will return to composition later, but for the moment, let's focus on
a one hidden layer neural network. We are interested in the prediction
function, so we'll ignore the objective function (which is often called
an error function) for the moment, and just describe the mathematical
object of interest

$$
\mappingFunction(\inputVector) = \mappingMatrix^\top \activationVector(\mappingMatrixTwo, \inputVector)
$$

Where in this case $\mappingFunction(\cdot)$ is a scalar function with
vector inputs, and $\activationVector(\cdot)$ is a vector function with
vector inputs. The dimensionality of the vector function is known as the
number of hidden units, or the number of neurons. The elements of this
vector function are known as the *activation* function of the neural
network and $\mappingMatrixTwo$ are the parameters of the activation
functions.

### Relations with Classical Statistics

In statistics activation functions are traditionally known as *basis
functions*. And we would think of this as a *linear model*. It's doesn't
make linear predictions, but it's linear because in statistics
estimation focuses on the parameters, $\mappingMatrix$, not the
parameters, $\mappingMatrixTwo$. The linear model terminology refers to
the fact that the model is *linear in the parameters*, but it is *not*
linear in the data unless the activation functions are chosen to be
linear.

### Adaptive Basis Functions

The first difference in the (early) neural network literature to the
classical statistical literature is the decision to optimize these
parameters, $\mappingMatrixTwo$, as well as the parameters,
$\mappingMatrix$ (which would normally be denoted in statistics by
$\boldsymbol{\beta}$)[^1].

In this tutorial, we're going to go revisit that decision, and follow
the path of Radford Neal [@Neal:bayesian94] who, inspired by work of
David MacKay [@MacKay:bayesian92] and others did his PhD thesis on
Bayesian Neural Networks. If we take a Bayesian approach to parameter
inference (note I am using inference here in the classical sense, not in
the sense of prediction of test data, which seems to be a newer usage),
then we don't wish to fit parameters at all, rather we wish to integrate
them away and understand the family of functions that the model
describes.

### Probabilistic Modelling

This Bayesian approach is designed to deal with uncertainty arising from
fitting our prediction function to the data we have, a reduced data set.

The Bayesian approach can be derived from a broader understanding of
what our objective is. If we accept that we can jointly represent all
things that happen in the world with a probability distribution, then we
can interogate that probability to make predictions. So, if we are
interested in predictions, $\dataScalar_*$ at future points input
locations of interest, $\inputVector_*$ given previously training data,
$\dataVector$ and corresponding inputs, $\inputMatrix$, then we are
really interogating the following probability density, $$
p(\dataScalar_*|\dataVector, \inputMatrix, \inputVector_*),
$$ there is nothing controversial here, as long as you accept that you
have a good joint model of the world around you that relates test data
to training data,
$p(\dataScalar_*, \dataVector, \inputMatrix, \inputVector_*)$ then this
conditional distribution can be recovered through standard rules of
probability
($\text{data} + \text{model} \rightarrow \text{prediction}$).

We can construct this joint density through the use of the following
decomposition: $$
p(\dataScalar_*|\dataVector, \inputMatrix, \inputVector_*) = \int p(\dataScalar_*|\inputVector_*, \mappingMatrix) p(\mappingMatrix | \dataVector, \inputMatrix) \text{d} \mappingMatrix
$$

where, for convenience, we are assuming *all* the parameters of the
model are now represented by $\parameterVector$ (which contains
$\mappingMatrix$ and $\mappingMatrixTwo$) and
$p(\parameterVector | \dataVector, \inputMatrix)$ is recognised as the
posterior density of the parameters given data and
$p(\dataScalar_*|\inputVector_*, \parameterVector)$ is the *likelihood*
of an individual test data point given the parameters.

The likelihood of the data is normally assumed to be independent across
the parameters, $$
p(\dataVector|\inputMatrix, \mappingMatrix) \prod_{i=1}^\numData p(\dataScalar_i|\inputVector_i, \mappingMatrix),$$

and if that is so, it is easy to extend our predictions across all
future, potential, locations, $$
p(\dataVector_*|\dataVector, \inputMatrix, \inputMatrix_*) = \int p(\dataVector_*|\inputMatrix_*, \parameterVector) p(\parameterVector | \dataVector, \inputMatrix) \text{d} \parameterVector.
$$

The likelihood is also where the *prediction function* is incorporated.
For example in the regression case, we consider an objective based
around the Gaussian density, $$
p(\dataScalar_i | \mappingFunction(\inputVector_i)) = \frac{1}{\sqrt{2\pi \dataStd^2}} \exp\left(-\frac{\left(\dataScalar_i - \mappingFunction(\inputVector_i)\right)^2}{2\dataStd^2}\right)
$$

In short, that is the classical approach to probabilistic inference, and
all approaches to Bayesian neural networks fall within this path. For a
deep probabilistic model, we can simply take this one stage further and
place a probability distribution over the input locations, $$
p(\dataVector_*|\dataVector) = \int p(\dataVector_*|\inputMatrix_*, \parameterVector) p(\parameterVector | \dataVector, \inputMatrix) p(\inputMatrix) p(\inputMatrix_*) \text{d} \parameterVector \text{d} \inputMatrix \text{d}\inputMatrix_*
$$ and we have *unsupervised learning* (from where we can get deep
generative models).

### Graphical Models

One way of representing a joint distribution is to consider conditional
dependencies between data. Conditional dependencies allow us to
factorize the distribution. For example, a Markov chain is a
factorization of a distribution into components that represent the
conditional relationships between points that are neighboring, often in
time or space. It can be decomposed in the following form.
$$p(\dataVector) = p(\dataScalar_\numData | \dataScalar_{\numData-1}) p(\dataScalar_{\numData-1}|\dataScalar_{\numData-2}) \dots p(\dataScalar_{2} | \dataScalar_{1})$$

```{code-cell}
import daft
from matplotlib import rc

rc("font", **{'family':'sans-serif','sans-serif':['Helvetica']}, size=30)
rc("text", usetex=True)
```

```{code-cell}
pgm = daft.PGM(shape=[3, 1],
               origin=[0, 0], 
               grid_unit=5, 
               node_unit=1.9, 
               observed_style='shaded',
              line_width=3)


pgm.add_node(daft.Node("y_1", r"$y_1$", 0.5, 0.5, fixed=False))
pgm.add_node(daft.Node("y_2", r"$y_2$", 1.5, 0.5, fixed=False))
pgm.add_node(daft.Node("y_3", r"$y_3$", 2.5, 0.5, fixed=False))
pgm.add_edge("y_1", "y_2")
pgm.add_edge("y_2", "y_3")

pgm.render().figure.savefig("../slides/diagrams/ml/markov.svg", transparent=True)
```

<img src="../slides/diagrams/ml/markov.svg" align="">

By specifying conditional independencies we can reduce the
parameterization required for our data, instead of directly specifying
the parameters of the joint distribution, we can specify each set of
parameters of the conditonal independently. This can also give an
advantage in terms of interpretability. Understanding a conditional
independence structure gives a structured understanding of data. If
developed correctly, according to causal methodology, it can even inform
how we should intervene in the system to drive a desired result
[@Pearl:causality95].

However, a challenge arise when the data becomes more complex. Consider
the graphical model shown below, used to predict the perioperative risk
of *C Difficile* infection following colon surgery
[@Steele:predictive12].

<img class="negate" src="../slides/diagrams/bayes-net-diagnosis.png" width="40%" align="center" style="background:none; border:none; box-shadow:none;">

To capture the complexity in the interelationship between the data the
graph becomes more complex, and less interpretable.

### Performing Inference

As far as combining our data and our model to form our prediction, the
devil is in the detail. While everything is easy to write in terms of
probability densities, as we move from $\text{data}$ and $\text{model}$
to $\text{prediction}$ there is that simple
$\xrightarrow{\text{compute}}$ sign, which is now burying a wealth of
difficulties. Each integral sign above is a high dimensional integral
which will typically need approximation. Approximations also come with
computational demands. As we consider more complex classes of functions,
the challenges around the integrals become harder and prediction of
future test data given our model and the data becomes so involved as to
be impractical or impossible.

Statisticians realized these challenges early on, indeed, so early that
they were actually physicists, both Laplace and Gauss worked on models
such as this, in Gauss's case he made his career on prediction of the
location of the lost planet (later reclassified as a asteroid, then
dwarf planet), Ceres. Gauss and Laplace made use of maximum a posteriori
estimates for simplifying their computations and Laplace developed
Laplace's method (and invented the Gaussian density) to expand around
that mode. But classical statistics needs better guarantees around model
performance and interpretation, and as a result has focussed more on the
*linear* model implied by $$
  \mappingFunction(\inputVector) = \left.\mappingVector^{(2)}\right.^\top \activationVector(\mappingMatrix_1, \inputVector)
  $$

$$
  \mappingVector^{(2)} \sim \gaussianSamp{\zerosVector}{\covarianceMatrix}.
  $$

The Gaussian likelihood given above implies that the data observation is
related to the function by noise corruption so we have, $$
  \dataScalar_i = \mappingFunction(\inputVector_i) + \noiseScalar_i,
  $$ where $$
  \noiseScalar_i \sim \gaussianSamp{0}{\dataStd^2}
  $$ and while normally integrating over high dimensional parameter
vectors is highly complex, here it is *trivial*. That is because of a
property of the multivariate Gaussian.

### Multivariate Gaussian Properties

Gaussian processes are initially of interest because

1.  linear Gaussian models are easier to deal with
2.  Even the parameters *within* the process can be handled, by
    considering a particular limit.

Let's first of all review the properties of the multivariate Gaussian
distribution that make linear Gaussian models easier to deal with. We'll
return to the, perhaps surprising, result on the parameters within the
nonlinearity, $\parameterVector$, shortly.

To work with linear Gaussian models, to find the marginal likelihood all
you need to know is the following rules. If $$
\dataVector = \mappingMatrix \inputVector + \noiseVector,
$$ where $\dataVector$, $\inputVector$ and $\noiseVector$ are vectors
and we assume that $\inputVector$ and $\noiseVector$ are drawn from
multivariate Gaussians, $$\begin{align}
\inputVector & \sim \gaussianSamp{\meanVector}{\covarianceMatrix}\\
\noiseVector & \sim \gaussianSamp{\zerosVector}{\covarianceMatrixTwo}
\end{align}$$ then we know that $\dataVector$ is also drawn from a
multivariate Gaussian with, $$
\dataVector \sim \gaussianSamp{\mappingMatrix\meanVector}{\mappingMatrix\covarianceMatrix\mappingMatrix^\top + \covarianceMatrixTwo}.
$$ With apprioriately defined covariance, $\covarianceTwoMatrix$, this
is actually the marginal likelihood for Factor Analysis, or
Probabilistic Principal Component Analysis [@Tipping:probpca99], because
we integrated out the inputs (or *latent* variables they would be called
in that case).

However, we are focussing on what happens in models which are non-linear
in the inputs, whereas the above would be *linear* in the inputs. To
consider these, we introduce a matrix, called the design matrix. We set
each activation function computed at each data point to be $$
\activationScalar_{i,j} = \activationScalar(\mappingVector^{(1)}_{j}, \inputVector_{i})
$$ and define the matrix of activations (known as the *design matrix* in
statistics) to be, $$
\activationMatrix = 
\begin{bmatrix}
\activationScalar_{1, 1} & \activationScalar_{1, 2} & \dots & \activationScalar_{1, \numHidden} \\
\activationScalar_{1, 2} & \activationScalar_{1, 2} & \dots & \activationScalar_{1, \numData} \\
\vdots & \vdots & \ddots & \vdots \\
\activationScalar_{\numData, 1} & \activationScalar_{\numData, 2} & \dots & \activationScalar_{\numData, \numHidden}
\end{bmatrix}.
$$ By convention this matrix always has $\numData$ rows and $\numHidden$
columns, now if we define the vector of all noise corruptions,
$\noiseVector = \left[\noiseScalar_1, \dots \noiseScalar_\numData\right]^\top$.

### Matrix Representation of a Neural Network

$$\dataScalar\left(\inputVector\right) = \activationVector\left(\inputVector\right)^\top \mappingVector + \noiseScalar$$

. . .

$$\dataVector = \activationMatrix\mappingVector + \noiseVector$$

. . .

$$\noiseVector \sim \gaussianSamp{\zerosVector}{\dataStd^2\eye}$$

{ If we define the prior distribution over the vector $\mappingVector$
to be Gaussian,} $$
\mappingVector \sim \gaussianSamp{\zerosVector}{\alpha\eye},
$$

{ then we can use rules of multivariate Gaussians to see that,} $$
\dataVector \sim \gaussianSamp{\zerosVector}{\alpha \activationMatrix \activationMatrix^\top + \dataStd^2 \eye}.
$$

In other words, our training data is distributed as a multivariate
Gaussian, with zero mean and a covariance given by $$
\kernelMatrix = \alpha \activationMatrix \activationMatrix^\top + \dataStd^2 \eye.
$$

This is an $\numData \times \numData$ size matrix. Its elements are in
the form of a function. The maths shows that any element, index by $i$
and $j$, is a function *only* of inputs associated with data points $i$
and $j$, $\dataVector_i$, $\dataVector_j$.
$\kernel_{i,j} = \kernel\left(\inputVector_i, \inputVector_j\right)$

If we look at the portion of this function associated only with
$\mappingFunction(\cdot)$, i.e. we remove the noise, then we can write
down the covariance associated with our neural network, $$
\kernel_\mappingFunction\left(\inputVector_i, \inputVector_j\right) = \alpha \activationVector\left(\mappingMatrix_1, \inputVector_i\right)^\top \activationVector\left(\mappingMatrix_1, \inputVector_j\right)
$$ so the elements of the covariance or *kernel* matrix are formed by
inner products of the rows of the *design matrix*.

This is the essence of a Gaussian process. Instead of making assumptions
about our density over each data point, $\dataScalar_i$ as i.i.d. we
make a joint Gaussian assumption over our data. The covariance matrix is
now a function of both the parameters of the activation function,
$\mappingMatrixTwo$, and the input variables, $\inputMatrix$. This comes
about through integrating out the parameters of the model,
$\mappingVector$.

We can basically put anything inside the basis functions, and many
people do. These can be deep kernels [@Cho:deep09] or we can learn the
parameters of a convolutional neural network inside there.

Viewing a neural network in this way is also what allows us to beform
sensible *batch* normalizations [@Ioffe:batch15].

### Non-degenerate Gaussian Processes

The process described above is degenerate. The covariance function is of
rank at most $\numHidden$ and since the theoretical amount of data could
always increase $\numData \rightarrow \infty$, the covariance function
is not full rank. This means as we increase the amount of data to
infinity, there will come a point where we can't normalize the process
because the multivariate Gaussian has the form, $$
\gaussianDist{\mappingFunctionVector}{\zerosVector}{\kernelMatrix} = \frac{1}{\left(2\pi\right)^{\frac{\numData}{2}}\det{\kernelMatrix}^\frac{1}{2}} \exp\left(-\frac{\mappingFunctionVector^\top\kernelMatrix \mappingFunctionVector}{2}\right)
$$ and a non-degenerate kernel matrix leads to $\det{\kernelMatrix} = 0$
defeating the normalization (it's equivalent to finding a projection in
the high dimensional Gaussian where the variance of the the resulting
univariate Gaussian is zero, i.e. there is a null space on the
covariance, or alternatively you can imagine there are one or more
directions where the Gaussian has become the delta function).

In the machine learning field, it was Radford Neal [@Neal:bayesian94]
that realized the potential of the next step. In his 1994 thesis, he was
considering Bayesian neural networks, of the type we described above,
and in considered what would happen if you took the number of hidden
nodes, or neurons, to infinity, i.e. $\numHidden \rightarrow \infty$.

[<img class="" src="../slides/diagrams/neal-infinite-priors.png" width="80%" align="" style="background:none; border:none; box-shadow:none;">](http://www.cs.toronto.edu/~radford/ftp/thesis.pdf)

*Page 37 of Radford Neal's 1994 thesis*

In loose terms, what Radford considers is what happens to the elements
of the covariance function, $$
  \begin{align*}
  \kernel_\mappingFunction\left(\inputVector_i, \inputVector_j\right) & = \alpha \activationVector\left(\mappingMatrix_1, \inputVector_i\right)^\top \activationVector\left(\mappingMatrix_1, \inputVector_j\right)\\
  & = \alpha \sum_k \activationScalar\left(\mappingVector^{(1)}_k, \inputVector_i\right) \activationScalar\left(\mappingVector^{(1)}_k, \inputVector_j\right)
  \end{align*}
  $$ if instead of considering a finite number you sample infinitely
many of these activation functions, sampling parameters from a prior
density, $p(\mappingVectorTwo)$, for each one, $$
\kernel_\mappingFunction\left(\inputVector_i, \inputVector_j\right) = \alpha \int \activationScalar\left(\mappingVector^{(1)}, \inputVector_i\right) \activationScalar\left(\mappingVector^{(1)}, \inputVector_j\right) p(\mappingVector^{(1)}) \text{d}\mappingVector^{(1)}
$$ And that's not *only* for Gaussian $p(\mappingVectorTwo)$. In fact
this result holds for a range of activations, and a range of prior
densities because of the *central limit theorem*.

To write it in the form of a probabilistic program, as long as the
distribution for $\phi_i$ implied by this short probabilistic program,
$$
  \begin{align*}
  \mappingVectorTwo & \sim p(\cdot)\\
  \phi_i & = \activationScalar\left(\mappingVectorTwo, \inputVector_i\right), 
  \end{align*}
  $$ has finite variance, then the result of taking the number of hidden
units to infinity, with appropriate scaling, is also a Gaussian process.

### Further Reading

To understand this argument in more detail, I highly recommend reading
chapter 2 of Neal's thesis, which remains easy to read and clear today.
Indeed, for readers interested in Bayesian neural networks, both Raford
Neal's and David MacKay's PhD thesis [@MacKay:bayesian92] remain
essential reading. Both theses embody a clarity of thought, and an
ability to weave together threads from different fields that was the
business of machine learning in the 1990s. Radford and David were also
pioneers in making their software widely available and publishing
material on the web.

```{code-cell}
import numpy as np
import teaching_plots as plot
```

```{code-cell}
%load -s compute_kernel mlai.py
```

```{code-cell}
%load -s eq_cov mlai.py
```

```{code-cell}
np.random.seed(10)
plot.rejection_samples(compute_kernel, kernel=eq_cov, 
                       lengthscale=0.25, diagrams='../slides/diagrams/gp')
```

```{code-cell}
import pods
pods.notebook.display_plots('gp_rejection_samples{sample:0>3}.svg', 
                            '../slides/diagrams/gp', sample=(1,5))
```

<!-- ### Two Dimensional Gaussian Distribution -->
<!-- include{_ml/includes/two-d-gaussian.md} -->
### Distributions over Functions

```{code-cell}
import numpy as np
np.random.seed(4949)
```

```{code-cell}
import teaching_plots as plot
import pods
```

### Sampling a Function {#sampling-a-function data-transition="none"}

**Multi-variate Gaussians**

-   We will consider a Gaussian with a particular structure of
    covariance matrix.

-   Generate a single sample from this 25 dimensional Gaussian
    distribution,
    $\mappingFunctionVector=\left[\mappingFunction_{1},\mappingFunction_{2}\dots \mappingFunction_{25}\right]$.

-   We will plot these points against their index.

```{code-cell}
%load -s compute_kernel mlai.py
```

```{code-cell}
%load -s polynomial_cov mlai.py
```

```{code-cell}
%load -s exponentiated_quadratic mlai.py
```

```{code-cell}
plot.two_point_sample(compute_kernel, kernel=exponentiated_quadratic, 
                      lengthscale=0.5, diagrams='../slides/diagrams/gp')
```

```{code-cell}
pods.notebook.display_plots('two_point_sample{sample:0>3}.svg', 
                            '../slides/diagrams/gp', sample=(0,13))
```

### Uluru

<img class="" src="../slides/diagrams/gp/799px-Uluru_Panorama.jpg" width="" align="" style="background:none; border:none; box-shadow:none;">

### Prediction with Correlated Gaussians

-   Prediction of $\mappingFunction_2$ from $\mappingFunction_1$
    requires *conditional density*.

-   Conditional density is *also* Gaussian. $$
    p(\mappingFunction_2|\mappingFunction_1) = \gaussianDist{\mappingFunction_2}{\frac{\kernelScalar_{1, 2}}{\kernelScalar_{1, 1}}\mappingFunction_1}{ \kernelScalar_{2, 2} - \frac{\kernelScalar_{1,2}^2}{\kernelScalar_{1,1}}}
    $$ where covariance of joint density is given by $$
    \kernelMatrix = \begin{bmatrix} \kernelScalar_{1, 1} & \kernelScalar_{1, 2}\\ \kernelScalar_{2, 1} & \kernelScalar_{2, 2}\end{bmatrix}
    $$

```{code-cell}
pods.notebook.display_plots('two_point_sample{sample:0>3}.svg', 
                            '../slides/diagrams/gp', sample=(13,17))
```

### Key Object

-   Covariance function, $\kernelMatrix$

-   Determines properties of samples.

-   Function of $\inputMatrix$,
    $$\kernelScalar_{i,j} = \kernelScalar(\inputVector_i, \inputVector_j)$$

### Linear Algebra

-   Posterior mean

```{code-cell}
$$\mappingFunction_D(\inputVector_*) = \kernelVector(\inputVector_*, \inputMatrix) \kernelMatrix^{-1}
\mathbf{y}$$
```

-   Posterior covariance
    $$\mathbf{C}_* = \kernelMatrix_{*,*} - \kernelMatrix_{*,\mappingFunctionVector}
    \kernelMatrix^{-1} \kernelMatrix_{\mappingFunctionVector, *}$$

### Linear Algebra

-   Posterior mean

```{code-cell}
$$\mappingFunction_D(\inputVector_*) = \kernelVector(\inputVector_*, \inputMatrix) \boldsymbol{\alpha}$$
```

-   Posterior covariance
    $$\covarianceMatrix_* = \kernelMatrix_{*,*} - \kernelMatrix_{*,\mappingFunctionVector}
    \kernelMatrix^{-1} \kernelMatrix_{\mappingFunctionVector, *}$$

### 

<img src="../slides/diagrams/gp_prior_samples_data.svg" align="">

### 

<img src="../slides/diagrams/gp_rejection_samples.svg" align="">

### 

<img src="../slides/diagrams/gp_prediction.svg" align="">

```{code-cell}
%load -s eq_cov mlai.py
```

```{code-cell}
import teaching_plots as plot
import mlai
import numpy as np
```

```{code-cell}
K, anim=plot.animate_covariance_function(mlai.compute_kernel, 
                                         kernel=eq_cov, lengthscale=0.2)
```

```{code-cell}
from IPython.core.display import HTML
```

```{code-cell}
HTML(anim.to_jshtml())
```

```{code-cell}
plot.save_animation(anim, 
                    diagrams='../slides/diagrams/kern', 
                    filename='eq_covariance.html')
```

### Exponentiated Quadratic Covariance

The exponentiated quadratic covariance, also known as the Gaussian
covariance or the RBF covariance and the squared exponential. Covariance
between two points is related to the negative exponential of the squared
distnace between those points. This covariance function can be derived
in a few different ways: as the infinite limit of a radial basis
function neural network, as diffusion in the heat equation, as a
Gaussian filter in *Fourier space* or as the composition as a series of
linear filters applied to a base function.

The covariance takes the following form, $$
\kernelScalar(\inputVector, \inputVector^\prime) = \alpha \exp\left(-\frac{\ltwoNorm{\inputVector - \inputVector^\prime}^2}{2\ell^2}\right)
$$ where $\ell$ is the *length scale* or *time scale* of the process and
$\alpha$ represents the overall process variance.
<table>
<tr>
<td width="50%">
<img src="../slides/diagrams/kern/eq_covariance.svg" align="">
</td>
<td width="50%">
<iframe src="../slides/diagrams/kern/eq_covariance.html" width="512" height="384" allowtransparency="true" frameborder="0">
</iframe>
</td>
</tr>
</table>

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
import pods
import teaching_plots as plot
import mlai
```

### Olympic Marathon Data

The first thing we will do is load a standard data set for regression
modelling. The data consists of the pace of Olympic Gold Medal Marathon
winners for the Olympics from 1896 to present. First we load in the data
and plot.

```{code-cell}
data = pods.datasets.olympic_marathon_men()
x = data['X']
y = data['Y']

offset = y.mean()
scale = np.sqrt(y.var())

xlim = (1875,2030)
ylim = (2.5, 6.5)
yhat = (y-offset)/scale

fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
_ = ax.plot(x, y, 'r.',markersize=10)
ax.set_xlabel('year', fontsize=20)
ax.set_ylabel('pace min/km', fontsize=20)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

mlai.write_figure(figure=fig, filename='../slides/diagrams/datasets/olympic-marathon.svg', transparent=True, frameon=True)
```

### Olympic Marathon Data

<table>
<tr>
<td width="70%">
-   Gold medal times for Olympic Marathon since 1896.

-   Marathons before 1924 didn’t have a standardised distance.

-   Present results using pace per km.

-   In 1904 Marathon was badly organised leading to very slow times.

</td>
<td width="30%">
![image](../slides/diagrams/Stephen_Kiprotich.jpg) <small>Image from
Wikimedia Commons <http://bit.ly/16kMKHQ></small>
</td>
</tr>
</table>
<img src="../slides/diagrams/datasets/olympic-marathon.svg" align="">

Things to notice about the data include the outlier in 1904, in this
year, the olympics was in St Louis, USA. Organizational problems and
challenges with dust kicked up by the cars following the race meant that
participants got lost, and only very few participants completed.

More recent years see more consistently quick marathons.

Our first objective will be to perform a Gaussian process fit to the
data, we'll do this using the [GPy
software](https://github.com/SheffieldML/GPy).

```{code-cell}
m_full = GPy.models.GPRegression(x,yhat)
_ = m_full.optimize() # Optimize parameters of covariance function
```

The first command sets up the model, then

```{code-cell}
m_full.optimize()
```

optimizes the parameters of the covariance function and the noise level
of the model. Once the fit is complete, we'll try creating some test
points, and computing the output of the GP model in terms of the mean
and standard deviation of the posterior functions between 1870 and 2030.
We plot the mean function and the standard deviation at 200 locations.
We can obtain the predictions using

```{code-cell}
y_mean, y_var = m_full.predict(xt)
```

```{code-cell}
xt = np.linspace(1870,2030,200)[:,np.newaxis]
yt_mean, yt_var = m_full.predict(xt)
yt_sd=np.sqrt(yt_var)
```

Now we plot the results using the helper function in `teaching_plots`.

```{code-cell}
import teaching_plots as plot
```

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
plot.model_output(m_full, scale=scale, offset=offset, ax=ax, xlabel='year', ylabel='pace min/km', fontsize=20, portion=0.2)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
mlai.write_figure(figure=fig,
                  filename='../slides/diagrams/gp/olympic-marathon-gp.svg', 
                  transparent=True, frameon=True)
```

<img src="../slides/diagrams/gp/olympic-marathon-gp.svg" align="">

### Fit Quality

In the fit we see that the error bars (coming mainly from the noise
variance) are quite large. This is likely due to the outlier point in
1904, ignoring that point we can see that a tighter fit is obtained. To
see this making a version of the model, `m_clean`, where that point is
removed.

```{code-cell}
x_clean=np.vstack((x[0:2, :], x[3:, :]))
y_clean=np.vstack((y[0:2, :], y[3:, :]))

m_clean = GPy.models.GPRegression(x_clean,y_clean)
_ = m_clean.optimize()
```

Data is fine for answering very specific questions, like "Who won the
Olympic Marathon in 2012?", because we have that answer stored, however,
we are not given the answer to many other questions. For example, Alan
Turing was a formidable marathon runner, in 1946 he ran a time 2 hours
46 minutes (just under four minutes per kilometer, faster than I and
most of the other [Endcliffe Park
Run](http://www.parkrun.org.uk/sheffieldhallam/) runners can do 5 km).
What is the probability he would have won an Olympics if one had been
held in 1946?

<table>
<tr>
<td width="40%">
<img class="" src="../slides/diagrams/turing-run.jpg" width="40%" align="" style="background:none; border:none; box-shadow:none;">
</td>
<td width="50%">
<img class="" src="../slides/diagrams/turing-times.gif" width="50%" align="" style="background:none; border:none; box-shadow:none;">
</td>
</tr>
</table>
<center>
*Alan Turing, in 1946 he was only 11 minutes slower than the winner of
the 1948 games. Would he have won a hypothetical games held in 1946?
Source: [Alan Turing Internet
Scrapbook](http://www.turing.org.uk/scrapbook/run.html) *
</center>
### Basis Function Covariance

The fixed basis function covariane just comes from the properties of a
multivariate Gaussian, if we decide $$
\mappingFunctionVector=\basisMatrix\mappingVector
$$ and then we assume $$
\mappingVector \sim \gaussianSamp{\zerosVector}{\alpha\eye}
$$ then it follows from the properties of a multivariate Gaussian that
$$
\mappingFunctionVector \sim \gaussianSamp{\zerosVector}{\alpha\basisMatrix\basisMatrix^\top}
$$ meaning that the vector of observations from the function is jointly
distributed as a Gaussian process and the covariance matrix is
$\kernelMatrix = \alpha\basisMatrix \basisMatrix^\top$, each element of
the covariance matrix can then be found as the inner product between two
rows of the basis funciton matrix. $$
\kernel(\inputVector, \inputVector^\prime) = \basisVector(\inputVector)^\top \basisVector(\inputVector^\prime)
$$

<table>
<tr>
<td width="45%">
<img src="../slides/diagrams/kern/basis_covariance.svg" align="">
</td>
<td width="45%">
<img class="negate" src="../slides/diagrams/kern/basis_covariance.gif" width="40%" align="center" style="background:none; border:none; box-shadow:none;">
</td>
</tr>
</table>
### Brownian Covariance

```{code-cell}
%load -s brownian_cov mlai.py
```

```{code-cell}
import teaching_plots as plot
import mlai
import numpy as np
```

```{code-cell}
t=np.linspace(0, 2, 200)[:, np.newaxis]
K, anim=plot.animate_covariance_function(mlai.compute_kernel, 
                                         t, 
                                         kernel=brownian_cov)
```

```{code-cell}
from IPython.core.display import HTML
```

```{code-cell}
HTML(anim.to_jshtml())
```

```{code-cell}
plot.save_animation(anim, 
                    diagrams='../slides/diagrams/kern', 
                    filename='brownian_covariance.html')
```

Brownian motion is also a Gaussian process. It follows a Gaussian random
walk, with diffusion occuring at each time point driven by a Gaussian
input. This implies it is both Markov and Gaussian. The covariane
function for Brownian motion has the form $$
\kernelScalar(t, t^\prime) = \alpha \min(t, t^\prime)
$$
<table>
<tr>
<td width="50%">
<img src="../slides/diagrams/kern/brownian_covariance.svg" align="">
</td>
<td width="50%">
<iframe src="../slides/diagrams/kern/brownian_covariance.html" width="512" height="384" allowtransparency="true" frameborder="0">
</iframe>
</td>
</tr>
</table>
<center>
*The covariance of Brownian motion, and some samples from the covariance
showing the functional form. *
</center>
### MLP Covariance

```{code-cell}
%load -s mlp_cov mlai.py
```

```{code-cell}
import teaching_plots as plot
import mlai
import numpy as np
```

```{code-cell}
K, anim=plot.animate_covariance_function(mlai.compute_kernel, 
                                         kernel=mlp_cov, lengthscale=0.2)
```

```{code-cell}
from IPython.core.display import HTML
```

```{code-cell}
HTML(anim.to_jshtml())
```

```{code-cell}
plot.save_animation(anim, 
                    diagrams='../slides/diagrams/kern', 
                    filename='mlp_covariance.html')
```

The multi-layer perceptron (MLP) covariance, also known as the neural
network covariance or the arcsin covariance, is derived by considering
the infinite limit of a neural network. $$
\kernelScalar(\inputVector, \inputVector^\prime) = \alpha \arcsin\left(\frac{w \inputVector^\top \inputVector^\prime + b}{\sqrt{\left(w \inputVector^\top \inputVector + b + 1\right)\left(w \left.\inputVector^\prime\right.^\top \inputVector^\prime + b + 1\right)}}\right)
$$

<table>
<tr>
<td width="50%">
<img src="../slides/diagrams/kern/mlp_covariance.svg" align="">
</td>
<td width="50%">
<iframe src="../slides/diagrams/kern/mlp_covariance.html" width="512" height="384" allowtransparency="true" frameborder="0">
</iframe>
</td>
</tr>
</table>
<center>
*The multi-layer perceptron covariance function. This is derived by
considering the infinite limit of a neural network with probit
activation functions. *
</center>
<div style="fontsize:120px;vertical-align:middle;">

<img src="../slides/diagrams/earth_PNG37.png" width="20%" style="display:inline-block;background:none;vertical-align:middle;border:none;box-shadow:none;">$=f\Bigg($
<img src="../slides/diagrams/Planck_CMB.png"  width="50%" style="display:inline-block;background:none;vertical-align:middle;border:none;box-shadow:none;">$\Bigg)$

</div>

<center>
*The cosmic microwave background is, to a very high degree of precision,
a Gaussian process. The parameters of its covariance function are given
by fundamental parameters of the universe, such as the amount of dark
matter and mass. *
</center>
### Deep Gaussian Processes

<center>
*Image credit: Kai Arulkumaran *
</center>

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import GPy

import mlai
import teaching_plots as plot 
from gp_tutorial import gpplot
```

```{code-cell}
np.random.seed(101)
```

### A Simple Regression Problem

Here we set up a simple one dimensional regression problem. The input
locations, $\inputMatrix$, are in two separate clusters. The response
variable, $\dataVector$, is sampled from a Gaussian process with an
exponentiated quadratic covariance.

```{code-cell}
N = 50
noise_var = 0.01
X = np.zeros((50, 1))
X[:25, :] = np.linspace(0,3,25)[:,None] # First cluster of inputs/covariates
X[25:, :] = np.linspace(7,10,25)[:,None] # Second cluster of inputs/covariates

# Sample response variables from a Gaussian process with exponentiated quadratic covariance.
k = GPy.kern.RBF(1)
y = np.random.multivariate_normal(np.zeros(N),k.K(X)+np.eye(N)*np.sqrt(noise_var)).reshape(-1,1)
```

First we perform a full Gaussian process regression on the data. We
create a GP model, `m_full`, and fit it to the data, plotting the
resulting fit.

```{code-cell}
m_full = GPy.models.GPRegression(X,y)
_ = m_full.optimize(messages=True) # Optimize parameters of covariance function
```

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
plot.model_output(m_full, ax=ax, xlabel='$x$', ylabel='$y$', fontsize=20, portion=0.2)
xlim = ax.get_xlim()
ylim = ax.get_ylim()
mlai.write_figure(figure=fig,
                  filename='../slides/diagrams/gp/sparse-demo-full-gp.svg', 
                  transparent=True, frameon=True)
```

<img src="../slides/diagrams/gp/sparse-demo-full-gp.svg" align="">
<center>
*Full Gaussian process fitted to the data set. *
</center>
Now we set up the inducing variables, $\mathbf{u}$. Each inducing
variable has its own associated input index, $\mathbf{Z}$, which lives
in the same space as $\inputMatrix$. Here we are using the true
covariance function parameters to generate the fit.

```{code-cell}
kern = GPy.kern.RBF(1)
Z = np.hstack(
        (np.linspace(2.5,4.,3),
        np.linspace(7,8.5,3)))[:,None]
m = GPy.models.SparseGPRegression(X,y,kernel=kern,Z=Z)
m.noise_var = noise_var
m.inducing_inputs.constrain_fixed()
display(m)
```

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
plot.model_output(m, ax=ax, xlabel='$x$', ylabel='$y$', fontsize=20, portion=0.2, xlim=xlim, ylim=ylim)
mlai.write_figure(figure=fig,
                  filename='../slides/diagrams/gp/sparse-demo-constrained-inducing-6-unlearned-gp.svg', 
                  transparent=True, frameon=True)
```

<img src="../slides/diagrams/gp/sparse-demo-constrained-inducing-6-unlearned-gp.svg" align="">
<center>
*Sparse Gaussian process fitted with six inducing variables, no
optimization of parameters or inducing variables. *
</center>

```{code-cell}
_ = m.optimize(messages=True)
display(m)
```

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
plot.model_output(m, ax=ax, xlabel='$x$', ylabel='$y$', fontsize=20, portion=0.2, xlim=xlim, ylim=ylim)
mlai.write_figure(figure=fig,
                  filename='../slides/diagrams/gp/sparse-demo-full-gp.svg', 
                  transparent=True, frameon=True)
```

<img src="../slides/diagrams/gp/sparse-demo-constrained-inducing-6-learned-gp.svg" align="">
<center>
*Gaussian process fitted with inducing variables fixed and parameters
optimized *
</center>

```{code-cell}
m.randomize()
m.inducing_inputs.unconstrain()
_ = m.optimize(messages=True)
```

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
plot.model_output(m, ax=ax, xlabel='$x$', ylabel='$y$', fontsize=20, portion=0.2,xlim=xlim, ylim=ylim)
mlai.write_figure(figure=fig,
                  filename='../slides/diagrams/gp/sparse-demo-unconstrained-inducing-6-gp.svg', 
                  transparent=True, frameon=True)
```

<img src="../slides/diagrams/gp/sparse-demo-unconstrained-inducing-6-gp.svg" align="">
<center>
*Gaussian process fitted with location of inducing variables and
parameters both optimized *
</center>
Now we will vary the number of inducing points used to form the
approximation.

```{code-cell}
m.num_inducing=8
m.randomize()
M = 8
m.set_Z(np.random.rand(M,1)*12)

_ = m.optimize(messages=True)
```

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
plot.model_output(m, ax=ax, xlabel='$x$', ylabel='$y$', fontsize=20, portion=0.2, xlim=xlim, ylim=ylim)
mlai.write_figure(figure=fig,
                  filename='../slides/diagrams/gp/sparse-demo-sparse-inducing-8-gp.svg', 
                  transparent=True, frameon=True)
```

<center>
*Comparison of the full Gaussian process fit with a sparse Gaussian
process using eight inducing varibles. Both inducing variables and
parameters are optimized. *
</center>
And we can compare the probability of the result to the full model.

```{code-cell}
print(m.log_likelihood(), m_full.log_likelihood())
```

### Modern Review

-   *A Unifying Framework for Gaussian Process Pseudo-Point
    Approximations using Power Expectation Propagation*
    @Thang:unifying17

-   *Deep Gaussian Processes and Variational Propagation of Uncertainty*
    @Damianou:thesis2015

```{code-cell}
import teaching_plots as plot
```

```{code-cell}
plot.deep_nn(diagrams='../slides/diagrams/deepgp/')
```

<center>
*A deep neural network. Input nodes are shown at the bottom. Each hidden
layer is the result of applying an affine transformation to the previous
layer and placing through an activation function. *
</center>
Mathematically, each layer of a neural network is given through
computing the activation function, $\basisFunction(\cdot)$, contingent
on the previous layer, or the inputs. In this way the activation
functions, are composed to generate more complex interactions than would
be possible with any single layer. $$
\begin{align}
    \hiddenVector_{1} &= \basisFunction\left(\mappingMatrix_1 \inputVector\right)\\
    \hiddenVector_{2} &=  \basisFunction\left(\mappingMatrix_2\hiddenVector_{1}\right)\\
    \hiddenVector_{3} &= \basisFunction\left(\mappingMatrix_3 \hiddenVector_{2}\right)\\
    \dataVector &= \mappingVector_4 ^\top\hiddenVector_{3}
\end{align}
$$

### Overfitting

One potential problem is that as the number of nodes in two adjacent
layers increases, the number of parameters in the affine transformation
between layers, $\mappingMatrix$, increases. If there are $k_{i-1}$
nodes in one layer, and $k_i$ nodes in the following, then that matrix
contains $k_i k_{i-1}$ parameters, when we have layer widths in the
1000s that leads to millions of parameters.

One proposed solution is known as *dropout* where only a sub-set of the
neural network is trained at each iteration. An alternative solution
would be to reparameterize $\mappingMatrix$ with its *singular value
decomposition*. $$
  \mappingMatrix = \eigenvectorMatrix\eigenvalueMatrix\eigenvectwoMatrix^\top
  $$ or $$
  \mappingMatrix = \eigenvectorMatrix\eigenvectwoMatrix^\top
  $$ where if $\mappingMatrix \in \Re^{k_1\times k_2}$ then
$\eigenvectorMatrix\in \Re^{k_1\times q}$ and
$\eigenvectwoMatrix \in \Re^{k_2\times q}$, i.e. we have a low rank
matrix factorization for the weights.

```{code-cell}
import teaching_plots as plot
```

```{code-cell}
plot.low_rank_approximation(diagrams='../slides/diagrams')
```

<img src="../slides/diagrams/wisuvt.svg" align="">
<center>
*Pictorial representation of the low rank form of the matrix
$\mappingMatrix$ *
</center>

```{code-cell}
import teaching_plots as plot
```

```{code-cell}
plot.deep_nn_bottleneck(diagrams='../slides/diagrams/deepgp')
```

Including the low rank decomposition of $\mappingMatrix$ in the neural
network, we obtain a new mathematical form. Effectively, we are adding
additional *latent* layers, $\latentVector$, in between each of the
existing hidden layers. In a neural network these are sometimes known as
*bottleneck* layers. The network can now be written mathematically as $$
\begin{align}
  \latentVector_{1} &= \eigenvectwoMatrix^\top_1 \inputVector\\
  \hiddenVector_{1} &= \basisFunction\left(\eigenvectorMatrix_1 \latentVector_{1}\right)\\
  \latentVector_{2} &= \eigenvectwoMatrix^\top_2 \hiddenVector_{1}\\
  \hiddenVector_{2} &= \basisFunction\left(\eigenvectorMatrix_2 \latentVector_{2}\right)\\
  \latentVector_{3} &= \eigenvectwoMatrix^\top_3 \hiddenVector_{2}\\
  \hiddenVector_{3} &= \basisFunction\left(\eigenvectorMatrix_3 \latentVector_{3}\right)\\
  \dataVector &= \mappingVector_4^\top\hiddenVector_{3}.
\end{align}
$$

### A Cascade of Neural Networks

$$
\begin{align}
  \latentVector_{1} &= \eigenvectwoMatrix^\top_1 \inputVector\\
  \latentVector_{2} &= \eigenvectwoMatrix^\top_2 \basisFunction\left(\eigenvectorMatrix_1 \latentVector_{1}\right)\\
  \latentVector_{3} &= \eigenvectwoMatrix^\top_3 \basisFunction\left(\eigenvectorMatrix_2 \latentVector_{2}\right)\\
  \dataVector &= \mappingVector_4 ^\top \latentVector_{3}
\end{align}
$$

### Cascade of Gaussian Processes

-   Replace each neural network with a Gaussian process $$
    \begin{align}
      \latentVector_{1} &= \mappingFunctionVector_1\left(\inputVector\right)\\
      \latentVector_{2} &= \mappingFunctionVector_2\left(\latentVector_{1}\right)\\
      \latentVector_{3} &= \mappingFunctionVector_3\left(\latentVector_{2}\right)\\
      \dataVector &= \mappingFunctionVector_4\left(\latentVector_{3}\right)
    \end{align}
    $$

-   Equivalent to prior over parameters, take width of each layer to
    infinity.

Mathematically, a deep Gaussian process can be seen as a composite
*multivariate* function, $$
  \mathbf{g}(\inputVector)=\mappingFunctionVector_5(\mappingFunctionVector_4(\mappingFunctionVector_3(\mappingFunctionVector_2(\mappingFunctionVector_1(\inputVector))))).
  $$ Or if we view it from the probabilistic perspective we can see that
a deep Gaussian process is specifying a factorization of the joint
density, the standard deep model takes the form of a Markov chain.

```{code-cell}
from matplotlib import rc

rc("font", **{'family':'sans-serif','sans-serif':['Helvetica'],'size':30})
rc("text", usetex=True)
```

```{code-cell}
pgm = plot.horizontal_chain(depth=5)
pgm.render().figure.savefig("../slides/diagrams/deepgp/deep-markov.svg", transparent=True)
```

$$
  p(\dataVector|\inputVector)= p(\dataVector|\mappingFunctionVector_5)p(\mappingFunctionVector_5|\mappingFunctionVector_4)p(\mappingFunctionVector_4|\mappingFunctionVector_3)p(\mappingFunctionVector_3|\mappingFunctionVector_2)p(\mappingFunctionVector_2|\mappingFunctionVector_1)p(\mappingFunctionVector_1|\inputVector)
  $$

<img src="../slides/diagrams/deepgp/deep-markov.svg" align="">
<center>
*Probabilistically the deep Gaussian process can be represented as a
Markov chain. *
</center>

```{code-cell}
from matplotlib import rc
rc("font", **{'family':'sans-serif','sans-serif':['Helvetica'], 'size':15})
rc("text", usetex=True)
```

```{code-cell}
pgm = plot.vertical_chain(depth=5)
pgm.render().figure.savefig("../slides/diagrams/deepgp/deep-markov-vertical.svg", transparent=True)
```

<img src="../slides/diagrams/deepgp/deep-markov-vertical.svg" align="">

### Why Deep?

If the result of composing many functions together is simply another
function, then why do we bother? The key point is that we can change the
class of functions we are modeling by composing in this manner. A
Gaussian process is specifying a prior over functions, and one with a
number of elegant properties. For example, the derivative process (if it
exists) of a Gaussian process is also Gaussian distributed. That makes
it easy to assimilate, for example, derivative observations. But that
also might raise some alarm bells. That implies that the *marginal
derivative distribution* is also Gaussian distributed. If that's the
case, then it means that functions which occasionally exhibit very large
derivatives are hard to model with a Gaussian process. For example, a
function with jumps in.

A one off discontinuity is easy to model with a Gaussian process, or
even multiple discontinuities. They can be introduced in the mean
function, or independence can be forced between two covariance functions
that apply in different areas of the input space. But in these cases we
will need to specify the number of discontinuities and where they occur.
In otherwords we need to *parameterise* the discontinuities. If we do
not know the number of discontinuities and don't wish to specify where
they occur, i.e. if we want a non-parametric representation of
discontinuities, then the standard Gaussian process doesn't help.

### Stochastic Process Composition

The deep Gaussian process leads to *non-Gaussian* models, and
non-Gaussian characteristics in the covariance function. In effect, what
we are proposing is that we change the properties of the functions we
are considering by \*composing stochastic processes\$. This is an
approach to creating new stochastic processes from well known processes.

<img src="../slides/diagrams/deepgp/deep-markov-vertical.svg" align="">

```{code-cell}
pgm = plot.vertical_chain(depth=5, shape=[2, 7])
pgm.add_node(daft.Node('y_2', r'$\mathbf{y}_2$', 1.5, 3.5, observed=True))
pgm.add_edge('f_2', 'y_2')
pgm.render().figure.savefig("../slides/diagrams/deepgp/deep-markov-vertical-side.svg", transparent=True)
```

Additionally, we are not constrained to the formalism of the chain. For
example, we can easily add single nodes emerging from some point in the
depth of the chain. This allows us to combine the benefits of the
graphical modelling formalism, but with a powerful framework for
relating one set of variables to another, that of Gaussian processes
<img src="../slides/diagrams/deepgp/deep-markov-vertical-side.svg" align="">

```{code-cell}
plot.non_linear_difficulty_plot_3(diagrams='../../slides/diagrams/dimred/')
```

### Difficulty for Probabilistic Approaches {#difficulty-for-probabilistic-approaches data-transition="None"}

-   Propagate a probability distribution through a non-linear mapping.

-   Normalisation of distribution becomes intractable.

<img src="../slides/diagrams/dimred/nonlinear-mapping-3d-plot.svg" align="center">

```{code-cell}
plot.non_linear_difficulty_plot_2(diagrams='../../slides/diagrams/dimred/')
```

### Difficulty for Probabilistic Approaches {#difficulty-for-probabilistic-approaches-1 data-transition="None"}

-   Propagate a probability distribution through a non-linear mapping.

-   Normalisation of distribution becomes intractable.

<img src="../slides/diagrams/dimred/nonlinear-mapping-2d-plot.svg" align="center">

```{code-cell}
plot.non_linear_difficulty_plot_1(diagrams='../../slides/diagrams/dimred')
```

### Difficulty for Probabilistic Approaches {#difficulty-for-probabilistic-approaches-2 data-transition="None"}

-   Propagate a probability distribution through a non-linear mapping.

-   Normalisation of distribution becomes intractable.

<img src="../slides/diagrams/dimred/gaussian-through-nonlinear.svg" align="center">

### Deep Gaussian Processes

-   Deep architectures allow abstraction of features
    [@Bengio:deep09; @Hinton:fast06; @Salakhutdinov:quantitative08]

-   We use variational approach to stack GP models.

```{code-cell}
plot.stack_gp_sample(kernel=GPy.kern.Linear,
                     diagrams="../../slides/diagrams/deepgp")
```

```{code-cell}
pods.notebook.display_plots('stack-gp-sample-Linear-{sample:0>1}.svg', 
                            directory='../../slides/diagrams/deepgp', sample=(0,4))
```

### Stacked PCA

<img src="../slides/diagrams/stack-pca-sample-4.svg" align="">

```{code-cell}
plot.stack_gp_sample(kernel=GPy.kern.RBF,
                     diagrams="../../slides/diagrams/deepgp")
```

```{code-cell}
pods.notebook.display_plots('stack-gp-sample-RBF-{sample:0>1}.svg', 
                            directory='../../slides/diagrams/deepgp', sample=(0,4))
```

### Stacked GP

<img src="../slides/diagrams/stack-gp-sample-4.svg" align="">

### Analysis of Deep GPs

-   *Avoiding pathologies in very deep networks* @Duvenaud:pathologies14
    show that the derivative distribution of the process becomes more
    *heavy tailed* as number of layers increase.

-   *How Deep Are Deep Gaussian Processes?* @Dunlop:deep2017 perform a
    theoretical analysis possible through conditional Gaussian Markov
    property.

```{code-cell}
from IPython.lib.display import YouTubeVideo
YouTubeVideo('XhIvygQYFFQ')
```

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
import pods
import teaching_plots as plot
import mlai
```

### Olympic Marathon Data

The first thing we will do is load a standard data set for regression
modelling. The data consists of the pace of Olympic Gold Medal Marathon
winners for the Olympics from 1896 to present. First we load in the data
and plot.

```{code-cell}
data = pods.datasets.olympic_marathon_men()
x = data['X']
y = data['Y']

offset = y.mean()
scale = np.sqrt(y.var())

xlim = (1875,2030)
ylim = (2.5, 6.5)
yhat = (y-offset)/scale

fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
_ = ax.plot(x, y, 'r.',markersize=10)
ax.set_xlabel('year', fontsize=20)
ax.set_ylabel('pace min/km', fontsize=20)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

mlai.write_figure(figure=fig, filename='../slides/diagrams/datasets/olympic-marathon.svg', transparent=True, frameon=True)
```

### Olympic Marathon Data

<table>
<tr>
<td width="70%">
-   Gold medal times for Olympic Marathon since 1896.

-   Marathons before 1924 didn’t have a standardised distance.

-   Present results using pace per km.

-   In 1904 Marathon was badly organised leading to very slow times.

</td>
<td width="30%">
![image](../slides/diagrams/Stephen_Kiprotich.jpg) <small>Image from
Wikimedia Commons <http://bit.ly/16kMKHQ></small>
</td>
</tr>
</table>
<img src="../slides/diagrams/datasets/olympic-marathon.svg" align="">

Things to notice about the data include the outlier in 1904, in this
year, the olympics was in St Louis, USA. Organizational problems and
challenges with dust kicked up by the cars following the race meant that
participants got lost, and only very few participants completed.

More recent years see more consistently quick marathons.

Our first objective will be to perform a Gaussian process fit to the
data, we'll do this using the [GPy
software](https://github.com/SheffieldML/GPy).

```{code-cell}
m_full = GPy.models.GPRegression(x,yhat)
_ = m_full.optimize() # Optimize parameters of covariance function
```

The first command sets up the model, then

```{code-cell}
m_full.optimize()
```

optimizes the parameters of the covariance function and the noise level
of the model. Once the fit is complete, we'll try creating some test
points, and computing the output of the GP model in terms of the mean
and standard deviation of the posterior functions between 1870 and 2030.
We plot the mean function and the standard deviation at 200 locations.
We can obtain the predictions using

```{code-cell}
y_mean, y_var = m_full.predict(xt)
```

```{code-cell}
xt = np.linspace(1870,2030,200)[:,np.newaxis]
yt_mean, yt_var = m_full.predict(xt)
yt_sd=np.sqrt(yt_var)
```

Now we plot the results using the helper function in `teaching_plots`.

```{code-cell}
import teaching_plots as plot
```

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
plot.model_output(m_full, scale=scale, offset=offset, ax=ax, xlabel='year', ylabel='pace min/km', fontsize=20, portion=0.2)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
mlai.write_figure(figure=fig,
                  filename='../slides/diagrams/gp/olympic-marathon-gp.svg', 
                  transparent=True, frameon=True)
```

<img src="../slides/diagrams/gp/olympic-marathon-gp.svg" align="">

### Fit Quality

In the fit we see that the error bars (coming mainly from the noise
variance) are quite large. This is likely due to the outlier point in
1904, ignoring that point we can see that a tighter fit is obtained. To
see this making a version of the model, `m_clean`, where that point is
removed.

```{code-cell}
x_clean=np.vstack((x[0:2, :], x[3:, :]))
y_clean=np.vstack((y[0:2, :], y[3:, :]))

m_clean = GPy.models.GPRegression(x_clean,y_clean)
_ = m_clean.optimize()
```

Data is fine for answering very specific questions, like "Who won the
Olympic Marathon in 2012?", because we have that answer stored, however,
we are not given the answer to many other questions. For example, Alan
Turing was a formidable marathon runner, in 1946 he ran a time 2 hours
46 minutes (just under four minutes per kilometer, faster than I and
most of the other [Endcliffe Park
Run](http://www.parkrun.org.uk/sheffieldhallam/) runners can do 5 km).
What is the probability he would have won an Olympics if one had been
held in 1946?

<table>
<tr>
<td width="40%">
<img class="" src="../slides/diagrams/turing-run.jpg" width="40%" align="" style="background:none; border:none; box-shadow:none;">
</td>
<td width="50%">
<img class="" src="../slides/diagrams/turing-times.gif" width="50%" align="" style="background:none; border:none; box-shadow:none;">
</td>
</tr>
</table>
<center>
*Alan Turing, in 1946 he was only 11 minutes slower than the winner of
the 1948 games. Would he have won a hypothetical games held in 1946?
Source: [Alan Turing Internet
Scrapbook](http://www.turing.org.uk/scrapbook/run.html) *
</center>
### Deep GP Fit

Let's see if a deep Gaussian process can help here. We will construct a
deep Gaussian process with one hidden layer (i.e. one Gaussian process
feeding into another).

Build a Deep GP with an additional hidden layer (one dimensional) to fit
the model.

```{code-cell}
hidden = 1
m = deepgp.DeepGP([y.shape[1],hidden,x.shape[1]],Y=yhat, X=x, inits=['PCA','PCA'], 
                  kernels=[GPy.kern.RBF(hidden,ARD=True),
                           GPy.kern.RBF(x.shape[1],ARD=True)], # the kernels for each layer
                  num_inducing=50, back_constraint=False)
```

Deep Gaussian process models also can require some thought in
initialization. Here we choose to start by setting the noise variance to
be one percent of the data variance.

Optimization requires moving variational parameters in the hidden layer
representing the mean and variance of the expected values in that layer.
Since all those values can be scaled up, and this only results in a
downscaling in the output of the first GP, and a downscaling of the
input length scale to the second GP. It makes sense to first of all fix
the scales of the covariance function in each of the GPs.

Sometimes, deep Gaussian processes can find a local minima which
involves increasing the noise level of one or more of the GPs. This
often occurs because it allows a minimum in the KL divergence term in
the lower bound on the likelihood. To avoid this minimum we habitually
train with the likelihood variance (the noise on the output of the GP)
fixed to some lower value for some iterations.

Let's create a helper function to initialize the models we use in the
notebook.

```{code-cell}
def initialize(self, noise_factor=0.01, linear_factor=1):
    """Helper function for deep model initialization."""
    self.obslayer.likelihood.variance = self.Y.var()*noise_factor
    for layer in self.layers:
        if type(layer.X) is GPy.core.parameterization.variational.NormalPosterior:
            if layer.kern.ARD:
                var = layer.X.mean.var(0)
            else:
                var = layer.X.mean.var()
        else:
            if layer.kern.ARD:
                var = layer.X.var(0)
            else:
                var = layer.X.var()

        # Average 0.5 upcrossings in four standard deviations. 
        layer.kern.lengthscale = linear_factor*np.sqrt(layer.kern.input_dim)*2*4*np.sqrt(var)/(2*np.pi)
# Bind the new method to the Deep GP object.
deepgp.DeepGP.initialize=initialize
```

```{code-cell}
# Call the initalization
m.initialize()
```

Now optimize the model. The first stage of optimization is working on
variational parameters and lengthscales only.

```{code-cell}
m.optimize(messages=False,max_iters=100)
```

Now we remove the constraints on the scale of the covariance functions
associated with each GP and optimize again.

```{code-cell}
for layer in m.layers:
    pass #layer.kern.variance.constrain_positive(warning=False)
m.obslayer.kern.variance.constrain_positive(warning=False)
m.optimize(messages=False,max_iters=100)
```

Finally, we allow the noise variance to change and optimize for a large
number of iterations.

```{code-cell}
for layer in m.layers:
    layer.likelihood.variance.constrain_positive(warning=False)
m.optimize(messages=True,max_iters=10000)
```

For our optimization process we define a new function.

```{code-cell}
def staged_optimize(self, iters=(1000,1000,10000), messages=(False, False, True)):
    """Optimize with parameters constrained and then with parameters released"""
    for layer in self.layers:
        # Fix the scale of each of the covariance functions.
        layer.kern.variance.fix(warning=False)
        layer.kern.lengthscale.fix(warning=False)

        # Fix the variance of the noise in each layer.
        layer.likelihood.variance.fix(warning=False)

    self.optimize(messages=messages[0],max_iters=iters[0])
    
    for layer in self.layers:
        layer.kern.lengthscale.constrain_positive(warning=False)
    self.obslayer.kern.variance.constrain_positive(warning=False)


    self.optimize(messages=messages[1],max_iters=iters[1])

    for layer in self.layers:
        layer.kern.variance.constrain_positive(warning=False)
        layer.likelihood.variance.constrain_positive(warning=False)
    self.optimize(messages=messages[2],max_iters=iters[2])
    
# Bind the new method to the Deep GP object.
deepgp.DeepGP.staged_optimize=staged_optimize
```

```{code-cell}
m.staged_optimize(messages=(True,True,True))
```

### Plot the prediction

The prediction of the deep GP can be extracted in a similar way to the
normal GP. Although, in this case, it is an approximation to the true
distribution, because the true distribution is not Gaussian.

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
plot.model_output(m, scale=scale, offset=offset, ax=ax, xlabel='year', ylabel='pace min/km', 
          fontsize=20, portion=0.2)
ax.set_xlim(xlim)

ax.set_ylim(ylim)
mlai.write_figure(figure=fig, filename='../slides/diagrams/deepgp/olympic-marathon-deep-gp.svg', 
                transparent=True, frameon=True)
```

### Olympic Marathon Data Deep GP

<img src="../slides/diagrams/deepgp/olympic-marathon-deep-gp.svg" align="">

```{code-cell}
def posterior_sample(self, X, **kwargs):
    """Give a sample from the posterior of the deep GP."""
    Z = X
    for i, layer in enumerate(reversed(self.layers)):
        Z = layer.posterior_samples(Z, size=1, **kwargs)[:, :, 0]
 
    return Z
deepgp.DeepGP.posterior_sample = posterior_sample
```

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
plot.model_sample(m, scale=scale, offset=offset, samps=10, ax=ax, 
                  xlabel='year', ylabel='pace min/km', portion = 0.225)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
mlai.write_figure(figure=fig, filename='../slides/diagrams/deepgp/olympic-marathon-deep-gp-samples.svg', 
                  transparent=True, frameon=True)
```

### Olympic Marathon Data Deep GP {#olympic-marathon-data-deep-gp-1 data-transition="None"}

<img src="../slides/diagrams/deepgp/olympic-marathon-deep-gp-samples.svg" align="">

### Fitted GP for each layer

Now we explore the GPs the model has used to fit each layer. First of
all, we look at the hidden layer.

```{code-cell}
def visualize(self, scale=1.0, offset=0.0, xlabel='input', ylabel='output', 
              xlim=None, ylim=None, fontsize=20, portion=0.2,dataset=None, 
              diagrams='../diagrams'):
    """Visualize the layers in a deep GP with one-d input and output."""
    depth = len(self.layers)
    if dataset is None:
        fname = 'deep-gp-layer'
    else:
        fname = dataset + '-deep-gp-layer'
    filename = os.path.join(diagrams, fname)
    last_name = xlabel
    last_x = self.X
    for i, layer in enumerate(reversed(self.layers)):
        if i>0:
            plt.plot(last_x, layer.X.mean, 'r.',markersize=10)
            last_x=layer.X.mean
            ax=plt.gca()
            name = 'layer ' + str(i)
            plt.xlabel(last_name, fontsize=fontsize)
            plt.ylabel(name, fontsize=fontsize)
            last_name=name
            mlai.write_figure(filename=filename + '-' + str(i-1) + '.svg', 
                              transparent=True, frameon=True)
            
        if i==0 and xlim is not None:
            xt = plot.pred_range(np.array(xlim), portion=0.0)
        elif i>0:
            xt = plot.pred_range(np.array(next_lim), portion=0.0)
        else:
            xt = plot.pred_range(last_x, portion=portion)
        yt_mean, yt_var = layer.predict(xt)
        if layer==self.obslayer:
            yt_mean = yt_mean*scale + offset
            yt_var *= scale*scale
        yt_sd = np.sqrt(yt_var)
        gpplot(xt,yt_mean,yt_mean-2*yt_sd,yt_mean+2*yt_sd)
        ax = plt.gca()
        if i>0:
            ax.set_xlim(next_lim)
        elif xlim is not None:
            ax.set_xlim(xlim)
        next_lim = plt.gca().get_ylim()
        
    plt.plot(last_x, self.Y*scale + offset, 'r.',markersize=10)
    plt.xlabel(last_name, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    mlai.write_figure(filename=filename + '-' + str(i) + '.svg', 
                      transparent=True, frameon=True)

    if ylim is not None:
        ax=plt.gca()
        ax.set_ylim(ylim)

# Bind the new method to the Deep GP object.
deepgp.DeepGP.visualize=visualize
```

```{code-cell}
m.visualize(scale=scale, offset=offset, xlabel='year',
            ylabel='pace min/km',xlim=xlim, ylim=ylim,
            dataset='olympic-marathon',
            diagrams='../slides/diagrams/deepgp')
```

```{code-cell}
import pods
pods.notebook.display_plots('olympic-marathon-deep-gp-layer-{sample:0>1}.svg', 
                            '../slides/diagrams/deepgp', sample=(0,1))
```

```{code-cell}
def scale_data(x, portion):     
    scale = (x.max()-x.min())/(1-2*portion)
    offset = x.min() - portion*scale
    return (x-offset)/scale, scale, offset

def visualize_pinball(self, ax=None, scale=1.0, offset=0.0, xlabel='input', ylabel='output', 
                  xlim=None, ylim=None, fontsize=20, portion=0.2, points=50, vertical=True):
    """Visualize the layers in a deep GP with one-d input and output."""

    if ax is None:
        fig, ax = plt.subplots(figsize=plot.big_wide_figsize)

    depth = len(self.layers)

    last_name = xlabel
    last_x = self.X

    # Recover input and output scales from output plot
    plot_model_output(self, scale=scale, offset=offset, ax=ax, 
                      xlabel=xlabel, ylabel=ylabel, 
                      fontsize=fontsize, portion=portion)
    xlim=ax.get_xlim()
    xticks=ax.get_xticks()
    xtick_labels=ax.get_xticklabels().copy()
    ylim=ax.get_ylim()
    yticks=ax.get_yticks()
    ytick_labels=ax.get_yticklabels().copy()

    # Clear axes and start again
    ax.cla()
    if vertical:
        ax.set_xlim((0, 1))
        ax.invert_yaxis()

        ax.set_ylim((depth, 0))
    else:
        ax.set_ylim((0, 1))
        ax.set_xlim((0, depth))
        
    ax.set_axis_off()#frame_on(False)


    def pinball(x, y, y_std, color_scale=None, 
                layer=0, depth=1, ax=None, 
                alpha=1.0, portion=0.0, vertical=True):  

        scaledx, xscale, xoffset = scale_data(x, portion=portion)
        scaledy, yscale, yoffset = scale_data(y, portion=portion)
        y_std /= yscale

        # Check whether data is anti-correlated on output
        if np.dot((scaledx-0.5).T, (scaledy-0.5))<0:
            scaledy=1-scaledy
            flip=-1
        else:
            flip=1

        if color_scale is not None:
            color_scale, _, _=scale_data(color_scale, portion=0)
        scaledy = (1-alpha)*scaledx + alpha*scaledy

        def color_value(x, cmap=None, width=None, centers=None):
            """Return color as a function of position along x axis"""
            if cmap is None:
                cmap = np.asarray([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
            ncenters = cmap.shape[0]
            if centers is None:
                centers = np.linspace(0+0.5/ncenters, 1-0.5/ncenters, ncenters)
            if width is None:
                width = 0.25/ncenters
            
            r = (x-centers)/width
            weights = np.exp(-0.5*r*r).flatten()
            weights /=weights.sum()
            weights = weights[:, np.newaxis]
            return np.dot(cmap.T, weights).flatten()


        for i in range(x.shape[0]):
            if color_scale is not None:
                color = color_value(color_scale[i])
            else:
                color=(1, 0, 0)
            x_plot = np.asarray((scaledx[i], scaledy[i])).flatten()
            y_plot = np.asarray((layer, layer+alpha)).flatten()
            if vertical:
                ax.plot(x_plot, y_plot, color=color, alpha=0.5, linewidth=3)
                ax.plot(x_plot, y_plot, color='k', alpha=0.5, linewidth=0.5)
            else:
                ax.plot(y_plot, x_plot, color=color, alpha=0.5, linewidth=3)
                ax.plot(y_plot, x_plot, color='k', alpha=0.5, linewidth=0.5)

            # Plot error bars that increase as sqrt of distance from start.
            std_points = 50
            stdy = np.linspace(0, alpha,std_points)
            stdx = np.sqrt(stdy)*y_std[i]
            stdy += layer
            mean_vals = np.linspace(scaledx[i], scaledy[i], std_points)
            upper = mean_vals+stdx 
            lower = mean_vals-stdx 
            fillcolor=color
            x_errorbars=np.hstack((upper,lower[::-1]))
            y_errorbars=np.hstack((stdy,stdy[::-1]))
            if vertical:
                ax.fill(x_errorbars,y_errorbars,
                        color=fillcolor, alpha=0.1)
                ax.plot(scaledy[i], layer+alpha, '.',markersize=10, color=color, alpha=0.5)
            else:
                ax.fill(y_errorbars,x_errorbars,
                        color=fillcolor, alpha=0.1)
                ax.plot(layer+alpha, scaledy[i], '.',markersize=10, color=color, alpha=0.5)
            # Marker to show end point
        return flip


    # Whether final axis is flipped
    flip = 1
    first_x=last_x
    for i, layer in enumerate(reversed(self.layers)):     
        if i==0:
            xt = plot.pred_range(last_x, portion=portion, points=points)
            color_scale=xt
        yt_mean, yt_var = layer.predict(xt)
        if layer==self.obslayer:
            yt_mean = yt_mean*scale + offset
            yt_var *= scale*scale
        yt_sd = np.sqrt(yt_var)
        flip = flip*pinball(xt,yt_mean,yt_sd,color_scale,portion=portion, 
                            layer=i, ax=ax, depth=depth,vertical=vertical)#yt_mean-2*yt_sd,yt_mean+2*yt_sd)
        xt = yt_mean
    # Make room for axis labels
    if vertical:
        ax.set_ylim((2.1, -0.1))
        ax.set_xlim((-0.02, 1.02))
        ax.set_yticks(range(depth,0,-1))
    else:
        ax.set_xlim((-0.1, 2.1))
        ax.set_ylim((-0.02, 1.02))
        ax.set_xticks(range(0, depth))
        
    def draw_axis(ax, scale=1.0, offset=0.0, level=0.0, flip=1, 
                  label=None,up=False, nticks=10, ticklength=0.05,
                  vertical=True,
                 fontsize=20):
        def clean_gap(gap):
            nsf = np.log10(gap)
            if nsf>0:
                nsf = np.ceil(nsf)
            else:
                nsf = np.floor(nsf)
            lower_gaps = np.asarray([0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 
                                     0.1, 0.25, 0.5, 
                                     1, 1.5, 2, 2.5, 3, 4, 5, 10, 25, 50, 100])
            upper_gaps = np.asarray([1, 2, 3, 4, 5, 10])
            if nsf >2 or nsf<-2:
                d = np.abs(gap-upper_gaps*10**nsf)
                ind = np.argmin(d)
                return upper_gaps[ind]*10**nsf
            else:
                d = np.abs(gap-lower_gaps)
                ind = np.argmin(d)
                return lower_gaps[ind]
            
        tickgap = clean_gap(scale/(nticks-1))
        nticks = int(scale/tickgap) + 1
        tickstart = np.round(offset/tickgap)*tickgap
        ticklabels = np.asarray(range(0, nticks))*tickgap + tickstart
        ticks = (ticklabels-offset)/scale
        axargs = {'color':'k', 'linewidth':1}
        
        if not up:
            ticklength=-ticklength
        tickspot = np.linspace(0, 1, nticks)
        if flip < 0:
            ticks = 1-ticks
        for tick, ticklabel in zip(ticks, ticklabels):
            if vertical:
                ax.plot([tick, tick], [level, level-ticklength], **axargs)
                ax.text(tick, level-ticklength*2, ticklabel, horizontalalignment='center', 
                        fontsize=fontsize/2)
                ax.text(0.5, level-5*ticklength, label, horizontalalignment='center', fontsize=fontsize)
            else:
                ax.plot([level, level-ticklength], [tick, tick],  **axargs)
                ax.text(level-ticklength*2, tick, ticklabel, horizontalalignment='center', 
                        fontsize=fontsize/2)
                ax.text(level-5*ticklength, 0.5, label, horizontalalignment='center', fontsize=fontsize)
        
        if vertical:
            xlim = list(ax.get_xlim())
            if ticks.min()<xlim[0]:
                xlim[0] = ticks.min()
            if ticks.max()>xlim[1]:
                xlim[1] = ticks.max()
            ax.set_xlim(xlim)
            
            ax.plot([ticks.min(), ticks.max()], [level, level], **axargs)
        else:
            ylim = list(ax.get_ylim())
            if ticks.min()<ylim[0]:
                ylim[0] = ticks.min()
            if ticks.max()>ylim[1]:
                ylim[1] = ticks.max()
            ax.set_ylim(ylim)
            ax.plot([level, level], [ticks.min(), ticks.max()], **axargs)


    _, xscale, xoffset = scale_data(first_x, portion=portion)
    _, yscale, yoffset = scale_data(yt_mean, portion=portion)
    draw_axis(ax=ax, scale=xscale, offset=xoffset, level=0.0, label=xlabel, 
              up=True, vertical=vertical)
    draw_axis(ax=ax, scale=yscale, offset=yoffset, 
              flip=flip, level=depth, label=ylabel, up=False, vertical=vertical)
    
    #for txt in xticklabels:
    #    txt.set
# Bind the new method to the Deep GP object.
deepgp.DeepGP.visualize_pinball=visualize_pinball
```

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
m.visualize_pinball(ax=ax, scale=scale, offset=offset, points=30, portion=0.1,
                    xlabel='year', ylabel='pace km/min', vertical=True)
mlai.write_figure(figure=fig, filename='../slides/diagrams/deepgp/olympic-marathon-deep-gp-pinball.svg', 
                  transparent=True, frameon=True)
```

### Olympic Marathon Pinball Plot

<img src="../slides/diagrams/deepgp/olympic-marathon-deep-gp-pinball.svg" align="">

The pinball plot shows the flow of any input ball through the deep
Gaussian process. In a pinball plot a series of vertical parallel lines
would indicate a purely linear function. For the olypmic marathon data
we can see the first layer begins to shift from input towards the right.
Note it also does so with some uncertainty (indicated by the shaded
backgrounds). The second layer has less uncertainty, but bunches the
inputs more strongly to the right. This input layer of uncertainty,
followed by a layer that pushes inputs to the right is what gives the
heteroschedastic noise.

### Step Function

Next we consider a simple step function data set.

```{code-cell}
num_low=25
num_high=25
gap = -.1
noise=0.0001
x = np.vstack((np.linspace(-1, -gap/2.0, num_low)[:, np.newaxis],
              np.linspace(gap/2.0, 1, num_high)[:, np.newaxis]))
y = np.vstack((np.zeros((num_low, 1)), np.ones((num_high,1))))
scale = np.sqrt(y.var())
offset = y.mean()
yhat = (y-offset)/scale
```

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
_ = ax.plot(x, y, 'r.',markersize=10)
_ = ax.set_xlabel('$x$', fontsize=20)
_ = ax.set_ylabel('$y$', fontsize=20)
xlim = (-2, 2)
ylim = (-0.6, 1.6)
ax.set_ylim(ylim)
ax.set_xlim(xlim)
mlai.write_figure(figure=fig, filename='../../slides/diagrams/datasets/step-function.svg', 
            transparent=True, frameon=True)
```

### Step Function Data {#step-function-data data-transition="None"}

<img src="../slides/diagrams/datasets/step-function.svg" align="">

```{code-cell}
m_full = GPy.models.GPRegression(x,yhat)
_ = m_full.optimize() # Optimize parameters of covariance function
```

```{code-cell}
fig, ax=plt.subplots(figsize=plot.big_wide_figsize)
plot_model_output(m_full, scale=scale, offset=offset, ax=ax, fontsize=20, portion=0.5)
ax.set_ylim(ylim)
ax.set_xlim(xlim)

mlai.write_figure(figure=fig,filename='../../slides/diagrams/gp/step-function-gp.svg', 
            transparent=True, frameon=True)
```

### Step Function Data GP {#step-function-data-gp data-transition="None"}

<img src="../slides/diagrams/gp/step-function-gp.svg" align="">

```{code-cell}
layers = [y.shape[1], 1, 1, 1,x.shape[1]]
inits = ['PCA']*(len(layers)-1)
kernels = []
for i in layers[1:]:
    kernels += [GPy.kern.RBF(i)]
m = deepgp.DeepGP(layers,Y=yhat, X=x, 
                  inits=inits, 
                  kernels=kernels, # the kernels for each layer
                  num_inducing=20, back_constraint=False)
```

```{code-cell}
m.initialize()
m.staged_optimize()
```

```{code-cell}
fig, ax=plt.subplots(figsize=plot.big_wide_figsize)
plot_model_output(m, scale=scale, offset=offset, ax=ax, fontsize=20, portion=0.5)
ax.set_ylim(ylim)
ax.set_xlim(xlim)
mlai.write_figure(filename='../../slides/diagrams/deepgp/step-function-deep-gp.svg', 
            transparent=True, frameon=True)
```

### Step Function Data Deep GP {#step-function-data-deep-gp data-transition="None"}

<img src="../slides/diagrams/deepgp/step-function-deep-gp.svg" align="">

```{code-cell}
fig, ax=plt.subplots(figsize=plot.big_wide_figsize)
plot_model_sample(m, scale=scale, offset=offset, samps=10, ax=ax, portion = 0.5)
ax.set_ylim(ylim)
ax.set_xlim(xlim)
mlai.write_figure(figure=fig, filename='../../slides/diagrams/deepgp/step-function-deep-gp-samples.svg', 
                  transparent=True, frameon=True)
```

### Step Function Data Deep GP {#step-function-data-deep-gp-1 data-transition="None"}

<img src="../slides/diagrams/deepgp/step-function-deep-gp-samples.svg" align="">

```{code-cell}
m.visualize(offset=offset, scale=scale, xlim=xlim, ylim=ylim,
            dataset='step-function',
            diagrams='../../slides/diagrams/deepgp')
```

### Step Function Data Latent 1 {#step-function-data-latent-1 data-transition="None"}

<img src="../slides/diagrams/deepgp/step-function-deep-gp-layer-0.svg" align="">

### Step Function Data Latent 2 {#step-function-data-latent-2 data-transition="None"}

<img src="../slides/diagrams/deepgp/step-function-deep-gp-layer-1.svg" align="">

### Step Function Data Latent 3 {#step-function-data-latent-3 data-transition="None"}

<img src="../slides/diagrams/deepgp/step-function-deep-gp-layer-2.svg" align="">

### Step Function Data Latent 4 {#step-function-data-latent-4 data-transition="None"}

<img src="../slides/diagrams/deepgp/step-function-deep-gp-layer-3.svg" align="">

```{code-cell}
fig, ax=plt.subplots(figsize=plot.big_wide_figsize)
m.visualize_pinball(offset=offset, ax=ax, scale=scale, xlim=xlim, ylim=ylim, portion=0.1, points=50)
mlai.write_figure(figure=fig, filename='../../slides/diagrams/deepgp/step-function-deep-gp-pinball.svg', 
                  transparent=True, frameon=True, ax=ax)
```

### Step Function Pinball Plot {#step-function-pinball-plot data-transition="None"}

<img src="../slides/diagrams/deepgp/step-function-deep-gp-pinball.svg" align="">

```{code-cell}
import pods
data = pods.datasets.mcycle()
x = data['X']
y = data['Y']
scale=np.sqrt(y.var())
offset=y.mean()
yhat = (y - offset)/scale
```

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
_ = ax.plot(x, y, 'r.',markersize=10)
_ = ax.set_xlabel('time', fontsize=20)
_ = ax.set_ylabel('acceleration', fontsize=20)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
mlai.write_figure(filename='../../slides/diagrams/datasets/motorcycle-helmet.svg', 
            transparent=True, frameon=True)
```

### Motorcycle Helmet Data {#motorcycle-helmet-data data-transition="None"}

<img src="../slides/diagrams/datasets/motorcycle-helmet.svg" align="">

```{code-cell}
m_full = GPy.models.GPRegression(x,yhat)
_ = m_full.optimize() # Optimize parameters of covariance function
```

```{code-cell}
fig, ax=plt.subplots(figsize=plot.big_wide_figsize)
plot_model_output(m_full, scale=scale, offset=offset, ax=ax, xlabel='time', ylabel='acceleration/$g$', fontsize=20, portion=0.5)
xlim=(-20,80)
ylim=(-180,120)
ax.set_ylim(ylim)
ax.set_xlim(xlim)
mlai.write_figure(figure=fig,filename='../../slides/diagrams/gp/motorcycle-helmet-gp.svg', 
            transparent=True, frameon=True)
```

### Motorcycle Helmet Data GP {#motorcycle-helmet-data-gp data-transition="None"}

<img src="../slides/diagrams/gp/motorcycle-helmet-gp.svg" align="">

```{code-cell}
layers = [y.shape[1], 1, x.shape[1]]
inits = ['PCA']*(len(layers)-1)
kernels = []
for i in layers[1:]:
    kernels += [GPy.kern.RBF(i)]
m = deepgp.DeepGP(layers,Y=yhat, X=x, 
                  inits=inits, 
                  kernels=kernels, # the kernels for each layer
                  num_inducing=20, back_constraint=False)



m.initialize()
```

```{code-cell}
m.staged_optimize(iters=(1000,1000,10000), messages=(True, True, True))
```

```{code-cell}
fig, ax=plt.subplots(figsize=plot.big_wide_figsize)
plot_model_output(m, scale=scale, offset=offset, ax=ax, xlabel='time', ylabel='acceleration/$g$', fontsize=20, portion=0.5)
ax.set_ylim(ylim)
ax.set_xlim(xlim)
mlai.write_figure(filename='../../slides/diagrams/deepgp/motorcycle-helmet-deep-gp.svg', 
            transparent=True, frameon=True)
```

### Motorcycle Helmet Data Deep GP {#motorcycle-helmet-data-deep-gp data-transition="None"}

<img src="../slides/diagrams/deepgp/motorcycle-helmet-deep-gp.svg" align="">

```{code-cell}
fig, ax=plt.subplots(figsize=plot.big_wide_figsize)
plot_model_sample(m, scale=scale, offset=offset, samps=10, ax=ax, xlabel='time', ylabel='acceleration/$g$', portion = 0.5)
ax.set_ylim(ylim)
ax.set_xlim(xlim)

mlai.write_figure(figure=fig, filename='../../slides/diagrams/deepgp/motorcycle-helmet-deep-gp-samples.svg', 
                  transparent=True, frameon=True)
```

### Motorcycle Helmet Data Deep GP {#motorcycle-helmet-data-deep-gp-1 data-transition="None"}

<img src="../slides/diagrams/deepgp/motorcycle-helmet-deep-gp-samples.svg" align="">

```{code-cell}
m.visualize(xlim=xlim, ylim=ylim, scale=scale,offset=offset, 
            xlabel="time", ylabel="acceleration/$g$", portion=0.5,
            dataset='motorcycle-helmet',
            diagrams='../../slides/diagrams/deepgp')
```

### Motorcycle Helmet Data Latent 1 {#motorcycle-helmet-data-latent-1 data-transition="None"}

<img src="../slides/diagrams/deepgp/motorcycle-helmet-deep-gp-layer-0.svg" align="">

### Motorcycle Helmet Data Latent 2 {#motorcycle-helmet-data-latent-2 data-transition="None"}

<img src="../slides/diagrams/deepgp/motorcycle-helmet-deep-gp-layer-1.svg" align="">

```{code-cell}
fig, ax=plt.subplots(figsize=plot.big_wide_figsize)
m.visualize_pinball(ax=ax, xlabel='time', ylabel='acceleration/g', 
                    points=50, scale=scale, offset=offset, portion=0.1)
mlai.write_figure(figure=fig, filename='../../slides/diagrams/deepgp/motorcycle-helmet-deep-gp-pinball.svg', 
                  transparent=True, frameon=True)
```

### Motorcycle Helmet Pinball Plot {#motorcycle-helmet-pinball-plot data-transition="None"}

<img src="../slides/diagrams/deepgp/motorcycle-helmet-deep-gp-pinball.svg" align="">

### Robot Wireless Data

The robot wireless data is taken from an experiment run by Brian Ferris
at University of Washington. It consists of the measurements of WiFi
access point signal strengths as Brian walked in a loop.

```{code-cell}
data=pods.datasets.robot_wireless()

x = np.linspace(0,1,215)[:, np.newaxis]
y = data['Y']
offset = y.mean()
scale = np.sqrt(y.var())
yhat = (y-offset)/scale
```

The ground truth is recorded in the data, the actual loop is given in
the plot below.

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_figsize)
plt.plot(data['X'][:, 1], data['X'][:, 2], 'r.', markersize=5)
ax.set_xlabel('x position', fontsize=20)
ax.set_ylabel('y position', fontsize=20)
mlai.write_figure(figure=fig, filename='../../slides/diagrams/datasets/robot-wireless-ground-truth.svg', transparent=True, frameon=True)
```

### Robot Wireless Ground Truth {#robot-wireless-ground-truth data-transition="None"}

<img src="../slides/diagrams/datasets/robot-wireless-ground-truth.svg" align="">

We will ignore this ground truth in making our predictions, but see if
the model can recover something similar in one of the latent layers.

```{code-cell}
output_dim=1
xlim = (-0.3, 1.3)
fig, ax = plt.subplots(figsize=plot.big_wide_figsize)
_ = ax.plot(x.flatten(), y[:, output_dim], 
            'r.', markersize=5)

ax.set_xlabel('time', fontsize=20)
ax.set_ylabel('signal strength', fontsize=20)
xlim = (-0.2, 1.2)
ylim = (-0.6, 2.0)
ax.set_xlim(xlim)
ax.set_ylim(ylim)

mlai.write_figure(figure=fig, filename='../../slides/diagrams/datasets/robot-wireless-dim-' + str(output_dim) + '.svg', 
            transparent=True, frameon=True)
```

### Robot WiFi Data {#robot-wifi-data data-transition="None"}

<img src="../slides/diagrams/datasets/robot-wireless-dim-1.svg" align="">

Perform a Gaussian process fit on the data using GPy.

```{code-cell}
m_full = GPy.models.GPRegression(x,yhat)
_ = m_full.optimize() # Optimize parameters of covariance function
```

```{code-cell}
fig, ax=plt.subplots(figsize=plot.big_wide_figsize)
plot_model_output(m_full, output_dim=output_dim, scale=scale, offset=offset, ax=ax, 
                  xlabel='time', ylabel='signal strength', fontsize=20, portion=0.5)
ax.set_ylim(ylim)
ax.set_xlim(xlim)
mlai.write_figure(filename='../../slides/diagrams/gp/robot-wireless-gp-dim-' + str(output_dim)+ '.svg', 
            transparent=True, frameon=True)
```

### Robot WiFi Data GP {#robot-wifi-data-gp data-transition="None"}

<img src="../slides/diagrams/gp/robot-wireless-gp-dim-1.svg" align="">

```{code-cell}
layers = [y.shape[1], 10, 5, 2, 2, x.shape[1]]
inits = ['PCA']*(len(layers)-1)
kernels = []
for i in layers[1:]:
    kernels += [GPy.kern.RBF(i, ARD=True)]
```

```{code-cell}
m = deepgp.DeepGP(layers,Y=y, X=x, inits=inits, 
                  kernels=kernels,
                  num_inducing=50, back_constraint=False)
m.initialize()
```

```{code-cell}
m.staged_optimize(messages=(True,True,True))
```

```{code-cell}
fig, ax=plt.subplots(figsize=plot.big_wide_figsize)
plot_model_output(m, output_dim=output_dim, scale=scale, offset=offset, ax=ax, 
                  xlabel='time', ylabel='signal strength', fontsize=20, portion=0.5)
ax.set_ylim(ylim)
ax.set_xlim(xlim)
mlai.write_figure(figure=fig, filename='../../slides/diagrams/deepgp/robot-wireless-deep-gp-dim-' + str(output_dim)+ '.svg', 
                  transparent=True, frameon=True)
```

### Robot WiFi Data Deep GP {#robot-wifi-data-deep-gp data-transition="None"}

<img src="../slides/diagrams/deepgp/robot-wireless-deep-gp-dim-1.svg" align="">

```{code-cell}
fig, ax=plt.subplots(figsize=plot.big_wide_figsize)
plot_model_sample(m, output_dim=output_dim, scale=scale, offset=offset, samps=10, ax=ax,
                  xlabel='time', ylabel='signal strength', fontsize=20, portion=0.5)
ax.set_ylim(ylim)
ax.set_xlim(xlim)
mlai.write_figure(figure=fig, filename='../../slides/diagrams/deepgp/robot-wireless-deep-gp-samples-dim-' + str(output_dim)+ '.svg', 
                  transparent=True, frameon=True)
```

### Robot WiFi Data Deep GP {#robot-wifi-data-deep-gp-1 data-transition="None"}

<img src="../slides/diagrams/deepgp/robot-wireless-deep-gp-samples-dim-1.svg" align="">

### Robot WiFi Data Latent Space {#robot-wifi-data-latent-space data-transition="None"}

<img src="../slides/diagrams/deepgp/robot-wireless-ground-truth.svg" align="">

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_figsize)
ax.plot(m.layers[-2].latent_space.mean[:, 0], 
        m.layers[-2].latent_space.mean[:, 1], 
        'r.-', markersize=5)

ax.set_xlabel('latent dimension 1', fontsize=20)
ax.set_ylabel('latent dimension 2', fontsize=20)

mlai.write_figure(figure=fig, filename='../../slides/diagrams/deepgp/robot-wireless-latent-space.svg', 
            transparent=True, frameon=True)
```

### Robot WiFi Data Latent Space {#robot-wifi-data-latent-space-1 data-transition="None"}

<img src="../slides/diagrams/deepgp/robot-wireless-latent-space.svg" align="">

### Motion Capture {#motion-capture data-transition="none"}

-   ‘High five’ data.

-   Model learns structure between two interacting subjects.

### Shared LVM {#shared-lvm data-transition="none"}

<img src="../slides/diagrams/shared.svg" align="">

###  {#section-3 data-transition="none"}

<img class="negate" src="../slides/diagrams/deep-gp-high-five2.png" width="100%" align="" style="background:none; border:none; box-shadow:none;">

\credit{Zhenwen Dai and Neil D. Lawrence}

This notebook explores the deep Gaussian processes' capacity to perform
unsupervised learning.

We will look at a sub-sample of the MNIST digit data set.

This notebook depends on GPy and PyDeepGP. These libraries can be
installed via pip:

```{code-cell}
pip install GPy
pip install git+https://github.com/SheffieldML/PyDeepGP.git
```

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from IPython.display import display

import deepgp
import GPy

from gp_tutorial import ax_default, meanplot, gpplot
import mlai
import teaching_plots as plot
```

First load in the MNIST data set from scikit learn. This can take a
little while because it's large to download.

```{code-cell}
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
```

Sub-sample the dataset to make the training faster.

```{code-cell}
np.random.seed(0)
digits = [0,1,2,3,4]
N_per_digit = 100
Y = []
labels = []
for d in digits:
    imgs = mnist['data'][mnist['target']==d]
    Y.append(imgs[np.random.permutation(imgs.shape[0])][:N_per_digit])
    labels.append(np.ones(N_per_digit)*d)
Y = np.vstack(Y).astype(np.float64)
labels = np.hstack(labels)
Y /= 255.
```

### Fit a Deep GP

We're going to fit a Deep Gaussian process model to the MNIST data with
two hidden layers. Each of the two Gaussian processes (one from the
first hidden layer to the second, one from the second hidden layer to
the data) has an exponentiated quadratic covariance.

```{code-cell}
num_latent = 2
num_hidden_2 = 5
m = deepgp.DeepGP([Y.shape[1],num_hidden_2,num_latent],
                  Y,
                  kernels=[GPy.kern.RBF(num_hidden_2,ARD=True), 
                           GPy.kern.RBF(num_latent,ARD=False)], 
                  num_inducing=50, back_constraint=False, 
                  encoder_dims=[[200],[200]])
```

### Initialization

Just like deep neural networks, there are some tricks to intitializing
these models. The tricks we use here include some early training of the
model with model parameters constrained. This gives the variational
inducing parameters some scope to tighten the bound for the case where
the noise variance is small and the variances of the Gaussian processes
are around 1.

```{code-cell}
m.obslayer.likelihood.variance[:] = Y.var()*0.01
for layer in m.layers:
    layer.kern.variance.fix(warning=False)
    layer.likelihood.variance.fix(warning=False)
```

We now we optimize for a hundred iterations with the constrained model.

```{code-cell}
m.optimize(messages=False,max_iters=100)
```

Now we remove the fixed constraint on the kernel variance parameters,
but keep the noise output constrained, and run for a further 100
iterations.

```{code-cell}
for layer in m.layers:
    layer.kern.variance.constrain_positive(warning=False)
m.optimize(messages=False,max_iters=100)
```

Finally we unconstrain the layer likelihoods and allow the full model to
be trained for 1000 iterations.

```{code-cell}
for layer in m.layers:
    layer.likelihood.variance.constrain_positive(warning=False)
m.optimize(messages=True,max_iters=10000)
```

### Visualize the latent space of the top layer

Now the model is trained, let's plot the mean of the posterior
distributions in the top latent layer of the model.

```{code-cell}
rc("font", **{'family':'sans-serif','sans-serif':['Helvetica'],'size':20})
fig, ax = plt.subplots(figsize=plot.big_figsize)
for d in digits:
    ax.plot(m.layer_1.X.mean[labels==d,0],m.layer_1.X.mean[labels==d,1],'.',label=str(d))
_ = plt.legend()
mlai.write_figure(figure=fig, filename="../../slides/diagrams/deepgp/usps-digits-latent.svg", transparent=True)
```

<img src="../slides/diagrams/usps-digits-latent.svg" align="">

### Visualize the latent space of the intermediate layer

We can also visualize dimensions of the intermediate layer. First the
lengthscale of those dimensions is given by

```{code-cell}
m.obslayer.kern.lengthscale
```

```{code-cell}
fig, ax = plt.subplots(figsize=plot.big_figsize)
for i in range(5):
    for j in range(i):
        dims=[i, j]
        ax.cla()
        for d in digits:
            ax.plot(m.obslayer.X.mean[labels==d,dims[0]],
                 m.obslayer.X.mean[labels==d,dims[1]],
                 '.', label=str(d))
        plt.legend()
        plt.xlabel('dimension ' + str(dims[0]))
        plt.ylabel('dimension ' + str(dims[1]))
        mlai.write_figure(figure=fig, filename="../../slides/diagrams/deepgp/usps-digits-hidden-" + str(dims[0]) + '-' + str(dims[1]) + '.svg', transparent=True)
```

###  {#section-4 data-transition="none"}

<img src="../slides/diagrams/usps-digits-hidden-1-0.svg" align="">

###  {#section-5 data-transition="none"}

<img src="../slides/diagrams/usps-digits-hidden-2-0.svg" align="">

###  {#section-6 data-transition="none"}

<img src="../slides/diagrams/usps-digits-hidden-3-0.svg" align="">

###  {#section-7 data-transition="none"}

<img src="../slides/diagrams/usps-digits-hidden-4-0.svg" align="">

### Generate From Model

Now we can take a look at a sample from the model, by drawing a Gaussian
random sample in the latent space and propagating it through the model.

```{code-cell}

rows = 10
cols = 20
t=np.linspace(-1, 1, rows*cols)[:, None]
kern = GPy.kern.RBF(1,lengthscale=0.05)
cov = kern.K(t, t)
x = np.random.multivariate_normal(np.zeros(rows*cols), cov, num_latent).T
```

```{code-cell}
yt = m.predict(x)
fig, axs = plt.subplots(rows,cols,figsize=(10,6))
for i in range(rows):
    for j in range(cols):
        #v = np.random.normal(loc=yt[0][i*cols+j, :], scale=np.sqrt(yt[1][i*cols+j, :]))
        v = yt[0][i*cols+j, :]
        axs[i,j].imshow(v.reshape(28,28), 
                        cmap='gray', interpolation='none',
                        aspect='equal')
        axs[i,j].set_axis_off()
mlai.write_figure(figure=fig, filename="../../slides/diagrams/deepgp/digit-samples-deep-gp.svg", transparent=True)
```

###  {#section-8 data-transition="none"}

<img src="../slides/diagrams/digit-samples-deep-gp.svg" align="">

### Deep Health {#deep-health data-transition="None"}

<img src="../slides/diagrams/deep-health.svg" align="center">

### At this Year's NIPS

-   *Gaussian process based nonlinear latent structure discovery in
    multivariate spike train data* @Anqi:gpspike2017
-   *Doubly Stochastic Variational Inference for Deep Gaussian
    Processes* @Salimbeni:doubly2017
-   *Deep Multi-task Gaussian Processes for Survival Analysis with
    Competing Risks* @Alaa:deep2017
-   *Counterfactual Gaussian Processes for Reliable Decision-making and
    What-if Reasoning* @Schulam:counterfactual17

### Some Other Works

-   *Deep Survival Analysis* @Ranganath-survival16
-   *Recurrent Gaussian Processes* @Mattos:recurrent15
-   *Gaussian Process Based Approaches for Survival Analysis*
    @Saul:thesis2016

### Uncertainty Quantification

-   Deep nets are powerful approach to images, speech, language.

-   Proposal: Deep GPs may also be a great approach, but better to
    deploy according to natural strengths.

### Uncertainty Quantification

-   Probabilistic numerics, surrogate modelling, emulation, and UQ.

-   Not a fan of AI as a term.

-   But we are faced with increasing amounts of *algorithmic decision
    making*.

### ML and Decision Making

-   When trading off decisions: compute or acquire data?

-   There is a critical need for uncertainty.

### Uncertainty Quantification

> Uncertainty quantification (UQ) is the science of quantitative
> characterization and reduction of uncertainties in both computational
> and real world applications. It tries to determine how likely certain
> outcomes are if some aspects of the system are not exactly known.

-   Interaction between physical and virtual worlds of major interest
    for Amazon.

We will to illustrate different concepts of [Uncertainty
Quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification)
(UQ) and the role that Gaussian processes play in this field. Based on a
simple simulator of a car moving between a valley and a mountain, we are
going to illustrate the following concepts:

-   **Systems emulation**. Many real world decisions are based on
    simulations that can be computationally very demanding. We will show
    how simulators can be replaced by *emulators*: Gaussian process
    models fitted on a few simulations that can be used to replace the
    *simulator*. Emulators are cheap to compute, fast to run, and always
    provide ways to quantify the uncertainty of how precise they are
    compared the original simulator.

-   **Emulators in optimization problems**. We will show how emulators
    can be used to optimize black-box functions that are expensive to
    evaluate. This field is also called Bayesian Optimization and has
    gained an increasing relevance in machine learning as emulators can
    be used to optimize computer simulations (and machine learning
    algorithms) quite efficiently.

-   **Multi-fidelity emulation methods**. In many scenarios we have
    simulators of different quality about the same measure of interest.
    In these cases the goal is to merge all sources of information under
    the same model so the final emulator is cheaper and more accurate
    than an emulator fitted only using data from the most accurate and
    expensive simulator.

### Example: Formula One Racing

-   Designing an F1 Car requires CFD, Wind Tunnel, Track Testing etc.

-   How to combine them?

### Mountain Car Simulator

To illustrate the above mentioned concepts we we use the [mountain car
simulator](https://github.com/openai/gym/wiki/MountainCarContinuous-v0).
This simulator is widely used in machine learning to test reinforcement
learning algorithms. The goal is to define a control policy on a car
whose objective is to climb a mountain. Graphically, the problem looks
as follows:

<img class="" src="../slides/diagrams/uq/mountaincar.png" width="negate" align="" style="background:none; border:none; box-shadow:none;">

The goal is to define a sequence of actions (push the car right or left
with certain intensity) to make the car reach the flag after a number
$T$ of time steps.

At each time step $t$, the car is characterized by a vector
$\inputVector_{t} = (p_t,v_t)$ of states which are respectively the the
position and velocity of the car at time $t$. For a sequence of states
(an episode), the dynamics of the car is given by

$$\inputVector_{t+1} = \mappingFunction(\inputVector_{t},\textbf{u}_{t})$$

where $\textbf{u}_{t}$ is the value of an action force, which in this
example corresponds to push car to the left (negative value) or to the
right (positive value). The actions across a full episode are
represented in a policy $\textbf{u}_{t} = \pi(\inputVector_{t},\theta)$
that acts according to the current state of the car and some parameters
$\theta$. In the following examples we will assume that the policy is
linear which allows us to write $\pi(\inputVector_{t},\theta)$ as

$$\pi(\inputVector,\theta)= \theta_0 + \theta_p p + \theta_vv.$$

For $t=1,\dots,T$ now given some initial state $\inputVector_{0}$ and
some some values of each $\textbf{u}_{t}$, we can **simulate** the full
dynamics of the car for a full episode using
[Gym](https://gym.openai.com/envs/). The values of $\textbf{u}_{t}$ are
fully determined by the parameters of the linear controller.

After each episode of length $T$ is complete, a reward function
$R_{T}(\theta)$ is computed. In the mountain car example the reward is
computed as 100 for reaching the target of the hill on the right hand
side, minus the squared sum of actions (a real negative to push to the
left and a real positive to push to the right) from start to goal. Note
that our reward depend on $\theta$ as we make it dependent on the
parameters of the linear controller.

### Emulate the Mountain Car

```{code-cell}
import gym
```

```{code-cell}
env = gym.make('MountainCarContinuous-v0')
```

Our goal in this section is to find the parameters $\theta$ of the
linear controller such that

$$\theta^* = arg \max_{\theta} R_T(\theta).$$

In this section, we directly use Bayesian optimization to solve this
problem. We will use [GPyOpt](https://sheffieldml.github.io/GPyOpt/) so
we first define the objective function:

```{code-cell}
import mountain_car as mc
import GPyOpt
```

```{code-cell}
obj_func = lambda x: mc.run_simulation(env, x)[0]
objective = GPyOpt.core.task.SingleObjective(obj_func)
```

For each set of parameter values of the linear controller we can run an
episode of the simulator (that we fix to have a horizon of $T=500$) to
generate the reward. Using as input the parameters of the controller and
as outputs the rewards we can build a Gaussian process emulator of the
reward.

We start defining the input space, which is three-dimensional:

```{code-cell}
## --- We define the input space of the emulator

space= [{'name':'postion_parameter', 'type':'continuous', 'domain':(-1.2, +1)},
        {'name':'velocity_parameter', 'type':'continuous', 'domain':(-1/0.07, +1/0.07)},
        {'name':'constant', 'type':'continuous', 'domain':(-1, +1)}]

design_space = GPyOpt.Design_space(space=space)
```

Now we initizialize a Gaussian process emulator.

```{code-cell}
model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False, exact_feval=True, ARD=True)
```

In Bayesian optimization an acquisition function is used to balance
exploration and exploitation to evaluate new locations close to the
optimum of the objective. In this notebook we select the expected
improvement (EI). For further details have a look to the review paper of
[Shahriari et al
(2015)](http://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf).

```{code-cell}
aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(design_space)
acquisition = GPyOpt.acquisitions.AcquisitionEI(model, design_space, optimizer=aquisition_optimizer)
evaluator = GPyOpt.core.evaluators.Sequential(acquisition) # Collect points sequentially, no parallelization.
```

To initalize the model we start sampling some initial points (25) for
the linear controler randomly.

```{code-cell}
from GPyOpt.experiment_design.random_design import RandomDesign
```

```{code-cell}
n_initial_points = 25
random_design = RandomDesign(design_space)
initial_design = random_design.get_samples(n_initial_points)
```

Before we start any optimization, lets have a look to the behavior of
the car with the first of these initial points that we have selected
randomly.

```{code-cell}
import numpy as np
```

```{code-cell}
random_controller = initial_design[0,:]
_, _, _, frames = mc.run_simulation(env, np.atleast_2d(random_controller), render=True)
anim=mc.animate_frames(frames, 'Random linear controller')
```

```{code-cell}
from IPython.core.display import HTML
```

```{code-cell}
HTML(anim.to_jshtml())
```

```{code-cell}
mc.save_frames(frames, 
                  diagrams='../slides/diagrams/uq', 
                  filename='mountain_car_random.html')
```

<iframe src="../slides/diagrams/uq/mountain_car_random.html" width="1024" height="768" allowtransparency="true" frameborder="0">
</iframe>
As we can see the random linear controller does not manage to push the
car to the top of the mountain. Now, let's optimize the regret using
Bayesian optimization and the emulator for the reward. We try 50 new
parameters chosen by the EI.

```{code-cell}
max_iter = 50
bo = GPyOpt.methods.ModularBayesianOptimization(model, design_space, objective, acquisition, evaluator, initial_design)
bo.run_optimization(max_iter = max_iter )
```

Now we visualize the result for the best controller that we have found
with Bayesian optimization.

```{code-cell}
_, _, _, frames = mc.run_simulation(env, np.atleast_2d(bo.x_opt), render=True)
anim=mc.animate_frames(frames, 'Best controller after 50 iterations of Bayesian optimization')
```

```{code-cell}
HTML(anim.to_jshtml())
```

```{code-cell}
mc.save_frames(frames, 
                  diagrams='../slides/diagrams/uq', 
                  filename='mountain_car_simulated.html')
```

<iframe src="../slides/diagrams/uq/mountain_car_simulated.html" width="1024" height="768" allowtransparency="true" frameborder="0">
</iframe>
he car can now make it to the top of the mountain! Emulating the reward
function and using the EI helped as to find a linear controller that
solves the problem.

### Data Efficient Emulation

In the previous section we solved the mountain car problem by directly
emulating the reward but no considerations about the dynamics
$\inputVector_{t+1} = \mappingFunction(\inputVector_{t},\textbf{u}_{t})$
of the system were made. Note that we had to run 75 episodes of 500
steps each to solve the problem, which required to call the simulator
$500\times 75 =37500$ times. In this section we will show how it is
possible to reduce this number by building an emulator for $f$ that can
later be used to directly optimize the control.

The inputs of the model for the dynamics are the velocity, the position
and the value of the control so create this space accordingly.

```{code-cell}
import gym
```

```{code-cell}
env = gym.make('MountainCarContinuous-v0')
```

```{code-cell}
import GPyOpt
```

```{code-cell}
space_dynamics = [{'name':'position', 'type':'continuous', 'domain':[-1.2, +0.6]},
                  {'name':'velocity', 'type':'continuous', 'domain':[-0.07, +0.07]},
                  {'name':'action', 'type':'continuous', 'domain':[-1, +1]}]
design_space_dynamics = GPyOpt.Design_space(space=space_dynamics)
```

The outputs are the velocity and the position. Indeed our model will
capture the change in position and velocity on time. That is, we will
model

$$\Delta v_{t+1} = v_{t+1} - v_{t}$$

$$\Delta x_{t+1} = p_{t+1} - p_{t}$$

with Gaussian processes with prior mean $v_{t}$ and $p_{t}$
respectively. As a covariance function, we use a Matern52. We need
therefore two models to capture the full dynamics of the system.

```{code-cell}
position_model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False, exact_feval=True, ARD=True)
velocity_model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False, exact_feval=True, ARD=True)
```

Next, we sample some input parameters and use the simulator to compute
the outputs. Note that in this case we are not running the full
episodes, we are just using the simulator to compute
$\inputVector_{t+1}$ given $\inputVector_{t}$ and $\textbf{u}_{t}$.

```{code-cell}
import numpy as np
from GPyOpt.experiment_design.random_design import RandomDesign
import mountain_car as mc
```

```{code-cell}
### --- Random locations of the inputs
n_initial_points = 500
random_design_dynamics = RandomDesign(design_space_dynamics)
initial_design_dynamics = random_design_dynamics.get_samples(n_initial_points)
```

```{code-cell}
### --- Simulation of the (normalized) outputs
y = np.zeros((initial_design_dynamics.shape[0], 2))
for i in range(initial_design_dynamics.shape[0]):
    y[i, :] = mc.simulation(initial_design_dynamics[i, :])

# Normalize the data from the simulation
y_normalisation = np.std(y, axis=0)
y_normalised = y/y_normalisation
```

In general we might use much smarter strategies to design our emulation
of the simulator. For example, we could use the variance of the
predictive distributions of the models to collect points using
uncertainty sampling, which will give us a better coverage of the space.
For simplicity, we move ahead with the 500 randomly selected points.

Now that we have a data set, we can update the emulators for the
location and the velocity.

```{code-cell}
position_model.updateModel(initial_design_dynamics, y[:, [0]], None, None)
velocity_model.updateModel(initial_design_dynamics, y[:, [1]], None, None)
```

We can now have a look to how the emulator and the simulator match.
First, we show a contour plot of the car aceleration for each pair of
can position and velocity. You can use the bar bellow to play with the
values of the controler to compare the emulator and the simulator.

```{code-cell}
from IPython.html.widgets import interact
```

```{code-cell}
control = mc.plot_control(velocity_model)
interact(control.plot_slices, control=(-1, 1, 0.05))
```

<!---->
We can see how the emulator is doing a fairly good job approximating the
simulator. On the edges, however, it struggles to captures the dynamics
of the simulator.

Given some input parameters of the linear controlling, how do the
dynamics of the emulator and simulator match? In the following figure we
show the position and velocity of the car for the 500 time steps of an
episode in which the parameters of the linear controller have been fixed
beforehand. The value of the input control is also shown.

```{code-cell}
controller_gains = np.atleast_2d([0, .6, 1])  # change the valus of the linear controller to observe the trayectories.
```

```{code-cell}
mc.emu_sim_comparison(env, controller_gains, [position_model, velocity_model], 
                      max_steps=500, diagrams='../slides/diagrams/uq')
```

<img src="../slides/diagrams/uq/emu_sim_comparison.svg" align="">

We now make explicit use of the emulator, using it to replace the
simulator and optimize the linear controller. Note that in this
optimization, we don't need to query the simulator anymore as we can
reproduce the full dynamics of an episode using the emulator. For
illustrative purposes, in this example we fix the initial location of
the car.

We define the objective reward function in terms of the simulator.

```{code-cell}
### --- Optimize control parameters with emulator
car_initial_location = np.asarray([-0.58912799, 0]) 

### --- Reward objective function using the emulator
obj_func_emulator = lambda x: mc.run_emulation([position_model, velocity_model], x, car_initial_location)[0]
objective_emulator = GPyOpt.core.task.SingleObjective(obj_func_emulator)
```

And as before, we use Bayesian optimization to find the best possible
linear controller.

```{code-cell}
### --- Elements of the optimization that will use the multi-fidelity emulator
model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False, exact_feval=True, ARD=True)
```

The design space is the three continuous variables that make up the
linear controller.

```{code-cell}
space= [{'name':'linear_1', 'type':'continuous', 'domain':(-1/1.2, +1)},
        {'name':'linear_2', 'type':'continuous', 'domain':(-1/0.07, +1/0.07)},
        {'name':'constant', 'type':'continuous', 'domain':(-1, +1)}]

design_space         = GPyOpt.Design_space(space=space)
aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(design_space)

random_design = RandomDesign(design_space)
initial_design = random_design.get_samples(25)
```

We set the acquisition function to be expected improvement using
`GPyOpt`.

```{code-cell}
acquisition          = GPyOpt.acquisitions.AcquisitionEI(model, design_space, optimizer=aquisition_optimizer)
evaluator            = GPyOpt.core.evaluators.Sequential(acquisition)
```

```{code-cell}
bo_emulator = GPyOpt.methods.ModularBayesianOptimization(model, design_space, objective_emulator, acquisition, evaluator, initial_design)
bo_emulator.run_optimization(max_iter=50)
```

```{code-cell}
_, _, _, frames = mc.run_simulation(env, np.atleast_2d(bo_emulator.x_opt), render=True)
anim=mc.animate_frames(frames, 'Best controller using the emulator of the dynamics')
```

```{code-cell}
from IPython.core.display import HTML
```

```{code-cell}
HTML(anim.to_jshtml())
```

```{code-cell}
mc.save_frames(frames, 
                  diagrams='../slides/diagrams/uq', 
                  filename='mountain_car_emulated.html')
```

<iframe src="../slides/diagrams/uq/mountain_car_emulated.html" width="1024" height="768" allowtransparency="true" frameborder="0">
</iframe>
And the problem is again solved, but in this case we have replaced the
simulator of the car dynamics by a Gaussian process emulator that we
learned by calling the simulator only 500 times. Compared to the 37500
calls that we needed when applying Bayesian optimization directly on the
simulator this is a great gain.

In some scenarios we have simulators of the same environment that have
different fidelities, that is that reflect with different level of
accuracy the dynamics of the real world. Running simulations of the
different fidelities also have a different cost: hight fidelity
simulations are more expensive the cheaper ones. If we have access to
these simulators we can combine high and low fidelity simulations under
the same model.

So let's assume that we have two simulators of the mountain car
dynamics, one of high fidelity (the one we have used) and another one of
low fidelity. The traditional approach to this form of multi-fidelity
emulation is to assume that

$$\mappingFunction_i\left(\inputVector\right) = \rho\mappingFunction_{i-1}\left(\inputVector\right) + \delta_i\left(\inputVector \right)$$

where $\mappingFunction_{i-1}\left(\inputVector\right)$ is a low
fidelity simulation of the problem of interest and
$\mappingFunction_i\left(\inputVector\right)$ is a higher fidelity
simulation. The function $\delta_i\left(\inputVector \right)$ represents
the difference between the lower and higher fidelity simulation, which
is considered additive. The additive form of this covariance means that
if $\mappingFunction_{0}\left(\inputVector\right)$ and
$\left\{\delta_i\left(\inputVector \right)\right\}_{i=1}^m$ are all
Gaussian processes, then the process over all fidelities of simuation
will be a joint Gaussian process.

But with Deep Gaussian processes we can consider the form

$$\mappingFunction_i\left(\inputVector\right) = \mappingFunctionTwo_{i}\left(\mappingFunction_{i-1}\left(\inputVector\right)\right) + \delta_i\left(\inputVector \right),$$

where the low fidelity representation is non linearly transformed by
$\mappingFunctionTwo(\cdot)$ before use in the process. This is the
approach taken in @Perdikaris:multifidelity17. But once we accept that
these models can be composed, a highly flexible framework can emerge. A
key point is that the data enters the model at different levels, and
represents different aspects. For example these correspond to the two
fidelities of the mountain car simulator.

We start by sampling both of them at 250 random input locations.

```{code-cell}
import gym
```

```{code-cell}
env = gym.make('MountainCarContinuous-v0')
```

```{code-cell}
import GPyOpt
```

```{code-cell}
### --- Collect points from low and high fidelity simulator --- ###

space = GPyOpt.Design_space([
        {'name':'position', 'type':'continuous', 'domain':(-1.2, +1)},
        {'name':'velocity', 'type':'continuous', 'domain':(-0.07, +0.07)},
        {'name':'action', 'type':'continuous', 'domain':(-1, +1)}])

n_points = 250
random_design = GPyOpt.experiment_design.RandomDesign(space)
x_random = random_design.get_samples(n_points)
```

Next, we evaluate the high and low fidelity simualtors at those
locations.

```{code-cell}
import numpy as np
import mountain_car as mc
```

```{code-cell}
d_position_hf = np.zeros((n_points, 1))
d_velocity_hf = np.zeros((n_points, 1))
d_position_lf = np.zeros((n_points, 1))
d_velocity_lf = np.zeros((n_points, 1))

# --- Collect high fidelity points
for i in range(0, n_points):
    d_position_hf[i], d_velocity_hf[i] = mc.simulation(x_random[i, :])

# --- Collect low fidelity points  
for i in range(0, n_points):
    d_position_lf[i], d_velocity_lf[i] = mc.low_cost_simulation(x_random[i, :])
```

It is time to build the multi-fidelity model for both the position and
the velocity.

As we did in the previous section we use the emulator to optimize the
simulator. In this case we use the high fidelity output of the emulator.

```{code-cell}
### --- Optimize controller parameters 
obj_func = lambda x: mc.run_simulation(env, x)[0]
obj_func_emulator = lambda x: mc.run_emulation([position_model, velocity_model], x, car_initial_location)[0]
objective_multifidelity = GPyOpt.core.task.SingleObjective(obj_func)
```

And we optimize using Bayesian optimzation.

```{code-cell}
from GPyOpt.experiment_design.random_design import RandomDesign
```

```{code-cell}
model = GPyOpt.models.GPModel(optimize_restarts=5, verbose=False, exact_feval=True, ARD=True)
space= [{'name':'linear_1', 'type':'continuous', 'domain':(-1/1.2, +1)},
        {'name':'linear_2', 'type':'continuous', 'domain':(-1/0.07, +1/0.07)},
        {'name':'constant', 'type':'continuous', 'domain':(-1, +1)}]

design_space = GPyOpt.Design_space(space=space)
aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(design_space)

n_initial_points = 25
random_design = RandomDesign(design_space)
initial_design = random_design.get_samples(n_initial_points)
acquisition = GPyOpt.acquisitions.AcquisitionEI(model, design_space, optimizer=aquisition_optimizer)
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
```

```{code-cell}
bo_multifidelity = GPyOpt.methods.ModularBayesianOptimization(model, design_space, objective_multifidelity, acquisition, evaluator, initial_design)
bo_multifidelity.run_optimization(max_iter=50)
```

```{code-cell}
_, _, _, frames = mc.run_simulation(env, np.atleast_2d(bo_multifidelity.x_opt), render=True)
anim=mc.animate_frames(frames, 'Best controller with multi-fidelity emulator')
```

```{code-cell}
from IPython.core.display import HTML
```

```{code-cell}
HTML(anim.to_jshtml())
```

```{code-cell}
mc.save_frames(frames, 
                  diagrams='../slides/diagrams/uq', 
                  filename='mountain_car_multi_fidelity.html')
```

<iframe src="../slides/diagrams/uq/mountain_car_multi_fidelity.html" width="1024" height="768" allowtransparency="true" frameborder="0">
</iframe>
And problem solved! We see how the problem is also solved with 250
observations of the high fidelity simulator and 250 of the low fidelity
simulator.

### Acknowledgments

Stefanos Eleftheriadis, John Bronskill, Hugh Salimbeni, Rich Turner,
Zhenwen Dai, Javier Gonzalez, Andreas Damianou, Mark Pullin.

### Ongoing Code

-   Powerful framework but

-   Software isn't there yet.

-   Our focus: Gaussian Processes driven by MXNet

-   Composition of GPs, Neural Networks, Other Models

### Thanks!

-   twitter: @lawrennd
-   blog:
    [http://inverseprobability.com](http://inverseprobability.com/blog.html)

### References {#references .unnumbered .allowframebreaks}

[^1]: In classical statistics we often interpret these parameters,
    $\beta$, whereas in machine learning we are normally more interested
    in the result of the prediction, and less in the prediction.
    Although this is changing with more need for accountability. In
    honour of this I normally use $\boldsymbol{\beta}$ when I care about
    the value of these parameters, and $\mappingVector$ when I care more
    about the quality of the prediction.
