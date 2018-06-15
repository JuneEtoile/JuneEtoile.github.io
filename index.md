
<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
tex2jax: {
inlineMath: [['$','$'], ['\\(','\\)']],
processEscapes: true},
jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
TeX: {
extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
equationNumbers: {
autoNumber: "AMS"
}
}
});
</script>

# Table of Contents

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Table of Contents](#table-of-contents)
	- [$L_1$, $L_2$, $L_{\infty}$ Norm](#l1-l2-linfty-norm)
	- [Convex Optimization in Machine learning](#convex-optimization-in-machine-learning)
	- [Sequential Minimal Optimization (SMO)](#sequential-minimal-optimization-smo)
	- [Expectation Maximization Algorithm (EM) and Mixture Models (Mixture of Gaussians)](#expectation-maximization-algorithm-em-and-mixture-models-mixture-of-gaussians)
	- [Activation Functions in Deep learning](#activation-functions-in-deep-learning)
	- [Cost Functions](#cost-functions)

<!-- /TOC -->

## $L_1$, $L_2$, $L_{\infty}$ Norm
In machine learning, we may have $L_0$, $L_1$, $L_2$, and $L_{\infty}$ norm. Different norm has different purpose and property. Lets take a look:

- $L_0$ norm is used to count the number of non zeros entries.

**Regularization** is mainly used to choose a prefered model, so that the model is neither underfitting (too simple model) nor overfitting (too complex model).

How to do it? It just adds a **penalty term** to the objective function by controlling the **regularization's magnitude** (which is related to the model complexity, large magnitude of the regularization corresponds to complex model and vice versa) to control the model complexity.

Both $L_1$ and $L_2$ regularizations help to sovle the overfitting problem in machine learning.

L1 and L2 regularization prevents overfitting by shrinking on the coefficients. L2 (Ridge) shrinks all the coefficient to be very small but eliminates none, while L1 (Lasso) can shrink some coefficients to zero, performing variable selection.

  - $L_1$ norm (lasso loss) will make the solution to have the sparse property, such as the $L_1$ norm soft margin SVM. It is a great property for high dimensional data (model compressing, feature selection). L1 is not differentiable.
  - $L_2$ norm (ridge loss) tends to have smooth solution compared to $L_1$ norm. L2 is differentiable, gradient desent can be used.

A figure is shown for these two regulrizations:
![Image of $L_1$ and $L_2$ regularization](https://JuneEtoile.github.io/images/l1_l2_norm.png)


## Convex Optimization in Machine learning
* How to know that a function is a convex function?
    * First order and second order (Hessian matrix: nonnegative semi-definite) conditions (characterizations).
* Convex problem formulation:


## Sequential Minimal Optimization (SMO)



## Expectation Maximization Algorithm (EM) and Mixture Models (Mixture of Gaussians)
Used in the situation when latent variables (or missing data) are presented, such as mixture Gaussian model.

>EM exploits the fact that if the data were fully observed, then the ML/ MAP estimate would be
easy to compute. In particular, EM is an iterative algorithm which alternates between inferring
the missing values given the parameters (E step), and then optimizing the parameters given the
“filled in” data (M step). [1]



* E step: compute the expectation of the conditional latent variables.

  Calculate the expected value of the log likelihood function, with respect to the conditional distribution of $ \mathbf {Z} $  given $ \mathbf {X} $  under the current estimate of the parameters.

  Q function: for mixture Gaussian model: it is named as responsibility.

* M step: find the parameters that maximized the log likelihood function.


[1]: Machine Learning: A Probabilistic Perspective

## Activation Functions in Deep learning
* Sigmoid function: two class (to probability 0, 1)
* Softmax function: multi-class situation
* Tanh function: map to range (-1, 1)

>The advantage is that the negative inputs will be mapped strongly negative and the zero inputs will be mapped near zero in the tanh graph.

>The function is differentiable and monotonic while its derivative is not monotonic, mainly used classification between two classes.

Yann LeCun: For the learning time to be minimized, **the use of non-zero mean inputs should be avoided**.
>If the activation function is non-symmetric, as in the case of the sigmoid function, the output of of each neuron is restricted to the interval [0,1]. Such a choice **introduces a source of systematic bias** for those neurons located beyond the first layer of the network. To overcome this problem we need to use an antisymmetric activation function such as the hyperbolic tangent function. With this latter choice, the output of each neuron is permitted to assume both positive and negative values in the interval [−1,1], in which case it is likely for **its mean to be zero**. If the network connectivity is large, back-propagation learning with antisymmetric activation functions can yield **faster convergence** than a similar process with non-symmetric activation functions, for which there is also empirical evidence.


![Tanh and logistic sigmoid](https://JuneEtoile.github.io/images/relu_sigmoid.png)

* Relu /leaky Relu(rectified linear unit) activation function

Relu makes the activations to be **sparse and efficient**. Leaky relu is designed to make the gradient be non zero. Because the zero derivative will cause several neurons to just die and not respond to the variation in error.

>ReLu is **less computationally expensive** than tanh and sigmoid because it involves simpler mathematical operations. That is a good point to consider when we are designing **deep neural nets**.

1. Question: Why do not use relu in LSTM (RNN)?

Answer: 1. It exits, such Hinto's IRNN. 2. Nonnegative output of RELU (non-symmetric)

2. Question: Since Relu has non-symmetric property, why it performs superb on image data?

Reference:

1: https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

2: https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0

## Cost Functions

* Cross Entropy loss
* Least Square Loss(L2 loss), L1 Loss
* Maximum Log likelihood Loss
