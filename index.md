
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

# Table of contents
  * [$L_1$, $L_2$, $L_{\infty}$ Norm]( ## $L_1$, $L_2$, $L_{\infty}$ Norm)
  * [Convex Optimization](## Convex Optimization in Machine Learning)
  * [Sequential Minimal Optimization (SMO)](#SMO)
  * [Expectation Maximization Algorithm (EM) and Mixture Models (Mixture of Gaussians)](#EM)
  * [Activation Functions](#Activation Functions in Deep Learning)

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

## Cost Functions
