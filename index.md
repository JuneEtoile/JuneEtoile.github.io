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

## Table of contents
  * [$L_1$, $L_2$, $L_{\infty}$ Norm]( # $L_1$, $L_2$, $L_{\infty}$ Norm)
  * [Convex Optimization](# Convex Optimization in Machine Learning) 
  * [SMO](# SMO)

## $L_1$, $L_2$, $L_{\infty}$ Norm
In machine learning, we may have $L_0$, $L_1$, $L_2$, and $L_{\infty}$ norm. Different norm has different purpose and property. Lets take a look:

- $L_0$ norm is used to count the number of non zeros entries.

**Regularization** is mainly used to choose the prefered model. so that the model is neither underfitting (two simple model) nor overfitting (two complex model). 

How to do it? It just add a penalty term to the objective function by controlling the **regularization's magnitude** (which is related to the model complexity, large magnitude of the regularization corresponds to complex model and vice versa) to control the model complexity.

Both $L_1$ and $L_2$ regularizations help to sovle the overfitting problem in machine learning.

L1 and L2 regularization prevents overfitting by shrinking on the coefficients. L2 (Ridge) shrinks all the coefficient to be very small but eliminates none, while L1 (Lasso) can shrink some coefficients to zero, performing variable selection.

  - $L_1$ norm (lasso loss) will make the solution to have the sparse property, such as the $L_1$ norm soft margin SVM. It is a greaty property for high dimensional data (model compressing, feature selection). L1 is not differentiable.
  - $L_2$ norm (ridge loss) tends to have smooth solution compared to $L_1$ norm. L2 is differentiable, gradient desent can be used.

## Convex Optimization in Machine learning

## SMO
