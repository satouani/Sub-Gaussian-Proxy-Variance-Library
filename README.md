# Optimal sub-Gaussian constants for discrete noise models in learning
Soufiane Atouani, Olivier Marchal, Julyan Arbel
2027-05-01

# Optimal sub-Gaussian constants for discrete noise models in learning

[![build and
publish](https://github.com/satouani/Sub-Gaussian-Proxy-Variance-Library.git/actions/workflows/build.yml/badge.svg)](https://github.com/satouani/Sub-Gaussian-Proxy-Variance-Library.git/actions/workflows/build.yml)
[![Creative Commons
License](https://i.creativecommons.org/l/by/4.0/80x15.png)](http://creativecommons.org/licenses/by/4.0/)
![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)

## Authors

- [Soufiane Atouani](https://www-ljk.imag.fr/) — Université Grenoble
  Alpes, Inria, CNRS, Grenoble INP, LJK, 38000 Grenoble, France
- [Olivier Marchal](https://math.univ-lyon1.fr/~marchal/index.html) —
  Université Jean Monnet Saint-Étienne, CNRS, Institut Camille Jordan
  UMR 5208; Institut Universitaire de France
- [Julyan Arbel](https://www.julyanarbel.com/) — Université Grenoble
  Alpes, Inria, CNRS, Grenoble INP, LJK, 38000 Grenoble, France

## Abstract

Sub-Gaussian assumptions are a cornerstone of learning theory,
underpinning concentration inequalities, generalization bounds, and
regret analyses in online learning and bandit problems. While such
results crucially depend on sub-Gaussian constants, these parameters are
often chosen conservatively, overlooking the fine structure of the
underlying noise distribution. In this work, we characterize optimal
sub-Gaussian variance proxies for a class of discrete random variables
with equally spaced support. We develop a general framework that yields
a practical procedure to compute optimal variance proxies from the
moment-generating function. As a first application, we provide a
complete analysis of three-point distributions, thereby extending the
classical Bernoulli case. We uncover sharp phase transitions between
strict and non-strict sub-Gaussian regimes and derive explicit
expressions for the optimal variance proxy in both symmetric and
asymmetric settings. We further show that the discrete uniform
distribution on $N$ points is strictly sub-Gaussian for all $N \geq 2$.
Building on these characterizations, we demonstrate how optimal variance
proxies can be plugged into standard concentration inequalities and
stochastic bandit analyses, leading to tighter confidence bounds and
improved constants in regret guarantees without modifying the underlying
algorithms. Finally, we provide an open-source Python package that
implements our theoretical results and enables the computation of
optimal sub-Gaussian variance proxies for a wide range of distributions.

## Guidelines

All figures in the paper are automatically reproduced during compilation
via the `pre-render` step defined in `_quarto.yml`. They are saved as
PDF and PNG in `figures/` and embedded directly in the document.

This repository also provides an open-source Python library
(`src/variance_proxy.py`) for computing optimal sub-Gaussian variance
proxies across a wide range of distributions. The library implements
both the theoretical results derived in this paper and complementary
results from the literature, covering Bernoulli, Binomial, Uniform
(continuous and discrete), Truncated Normal, Triangular, 3-mass discrete
(symmetric and asymmetric), Beta, and Kumaraswamy distributions.

To install the dependencies (Python \>= 3.10 required):

``` bash
pip install -r src/requirements.txt
```
