This is a clean, professional `README.md` tailored for your **Computo** submission. It integrates your authors' info, abstract, the specific logic about the  case, and the different formulas used for symmetric and asymmetric distributions.

---

# Sub-Gaussian Variance Proxy for Discrete and Continuous Distributions

### Authors

* **Soufiane Atouani** Université Grenoble Alpes, Inria, CNRS, Grenoble INP, LJK, 38000 Grenoble, France
* **Olivier Marchal** Université Jean Monnet Saint-Étienne, CNRS, Institut Camille Jordan UMR 5208
Institut Universitaire de France, Les Forges 2, 20 Rue du Dr Annino, 42000 Saint-Étienne, France
* **Julyan Arbel** Université Grenoble Alpes, Inria, CNRS, Grenoble INP, LJK, 38000 Grenoble, France

### Abstract

We investigate the problem of characterizing the optimal variance proxy for sub-Gaussian random variables, whose moment-generating function exhibits bounded growth at infinity. We apply a general characterization method to discrete random variables with equally spaced atoms. We thoroughly study 3-mass distributions, thereby generalizing the well-studied Bernoulli case. We also prove that the discrete uniform distribution over  points is strictly sub-Gaussian. Finally, we provide an open-source Python package that combines analytical and numerical approaches to compute optimal sub-Gaussian variance proxies across a wide range of distributions.

---

## 🛠 Project Overview

This repository contains the implementation of the optimal sub-Gaussian variance proxy  for:

* **3-Mass Distributions:** Support on  with probabilities .
* **Beta Distributions:** .
* **Bernoulli Distributions:** A special case of both 3-mass (with one mass zeroed) and Beta distributions.

### Symmetric vs. Asymmetric Formulas

One of the key features of this implementation is the handling of **3-mass distributions**:

1. **Symmetric Case ():** Uses a specialized analytical formula derived specifically for symmetry.
2. **Asymmetric Case ():** Uses a generalized numerical optimization approach to solve for the proxy across the probability simplex.

We demonstrate that in the special case where , the generalized asymmetric implementation converges to the same results as the symmetric formula, particularly for the regime .

---

## 🚀 How to Run the Tests

To ensure reproducibility (as required by Computo), we provide a validation suite that replicates the paper's figures and checks numerical consistency.

### Requirements

* `numpy`
* `scipy`
* `matplotlib`

### Execution

**Via Command Line:**

```bash
python -m tests.test_variance_proxy

```

**Via VS Code Debugger:**
Use the provided `launch.json` configuration to run the `test_variance_proxy` module directly.

---

## 📊 Expected Outputs

The test script produces several plots and logs that validate the theoretical findings:

### 1. Numerical Equivalence

The script compares `SubGaussian3MassAsymmetricProxy` and `SubGaussian3MassSymmetricProxy`. For , it calculates the maximum absolute error between the two different formulas.

* **Output:** `Numerical equivalence (rtol=1e-10): True`
* **Significance:** Confirms that the two distinct mathematical approaches yield consistent results in the symmetric limit.

### 2. Beta & Bernoulli Reproduction

Reproduces Figure 1 from *Marchal and Arbel (2017)*, comparing the optimal proxy for Beta distributions against the Bernoulli case and the theoretical upper bound:


### 3. 3-Mass Stability Sweep

An exhaustive test runs through nearly 1,000,000 pairs of  to ensure that the numerical solver for the asymmetric case remains stable (Success rate: 1.0) and never produces `NaN` values within the valid simplex ().

---

## 📂 Repository Structure

* `src/`: Contains the core logic for `variance_proxy`.
* `tests/`: Contains `test_variance_proxy.py` used for validation and figure reproduction.
* `notebooks/`: (Optional) If you have Quarto/Jupyter examples for Computo.

---