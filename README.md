# Statistical-Learning
UC San Diego ECE 271A coursework: MATLAB implementations of statistical learning algorithms (Bayesian Decision Theory, MLE/MAP, GMM via EM Algorithm) applied to image segmentation tasks. 

# Probabilistic Machine Learning & Pattern Recognition (ECE 271A)

This repository contains a collection of projects developed for ECE 271A: Statistical Learning I. The projects focus on implementing fundamental probabilistic models from scratch in MATLAB, covering Bayesian decision theory, parameter estimation, and latent-variable learning.

The core application of these projects is using different methods and probability models to segment a Cheetah(foreground) from Grass(background) in images using 64-dimensional DCT features.

---

## Key Skills Demonstrated
* **Probabilistic Modeling:** Gaussian Mixture Models (GMM), Multivariate Gaussians.
* **Parameter Estimation:** Maximum Likelihood (ML), Maximum A Posteriori (MAP), Bayesian Predictive Distributions.
* **Optimization:** Expectation-Maximization (EM) algorithm derivation and implementation.
* **Dimensionality Analysis:** Feature selection and the curse of dimensionality.

---

## Project Overview

### [Homework 1: Feature Analysis & Priors]
* **Focus:** Preprocessing and Prior Estimation.
* **Details:**
  * Processed raw image blocks using Discrete Cosine Transform (DCT) and Zig-Zag scanning to create 64-dimensional feature vectors.
  * Estimated class priors $P(Cheetah)$ and $P(Grass)$ from training samples.
  * Visualized feature marginal densities to assess separability.

### [Homework 2: Bayesian Classification (Gaussian)]
* **Focus:** Maximum Likelihood Estimation (MLE) & Dimensionality.
* **Details:**
  * Modeled class-conditional densities $P(x|Class)$ using a Single Multivariate Gaussian.
  * Compared classification performance between the best 8 features vs. the full 64 features.
  * **Result:** The 64D model significantly outperformed the 8D model, demonstrating that even weak features contribute to classification in a Bayesian framework.

### [Homework 3: Bayesian Parameter Estimation]
* **Focus:** ML vs. MAP vs. Predictive Distribution.
* **Details:**
  * Implemented three different classifiers to handle parameter uncertainty:
    1. **Plug-in Classifier** using MLE.
    2. **Plug-in Classifier** using MAP estimation.
    3. **Bayesian Predictive Classifier** (integrating out parameters).
  * **Result:** Analyzed convergence behavior across datasets of varying sizes. Demonstrated that as prior uncertainty increases (larger $\alpha$), MAP and Predictive estimates converge toward the ML estimate.

### [Homework 5: EM for Gaussian Mixture Models]
* **Focus:** Latent Variables & Model Complexity.
* **Details:**
  * Derived and implemented the Expectation-Maximization (EM) algorithm for GMMs with diagonal covariance.
  * **Initialization:** Investigated the sensitivity of EM to random initializations and its impact on the decision boundary.
  * **Model Selection:** Evaluated the Probability of Error (PoE) for mixture components $C \in \{1, 2, ..., 32\}$.
  * **Result:** Found that $C=8$ provides the best balance between model flexibility and overfitting.

---

## Sample Results

---

## Disclaimer

This repository is intended for educational and portfolio demonstration purposes. The datasets and problem statements are property of the UCSD ECE department.
