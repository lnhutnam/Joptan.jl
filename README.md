# joptan

[![Build Status](https://github.com/lnhutnam/joptan.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lnhutnam/joptan.jl/actions/workflows/CI.yml?query=branch%3Amain)


## First-order methods

- Gradient-based algorithms:
    - Gradient Descent (GD) [paper](./docs/40_lemarechal-claude.pdf)
    - Polyak's Heavy-ball
    - Incremental Gradient (IG)
    - Nesterov's Acceleration
    - Nesterov's Acceleration with a special line search
    - Nesterov's Acceleration with restarts (RestNest)
    - Optimized Gradient Method (OGM)
- Adaptive:
    - AdaGrad
    - Adaptive GD (AdGD)
    - Accelerated AdGD (AdgdAccel)
    - Polyak

## Quasi-Newton

- Broyden–Fletcher–Goldfarb–Shanno algorithm (BFGS): is an iterative method for solving unconstrained nonlinear optimization problems. It determines the descent direction by preconditioning the gradient with curvature information. 
- Davidon–Fletcher–Powell algorithm (DFP): is a quasi-Newton method used for solving unconstrained optimization problems. It iteratively approximates the inverse of the Hessian matrix, which is essential for finding the optimal solution in optimization problems.
- Limited-memory BFGS (L-BFGS): is a limited-memory version of the BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm, which is a quasi-Newton method used for numerical optimization.
- Shor's R algorithm: is an iterative method for unconstrained optimization, designed for minimizing non- smooth functions, for which its reported success has been considerable.
- Symmetric Rank 1 (SR1): is a quasi-Newton method to update the second derivative (Hessian) based on the derivatives (gradients) calculated at two points.

## Second-order methods

(use second-order information (second derivatives) or their approximations.)

- Newton
- Cubic Newton
- Regularized (Global) Newton


## Stochastic first-order

- Stochastic Gradient Descent (SGD): is an iterative optimization algorithm used to train machine learning models.
- Root-Stochastic Gradient Descent (Root-SGD)
- Stochastic Variance Reduced Gradient (SVRG)
- Random Reshuffling (RR).


## Stochastic second-order

