# joptan

[![Build Status](https://github.com/lnhutnam/joptan.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lnhutnam/joptan.jl/actions/workflows/CI.yml?query=branch%3Amain)


## First-order methods

- Gradient-based algorithms:
    - Gradient Descent (GD) [1, 2, 3, 4]
        - [docs](./docs/40_lemarechal-claude.pdf)
    - [Polyak's Heavy-ball](https://mitliagkas.github.io/ift6085-2019/ift-6085-lecture-5-notes.pdf) [5]
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


## References

[1] Lemaréchal, C. (1 January 2012). "Cauchy and the gradient method" ([PDF](https://web.archive.org/web/20181229073335/https://www.math.uni-bielefeld.de/documenta/vol-ismp/40_lemarechal-claude.pdf)). In Grötschel, M. (ed.). Optimization Stories. Documenta Mathematica Series. Vol. 6 (1st ed.). EMS Press. pp. 251–254. doi:10.4171/dms/6/27. ISBN 978-3-936609-58-5. Archived from the original ([PDF](https://www.math.uni-bielefeld.de/documenta/vol-ismp/40_lemarechal-claude.pdf)) on 2018-12-29. Retrieved 2020-01-26.

[2] Curry, Haskell B. (1944). "[The Method of Steepest Descent for Non-linear Minimization Problems](https://doi.org/10.1090%2Fqam%2F10667)". Quart. Appl. Math. 2 (3): 258–261. doi:10.1090/qam/10667

[3] Polyak, Boris (1987). [Introduction to Optimization](https://www.researchgate.net/publication/342978480)

[4] Akilov, G. P.; Kantorovich, L. V. (1982). Functional Analysis (2nd ed.). Pergamon Press. ISBN 0-08-023036-9.

[5] Polyak, Boris T. "[Some methods of speeding up the convergence of iteration methods.](https://hengshuaiyao.github.io/papers/polyak64.pdf)" Ussr computational mathematics and mathematical physics 4.5 (1964): 1-17.