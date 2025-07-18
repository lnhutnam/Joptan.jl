"""
Loss functions module for the Joptan optimization package.

This module provides common test functions used in optimization research,
including their gradients and Hessians when available, as well as the
oracle base class system for structured loss function implementations.
"""

# Include oracle system first
include("utils.jl")
include("regularizer.jl") 
include("loss_oracle.jl")
include("bounded_l2.jl")

# Include all loss function implementations
include("rosenbrock.jl")
include("rastrigin.jl")
include("linear_regression.jl")

# Export oracle system
export AbstractOracle, Oracle, Regularizer
export value, gradient, hessian, hess_vec_prod
export smoothness, max_smoothness, average_smoothness, batch_smoothness
export set_seed!, get_best_point, reset_best!
export prox, prox_l1, prox_l2, is_zero

# Export bounded L2 regularizer
export BoundedL2Regularizer
export hessian_diagonal

# Export utility functions
export safe_sparse_add, safe_sparse_multiply, safe_sparse_norm
export safe_sparse_inner_prod, safe_outer_prod, safe_is_equal

# Export test functions
export rosenbrock, rosenbrock_gradient, rosenbrock_hessian
export rastrigin, rastrigin_gradient, rastrigin_hessian

# Export linear regression functions
export LinearRegressionOracle, LinearRegressionLoss
export mat_vec_product, stochastic_gradient
export linear_regression_loss, linear_regression_gradient, linear_regression_hessian