"""
Loss functions module for the Joptan optimization package.

This module provides common test functions used in optimization research,
including their gradients and Hessians when available, as well as the
oracle base class system for structured loss function implementations.
"""

include("loss_oracle.jl")
include("bounded_l2.jl")
include("linear_regression.jl")
include("logistic_regression.jl")
include("log_sum_exp.jl")
include("regularizer.jl")
include("utils.jl")

# Export linear regression functions
export LinearRegressionLoss
export linear_regression_loss, linear_regression_gradient, linear_regression_hessian
export linear_regression_stochastic_gradient
export linear_regression_smoothness, linear_regression_max_smoothness, linear_regression_average_smoothness