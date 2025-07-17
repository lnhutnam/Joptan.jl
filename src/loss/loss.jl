"""
Loss functions module for the Joptan optimization package.

This module provides common test functions used in optimization research,
including their gradients and Hessians when available.
"""

# Include all loss function implementations
include("rosenbrock.jl")
include("rastrigin.jl")
include("linear_regression.jl")

# Export all functions
export rosenbrock, rosenbrock_gradient, rosenbrock_hessian
export rastrigin, rastrigin_gradient, rastrigin_hessian

# Export linear regression functions
export LinearRegressionLoss
export linear_regression_loss, linear_regression_gradient, linear_regression_hessian
export linear_regression_stochastic_gradient
export linear_regression_smoothness, linear_regression_max_smoothness, linear_regression_average_smoothness
export linear_regression_simple, linear_regression_gradient_simple, linear_regression_hessian_simple