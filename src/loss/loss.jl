"""
Loss functions module for the Joptan optimization package.

This module provides common test functions used in optimization research,
including their gradients and Hessians when available.
"""

# Include all loss function implementations
include("rosenbrock.jl")
include("rastrigin.jl")

# Export all functions
export rosenbrock, rosenbrock_gradient, rosenbrock_hessian
export rastrigin, rastrigin_gradient, rastrigin_hessian