module Joptan

"""
Joptan.jl - A Julia Optimization Package

This package provides optimization algorithms and test functions for numerical optimization.
"""

# Include optimizer framework in correct order
include("optimizer_trace.jl")
include("base_optimizer.jl")

# Include all submodules
include("functions.jl")

# Include loss functions
include("loss/loss.jl")

# Include optimization algorithm modules
include("first_order/first_order.jl")
# include("second_order/second_order.jl")
# include("quasi_newton/quasi_newton.jl")
# include("stochastic_first_order/stochastic_first_order.jl")
# include("stochastic_second_order/stochastic_second_order.jl")
# include("line_search/line_search.jl")

# Export base optimizer types and functions
export AbstractOptimizer, Optimizer, StochasticOptimizer
export OptimizationTrace, StochasticTrace
export init_run!, check_convergence, should_update_trace, save_checkpoint!, step!, run!, reset!
export compute_loss_of_iterates!, compute_grad_norms!, get_losses, get_grad_norms
export add_checkpoint!, init_seed!, get_mean_losses, get_std_losses, get_best_seed
export get_convergence_statistics, add_seeds!

# Export main functions from functions.jl
# export your_main_functions_here

# Export loss functions
export rosenbrock, rosenbrock_gradient, rosenbrock_hessian
export rastrigin, rastrigin_gradient, rastrigin_hessian

# Export linear regression functions
export LinearRegressionLoss
export linear_regression_loss, linear_regression_gradient, linear_regression_hessian
export linear_regression_stochastic_gradient
export linear_regression_smoothness, linear_regression_max_smoothness, linear_regression_average_smoothness

# Export first-order optimization algorithms
export AdagradOptimizer, AdagradStochasticOptimizer
export estimate_stepsize!, optimize_adagrad, compare_adagrad_variants
export get_best_result, print_results_summary

# Export optimization algorithms (when implemented)
# export gradient_descent, newton_method, bfgs, etc.

end # module Joptan