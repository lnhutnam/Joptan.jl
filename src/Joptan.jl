module Joptan

"""
Joptan.jl - A Julia Optimization Package

This package provides optimization algorithms and test functions for numerical optimization.
"""
# Include test functions
include("test_functions/test_functions.jl")

# Include optimizer framework in correct order
include("optimizer_trace.jl")
include("base_optimizer.jl")

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

end # module Joptan