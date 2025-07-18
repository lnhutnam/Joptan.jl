"""
First-order optimization methods module for Joptan.jl

This module contains implementations of first-order optimization algorithms
that use only gradient information (no Hessian).
"""

# Include utility functions first
include("utils.jl")

# Include all first-order method implementations
include("adagrad.jl")

# Export utility functions
export optimize_with_legacy_interface, compare_optimizer_variants
export get_best_result, print_results_summary, create_convergence_plot
export save_results_to_file

# Export Adagrad functions and types
export AdagradOptimizer, AdagradStochasticOptimizer
export estimate_stepsize!, optimize_adagrad, compare_adagrad_variants