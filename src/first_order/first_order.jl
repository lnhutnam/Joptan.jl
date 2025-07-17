"""
First-order optimization methods module for Joptan.jl

This module contains implementations of first-order optimization algorithms
that use only gradient information (no Hessian).
"""

# Include all first-order method implementations
include("adagrad.jl")

# Export Adagrad functions and types
export AdagradOptimizer, AdagradStochasticOptimizer
export estimate_stepsize!, optimize_adagrad, compare_adagrad_variants
export get_best_result, print_results_summary