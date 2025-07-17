"""
Base optimizer classes for Joptan.jl

This module provides the foundation for all optimization algorithms,
including deterministic and stochastic variants.
"""

using LinearAlgebra
using Random
using Statistics

"""
    AbstractOptimizer

Abstract base type for all optimizers.
"""
abstract type AbstractOptimizer end

"""
    Optimizer

Base class for deterministic optimization algorithms.

# Fields
- `loss_func::Function`: Loss function to minimize
- `grad_func::Function`: Gradient function
- `trace_len::Int`: Number of checkpoints to store in trace
- `tolerance::Float64`: Stationarity tolerance for convergence
- `save_first_iterations::Int`: Number of first iterations to always save
- `label::Union{String, Nothing}`: Label for the optimizer
- `trace::OptimizationTrace`: Optimization trace
- `rng::MersenneTwister`: Random number generator
- `initialized::Bool`: Whether optimizer has been initialized
- `x_old_tol::Union{Vector{Float64}, Nothing}`: Previous iterate for tolerance check
"""
mutable struct Optimizer <: AbstractOptimizer
    loss_func::Function
    grad_func::Function
    trace_len::Int
    tolerance::Float64
    save_first_iterations::Int
    label::Union{String, Nothing}
    trace::OptimizationTrace
    rng::MersenneTwister
    initialized::Bool
    x_old_tol::Union{Vector{Float64}, Nothing}
    
    # Runtime variables
    x::Vector{Float64}
    dim::Int
    t_max::Float64
    it_max::Int
    it::Int
    t::Float64
    t_start::Float64
    time_progress::Int
    iterations_progress::Int
    max_progress::Int
    
    function Optimizer(loss_func::Function, grad_func::Function;
                      trace_len::Int=200, tolerance::Float64=0.0,
                      save_first_iterations::Int=5, label::Union{String, Nothing}=nothing,
                      seed::Int=42)
        rng = MersenneTwister(seed)
        trace = OptimizationTrace(loss_func=loss_func, grad_func=grad_func, label=label)
        
        new(loss_func, grad_func, trace_len, tolerance, save_first_iterations, label,
            trace, rng, false, nothing,
            Float64[], 0, Inf, typemax(Int), 0, 0.0, 0.0, 0, 0, 0)
    end
end

"""
    StochasticOptimizer

Base class for stochastic optimization algorithms.

# Fields
- `loss_func::Function`: Loss function to minimize
- `grad_func::Function`: Gradient function
- `n_seeds::Int`: Number of random seeds to use
- `seeds::Vector{Int}`: List of random seeds
- `trace_len::Int`: Number of checkpoints to store in trace
- `tolerance::Float64`: Stationarity tolerance for convergence
- `save_first_iterations::Int`: Number of first iterations to always save
- `label::Union{String, Nothing}`: Label for the optimizer
- `trace::StochasticTrace`: Stochastic optimization trace
- `finished_seeds::Vector{Int}`: Seeds that have been completed
- `current_seed::Union{Int, Nothing}`: Currently active seed
"""
mutable struct StochasticOptimizer <: AbstractOptimizer
    loss_func::Function
    grad_func::Function
    n_seeds::Int
    seeds::Vector{Int}
    trace_len::Int
    tolerance::Float64
    save_first_iterations::Int
    label::Union{String, Nothing}
    trace::StochasticTrace
    finished_seeds::Vector{Int}
    current_seed::Union{Int, Nothing}
    
    # Runtime variables (same as Optimizer)
    x::Vector{Float64}
    dim::Int
    t_max::Float64
    it_max::Int
    it::Int
    t::Float64
    t_start::Float64
    time_progress::Int
    iterations_progress::Int
    max_progress::Int
    initialized::Bool
    x_old_tol::Union{Vector{Float64}, Nothing}
    rng::MersenneTwister
    
    function StochasticOptimizer(loss_func::Function, grad_func::Function;
                               n_seeds::Int=1, seeds::Union{Vector{Int}, Nothing}=nothing,
                               trace_len::Int=200, tolerance::Float64=0.0,
                               save_first_iterations::Int=5, 
                               label::Union{String, Nothing}=nothing)
        
        if seeds === nothing
            Random.seed!(42)
            seeds = rand(1:10000000, n_seeds)
        end
        
        trace = StochasticTrace(loss_func=loss_func, grad_func=grad_func, label=label)
        
        new(loss_func, grad_func, n_seeds, seeds, trace_len, tolerance, 
            save_first_iterations, label, trace, Int[], nothing,
            Float64[], 0, Inf, typemax(Int), 0, 0.0, 0.0, 0, 0, 0, false, nothing,
            MersenneTwister(42))
    end
end

"""
    init_run!(optimizer::Optimizer, x0::Vector{Float64})

Initialize the optimizer for a new run.
"""
function init_run!(optimizer::Optimizer, x0::Vector{Float64})
    optimizer.dim = length(x0)
    optimizer.x = copy(x0)
    optimizer.it = 0
    optimizer.t = 0.0
    optimizer.t_start = time()
    optimizer.time_progress = 0
    optimizer.iterations_progress = 0
    optimizer.max_progress = 0
    optimizer.x_old_tol = nothing
    
    # Initialize trace
    optimizer.trace = OptimizationTrace(
        loss_func=optimizer.loss_func, 
        grad_func=optimizer.grad_func, 
        label=optimizer.label
    )
    optimizer.trace.xs = [copy(x0)]
    optimizer.trace.ts = [0.0]
    optimizer.trace.its = [0]
    optimizer.trace.ls_its = [0]
    optimizer.trace.lrs = [0.0]
    
    optimizer.initialized = true
end

"""
    init_run!(optimizer::StochasticOptimizer, x0::Vector{Float64})

Initialize the stochastic optimizer for a new run.
"""
function init_run!(optimizer::StochasticOptimizer, x0::Vector{Float64})
    optimizer.dim = length(x0)
    optimizer.x = copy(x0)
    optimizer.it = 0
    optimizer.t = 0.0
    optimizer.t_start = time()
    optimizer.time_progress = 0
    optimizer.iterations_progress = 0
    optimizer.max_progress = 0
    optimizer.x_old_tol = nothing
    
    # Initialize trace for current seed
    if optimizer.current_seed !== nothing
        init_seed!(optimizer.trace, optimizer.current_seed)
        add_checkpoint!(optimizer.trace, x0, 0.0, 0)
    end
    
    optimizer.initialized = true
end

"""
    check_convergence(optimizer::AbstractOptimizer)

Check if the optimizer should stop based on convergence criteria.
"""
function check_convergence(optimizer::AbstractOptimizer)
    # Check iteration limit
    no_it_left = optimizer.it >= optimizer.it_max
    
    # Check time limit
    no_time_left = (time() - optimizer.t_start) >= optimizer.t_max
    
    # Check tolerance
    tolerance_met = false
    if optimizer.tolerance > 0 && optimizer.x_old_tol !== nothing
        tolerance_met = norm(optimizer.x - optimizer.x_old_tol) < optimizer.tolerance
    end
    
    return no_it_left || no_time_left || tolerance_met
end

"""
    should_update_trace(optimizer::AbstractOptimizer)

Determine if the trace should be updated at the current iteration.
"""
function should_update_trace(optimizer::AbstractOptimizer)
    # Always save first few iterations
    if optimizer.it <= optimizer.save_first_iterations
        return true
    end
    
    # Calculate progress
    if optimizer.t_max != Inf
        optimizer.time_progress = Int(floor((optimizer.trace_len - optimizer.save_first_iterations) * 
                                          optimizer.t / optimizer.t_max))
    end
    
    if optimizer.it_max != typemax(Int)
        optimizer.iterations_progress = Int(floor((optimizer.trace_len - optimizer.save_first_iterations) * 
                                                optimizer.it / optimizer.it_max))
    end
    
    # Check if enough progress has been made
    enough_progress = max(optimizer.time_progress, optimizer.iterations_progress) > optimizer.max_progress
    
    return enough_progress
end

"""
    save_checkpoint!(optimizer::Optimizer)

Save a checkpoint in the optimization trace.
"""
function save_checkpoint!(optimizer::Optimizer)
    optimizer.it += 1
    optimizer.t = time() - optimizer.t_start
    
    if should_update_trace(optimizer)
        add_checkpoint!(optimizer.trace, optimizer.x, optimizer.t, optimizer.it)
        optimizer.max_progress = max(optimizer.time_progress, optimizer.iterations_progress)
    end
end

"""
    save_checkpoint!(optimizer::StochasticOptimizer)

Save a checkpoint in the stochastic optimization trace.
"""
function save_checkpoint!(optimizer::StochasticOptimizer)
    optimizer.it += 1
    optimizer.t = time() - optimizer.t_start
    
    if should_update_trace(optimizer)
        add_checkpoint!(optimizer.trace, optimizer.x, optimizer.t, optimizer.it)
        optimizer.max_progress = max(optimizer.time_progress, optimizer.iterations_progress)
    end
end

"""
    step!(optimizer::AbstractOptimizer)

Perform one optimization step. This method should be implemented by concrete optimizer types.
"""
function step!(optimizer::AbstractOptimizer)
    error("step! method must be implemented by concrete optimizer types, not $(typeof(optimizer))")
end

"""
    run!(optimizer::Optimizer, x0::Vector{Float64}; 
         t_max::Float64=Inf, it_max::Int=100, verbose::Bool=false)

Run the optimization algorithm.
"""
function run!(optimizer::Optimizer, x0::Vector{Float64}; 
              t_max::Float64=Inf, it_max::Int=100, verbose::Bool=false)
    
    if t_max == Inf && it_max == typemax(Int)
        it_max = 100
        if verbose
            println("$(optimizer.label): The number of iterations is set to $it_max.")
        end
    end
    
    optimizer.t_max = t_max
    optimizer.it_max = it_max
    
    # Initialize the run
    init_run!(optimizer, x0)
    
    if verbose
        println("Starting optimization: $(optimizer.label)")
        println("Max iterations: $it_max, Max time: $t_max")
    end
    
    # Main optimization loop
    while !check_convergence(optimizer)
        # Store previous iterate for tolerance check
        if optimizer.tolerance > 0
            optimizer.x_old_tol = copy(optimizer.x)
        end
        
        # Perform optimization step - this calls the step! method of the actual optimizer type
        step!(optimizer)
        
        # Save checkpoint
        save_checkpoint!(optimizer)
        
        # Print progress
        if verbose && (optimizer.it % 100 == 0 || optimizer.it == 1)
            current_loss = optimizer.loss_func(optimizer.x)
            current_grad_norm = norm(optimizer.grad_func(optimizer.x))
            println("Iter $(optimizer.it): loss = $(round(current_loss, digits=6)), " *
                   "||∇f|| = $(round(current_grad_norm, digits=6))")
        end
    end
    
    # Compute final statistics
    compute_loss_of_iterates!(optimizer.trace)
    compute_grad_norms!(optimizer.trace)
    
    if verbose
        final_loss = optimizer.loss_func(optimizer.x)
        final_grad_norm = norm(optimizer.grad_func(optimizer.x))
        println("Optimization completed!")
        println("Final loss: $(round(final_loss, digits=6))")
        println("Final gradient norm: $(round(final_grad_norm, digits=6))")
        println("Total iterations: $(optimizer.it)")
        println("Total time: $(round(optimizer.t, digits=3)) seconds")
    end
    
    return optimizer.trace
end

"""
    run!(optimizer::StochasticOptimizer, x0::Vector{Float64}; 
         t_max::Float64=Inf, it_max::Int=100, verbose::Bool=false)

Run the stochastic optimization algorithm with multiple seeds.
"""
function run!(optimizer::StochasticOptimizer, x0::Vector{Float64}; 
              t_max::Float64=Inf, it_max::Int=100, verbose::Bool=false)
    
    if t_max == Inf && it_max == typemax(Int)
        it_max = 100
        if verbose
            println("$(optimizer.label): The number of iterations is set to $it_max.")
        end
    end
    
    optimizer.t_max = t_max
    optimizer.it_max = it_max
    
    # Run optimization for each seed
    for seed in optimizer.seeds
        if seed in optimizer.finished_seeds
            continue
        end
        
        if verbose && length(optimizer.seeds) > 1
            println("$(optimizer.label): Running seed $seed")
        end
        
        # Set random seed
        Random.seed!(seed)
        optimizer.rng = MersenneTwister(seed)
        optimizer.current_seed = seed
        
        # Initialize the run
        init_run!(optimizer, x0)
        
        # Main optimization loop
        while !check_convergence(optimizer)
            # Store previous iterate for tolerance check
            if optimizer.tolerance > 0
                optimizer.x_old_tol = copy(optimizer.x)
            end
            
            # Perform optimization step
            step!(optimizer)
            
            # Save checkpoint
            save_checkpoint!(optimizer)
        end
        
        # Mark seed as finished
        push!(optimizer.finished_seeds, seed)
        optimizer.initialized = false
        
        if verbose
            final_loss = optimizer.loss_func(optimizer.x)
            println("Seed $seed completed: final loss = $(round(final_loss, digits=6))")
        end
    end
    
    # Compute final statistics for all seeds
    for seed in optimizer.seeds
        if haskey(optimizer.trace.seed_traces, seed)
            compute_loss_of_iterates!(optimizer.trace.seed_traces[seed])
            compute_grad_norms!(optimizer.trace.seed_traces[seed])
        end
    end
    
    if verbose
        stats = get_convergence_statistics(optimizer.trace)
        if haskey(stats, "mean_final_loss")
            println("Stochastic optimization completed!")
            println("Mean final loss: $(round(stats["mean_final_loss"], digits=6))")
            println("Std final loss: $(round(stats["std_final_loss"], digits=6))")
            println("Best final loss: $(round(stats["best_final_loss"], digits=6))")
        end
    end
    
    return optimizer.trace
end

"""
    reset!(optimizer::AbstractOptimizer)

Reset the optimizer to initial state.
"""
function reset!(optimizer::Optimizer)
    optimizer.initialized = false
    optimizer.x_old_tol = nothing
    optimizer.trace = OptimizationTrace(
        loss_func=optimizer.loss_func, 
        grad_func=optimizer.grad_func, 
        label=optimizer.label
    )
end

function reset!(optimizer::StochasticOptimizer)
    optimizer.initialized = false
    optimizer.x_old_tol = nothing
    optimizer.finished_seeds = Int[]
    optimizer.current_seed = nothing
    optimizer.trace = StochasticTrace(
        loss_func=optimizer.loss_func, 
        grad_func=optimizer.grad_func, 
        label=optimizer.label
    )
end

"""
    add_seeds!(optimizer::StochasticOptimizer, n_extra_seeds::Int)

Add additional seeds to the stochastic optimizer.
"""
function add_seeds!(optimizer::StochasticOptimizer, n_extra_seeds::Int)
    Random.seed!(42)
    new_seeds = rand(1:10000000, n_extra_seeds)
    
    # Make sure seeds are unique
    for seed in new_seeds
        if !(seed in optimizer.seeds)
            push!(optimizer.seeds, seed)
        end
    end
    
    optimizer.n_seeds = length(optimizer.seeds)
end

"""
    add_seeds!(optimizer::Optimizer, n_extra_seeds::Int)

Add seeds to a deterministic optimizer (not supported - throws error).
"""
function add_seeds!(optimizer::Optimizer, n_extra_seeds::Int)
    error("Cannot add seeds to a deterministic Optimizer. Use StochasticOptimizer instead.")
end