"""
Adagrad Optimizer Implementation

Implements Adagrad from Duchi et. al, 2011:
"Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

This implementation supports both the standard gradient descent update and
the dual averaging method of Nesterov.
"""

using LinearAlgebra
using Statistics
using Random

# No need to include base_optimizer.jl here since it's included in main module

"""
    AdagradOptimizer

Adagrad optimizer implementation using the base optimizer framework.

# Fields
- `primal_dual::Bool`: If true, uses dual averaging method, otherwise gradient descent
- `lr::Float64`: Learning rate coefficient
- `delta::Float64`: Small constant for numerical stability
- `optimizer::Optimizer`: Base optimizer instance
- `x0::Vector{Float64}`: Initial parameters (for dual averaging)
- `s::Vector{Float64}`: Accumulated squared gradients
- `sum_grad::Vector{Float64}`: Sum of gradients (for dual averaging)
- `grad::Vector{Float64}`: Current gradient
- `inv_lr::Vector{Float64}`: Inverse learning rate for each parameter
"""
mutable struct AdagradOptimizer <: AbstractOptimizer
    primal_dual::Bool
    lr::Float64
    delta::Float64
    optimizer::Optimizer
    x0::Vector{Float64}
    s::Vector{Float64}
    sum_grad::Vector{Float64}
    grad::Vector{Float64}
    inv_lr::Vector{Float64}
    
    function AdagradOptimizer(loss_func::Function, grad_func::Function; 
                             primal_dual::Bool=false, lr::Float64=1.0, delta::Float64=0.0,
                             trace_len::Int=200, tolerance::Float64=0.0,
                             save_first_iterations::Int=5, label::Union{String, Nothing}=nothing,
                             seed::Int=42)
        
        base_optimizer = Optimizer(loss_func, grad_func, 
                                  trace_len=trace_len, tolerance=tolerance,
                                  save_first_iterations=save_first_iterations, 
                                  label=label, seed=seed)
        
        new(primal_dual, lr, delta, base_optimizer, 
            Float64[], Float64[], Float64[], Float64[], Float64[])
    end
end

"""
    AdagradStochasticOptimizer

Stochastic Adagrad optimizer implementation.
"""
mutable struct AdagradStochasticOptimizer <: AbstractOptimizer
    primal_dual::Bool
    lr::Float64
    delta::Float64
    optimizer::StochasticOptimizer
    x0::Vector{Float64}
    s::Vector{Float64}
    sum_grad::Vector{Float64}
    grad::Vector{Float64}
    inv_lr::Vector{Float64}
    
    function AdagradStochasticOptimizer(loss_func::Function, grad_func::Function; 
                                       primal_dual::Bool=false, lr::Float64=1.0, delta::Float64=0.0,
                                       n_seeds::Int=1, seeds::Union{Vector{Int}, Nothing}=nothing,
                                       trace_len::Int=200, tolerance::Float64=0.0,
                                       save_first_iterations::Int=5, 
                                       label::Union{String, Nothing}=nothing)
        
        base_optimizer = StochasticOptimizer(loss_func, grad_func,
                                           n_seeds=n_seeds, seeds=seeds,
                                           trace_len=trace_len, tolerance=tolerance,
                                           save_first_iterations=save_first_iterations,
                                           label=label)
        
        new(primal_dual, lr, delta, base_optimizer,
            Float64[], Float64[], Float64[], Float64[], Float64[])
    end
end

# Forward base optimizer methods (except step!, init_run!, and run! which we implement ourselves)
for method in [:check_convergence, :should_update_trace, :save_checkpoint!, :reset!, :add_seeds!]
    @eval $method(adagrad::AdagradOptimizer, args...; kwargs...) = $method(adagrad.optimizer, args...; kwargs...)
    @eval $method(adagrad::AdagradStochasticOptimizer, args...; kwargs...) = $method(adagrad.optimizer, args...; kwargs...)
end

# Forward properties with proper handling
function Base.getproperty(adagrad::AdagradOptimizer, prop::Symbol)
    if prop in fieldnames(AdagradOptimizer)
        return getfield(adagrad, prop)
    else
        return getproperty(adagrad.optimizer, prop)
    end
end

function Base.getproperty(adagrad::AdagradStochasticOptimizer, prop::Symbol)
    if prop in fieldnames(AdagradStochasticOptimizer)
        return getfield(adagrad, prop)
    else
        return getproperty(adagrad.optimizer, prop)
    end
end

function Base.setproperty!(adagrad::AdagradOptimizer, prop::Symbol, value)
    if prop in fieldnames(AdagradOptimizer)
        setfield!(adagrad, prop, value)
    else
        setproperty!(adagrad.optimizer, prop, value)
    end
end

function Base.setproperty!(adagrad::AdagradStochasticOptimizer, prop::Symbol, value)
    if prop in fieldnames(AdagradStochasticOptimizer)
        setfield!(adagrad, prop, value)
    else
        setproperty!(adagrad.optimizer, prop, value)
    end
end

"""
    init_adagrad!(adagrad::Union{AdagradOptimizer, AdagradStochasticOptimizer}, x0::Vector{Float64})

Initialize Adagrad-specific variables.
"""
function init_adagrad!(adagrad::Union{AdagradOptimizer, AdagradStochasticOptimizer}, x0::Vector{Float64})
    adagrad.x0 = copy(x0)
    adagrad.s = zeros(Float64, length(x0))
    adagrad.sum_grad = zeros(Float64, length(x0))
    adagrad.grad = zeros(Float64, length(x0))
    adagrad.inv_lr = zeros(Float64, length(x0))
end

"""
    estimate_stepsize!(adagrad::Union{AdagradOptimizer, AdagradStochasticOptimizer})

Update the accumulated squared gradients and compute inverse learning rates.
"""
function estimate_stepsize!(adagrad::Union{AdagradOptimizer, AdagradStochasticOptimizer})
    # Update accumulated squared gradients
    adagrad.s = sqrt.(adagrad.s .^ 2 + adagrad.grad .^ 2)
    
    # Compute inverse learning rate for each parameter
    adagrad.inv_lr = adagrad.delta .+ adagrad.s
end

"""
    init_run!(adagrad::AdagradOptimizer, x0::Vector{Float64})

Initialize the Adagrad optimizer for a new run.
"""
function init_run!(adagrad::AdagradOptimizer, x0::Vector{Float64})
    init_run!(adagrad.optimizer, x0)
    init_adagrad!(adagrad, x0)
end

"""
    init_run!(adagrad::AdagradStochasticOptimizer, x0::Vector{Float64})

Initialize the stochastic Adagrad optimizer for a new run.
"""
function init_run!(adagrad::AdagradStochasticOptimizer, x0::Vector{Float64})
    init_run!(adagrad.optimizer, x0)
    init_adagrad!(adagrad, x0)
end

"""
    step!(adagrad::AdagradOptimizer)

Perform one Adagrad optimization step.
"""
function step!(adagrad::AdagradOptimizer)
    # Compute gradient
    adagrad.grad = adagrad.optimizer.grad_func(adagrad.optimizer.x)
    
    # Update stepsize estimates
    estimate_stepsize!(adagrad)
    
    # Update parameters
    if adagrad.primal_dual
        # Dual averaging method
        adagrad.sum_grad += adagrad.grad
        
        # Avoid division by zero
        update_mask = adagrad.inv_lr .!= 0
        adagrad.optimizer.x = copy(adagrad.x0)
        adagrad.optimizer.x[update_mask] -= adagrad.lr * (adagrad.sum_grad[update_mask] ./ adagrad.inv_lr[update_mask])
    else
        # Standard gradient descent update
        update_mask = adagrad.inv_lr .!= 0
        adagrad.optimizer.x[update_mask] -= adagrad.lr * (adagrad.grad[update_mask] ./ adagrad.inv_lr[update_mask])
    end
end

"""
    step!(adagrad::AdagradStochasticOptimizer)

Perform one Adagrad optimization step for stochastic optimizer.
"""
function step!(adagrad::AdagradStochasticOptimizer)
    # Compute gradient
    adagrad.grad = adagrad.optimizer.grad_func(adagrad.optimizer.x)
    
    # Update stepsize estimates
    estimate_stepsize!(adagrad)
    
    # Update parameters
    if adagrad.primal_dual
        # Dual averaging method
        adagrad.sum_grad += adagrad.grad
        
        # Avoid division by zero
        update_mask = adagrad.inv_lr .!= 0
        adagrad.optimizer.x = copy(adagrad.x0)
        adagrad.optimizer.x[update_mask] -= adagrad.lr * (adagrad.sum_grad[update_mask] ./ adagrad.inv_lr[update_mask])
    else
        # Standard gradient descent update
        update_mask = adagrad.inv_lr .!= 0
        adagrad.optimizer.x[update_mask] -= adagrad.lr * (adagrad.grad[update_mask] ./ adagrad.inv_lr[update_mask])
    end
end

"""
    run!(adagrad::AdagradOptimizer, x0::Vector{Float64}; kwargs...)

Run Adagrad optimization with custom optimization loop.
"""
function run!(adagrad::AdagradOptimizer, x0::Vector{Float64}; 
              t_max::Float64=Inf, it_max::Int=100, verbose::Bool=false)
    
    # Set up optimizer parameters
    if t_max == Inf && it_max == typemax(Int)
        it_max = 100
        if verbose
            println("$(adagrad.optimizer.label): The number of iterations is set to $it_max.")
        end
    end
    
    adagrad.optimizer.t_max = t_max
    adagrad.optimizer.it_max = it_max
    
    # Initialize the run
    init_run!(adagrad, x0)
    
    if verbose
        println("Starting optimization: $(adagrad.optimizer.label)")
        println("Max iterations: $it_max, Max time: $t_max")
    end
    
    # Main optimization loop - call step! on AdagradOptimizer directly
    while !check_convergence(adagrad)
        # Store previous iterate for tolerance check
        if adagrad.optimizer.tolerance > 0
            adagrad.optimizer.x_old_tol = copy(adagrad.optimizer.x)
        end
        
        # Perform optimization step - this calls step! on AdagradOptimizer
        step!(adagrad)
        
        # Save checkpoint
        save_checkpoint!(adagrad)
        
        # Print progress
        if verbose && (adagrad.optimizer.it % 100 == 0 || adagrad.optimizer.it == 1)
            current_loss = adagrad.optimizer.loss_func(adagrad.optimizer.x)
            current_grad_norm = norm(adagrad.optimizer.grad_func(adagrad.optimizer.x))
            println("Iter $(adagrad.optimizer.it): loss = $(round(current_loss, digits=6)), " *
                   "||∇f|| = $(round(current_grad_norm, digits=6))")
        end
    end
    
    # Compute final statistics
    compute_loss_of_iterates!(adagrad.optimizer.trace)
    compute_grad_norms!(adagrad.optimizer.trace)
    
    if verbose
        final_loss = adagrad.optimizer.loss_func(adagrad.optimizer.x)
        final_grad_norm = norm(adagrad.optimizer.grad_func(adagrad.optimizer.x))
        println("Optimization completed!")
        println("Final loss: $(round(final_loss, digits=6))")
        println("Final gradient norm: $(round(final_grad_norm, digits=6))")
        println("Total iterations: $(adagrad.optimizer.it)")
        println("Total time: $(round(adagrad.optimizer.t, digits=3)) seconds")
    end
    
    return adagrad.optimizer.trace
end

"""
    run!(adagrad::AdagradStochasticOptimizer, x0::Vector{Float64}; kwargs...)

Run stochastic Adagrad optimization with custom optimization loop.
"""
function run!(adagrad::AdagradStochasticOptimizer, x0::Vector{Float64}; 
              t_max::Float64=Inf, it_max::Int=100, verbose::Bool=false)
    
    # Set up optimizer parameters
    if t_max == Inf && it_max == typemax(Int)
        it_max = 100
        if verbose
            println("$(adagrad.optimizer.label): The number of iterations is set to $it_max.")
        end
    end
    
    adagrad.optimizer.t_max = t_max
    adagrad.optimizer.it_max = it_max
    
    # Run optimization for each seed
    for seed in adagrad.optimizer.seeds
        if seed in adagrad.optimizer.finished_seeds
            continue
        end
        
        if verbose && length(adagrad.optimizer.seeds) > 1
            println("$(adagrad.optimizer.label): Running seed $seed")
        end
        
        # Set random seed
        Random.seed!(seed)
        adagrad.optimizer.rng = MersenneTwister(seed)
        adagrad.optimizer.current_seed = seed
        
        # Initialize the run
        init_run!(adagrad, x0)
        
        # Main optimization loop - call step! on AdagradStochasticOptimizer directly
        while !check_convergence(adagrad)
            # Store previous iterate for tolerance check
            if adagrad.optimizer.tolerance > 0
                adagrad.optimizer.x_old_tol = copy(adagrad.optimizer.x)
            end
            
            # Perform optimization step - this calls step! on AdagradStochasticOptimizer
            step!(adagrad)
            
            # Save checkpoint
            save_checkpoint!(adagrad)
        end
        
        # Mark seed as finished
        push!(adagrad.optimizer.finished_seeds, seed)
        adagrad.optimizer.initialized = false
        
        if verbose
            final_loss = adagrad.optimizer.loss_func(adagrad.optimizer.x)
            println("Seed $seed completed: final loss = $(round(final_loss, digits=6))")
        end
    end
    
    # Compute final statistics for all seeds
    for seed in adagrad.optimizer.seeds
        if haskey(adagrad.optimizer.trace.seed_traces, seed)
            compute_loss_of_iterates!(adagrad.optimizer.trace.seed_traces[seed])
            compute_grad_norms!(adagrad.optimizer.trace.seed_traces[seed])
        end
    end
    
    if verbose
        stats = get_convergence_statistics(adagrad.optimizer.trace)
        if haskey(stats, "mean_final_loss")
            println("Stochastic optimization completed!")
            println("Mean final loss: $(round(stats["mean_final_loss"], digits=6))")
            println("Std final loss: $(round(stats["std_final_loss"], digits=6))")
            println("Best final loss: $(round(stats["best_final_loss"], digits=6))")
        end
    end
    
    return adagrad.optimizer.trace
end

# Legacy interface functions for backward compatibility
"""
    optimize_adagrad(loss_func, grad_func, x0::Vector{Float64}; kwargs...)

Optimize a function using Adagrad algorithm (legacy interface).
"""
function optimize_adagrad(loss_func::Function, grad_func::Function, x0::Vector{Float64}; 
                         primal_dual::Bool=false, lr::Float64=1.0, delta::Float64=0.0,
                         max_iter::Int=1000, tol::Float64=1e-6, verbose::Bool=false)
    
    # Create optimizer
    optimizer = AdagradOptimizer(loss_func, grad_func,
                               primal_dual=primal_dual, lr=lr, delta=delta,
                               tolerance=tol, label="Adagrad")
    
    # Run optimization
    trace = run!(optimizer, x0, it_max=max_iter, verbose=verbose)
    
    # Extract results
    x_opt = optimizer.x
    
    # Convert trace to history format
    history = []
    for i in 1:length(trace.xs)
        push!(history, Dict(
            "iteration" => trace.its[i],
            "loss" => trace.losses_computed ? trace.losses[i] : loss_func(trace.xs[i]),
            "gradient_norm" => trace.grad_norms_computed ? trace.grad_norms[i] : norm(grad_func(trace.xs[i])),
            "parameters" => trace.xs[i]
        ))
    end
    
    return x_opt, history
end

"""
    compare_adagrad_variants(loss_func, grad_func, x0::Vector{Float64}; kwargs...)

Compare different Adagrad configurations (legacy interface).
"""
function compare_adagrad_variants(loss_func::Function, grad_func::Function, x0::Vector{Float64}; 
                                 lr_values=[0.1, 1.0, 10.0], delta_values=[0.0, 1e-8, 1e-6],
                                 max_iter::Int=1000, tol::Float64=1e-6)
    
    results = Dict[]
    
    println("Comparing Adagrad variants...")
    println("=" ^ 50)
    
    for primal_dual in [false, true]
        for lr in lr_values
            for delta in delta_values
                method_name = primal_dual ? "Dual Averaging" : "Gradient Descent"
                
                println("Testing $method_name with lr=$lr, delta=$delta")
                
                # Create and run optimizer
                optimizer = AdagradOptimizer(loss_func, grad_func,
                                           primal_dual=primal_dual, lr=lr, delta=delta,
                                           tolerance=tol, label=method_name)
                
                trace = run!(optimizer, x0, it_max=max_iter, verbose=false)
                
                # Extract results
                x_opt = optimizer.x
                final_loss = loss_func(x_opt)
                final_grad_norm = norm(grad_func(x_opt))
                converged = final_grad_norm < tol
                iterations = optimizer.it
                
                result = Dict(
                    "method" => method_name,
                    "primal_dual" => primal_dual,
                    "lr" => lr,
                    "delta" => delta,
                    "final_loss" => final_loss,
                    "final_grad_norm" => final_grad_norm,
                    "converged" => converged,
                    "iterations" => iterations,
                    "x_opt" => x_opt,
                    "trace" => trace
                )
                
                push!(results, result)
                
                println("  Final loss: $(round(final_loss, digits=6))")
                println("  Gradient norm: $(round(final_grad_norm, digits=6))")
                println("  Converged: $converged")
                println("  Iterations: $iterations")
                println()
            end
        end
    end
    
    return results
end

"""
    get_best_result(results::Vector{Dict})

Find the best result from a comparison of Adagrad variants.
"""
function get_best_result(results::Vector{Dict})
    best_result = results[1]
    
    for result in results
        if result["final_loss"] < best_result["final_loss"]
            best_result = result
        end
    end
    
    return best_result
end

"""
    print_results_summary(results::Vector{Dict})

Print a summary table of Adagrad comparison results.
"""
function print_results_summary(results::Vector{Dict})
    println("Adagrad Results Summary")
    println("=" ^ 80)
    println("Method\t\t\tLR\tDelta\t\tFinal Loss\tGrad Norm\tConverged\tIters")
    println("-" ^ 80)
    
    for result in results
        method = result["primal_dual"] ? "Dual Avg" : "Grad Desc"
        lr = result["lr"]
        delta = result["delta"]
        loss = round(result["final_loss"], digits=6)
        grad_norm = round(result["final_grad_norm"], digits=6)
        converged = result["converged"] ? "Yes" : "No"
        iters = result["iterations"]
        
        println("$method\t\t$lr\t$delta\t\t$loss\t$grad_norm\t$converged\t\t$iters")
    end
    
    println("-" ^ 80)
    
    # Find and highlight best result
    best_result = get_best_result(results)
    println("Best result: $(best_result["method"]) with lr=$(best_result["lr"]), delta=$(best_result["delta"])")
    println("Final loss: $(round(best_result["final_loss"], digits=6))")
end