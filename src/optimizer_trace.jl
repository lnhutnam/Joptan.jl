"""
Optimization trace utilities for Joptan.jl

This module provides classes to store and analyze optimization traces,
including support for stochastic optimization with multiple seeds.
"""

using Statistics
using LinearAlgebra

"""
    OptimizationTrace

Stores the trace of an optimization run, including iterates, function values,
timing information, and other relevant statistics.

# Fields
- `loss_func::Union{Function, Nothing}`: Loss function being optimized
- `grad_func::Union{Function, Nothing}`: Gradient function
- `label::Union{String, Nothing}`: Label for the trace
- `xs::Vector{Vector{Float64}}`: Sequence of iterates
- `ts::Vector{Float64}`: Time stamps
- `its::Vector{Int}`: Iteration numbers
- `ls_its::Vector{Int}`: Line search iteration numbers (if applicable)
- `lrs::Vector{Float64}`: Learning rates (if applicable)
- `losses::Vector{Float64}`: Function values (computed lazily)
- `grad_norms::Vector{Float64}`: Gradient norms (computed lazily)
- `losses_computed::Bool`: Whether losses have been computed
- `grad_norms_computed::Bool`: Whether gradient norms have been computed
"""
mutable struct OptimizationTrace
    loss_func::Union{Function, Nothing}
    grad_func::Union{Function, Nothing}
    label::Union{String, Nothing}
    xs::Vector{Vector{Float64}}
    ts::Vector{Float64}
    its::Vector{Int}
    ls_its::Vector{Int}
    lrs::Vector{Float64}
    losses::Vector{Float64}
    grad_norms::Vector{Float64}
    losses_computed::Bool
    grad_norms_computed::Bool
    
    function OptimizationTrace(; loss_func=nothing, grad_func=nothing, label=nothing)
        new(loss_func, grad_func, label, 
            Vector{Vector{Float64}}(), Float64[], Int[], Int[], Float64[],
            Float64[], Float64[], false, false)
    end
end

"""
    compute_loss_of_iterates!(trace::OptimizationTrace)

Compute loss values for all iterates in the trace.
"""
function compute_loss_of_iterates!(trace::OptimizationTrace)
    if trace.loss_func === nothing
        @warn "Loss function not provided, cannot compute losses"
        return
    end
    
    if !trace.losses_computed
        trace.losses = [trace.loss_func(x) for x in trace.xs]
        trace.losses_computed = true
    end
end

"""
    compute_grad_norms!(trace::OptimizationTrace)

Compute gradient norms for all iterates in the trace.
"""
function compute_grad_norms!(trace::OptimizationTrace)
    if trace.grad_func === nothing
        @warn "Gradient function not provided, cannot compute gradient norms"
        return
    end
    
    if !trace.grad_norms_computed
        trace.grad_norms = [norm(trace.grad_func(x)) for x in trace.xs]
        trace.grad_norms_computed = true
    end
end

"""
    get_losses(trace::OptimizationTrace)

Get loss values, computing them if necessary.
"""
function get_losses(trace::OptimizationTrace)
    if !trace.losses_computed
        compute_loss_of_iterates!(trace)
    end
    return trace.losses
end

"""
    get_grad_norms(trace::OptimizationTrace)

Get gradient norms, computing them if necessary.
"""
function get_grad_norms(trace::OptimizationTrace)
    if !trace.grad_norms_computed
        compute_grad_norms!(trace)
    end
    return trace.grad_norms
end

"""
    add_checkpoint!(trace::OptimizationTrace, x::Vector{Float64}, t::Float64, it::Int;
                   ls_it::Int=0, lr::Float64=0.0)

Add a checkpoint to the trace.
"""
function add_checkpoint!(trace::OptimizationTrace, x::Vector{Float64}, t::Float64, it::Int;
                        ls_it::Int=0, lr::Float64=0.0)
    push!(trace.xs, copy(x))
    push!(trace.ts, t)
    push!(trace.its, it)
    push!(trace.ls_its, ls_it)
    push!(trace.lrs, lr)
    
    # Invalidate cached computations
    trace.losses_computed = false
    trace.grad_norms_computed = false
end

"""
    StochasticTrace

Stores traces from multiple stochastic optimization runs with different seeds.

# Fields
- `loss_func::Union{Function, Nothing}`: Loss function being optimized
- `grad_func::Union{Function, Nothing}`: Gradient function
- `label::Union{String, Nothing}`: Label for the trace
- `seed_traces::Dict{Int, OptimizationTrace}`: Traces for each seed
- `current_seed::Union{Int, Nothing}`: Currently active seed
- `seeds::Vector{Int}`: List of all seeds used
"""
mutable struct StochasticTrace
    loss_func::Union{Function, Nothing}
    grad_func::Union{Function, Nothing}
    label::Union{String, Nothing}
    seed_traces::Dict{Int, OptimizationTrace}
    current_seed::Union{Int, Nothing}
    seeds::Vector{Int}
    
    function StochasticTrace(; loss_func=nothing, grad_func=nothing, label=nothing)
        new(loss_func, grad_func, label, Dict{Int, OptimizationTrace}(), nothing, Int[])
    end
end

"""
    init_seed!(trace::StochasticTrace, seed::Int)

Initialize a new seed for stochastic optimization.
"""
function init_seed!(trace::StochasticTrace, seed::Int)
    trace.current_seed = seed
    if !(seed in trace.seeds)
        push!(trace.seeds, seed)
    end
    trace.seed_traces[seed] = OptimizationTrace(
        loss_func=trace.loss_func, 
        grad_func=trace.grad_func, 
        label=trace.label
    )
end

"""
    add_checkpoint!(trace::StochasticTrace, x::Vector{Float64}, t::Float64, it::Int;
                   ls_it::Int=0, lr::Float64=0.0)

Add a checkpoint to the current seed's trace.
"""
function add_checkpoint!(trace::StochasticTrace, x::Vector{Float64}, t::Float64, it::Int;
                        ls_it::Int=0, lr::Float64=0.0)
    if trace.current_seed === nothing
        error("No active seed. Call init_seed! first.")
    end
    
    add_checkpoint!(trace.seed_traces[trace.current_seed], x, t, it, ls_it=ls_it, lr=lr)
end

"""
    get_mean_losses(trace::StochasticTrace)

Get mean loss values across all seeds.
"""
function get_mean_losses(trace::StochasticTrace)
    if isempty(trace.seed_traces)
        return Float64[]
    end
    
    # Get losses for all seeds
    all_losses = []
    for seed in trace.seeds
        if haskey(trace.seed_traces, seed)
            losses = get_losses(trace.seed_traces[seed])
            push!(all_losses, losses)
        end
    end
    
    if isempty(all_losses)
        return Float64[]
    end
    
    # Find minimum length
    min_len = minimum(length(losses) for losses in all_losses)
    
    # Compute mean
    mean_losses = zeros(min_len)
    for i in 1:min_len
        mean_losses[i] = mean(losses[i] for losses in all_losses)
    end
    
    return mean_losses
end

"""
    get_std_losses(trace::StochasticTrace)

Get standard deviation of loss values across all seeds.
"""
function get_std_losses(trace::StochasticTrace)
    if isempty(trace.seed_traces) || length(trace.seeds) < 2
        return Float64[]
    end
    
    # Get losses for all seeds
    all_losses = []
    for seed in trace.seeds
        if haskey(trace.seed_traces, seed)
            losses = get_losses(trace.seed_traces[seed])
            push!(all_losses, losses)
        end
    end
    
    if length(all_losses) < 2
        return Float64[]
    end
    
    # Find minimum length
    min_len = minimum(length(losses) for losses in all_losses)
    
    # Compute standard deviation
    std_losses = zeros(min_len)
    for i in 1:min_len
        values = [losses[i] for losses in all_losses]
        std_losses[i] = std(values)
    end
    
    return std_losses
end

"""
    get_best_seed(trace::StochasticTrace)

Get the seed that achieved the best (lowest) final loss.
"""
function get_best_seed(trace::StochasticTrace)
    if isempty(trace.seed_traces)
        return nothing
    end
    
    best_seed = nothing
    best_loss = Inf
    
    for seed in trace.seeds
        if haskey(trace.seed_traces, seed)
            losses = get_losses(trace.seed_traces[seed])
            if !isempty(losses) && losses[end] < best_loss
                best_loss = losses[end]
                best_seed = seed
            end
        end
    end
    
    return best_seed
end

"""
    get_convergence_statistics(trace::Union{OptimizationTrace, StochasticTrace})

Get convergence statistics from the trace.
"""
function get_convergence_statistics(trace::OptimizationTrace)
    if isempty(trace.xs)
        return Dict()
    end
    
    stats = Dict(
        "total_iterations" => length(trace.xs) - 1,
        "total_time" => isempty(trace.ts) ? 0.0 : trace.ts[end],
        "final_iterate" => copy(trace.xs[end])
    )
    
    if trace.losses_computed || trace.loss_func !== nothing
        losses = get_losses(trace)
        if !isempty(losses)
            stats["initial_loss"] = losses[1]
            stats["final_loss"] = losses[end]
            stats["loss_reduction"] = losses[1] - losses[end]
        end
    end
    
    if trace.grad_norms_computed || trace.grad_func !== nothing
        grad_norms = get_grad_norms(trace)
        if !isempty(grad_norms)
            stats["initial_grad_norm"] = grad_norms[1]
            stats["final_grad_norm"] = grad_norms[end]
        end
    end
    
    return stats
end

function get_convergence_statistics(trace::StochasticTrace)
    if isempty(trace.seed_traces)
        return Dict()
    end
    
    # Get statistics for each seed
    seed_stats = Dict()
    for seed in trace.seeds
        if haskey(trace.seed_traces, seed)
            seed_stats[seed] = get_convergence_statistics(trace.seed_traces[seed])
        end
    end
    
    # Aggregate statistics
    stats = Dict(
        "num_seeds" => length(trace.seeds),
        "seed_statistics" => seed_stats
    )
    
    # Compute mean and std of key metrics
    if !isempty(seed_stats)
        final_losses = [s["final_loss"] for s in values(seed_stats) if haskey(s, "final_loss")]
        if !isempty(final_losses)
            stats["mean_final_loss"] = mean(final_losses)
            stats["std_final_loss"] = length(final_losses) > 1 ? std(final_losses) : 0.0
            stats["best_final_loss"] = minimum(final_losses)
        end
        
        final_grad_norms = [s["final_grad_norm"] for s in values(seed_stats) if haskey(s, "final_grad_norm")]
        if !isempty(final_grad_norms)
            stats["mean_final_grad_norm"] = mean(final_grad_norms)
            stats["std_final_grad_norm"] = length(final_grad_norms) > 1 ? std(final_grad_norms) : 0.0
        end
    end
    
    return stats
end