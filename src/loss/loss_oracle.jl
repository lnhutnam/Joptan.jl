"""
Base oracle class for loss functions in Joptan.jl

This module provides the base class for all loss functions, including
regularization support and optimal point tracking.
"""

using LinearAlgebra
using Random
using Statistics
include("utils.jl")
include("regularizer.jl")

"""
    AbstractOracle

Abstract base type for all loss function oracles.
"""
abstract type AbstractOracle end

"""
    Oracle

Base class for all objectives. Can provide objective values, gradients 
and Hessians as functions that take parameters as input.

# Fields
- `l1::Float64`: L1 regularization coefficient
- `l2::Float64`: L2 regularization coefficient  
- `l2_in_prox::Bool`: Whether L2 regularization is handled in proximal operator
- `regularizer::Union{Regularizer, Nothing}`: Regularizer object
- `x_opt::Union{Vector{Float64}, Nothing}`: Best point found so far
- `f_opt::Float64`: Best function value found so far
- `seed::Int`: Random seed
- `rng::MersenneTwister`: Random number generator
- `_smoothness::Union{Float64, Nothing}`: Cached smoothness constant
- `_max_smoothness::Union{Float64, Nothing}`: Cached maximum smoothness
- `_ave_smoothness::Union{Float64, Nothing}`: Cached average smoothness
"""
mutable struct Oracle <: AbstractOracle
    l1::Float64
    l2::Float64
    l2_in_prox::Bool
    regularizer::Union{Regularizer, Nothing}
    x_opt::Union{Vector{Float64}, Nothing}
    f_opt::Float64
    seed::Int
    rng::MersenneTwister
    _smoothness::Union{Float64, Nothing}
    _max_smoothness::Union{Float64, Nothing}
    _ave_smoothness::Union{Float64, Nothing}
    
    function Oracle(; l1::Float64=0.0, l2::Float64=0.0, l2_in_prox::Bool=false,
                   regularizer::Union{Regularizer, Nothing}=nothing, seed::Int=42)
        
        # Validate regularization parameters
        if l1 < 0.0
            throw(ArgumentError("Invalid value for l1 regularization: $l1"))
        end
        if l2 < 0.0
            throw(ArgumentError("Invalid value for l2 regularization: $l2"))
        end
        if l2 == 0.0 && l2_in_prox
            @warn "The value of l2 is set to 0, so l2_in_prox is changed to False."
            l2_in_prox = false
        end
        
        # Set up L2 regularization
        effective_l2 = l2_in_prox ? 0.0 : l2
        
        # Create regularizer if needed
        if (l1 > 0 || l2_in_prox) && regularizer === nothing
            l2_prox = l2_in_prox ? l2 : 0.0
            regularizer = Regularizer(l1=l1, l2=l2_prox)
        end
        
        # Initialize random number generator
        rng = MersenneTwister(seed)
        
        new(l1, effective_l2, l2_in_prox, regularizer, nothing, Inf, seed, rng,
            nothing, nothing, nothing)
    end
end

"""
    set_seed!(oracle::Oracle, seed::Int)

Set the random seed for the oracle.

# Arguments
- `oracle::Oracle`: Oracle instance
- `seed::Int`: New random seed
"""
function set_seed!(oracle::Oracle, seed::Int)
    oracle.seed = seed
    oracle.rng = MersenneTwister(seed)
end

"""
    _value(oracle::Oracle, x::Vector{Float64})

Compute the base function value (without regularization).
This method should be implemented by subclasses.

# Arguments
- `oracle::Oracle`: Oracle instance
- `x::Vector{Float64}`: Input parameters

# Returns
- `Float64`: Base function value
"""
function _value(oracle::Oracle, x::Vector{Float64})
    error("_value method must be implemented by subclasses")
end

"""
    value(oracle::Oracle, x::Vector{Float64})

Compute the full objective value including regularization.

# Arguments
- `oracle::Oracle`: Oracle instance
- `x::Vector{Float64}`: Input parameters

# Returns
- `Float64`: Full objective value

# Note
This method automatically tracks the best point found so far.
"""
function value(oracle::Oracle, x::Vector{Float64})
    # Compute base value
    base_value = _value(oracle, x)
    
    # Add regularization
    total_value = base_value
    if oracle.regularizer !== nothing
        total_value += oracle.regularizer(x)
    else
        # Add L2 regularization if not in proximal operator
        if oracle.l2 > 0
            total_value += 0.5 * oracle.l2 * safe_sparse_norm(x)^2
        end
    end
    
    # Track best solution
    if total_value < oracle.f_opt
        oracle.x_opt = copy(x)
        oracle.f_opt = total_value
    end
    
    return total_value
end

"""
    gradient(oracle::Oracle, x::Vector{Float64})

Compute the gradient of the objective.
This method should be implemented by subclasses.

# Arguments
- `oracle::Oracle`: Oracle instance
- `x::Vector{Float64}`: Input parameters

# Returns
- `Vector{Float64}`: Gradient vector
"""
function gradient(oracle::Oracle, x::Vector{Float64})
    error("gradient method must be implemented by subclasses")
end

"""
    hessian(oracle::Oracle, x::Vector{Float64})

Compute the Hessian of the objective.
This method should be implemented by subclasses.

# Arguments
- `oracle::Oracle`: Oracle instance
- `x::Vector{Float64}`: Input parameters

# Returns
- `Matrix{Float64}`: Hessian matrix
"""
function hessian(oracle::Oracle, x::Vector{Float64})
    error("hessian method must be implemented by subclasses")
end

"""
    hess_vec_prod(oracle::Oracle, x::Vector{Float64}, v::Vector{Float64}; 
                  grad_dif::Bool=false, eps::Union{Float64, Nothing}=nothing)

Compute Hessian-vector product.

# Arguments
- `oracle::Oracle`: Oracle instance
- `x::Vector{Float64}`: Input parameters
- `v::Vector{Float64}`: Vector to multiply with Hessian
- `grad_dif::Bool`: Whether to use gradient differences for approximation
- `eps::Union{Float64, Nothing}`: Step size for finite differences

# Returns
- `Vector{Float64}`: Hessian-vector product

# Note
Default implementation uses finite differences. Subclasses can override
for more efficient implementations.
"""
function hess_vec_prod(oracle::Oracle, x::Vector{Float64}, v::Vector{Float64}; 
                      grad_dif::Bool=false, eps::Union{Float64, Nothing}=nothing)
    if grad_dif
        if eps === nothing
            eps = 1e-7
        end
        
        # Finite difference approximation: (∇f(x + εv) - ∇f(x)) / ε
        grad_plus = gradient(oracle, x + eps * v)
        grad_base = gradient(oracle, x)
        return (grad_plus - grad_base) / eps
    else
        # Use explicit Hessian
        H = hessian(oracle, x)
        return H * v
    end
end

"""
    smoothness(oracle::Oracle)

Get the smoothness constant (Lipschitz constant of the gradient).
This method should be implemented by subclasses.

# Returns
- `Float64`: Smoothness constant
"""
function smoothness(oracle::Oracle)
    if oracle._smoothness === nothing
        error("smoothness method must be implemented by subclasses")
    end
    return oracle._smoothness
end

"""
    max_smoothness(oracle::Oracle)

Get the maximum smoothness constant over all samples.
This method should be implemented by subclasses.

# Returns
- `Float64`: Maximum smoothness constant
"""
function max_smoothness(oracle::Oracle)
    if oracle._max_smoothness === nothing
        error("max_smoothness method must be implemented by subclasses")
    end
    return oracle._max_smoothness
end

"""
    average_smoothness(oracle::Oracle)

Get the average smoothness constant.
This method should be implemented by subclasses.

# Returns
- `Float64`: Average smoothness constant
"""
function average_smoothness(oracle::Oracle)
    if oracle._ave_smoothness === nothing
        error("average_smoothness method must be implemented by subclasses")
    end
    return oracle._ave_smoothness
end

"""
    batch_smoothness(oracle::Oracle, batch_size::Int)

Get the smoothness constant for a given batch size.
Default implementation returns the average smoothness.

# Arguments
- `oracle::Oracle`: Oracle instance
- `batch_size::Int`: Batch size

# Returns
- `Float64`: Batch smoothness constant
"""
function batch_smoothness(oracle::Oracle, batch_size::Int)
    return average_smoothness(oracle)
end

"""
    norm(oracle::Oracle, x)

Compute norm of input (static method equivalent).

# Arguments
- `oracle::Oracle`: Oracle instance (not used, for interface compatibility)
- `x`: Input to compute norm of

# Returns
- `Float64`: Norm of x
"""
function norm(oracle::Oracle, x)
    return safe_sparse_norm(x)
end

"""
    inner_prod(oracle::Oracle, x, y)

Compute inner product (static method equivalent).

# Arguments
- `oracle::Oracle`: Oracle instance (not used, for interface compatibility)
- `x`: First vector
- `y`: Second vector

# Returns
- `Float64`: Inner product of x and y
"""
function inner_prod(oracle::Oracle, x, y)
    return safe_sparse_inner_prod(x, y)
end

"""
    outer_prod(oracle::Oracle, x, y)

Compute outer product (static method equivalent).

# Arguments
- `oracle::Oracle`: Oracle instance (not used, for interface compatibility)
- `x`: First vector
- `y`: Second vector

# Returns
- `Matrix{Float64}`: Outer product of x and y
"""
function outer_prod(oracle::Oracle, x, y)
    return safe_outer_prod(x, y)
end

"""
    is_equal(oracle::Oracle, x, y; kwargs...)

Check if two arrays are equal (static method equivalent).

# Arguments
- `oracle::Oracle`: Oracle instance (not used, for interface compatibility)
- `x`: First array
- `y`: Second array
- `kwargs...`: Additional arguments for comparison

# Returns
- `Bool`: Whether arrays are equal
"""
function is_equal(oracle::Oracle, x, y; kwargs...)
    return safe_is_equal(x, y; kwargs...)
end

"""
    get_best_point(oracle::Oracle)

Get the best point found during optimization.

# Arguments
- `oracle::Oracle`: Oracle instance

# Returns
- `Tuple{Union{Vector{Float64}, Nothing}, Float64}`: Best point and its function value
"""
function get_best_point(oracle::Oracle)
    return oracle.x_opt, oracle.f_opt
end

"""
    reset_best!(oracle::Oracle)

Reset the best point tracking.

# Arguments
- `oracle::Oracle`: Oracle instance
"""
function reset_best!(oracle::Oracle)
    oracle.x_opt = nothing
    oracle.f_opt = Inf
end