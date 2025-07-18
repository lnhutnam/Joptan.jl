"""
Bounded L2 Regularizer Implementation

This module implements the bounded L2 regularization function:
    R(x) = sum_{i=1}^d x_i^2 / (x_i^2 + 1)

This penalty is attractive for benchmarking purposes since it is smooth 
(has Lipschitz gradient) and nonconvex.

References:
- https://arxiv.org/pdf/1905.05920.pdf
- https://arxiv.org/pdf/1810.10690.pdf
"""

using LinearAlgebra
using SparseArrays
include("utils.jl")

"""
    BoundedL2Regularizer

Bounded L2 regularizer that implements R(x) = coef * 0.5 * sum(x_i^2 / (x_i^2 + 1)).

This regularizer is smooth and nonconvex, making it useful for benchmarking 
optimization algorithms. It follows the same interface as the standard Regularizer.

# Fields
- `l1::Float64`: L1 regularization coefficient (always 0.0 for bounded L2)
- `l2::Float64`: L2 regularization coefficient (always 0.0 for bounded L2)
- `coef::Union{Vector{Float64}, Nothing}`: Optional coefficient vector
- `bounded_l2_coef::Float64`: Bounded L2 regularization coefficient
"""
mutable struct BoundedL2Regularizer
    l1::Float64
    l2::Float64
    coef::Union{Vector{Float64}, Nothing}
    bounded_l2_coef::Float64
    
    function BoundedL2Regularizer(bounded_l2_coef::Float64; coef::Union{Vector{Float64}, Nothing}=nothing)
        if bounded_l2_coef < 0.0
            throw(ArgumentError("Bounded L2 regularization coefficient must be non-negative, got $bounded_l2_coef"))
        end
        new(0.0, 0.0, coef, bounded_l2_coef)
    end
end

"""
    (reg::BoundedL2Regularizer)(x::Vector{Float64})

Function call operator for regularizer evaluation.

# Arguments
- `x::Vector{Float64}`: Input vector

# Returns
- `Float64`: Regularization value
"""
function (reg::BoundedL2Regularizer)(x::Vector{Float64})
    return value(reg, x)
end

"""
    value(reg::BoundedL2Regularizer, x::Vector{Float64})

Compute the bounded L2 regularization value.

# Arguments
- `reg::BoundedL2Regularizer`: Regularizer instance
- `x::Vector{Float64}`: Input vector

# Returns
- `Float64`: Bounded L2 regularization value

# Formula
R(x) = bounded_l2_coef * 0.5 * sum(x_i^2 / (x_i^2 + 1))
"""
function value(reg::BoundedL2Regularizer, x::Vector{Float64})
    if issparse(x)
        result = 0.0
        for (i, val) in zip(findnz(x)...)
            x2 = val * val
            result += x2 / (x2 + 1)
        end
        return reg.bounded_l2_coef * 0.5 * result
    else
        x2 = x .* x
        return reg.bounded_l2_coef * 0.5 * sum(x2 ./ (x2 .+ 1))
    end
end

"""
    gradient(reg::BoundedL2Regularizer, x::Vector{Float64})

Compute the gradient of the bounded L2 regularization.

# Arguments
- `reg::BoundedL2Regularizer`: Regularizer instance
- `x::Vector{Float64}`: Input vector

# Returns
- `Vector{Float64}`: Gradient vector

# Formula
∇R(x) = bounded_l2_coef * x_i / (x_i^2 + 1)^2 for each component i
"""
function gradient(reg::BoundedL2Regularizer, x::Vector{Float64})
    if issparse(x)
        I, V = findnz(x)
        grad_values = similar(V)
        
        for (idx, val) in enumerate(V)
            denominator = (val * val + 1)^2
            grad_values[idx] = reg.bounded_l2_coef * val / denominator
        end
        
        return sparsevec(I, grad_values, length(x))
    else
        denominator = (x .* x .+ 1) .^ 2
        return reg.bounded_l2_coef * x ./ denominator
    end
end

"""
    prox(reg::BoundedL2Regularizer, x::Vector{Float64}, lr::Float64)

Proximal operator for bounded L2 regularization.

# Arguments
- `reg::BoundedL2Regularizer`: Regularizer instance
- `x::Vector{Float64}`: Input vector
- `lr::Float64`: Learning rate/step size

# Throws
- `ErrorException`: The exact proximal operator does not exist for bounded L2

# Note
The exact proximal operator for bounded L2 regularization does not have a 
closed-form solution. Consider using gradient-based methods instead.
"""
function prox(reg::BoundedL2Regularizer, x::Vector{Float64}, lr::Float64)
    throw(ErrorException("Exact proximal operator for bounded L2 does not exist. Consider using gradients."))
end

"""
    prox_l1(reg::BoundedL2Regularizer, x::Vector{Float64}, lr::Float64)

Proximal operator for L1 regularization (not applicable for bounded L2).

# Arguments
- `reg::BoundedL2Regularizer`: Regularizer instance
- `x::Vector{Float64}`: Input vector
- `lr::Float64`: Learning rate/step size

# Returns
- `Vector{Float64}`: Input vector unchanged (no L1 regularization)
"""
function prox_l1(reg::BoundedL2Regularizer, x::Vector{Float64}, lr::Float64)
    return x  # No L1 regularization in bounded L2
end

"""
    prox_l2(reg::BoundedL2Regularizer, x::Vector{Float64}, lr::Float64)

Proximal operator for L2 regularization (not applicable for bounded L2).

# Arguments
- `reg::BoundedL2Regularizer`: Regularizer instance
- `x::Vector{Float64}`: Input vector
- `lr::Float64`: Learning rate/step size

# Returns
- `Vector{Float64}`: Input vector unchanged (no standard L2 regularization)
"""
function prox_l2(reg::BoundedL2Regularizer, x::Vector{Float64}, lr::Float64)
    return x  # No standard L2 regularization in bounded L2
end

"""
    smoothness(reg::BoundedL2Regularizer)

Get the smoothness constant (Lipschitz constant of the gradient).

# Arguments
- `reg::BoundedL2Regularizer`: Regularizer instance

# Returns
- `Float64`: Smoothness constant

For bounded L2 regularization, the smoothness constant equals the coefficient.
"""
function smoothness(reg::BoundedL2Regularizer)
    return reg.bounded_l2_coef
end

"""
    hessian(reg::BoundedL2Regularizer, x::Vector{Float64})

Compute the Hessian of the bounded L2 regularization.

# Arguments
- `reg::BoundedL2Regularizer`: Regularizer instance
- `x::Vector{Float64}`: Input vector

# Returns
- `Matrix{Float64}`: Hessian matrix (diagonal)

# Formula
∇²R(x)_ii = bounded_l2_coef * (1 - 3*x_i^2) / (x_i^2 + 1)^3
∇²R(x)_ij = 0 for i ≠ j
"""
function hessian(reg::BoundedL2Regularizer, x::Vector{Float64})
    n = length(x)
    H = zeros(n, n)
    
    for i in 1:n
        x2 = x[i] * x[i]
        denominator = (x2 + 1)^3
        H[i, i] = reg.bounded_l2_coef * (1 - 3*x2) / denominator
    end
    
    return H
end

"""
    hessian_diagonal(reg::BoundedL2Regularizer, x::Vector{Float64})

Compute the diagonal of the Hessian matrix.

# Arguments
- `reg::BoundedL2Regularizer`: Regularizer instance
- `x::Vector{Float64}`: Input vector

# Returns
- `Vector{Float64}`: Diagonal elements of the Hessian

This is more efficient than computing the full Hessian when only the diagonal is needed.
"""
function hessian_diagonal(reg::BoundedL2Regularizer, x::Vector{Float64})
    x2 = x .* x
    denominator = (x2 .+ 1) .^ 3
    return reg.bounded_l2_coef * (1 .- 3 .* x2) ./ denominator
end

"""
    is_zero(reg::BoundedL2Regularizer)

Check if regularizer is effectively zero.

# Arguments
- `reg::BoundedL2Regularizer`: Regularizer instance

# Returns
- `Bool`: True if bounded L2 coefficient is zero
"""
function is_zero(reg::BoundedL2Regularizer)
    return reg.bounded_l2_coef == 0.0
end

# Export the main type and functions
export BoundedL2Regularizer, value, gradient, prox, prox_l1, prox_l2, smoothness, hessian, hessian_diagonal, is_zero