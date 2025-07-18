"""
Regularizer implementation for Joptan.jl

This module provides regularization functionality including L1, L2 regularization
and their proximal operators.
"""

using LinearAlgebra
using SparseArrays
include("utils.jl")

"""
    Regularizer

A simple oracle class for regularizers that have proximal operator
and can be evaluated during training. By default, L1+L2 regularization is implemented.

# Fields
- `l1::Float64`: L1 regularization coefficient
- `l2::Float64`: L2 regularization coefficient  
- `coef::Union{Vector{Float64}, Nothing}`: Optional coefficient vector
"""
mutable struct Regularizer
    l1::Float64
    l2::Float64
    coef::Union{Vector{Float64}, Nothing}
    
    function Regularizer(; l1::Float64=0.0, l2::Float64=0.0, coef::Union{Vector{Float64}, Nothing}=nothing)
        if l1 < 0.0
            throw(ArgumentError("L1 regularization coefficient must be non-negative, got $l1"))
        end
        if l2 < 0.0
            throw(ArgumentError("L2 regularization coefficient must be non-negative, got $l2"))
        end
        new(l1, l2, coef)
    end
end

"""
    (reg::Regularizer)(x)

Function call operator for regularizer evaluation.

# Arguments
- `x::Vector{Float64}`: Input vector

# Returns
- `Float64`: Regularization value
"""
function (reg::Regularizer)(x::Vector{Float64})
    return value(reg, x)
end

"""
    value(reg::Regularizer, x::Vector{Float64})

Compute the regularization value.

# Arguments
- `reg::Regularizer`: Regularizer instance
- `x::Vector{Float64}`: Input vector

# Returns
- `Float64`: L1 + L2 regularization value

# Formula
reg(x) = l1 * ||x||₁ + (l2/2) * ||x||₂²
"""
function value(reg::Regularizer, x::Vector{Float64})
    l1_term = reg.l1 * safe_sparse_norm(x, ord=1)
    l2_term = 0.5 * reg.l2 * safe_sparse_norm(x)^2
    return l1_term + l2_term
end

"""
    prox_l1(reg::Regularizer, x::Vector{Float64}, lr::Float64)

Compute the proximal operator of L1 regularization.

# Arguments
- `reg::Regularizer`: Regularizer instance
- `x::Vector{Float64}`: Input vector
- `lr::Float64`: Learning rate/step size

# Returns
- `Vector{Float64}`: Result of proximal operator

# Formula
prox_l1(x) = sign(x) * max(|x| - l1*lr, 0)
"""
function prox_l1(reg::Regularizer, x::Vector{Float64}, lr::Float64)
    threshold = reg.l1 * lr
    
    if issparse(x)
        # Handle sparse case
        abs_x = abs.(x)
        prox_res = max.(abs_x .- threshold, 0.0)
        prox_res .*= sign.(x)
        # Remove zeros for sparsity
        return prox_res
    else
        # Dense case
        abs_x = abs.(x)
        prox_res = max.(abs_x .- threshold, 0.0)
        prox_res .*= sign.(x)
        return prox_res
    end
end

"""
    prox_l2(reg::Regularizer, x::Vector{Float64}, lr::Float64)

Compute the proximal operator of L2 regularization.

# Arguments
- `reg::Regularizer`: Regularizer instance
- `x::Vector{Float64}`: Input vector
- `lr::Float64`: Learning rate/step size

# Returns
- `Vector{Float64}`: Result of proximal operator

# Formula
prox_l2(x) = x / (1 + lr * l2)
"""
function prox_l2(reg::Regularizer, x::Vector{Float64}, lr::Float64)
    return x ./ (1.0 + lr * reg.l2)
end

"""
    prox(reg::Regularizer, x::Vector{Float64}, lr::Float64)

Compute the proximal operator of L1 + L2 regularization.

The proximal operator of l1||x||₁ + l2/2||x||₂² is equal to the
combination of the proximal operator of l1||x||₁ and then the 
proximal operator of l2/2||x||₂².

# Arguments
- `reg::Regularizer`: Regularizer instance
- `x::Vector{Float64}`: Input vector
- `lr::Float64`: Learning rate/step size

# Returns
- `Vector{Float64}`: Result of combined proximal operator
"""
function prox(reg::Regularizer, x::Vector{Float64}, lr::Float64)
    # Apply L1 prox first
    prox_l1_result = prox_l1(reg, x, lr)
    
    # Then apply L2 prox
    return prox_l2(reg, prox_l1_result, lr)
end

"""
    gradient(reg::Regularizer, x::Vector{Float64})

Compute the gradient of the regularizer (where it exists).

# Arguments
- `reg::Regularizer`: Regularizer instance
- `x::Vector{Float64}`: Input vector

# Returns
- `Vector{Float64}`: Gradient (only L2 part, L1 is non-differentiable at 0)

# Note
This only returns the L2 gradient. L1 regularization is non-differentiable
at zero and requires subgradient methods.
"""
function gradient(reg::Regularizer, x::Vector{Float64})
    # Only L2 part is differentiable everywhere
    l2_grad = reg.l2 * x
    
    # L1 subgradient (sign function, undefined at 0)
    l1_subgrad = reg.l1 * sign.(x)
    
    return l2_grad + l1_subgrad
end

"""
    is_zero(reg::Regularizer)

Check if regularizer is effectively zero.

# Arguments
- `reg::Regularizer`: Regularizer instance

# Returns
- `Bool`: True if both L1 and L2 coefficients are zero
"""
function is_zero(reg::Regularizer)
    return reg.l1 == 0.0 && reg.l2 == 0.0
end