"""
Linear Regression Loss Function Implementation

This module implements linear regression loss functions for optimization,
including regularized variants with L1 and L2 penalties.

The loss function is defined as:
f(x) = (1/2n) * ||Ax - b||² + λ₁||x||₁ + (λ₂/2)||x||²

where:
- A is the design matrix (n × d)
- b is the target vector (n × 1)
- x is the parameter vector (d × 1)
- λ₁ is the L1 regularization parameter
- λ₂ is the L2 regularization parameter
- n is the number of samples
- d is the number of features
"""

using LinearAlgebra
using SparseArrays
using Statistics
using Random

"""
    LinearRegressionLoss

Mutable struct to store linear regression problem data and regularization parameters.

# Fields
- `A::Matrix{Float64}`: Design matrix (n × d)
- `b::Vector{Float64}`: Target vector (n × 1)
- `l1::Float64`: L1 regularization parameter (default: 0.0)
- `l2::Float64`: L2 regularization parameter (default: 0.0)
- `n::Int`: Number of samples
- `d::Int`: Number of features
- `store_mat_vec_prod::Bool`: Whether to cache matrix-vector products
- `x_last::Vector{Float64}`: Last x for which mat-vec product was computed
- `mat_vec_prod::Vector{Float64}`: Cached matrix-vector product Ax
"""
mutable struct LinearRegressionLoss
    A::Matrix{Float64}
    b::Vector{Float64}
    l1::Float64
    l2::Float64
    n::Int
    d::Int
    store_mat_vec_prod::Bool
    x_last::Vector{Float64}
    mat_vec_prod::Vector{Float64}
    
    function LinearRegressionLoss(A::Matrix{Float64}, b::Vector{Float64}; 
                                 l1::Float64=0.0, l2::Float64=0.0, 
                                 store_mat_vec_prod::Bool=true)
        n, d = size(A)
        
        # Validate dimensions
        if length(b) != n
            throw(DimensionMismatch("Length of b ($(length(b))) must match number of rows in A ($n)"))
        end
        
        # Initialize cache variables
        x_last = zeros(d)
        mat_vec_prod = zeros(n)
        
        new(A, b, l1, l2, n, d, store_mat_vec_prod, x_last, mat_vec_prod)
    end
end

"""
    mat_vec_product(lrl::LinearRegressionLoss, x::Vector{Float64})

Compute matrix-vector product Ax with optional caching.

# Arguments
- `lrl::LinearRegressionLoss`: Linear regression loss object
- `x::Vector{Float64}`: Input vector

# Returns
- `Vector{Float64}`: Matrix-vector product Ax
"""
function mat_vec_product(lrl::LinearRegressionLoss, x::Vector{Float64})::Vector{Float64}
    if !lrl.store_mat_vec_prod || norm(x - lrl.x_last) != 0
        z = lrl.A * x
        if lrl.store_mat_vec_prod
            lrl.mat_vec_prod = z
            lrl.x_last = copy(x)
        end
        return z
    else
        return lrl.mat_vec_prod
    end
end

"""
    linear_regression_loss(lrl::LinearRegressionLoss, x::Vector{Float64})

Compute the linear regression loss function value.

# Arguments
- `lrl::LinearRegressionLoss`: Linear regression loss object
- `x::Vector{Float64}`: Parameter vector

# Returns
- `Float64`: Loss function value

# Formula
f(x) = (1/2n) * ||Ax - b||² + λ₁||x||₁ + (λ₂/2)||x||²
"""
function linear_regression_loss(lrl::LinearRegressionLoss, x::Vector{Float64})::Float64
    residual = mat_vec_product(lrl, x) - lrl.b
    mse_loss = 0.5 * norm(residual)^2 / lrl.n
    
    # Add regularization
    l1_reg = lrl.l1 * norm(x, 1)
    l2_reg = 0.5 * lrl.l2 * norm(x)^2
    
    return mse_loss + l1_reg + l2_reg
end

"""
    linear_regression_gradient(lrl::LinearRegressionLoss, x::Vector{Float64})

Compute the gradient of the linear regression loss function.

# Arguments
- `lrl::LinearRegressionLoss`: Linear regression loss object
- `x::Vector{Float64}`: Parameter vector

# Returns
- `Vector{Float64}`: Gradient vector

# Formula
∇f(x) = (1/n) * A^T(Ax - b) + λ₁ * sign(x) + λ₂ * x

Note: For L1 regularization, we use the subgradient (sign function).
"""
function linear_regression_gradient(lrl::LinearRegressionLoss, x::Vector{Float64})::Vector{Float64}
    residual = mat_vec_product(lrl, x) - lrl.b
    grad = lrl.A' * residual / lrl.n
    
    # Add L2 regularization
    grad += lrl.l2 * x
    
    # Add L1 regularization (subgradient)
    if lrl.l1 > 0
        grad += lrl.l1 * sign.(x)
    end
    
    return grad
end

"""
    linear_regression_hessian(lrl::LinearRegressionLoss, x::Vector{Float64})

Compute the Hessian of the linear regression loss function.

# Arguments
- `lrl::LinearRegressionLoss`: Linear regression loss object
- `x::Vector{Float64}`: Parameter vector (not used for linear regression, but kept for consistency)

# Returns
- `Matrix{Float64}`: Hessian matrix

# Formula
∇²f(x) = (1/n) * A^T * A + λ₂ * I

Note: L1 regularization contributes zero to the Hessian (except at x=0 where it's undefined).
"""
function linear_regression_hessian(lrl::LinearRegressionLoss, x::Vector{Float64})::Matrix{Float64}
    hessian = lrl.A' * lrl.A / lrl.n
    
    # Add L2 regularization
    if lrl.l2 > 0
        hessian += lrl.l2 * I(lrl.d)
    end
    
    return hessian
end

"""
    linear_regression_stochastic_gradient(lrl::LinearRegressionLoss, x::Vector{Float64}, 
                                         idx::Union{Vector{Int}, Nothing}=nothing;
                                         batch_size::Int=1, replace::Bool=false)

Compute stochastic gradient using a batch of samples.

# Arguments
- `lrl::LinearRegressionLoss`: Linear regression loss object
- `x::Vector{Float64}`: Parameter vector
- `idx::Union{Vector{Int}, Nothing}`: Indices of samples to use (if nothing, randomly sample)
- `batch_size::Int`: Size of the batch (default: 1)
- `replace::Bool`: Whether to sample with replacement (default: false)

# Returns
- `Vector{Float64}`: Stochastic gradient vector

# Formula
∇f_batch(x) = (1/|batch|) * A_batch^T(A_batch*x - b_batch) + λ₁ * sign(x) + λ₂ * x
"""
function linear_regression_stochastic_gradient(lrl::LinearRegressionLoss, x::Vector{Float64}, 
                                             idx::Union{Vector{Int}, Nothing}=nothing;
                                             batch_size::Int=1, replace::Bool=false)::Vector{Float64}
    # Sample indices if not provided
    if idx === nothing
        idx = sample(1:lrl.n, batch_size; replace=replace)
    end
    
    # Compute stochastic gradient
    A_batch = lrl.A[idx, :]
    b_batch = lrl.b[idx]
    residual = A_batch * x - b_batch
    stoch_grad = A_batch' * residual / length(idx)
    
    # Add L2 regularization
    stoch_grad += lrl.l2 * x
    
    # Add L1 regularization (subgradient)
    if lrl.l1 > 0
        stoch_grad += lrl.l1 * sign.(x)
    end
    
    return stoch_grad
end

"""
    linear_regression_smoothness(lrl::LinearRegressionLoss)

Compute the smoothness constant (largest eigenvalue of the Hessian).

# Arguments
- `lrl::LinearRegressionLoss`: Linear regression loss object

# Returns
- `Float64`: Smoothness constant

The smoothness constant is λ_max(A^T*A/n) + λ₂
"""
function linear_regression_smoothness(lrl::LinearRegressionLoss)::Float64
    covariance = lrl.A' * lrl.A / lrl.n
    max_eigenvalue = maximum(eigvals(covariance))
    return max_eigenvalue + lrl.l2
end

"""
    linear_regression_max_smoothness(lrl::LinearRegressionLoss)

Compute the maximum smoothness constant over all samples.

# Arguments
- `lrl::LinearRegressionLoss`: Linear regression loss object

# Returns
- `Float64`: Maximum smoothness constant

This is the maximum row norm squared of A plus λ₂.
"""
function linear_regression_max_smoothness(lrl::LinearRegressionLoss)::Float64
    max_row_norm_squared = maximum(sum(abs2, lrl.A; dims=2))
    return max_row_norm_squared + lrl.l2
end

"""
    linear_regression_average_smoothness(lrl::LinearRegressionLoss)

Compute the average smoothness constant over all samples.

# Arguments
- `lrl::LinearRegressionLoss`: Linear regression loss object

# Returns
- `Float64`: Average smoothness constant

This is the average row norm squared of A plus λ₂.
"""
function linear_regression_average_smoothness(lrl::LinearRegressionLoss)::Float64
    avg_row_norm_squared = mean(sum(abs2, lrl.A; dims=2))
    return avg_row_norm_squared + lrl.l2
end

# Convenience functions that work with matrices directly (similar to Rosenbrock/Rastrigin)

"""
    linear_regression_simple(A::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64}; 
                            l1::Float64=0.0, l2::Float64=0.0)

Simple linear regression loss function that works directly with matrices.

# Arguments
- `A::Matrix{Float64}`: Design matrix
- `b::Vector{Float64}`: Target vector
- `x::Vector{Float64}`: Parameter vector
- `l1::Float64`: L1 regularization parameter (default: 0.0)
- `l2::Float64`: L2 regularization parameter (default: 0.0)

# Returns
- `Float64`: Loss function value
"""
function linear_regression_simple(A::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64}; 
                                l1::Float64=0.0, l2::Float64=0.0)::Float64
    n = size(A, 1)
    residual = A * x - b
    mse_loss = 0.5 * norm(residual)^2 / n
    l1_reg = l1 * norm(x, 1)
    l2_reg = 0.5 * l2 * norm(x)^2
    return mse_loss + l1_reg + l2_reg
end

"""
    linear_regression_gradient_simple(A::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64}; 
                                    l1::Float64=0.0, l2::Float64=0.0)

Simple linear regression gradient function that works directly with matrices.

# Arguments
- `A::Matrix{Float64}`: Design matrix
- `b::Vector{Float64}`: Target vector
- `x::Vector{Float64}`: Parameter vector
- `l1::Float64`: L1 regularization parameter (default: 0.0)
- `l2::Float64`: L2 regularization parameter (default: 0.0)

# Returns
- `Vector{Float64}`: Gradient vector
"""
function linear_regression_gradient_simple(A::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64}; 
                                         l1::Float64=0.0, l2::Float64=0.0)::Vector{Float64}
    n = size(A, 1)
    residual = A * x - b
    grad = A' * residual / n + l2 * x
    
    if l1 > 0
        grad += l1 * sign.(x)
    end
    
    return grad
end

"""
    linear_regression_hessian_simple(A::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64}; 
                                    l1::Float64=0.0, l2::Float64=0.0)

Simple linear regression Hessian function that works directly with matrices.

# Arguments
- `A::Matrix{Float64}`: Design matrix
- `b::Vector{Float64}`: Target vector
- `x::Vector{Float64}`: Parameter vector (not used but kept for consistency)
- `l1::Float64`: L1 regularization parameter (default: 0.0, not used in Hessian)
- `l2::Float64`: L2 regularization parameter (default: 0.0)

# Returns
- `Matrix{Float64}`: Hessian matrix
"""
function linear_regression_hessian_simple(A::Matrix{Float64}, b::Vector{Float64}, x::Vector{Float64}; 
                                        l1::Float64=0.0, l2::Float64=0.0)::Matrix{Float64}
    n, d = size(A)
    hessian = A' * A / n
    
    if l2 > 0
        hessian += l2 * I(d)
    end
    
    return hessian
end

# Helper function for sampling (simple implementation)
function sample(range::UnitRange{Int}, n::Int; replace::Bool=false)
    if replace
        return rand(range, n)
    else
        if n > length(range)
            throw(ArgumentError("Cannot sample $n elements without replacement from range of length $(length(range))"))
        end
        return shuffle(collect(range))[1:n]
    end
end