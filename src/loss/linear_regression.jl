"""
Linear Regression Oracle Implementation

This module implements linear regression loss functions using the Oracle base class,
including regularized variants with L1 and L2 penalties.

The loss function is defined as:
f(x) = (1/2n) * ||Ax - b||² + regularization

where:
- A is the design matrix (n × d)
- b is the target vector (n × 1)
- x is the parameter vector (d × 1)
- n is the number of samples
- d is the number of features
"""

using LinearAlgebra
using SparseArrays
using Statistics
using Random

include("loss_oracle.jl")

"""
    LinearRegressionOracle

Linear regression oracle that returns loss values, gradients and Hessians.
Inherits from Oracle base class for regularization support.

# Fields
- `A::Matrix{Float64}`: Design matrix (n × d)
- `b::Vector{Float64}`: Target vector (n × 1)
- `n::Int`: Number of samples
- `d::Int`: Number of features
- `store_mat_vec_prod::Bool`: Whether to cache matrix-vector products
- `x_last::Vector{Float64}`: Last x for which mat-vec product was computed
- `mat_vec_prod::Vector{Float64}`: Cached matrix-vector product Ax
"""
mutable struct LinearRegressionOracle <: AbstractOracle
    oracle::Oracle  # Base oracle for regularization
    A::Matrix{Float64}
    b::Vector{Float64}
    n::Int
    d::Int
    store_mat_vec_prod::Bool
    x_last::Vector{Float64}
    mat_vec_prod::Vector{Float64}
    
    function LinearRegressionOracle(A::Matrix{Float64}, b::Vector{Float64}; 
                                   l1::Float64=0.0, l2::Float64=0.0, 
                                   l2_in_prox::Bool=false,
                                   store_mat_vec_prod::Bool=true,
                                   seed::Int=42)
        n, d = size(A)
        
        # Validate dimensions
        if length(b) != n
            throw(DimensionMismatch("Length of b ($(length(b))) must match number of rows in A ($n)"))
        end
        
        # Handle label transformations (from Python version)
        b_processed = copy(b)
        unique_vals = unique(b)
        if length(unique_vals) == 2 && Set(unique_vals) == Set([1, 2])
            # Transform labels {1, 2} to {0, 1}
            b_processed = b .- 1
        elseif length(unique_vals) == 2 && Set(unique_vals) == Set([-1, 1])
            # Transform labels {-1, 1} to {0, 1}
            b_processed = (b .+ 1) ./ 2
        elseif !(Set(unique_vals) ⊆ Set([0, 1]))
            # For continuous targets, keep as is
            b_processed = b
        end
        
        # Create base oracle with regularization
        base_oracle = Oracle(l1=l1, l2=l2, l2_in_prox=l2_in_prox, seed=seed)
        
        # Initialize cache variables
        x_last = zeros(d)
        mat_vec_prod = zeros(n)
        
        new(base_oracle, A, b_processed, n, d, store_mat_vec_prod, x_last, mat_vec_prod)
    end
end

# Forward Oracle methods to base oracle
for method in [:set_seed!, :get_best_point, :reset_best!]
    @eval $method(lro::LinearRegressionOracle, args...; kwargs...) = $method(lro.oracle, args...; kwargs...)
end

# Forward properties
function Base.getproperty(lro::LinearRegressionOracle, prop::Symbol)
    if prop in fieldnames(LinearRegressionOracle)
        return getfield(lro, prop)
    else
        # Forward to base oracle
        return getproperty(lro.oracle, prop)
    end
end

function Base.setproperty!(lro::LinearRegressionOracle, prop::Symbol, value)
    if prop in fieldnames(LinearRegressionOracle)
        setfield!(lro, prop, value)
    else
        # Forward to base oracle
        setproperty!(lro.oracle, prop, value)
    end
end

"""
    mat_vec_product(lro::LinearRegressionOracle, x::Vector{Float64})

Compute matrix-vector product Ax with optional caching.

# Arguments
- `lro::LinearRegressionOracle`: Linear regression oracle
- `x::Vector{Float64}`: Input vector

# Returns
- `Vector{Float64}`: Matrix-vector product Ax
"""
function mat_vec_product(lro::LinearRegressionOracle, x::Vector{Float64})::Vector{Float64}
    if !lro.store_mat_vec_prod || safe_sparse_norm(x - lro.x_last) != 0
        z = lro.A * x
        if lro.store_mat_vec_prod
            lro.mat_vec_prod = z
            lro.x_last = copy(x)
        end
        return z
    else
        return lro.mat_vec_prod
    end
end

"""
    _value(lro::LinearRegressionOracle, x::Vector{Float64})

Compute the base linear regression loss (without regularization).

# Arguments
- `lro::LinearRegressionOracle`: Linear regression oracle
- `x::Vector{Float64}`: Parameter vector

# Returns
- `Float64`: Base loss function value

# Formula
f(x) = (1/2n) * ||Ax - b||²
"""
function _value(lro::LinearRegressionOracle, x::Vector{Float64})::Float64
    residual = mat_vec_product(lro, x) - lro.b
    return 0.5 * safe_sparse_norm(residual)^2 / lro.n
end

"""
    value(lro::LinearRegressionOracle, x::Vector{Float64})

Compute the full linear regression loss including regularization.

# Arguments
- `lro::LinearRegressionOracle`: Linear regression oracle
- `x::Vector{Float64}`: Parameter vector

# Returns
- `Float64`: Full loss function value including regularization
"""
function value(lro::LinearRegressionOracle, x::Vector{Float64})::Float64
    # Compute base value
    base_value = _value(lro, x)
    
    # Add regularization
    total_value = base_value
    if lro.oracle.regularizer !== nothing
        total_value += lro.oracle.regularizer(x)
    else
        # Add L2 regularization if not in proximal operator
        if lro.oracle.l2 > 0
            total_value += 0.5 * lro.oracle.l2 * safe_sparse_norm(x)^2
        end
    end
    
    # Track best solution
    if total_value < lro.oracle.f_opt
        lro.oracle.x_opt = copy(x)
        lro.oracle.f_opt = total_value
    end
    
    return total_value
end

"""
    gradient(lro::LinearRegressionOracle, x::Vector{Float64})

Compute the gradient of the linear regression loss.

# Arguments
- `lro::LinearRegressionOracle`: Linear regression oracle
- `x::Vector{Float64}`: Parameter vector

# Returns
- `Vector{Float64}`: Gradient vector

# Formula
∇f(x) = (1/n) * A^T(Ax - b) + regularization_gradient
"""
function gradient(lro::LinearRegressionOracle, x::Vector{Float64})::Vector{Float64}
    residual = mat_vec_product(lro, x) - lro.b
    grad = lro.A' * residual / lro.n
    
    # Add L2 regularization gradient (if not in proximal)
    if lro.oracle.l2 > 0
        grad = safe_sparse_add(grad, lro.oracle.l2 * x)
    end
    
    # Add L1 regularization (subgradient)
    if lro.oracle.l1 > 0
        grad = safe_sparse_add(grad, lro.oracle.l1 * sign.(x))
    end
    
    return grad
end

"""
    hessian(lro::LinearRegressionOracle, x::Vector{Float64})

Compute the Hessian of the linear regression loss.

# Arguments
- `lro::LinearRegressionOracle`: Linear regression oracle
- `x::Vector{Float64}`: Parameter vector (not used for linear regression, but kept for consistency)

# Returns
- `Matrix{Float64}`: Hessian matrix

# Formula
∇²f(x) = (1/n) * A^T * A + λ₂ * I

Note: L1 regularization contributes zero to the Hessian (except at x=0 where it's undefined).
"""
function hessian(lro::LinearRegressionOracle, x::Vector{Float64})::Matrix{Float64}
    hessian = lro.A' * lro.A / lro.n
    
    # Add L2 regularization
    if lro.oracle.l2 > 0
        hessian += lro.oracle.l2 * I(lro.d)
    end
    
    return hessian
end

"""
    stochastic_gradient(lro::LinearRegressionOracle, x::Vector{Float64}, 
                       idx::Union{Vector{Int}, Nothing}=nothing;
                       batch_size::Int=1, replace::Bool=false)

Compute stochastic gradient using a batch of samples.

# Arguments
- `lro::LinearRegressionOracle`: Linear regression oracle
- `x::Vector{Float64}`: Parameter vector
- `idx::Union{Vector{Int}, Nothing}`: Indices of samples to use (if nothing, randomly sample)
- `batch_size::Int`: Size of the batch (default: 1)
- `replace::Bool`: Whether to sample with replacement (default: false)

# Returns
- `Vector{Float64}`: Stochastic gradient vector

# Formula
∇f_batch(x) = (1/|batch|) * A_batch^T(A_batch*x - b_batch) + regularization_gradient
"""
function stochastic_gradient(lro::LinearRegressionOracle, x::Vector{Float64}, 
                            idx::Union{Vector{Int}, Nothing}=nothing;
                            batch_size::Int=1, replace::Bool=false)::Vector{Float64}
    # Sample indices if not provided
    if idx === nothing
        if replace
            idx = rand(lro.oracle.rng, 1:lro.n, batch_size)
        else
            if batch_size > lro.n
                throw(ArgumentError("Cannot sample $batch_size elements without replacement from $lro.n samples"))
            end
            idx = shuffle(lro.oracle.rng, collect(1:lro.n))[1:batch_size]
        end
    end
    
    # Compute stochastic gradient
    A_batch = lro.A[idx, :]
    b_batch = lro.b[idx]
    residual = A_batch * x - b_batch
    stoch_grad = A_batch' * residual / length(idx)
    
    # Add L2 regularization
    if lro.oracle.l2 > 0
        stoch_grad = safe_sparse_add(stoch_grad, lro.oracle.l2 * x)
    end
    
    # Add L1 regularization (subgradient)
    if lro.oracle.l1 > 0
        stoch_grad = safe_sparse_add(stoch_grad, lro.oracle.l1 * sign.(x))
    end
    
    return stoch_grad
end

"""
    smoothness(lro::LinearRegressionOracle)

Compute the smoothness constant (largest eigenvalue of the Hessian).

# Arguments
- `lro::LinearRegressionOracle`: Linear regression oracle

# Returns
- `Float64`: Smoothness constant

The smoothness constant is λ_max(A^T*A/n) + λ₂
"""
function smoothness(lro::LinearRegressionOracle)::Float64
    if lro.oracle._smoothness === nothing
        covariance = lro.A' * lro.A / lro.n
        max_eigenvalue = maximum(eigvals(covariance))
        lro.oracle._smoothness = max_eigenvalue + lro.oracle.l2
    end
    return lro.oracle._smoothness
end

"""
    max_smoothness(lro::LinearRegressionOracle)

Compute the maximum smoothness constant over all samples.

# Arguments
- `lro::LinearRegressionOracle`: Linear regression oracle

# Returns
- `Float64`: Maximum smoothness constant

This is the maximum row norm squared of A plus λ₂.
"""
function max_smoothness(lro::LinearRegressionOracle)::Float64
    if lro.oracle._max_smoothness === nothing
        max_row_norm_squared = maximum(sum(abs2, lro.A; dims=2))
        lro.oracle._max_smoothness = max_row_norm_squared + lro.oracle.l2
    end
    return lro.oracle._max_smoothness
end

"""
    average_smoothness(lro::LinearRegressionOracle)

Compute the average smoothness constant over all samples.

# Arguments
- `lro::LinearRegressionOracle`: Linear regression oracle

# Returns
- `Float64`: Average smoothness constant

This is the average row norm squared of A plus λ₂.
"""
function average_smoothness(lro::LinearRegressionOracle)::Float64
    if lro.oracle._ave_smoothness === nothing
        avg_row_norm_squared = mean(sum(abs2, lro.A; dims=2))
        lro.oracle._ave_smoothness = avg_row_norm_squared + lro.oracle.l2
    end
    return lro.oracle._ave_smoothness
end

# Convenience functions that work with matrices directly (for backward compatibility)

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
    oracle = LinearRegressionOracle(A, b, l1=l1, l2=l2)
    return value(oracle, x)
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
    oracle = LinearRegressionOracle(A, b, l1=l1, l2=l2)
    return gradient(oracle, x)
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
    oracle = LinearRegressionOracle(A, b, l1=l1, l2=l2)
    return hessian(oracle, x)
end

# For backward compatibility, keep the old struct name as an alias
const LinearRegressionLoss = LinearRegressionOracle

# Legacy function names
const linear_regression_loss = linear_regression_simple
const linear_regression_gradient = linear_regression_gradient_simple
const linear_regression_hessian = linear_regression_hessian_simple

# Export the main oracle type and functions
export LinearRegressionOracle, LinearRegressionLoss
export mat_vec_product, stochastic_gradient
export smoothness, max_smoothness, average_smoothness