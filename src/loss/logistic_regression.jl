"""
Logistic Regression Oracle Implementation

Implements logistic regression loss functions using the Oracle base class,
including regularized variants with L1 and L2 penalties.

The loss function is defined as:
f(x) = (1/n) * sum(log(1 + exp(-y_i * (a_i^T x)))) + regularization

where:
- A is the design matrix (n × d)
- b is the binary target vector (n × 1) with values in {0, 1}
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
    logsig(x::Real)

Compute the log-sigmoid function: log(sigmoid(x)) = log(1/(1+exp(-x)))
Uses numerically stable implementation.
See http://fa.bianp.net/blog/2019/evaluate_logistic/ for more details.

# Arguments
- `x::Real`: Input value

# Returns
- `Real`: log(sigmoid(x))
"""
function logsig(x::Real)
    if x < -33
        return x
    elseif x >= -33 && x < -18
        return x - exp(x)
    elseif x >= -18 && x < 37
        return -log1p(exp(-x))
    else  # x >= 37
        return -exp(-x)
    end
end

"""
    logsig(x::AbstractVector)

Vectorized version of logsig function.
"""
function logsig(x::AbstractVector)
    return [logsig(xi) for xi in x]
end

"""
    sigmoid(x::Real)

Compute the sigmoid function: 1/(1+exp(-x))
Uses numerically stable implementation.
"""
function sigmoid(x::Real)
    if x >= 0
        exp_neg_x = exp(-x)
        return 1.0 / (1.0 + exp_neg_x)
    else
        exp_x = exp(x)
        return exp_x / (1.0 + exp_x)
    end
end

"""
    sigmoid(x::AbstractVector)

Vectorized version of sigmoid function.
"""
function sigmoid(x::AbstractVector)
    return [sigmoid(xi) for xi in x]
end

"""
    LogisticRegressionOracle

Logistic regression oracle that returns loss values, gradients, Hessians,
their stochastic analogues as well as smoothness constants.
Inherits from Oracle base class for regularization support.

# Fields
- `A::Matrix{Float64}`: Design matrix (n × d)
- `b::Vector{Float64}`: Binary target vector (n × 1) with values in {0, 1}
- `n::Int`: Number of samples
- `d::Int`: Number of features
- `store_mat_vec_prod::Bool`: Whether to cache matrix-vector products
- `x_last::Vector{Float64}`: Last x for which mat-vec product was computed
- `mat_vec_prod::Vector{Float64}`: Cached matrix-vector product Ax
- `_individ_smoothness::Union{Vector{Float64}, Nothing}`: Individual smoothness constants
- `_importance_probs::Union{Vector{Float64}, Nothing}`: Importance sampling probabilities
"""
mutable struct LogisticRegressionOracle <: AbstractOracle
    oracle::Oracle  # Base oracle for regularization
    A::Matrix{Float64}
    b::Vector{Float64}
    n::Int
    d::Int
    store_mat_vec_prod::Bool
    x_last::Vector{Float64}
    mat_vec_prod::Vector{Float64}
    _individ_smoothness::Union{Vector{Float64}, Nothing}
    _importance_probs::Union{Vector{Float64}, Nothing}
    
    function LogisticRegressionOracle(A::Matrix{Float64}, b::Vector{Float64}; 
                                     l1::Float64=0.0, l2::Float64=0.0, 
                                     l2_in_prox::Bool=false,
                                     store_mat_vec_prod::Bool=true,
                                     seed::Int=42)
        n, d = size(A)
        
        # Validate dimensions
        if length(b) != n
            throw(DimensionMismatch("Length of b ($(length(b))) must match number of rows in A ($n)"))
        end
        
        # Handle label transformations
        b_processed = copy(b)
        unique_vals = unique(b)
        
        if length(unique_vals) == 1
            @warn "The labels have only one unique value."
        elseif length(unique_vals) > 2
            throw(ArgumentError("The number of classes must be no more than 2 for binary classification."))
        elseif length(unique_vals) == 2 && !issubset(unique_vals, [0.0, 1.0])
            if Set(unique_vals) == Set([1.0, 2.0])
                println("The passed labels have values in the set {1, 2}. Changing them to {0, 1}")
                b_processed = b .- 1.0
            elseif Set(unique_vals) == Set([-1.0, 1.0])
                println("The passed labels have values in the set {-1, 1}. Changing them to {0, 1}")
                b_processed = (b .+ 1.0) ./ 2.0
            else
                println("Changing the labels from $(unique_vals[1]) to 1s and the rest to 0s")
                b_processed = Float64.(b .== unique_vals[1])
            end
        end
        
        # Create base oracle with regularization
        base_oracle = Oracle(l1=l1, l2=l2, l2_in_prox=l2_in_prox, seed=seed)
        
        # Initialize cache variables
        x_last = zeros(d)
        mat_vec_prod = zeros(n)
        
        new(base_oracle, A, b_processed, n, d, store_mat_vec_prod, x_last, mat_vec_prod,
            nothing, nothing)
    end
end

# Forward Oracle methods to base oracle
for method in [:set_seed!, :get_best_point, :reset_best!]
    @eval $method(lro::LogisticRegressionOracle, args...; kwargs...) = $method(lro.oracle, args...; kwargs...)
end

# Forward properties
function Base.getproperty(lro::LogisticRegressionOracle, prop::Symbol)
    if prop in fieldnames(LogisticRegressionOracle)
        return getfield(lro, prop)
    else
        # Forward to base oracle
        return getproperty(lro.oracle, prop)
    end
end

function Base.setproperty!(lro::LogisticRegressionOracle, prop::Symbol, value)
    if prop in fieldnames(LogisticRegressionOracle)
        setfield!(lro, prop, value)
    else
        # Forward to base oracle
        setproperty!(lro.oracle, prop, value)
    end
end

"""
    mat_vec_product(lro::LogisticRegressionOracle, x::Vector{Float64})

Compute matrix-vector product Ax with optional caching.

# Arguments
- `lro::LogisticRegressionOracle`: Logistic regression oracle
- `x::Vector{Float64}`: Input vector

# Returns
- `Vector{Float64}`: Matrix-vector product Ax
"""
function mat_vec_product(lro::LogisticRegressionOracle, x::Vector{Float64})::Vector{Float64}
    if !lro.store_mat_vec_prod || safe_sparse_norm(x - lro.x_last) != 0
        z = lro.A * x
        if issparse(z)
            z = Array(z)
        end
        z = vec(z)  # Ensure it's a vector
        
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
    _value(lro::LogisticRegressionOracle, x::Vector{Float64})

Compute the base logistic regression loss (without regularization).

# Arguments
- `lro::LogisticRegressionOracle`: Logistic regression oracle
- `x::Vector{Float64}`: Parameter vector

# Returns
- `Float64`: Base loss function value

# Formula
f(x) = (1/n) * sum((1-b_i)*a_i^T*x - log(sigmoid(a_i^T*x)))
"""
function _value(lro::LogisticRegressionOracle, x::Vector{Float64})::Float64
    Ax = mat_vec_product(lro, x)
    
    # Compute: mean((1-b) .* Ax - logsig.(Ax))
    term1 = safe_sparse_multiply(1.0 .- lro.b, Ax)
    term2 = logsig(Ax)
    
    return mean(term1 - term2)
end

"""
    value(lro::LogisticRegressionOracle, x::Vector{Float64})

Compute the full logistic regression loss including regularization.

# Arguments
- `lro::LogisticRegressionOracle`: Logistic regression oracle
- `x::Vector{Float64}`: Parameter vector

# Returns
- `Float64`: Full loss function value including regularization
"""
function value(lro::LogisticRegressionOracle, x::Vector{Float64})::Float64
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
    partial_value(lro::LogisticRegressionOracle, x::Vector{Float64}, idx::Union{Int, Vector{Int}};
                  include_reg::Bool=true, normalization::Union{Int, Nothing}=nothing)

Compute the partial loss on a subset of samples.

# Arguments
- `lro::LogisticRegressionOracle`: Logistic regression oracle
- `x::Vector{Float64}`: Parameter vector
- `idx::Union{Int, Vector{Int}}`: Indices of samples to use
- `include_reg::Bool`: Whether to include regularization (default: true)
- `normalization::Union{Int, Nothing}`: Normalization factor (default: batch size)

# Returns
- `Float64`: Partial loss value
"""
function partial_value(lro::LogisticRegressionOracle, x::Vector{Float64}, 
                      idx::Union{Int, Vector{Int}};
                      include_reg::Bool=true, 
                      normalization::Union{Int, Nothing}=nothing)::Float64
    
    idx_vec = isa(idx, Int) ? [idx] : idx
    batch_size = length(idx_vec)
    
    if normalization === nothing
        normalization = batch_size
    end
    
    Ax = lro.A[idx_vec, :] * x
    if issparse(Ax)
        Ax = vec(Array(Ax))
    end
    
    # Compute partial loss
    term1 = safe_sparse_multiply(1.0 .- lro.b[idx_vec], Ax)
    term2 = logsig(Ax)
    partial_loss = sum(term1 - term2) / normalization
    
    # Add regularization if requested
    if include_reg && lro.oracle.l2 > 0
        partial_loss += 0.5 * lro.oracle.l2 * safe_sparse_norm(x)^2
    end
    
    return partial_loss
end

"""
    gradient(lro::LogisticRegressionOracle, x::Vector{Float64})

Compute the gradient of the logistic regression loss.

# Arguments
- `lro::LogisticRegressionOracle`: Logistic regression oracle
- `x::Vector{Float64}`: Parameter vector

# Returns
- `Vector{Float64}`: Gradient vector

# Formula
∇f(x) = (1/n) * A^T(sigmoid(Ax) - b) + regularization_gradient
"""
function gradient(lro::LogisticRegressionOracle, x::Vector{Float64})::Vector{Float64}
    Ax = mat_vec_product(lro, x)
    activation = sigmoid(Ax)
    
    # Base gradient: A^T * (activation - b) / n
    grad = lro.A' * (activation - lro.b) / lro.n
    
    # Add L2 regularization gradient (if not in proximal)
    if lro.oracle.l2 > 0
        grad = safe_sparse_add(grad, lro.oracle.l2 * x)
    end
    
    # Add L1 regularization (subgradient)
    if lro.oracle.l1 > 0
        grad = safe_sparse_add(grad, lro.oracle.l1 * sign.(x))
    end
    
    # Handle sparse case
    if issparse(x)
        # Convert result to sparse if input was sparse
        return sparse(grad)
    end
    
    return grad
end

"""
    stochastic_gradient(lro::LogisticRegressionOracle, x::Vector{Float64}, 
                       idx::Union{Vector{Int}, Nothing}=nothing;
                       batch_size::Int=1, replace::Bool=false, 
                       normalization::Union{Int, Nothing}=nothing,
                       importance_sampling::Bool=false,
                       p::Union{Vector{Float64}, Nothing}=nothing)

Compute stochastic gradient using a batch of samples.

# Arguments
- `lro::LogisticRegressionOracle`: Logistic regression oracle
- `x::Vector{Float64}`: Parameter vector
- `idx::Union{Vector{Int}, Nothing}`: Indices of samples to use (if nothing, randomly sample)
- `batch_size::Int`: Size of the batch (default: 1)
- `replace::Bool`: Whether to sample with replacement (default: false)
- `normalization::Union{Int, Nothing}`: Normalization factor
- `importance_sampling::Bool`: Whether to use importance sampling (default: false)
- `p::Union{Vector{Float64}, Nothing}`: Importance sampling probabilities

# Returns
- `Vector{Float64}`: Stochastic gradient vector
"""
function stochastic_gradient(lro::LogisticRegressionOracle, x::Vector{Float64}, 
                            idx::Union{Vector{Int}, Nothing}=nothing;
                            batch_size::Int=1, replace::Bool=false,
                            normalization::Union{Int, Nothing}=nothing,
                            importance_sampling::Bool=false,
                            p::Union{Vector{Float64}, Nothing}=nothing)::Vector{Float64}
    
    # Handle full batch case
    if batch_size == lro.n
        return gradient(lro, x)
    end
    
    # Sample indices if not provided
    if idx === nothing
        if p === nothing && importance_sampling
            if lro._importance_probs === nothing
                lro._importance_probs = individ_smoothness(lro)
                lro._importance_probs ./= sum(lro._importance_probs)
            end
            p = lro._importance_probs
        end
        
        if replace
            if p === nothing
                idx = rand(lro.oracle.rng, 1:lro.n, batch_size)
            else
                # Importance sampling with replacement
                cumsum_p = cumsum(p)
                idx = Int[]
                for _ in 1:batch_size
                    r = rand(lro.oracle.rng)
                    push!(idx, findfirst(cumsum_p .>= r))
                end
            end
        else
            if batch_size > lro.n
                throw(ArgumentError("Cannot sample $batch_size elements without replacement from $(lro.n) samples"))
            end
            if p === nothing
                idx = shuffle(lro.oracle.rng, collect(1:lro.n))[1:batch_size]
            else
                @warn "Importance sampling without replacement not implemented, using uniform sampling"
                idx = shuffle(lro.oracle.rng, collect(1:lro.n))[1:batch_size]
            end
        end
    end
    
    actual_batch_size = length(idx)
    
    # Set normalization
    if normalization === nothing
        if p === nothing
            normalization = actual_batch_size
        else
            normalization = actual_batch_size * p[idx] * lro.n
        end
    end
    
    # Compute stochastic gradient
    A_idx = lro.A[idx, :]
    Ax = A_idx * x
    if issparse(Ax)
        Ax = vec(Array(Ax))
    end
    
    activation = sigmoid(Ax)
    
    if issparse(x)
        error_term = sparse((activation - lro.b[idx]) ./ normalization)
    else
        error_term = (activation - lro.b[idx]) ./ normalization
    end
    
    if length(error_term) == 1
        grad = lro.oracle.l2 * x + error_term[1] * A_idx'
    else
        grad = lro.oracle.l2 * x + A_idx' * error_term
    end
    
    return grad
end

"""
    hessian(lro::LogisticRegressionOracle, x::Vector{Float64})

Compute the Hessian of the logistic regression loss.

# Arguments
- `lro::LogisticRegressionOracle`: Logistic regression oracle
- `x::Vector{Float64}`: Parameter vector

# Returns
- `Matrix{Float64}`: Hessian matrix

# Formula
∇²f(x) = (1/n) * A^T * diag(sigmoid(Ax) .* (1 - sigmoid(Ax))) * A + λ₂ * I
"""
function hessian(lro::LogisticRegressionOracle, x::Vector{Float64})::Matrix{Float64}
    Ax = mat_vec_product(lro, x)
    activation = sigmoid(Ax)
    weights = activation .* (1.0 .- activation)
    
    # A_weighted = A^T * diag(weights)
    A_weighted = safe_sparse_multiply(lro.A', weights)
    hessian_matrix = A_weighted * lro.A / lro.n
    
    # Add L2 regularization
    if lro.oracle.l2 > 0
        hessian_matrix += lro.oracle.l2 * I(lro.d)
    end
    
    return hessian_matrix
end

"""
    stochastic_hessian(lro::LogisticRegressionOracle, x::Vector{Float64}, 
                       idx::Union{Vector{Int}, Nothing}=nothing;
                       batch_size::Int=1, replace::Bool=false,
                       normalization::Union{Int, Nothing}=nothing)

Compute stochastic Hessian using a batch of samples.
"""
function stochastic_hessian(lro::LogisticRegressionOracle, x::Vector{Float64}, 
                           idx::Union{Vector{Int}, Nothing}=nothing;
                           batch_size::Int=1, replace::Bool=false,
                           normalization::Union{Int, Nothing}=nothing)::Matrix{Float64}
    
    if batch_size == lro.n
        return hessian(lro, x)
    end
    
    # Sample indices if not provided
    if idx === nothing
        if replace
            idx = rand(lro.oracle.rng, 1:lro.n, batch_size)
        else
            if batch_size > lro.n
                throw(ArgumentError("Cannot sample $batch_size elements without replacement from $(lro.n) samples"))
            end
            idx = shuffle(lro.oracle.rng, collect(1:lro.n))[1:batch_size]
        end
    end
    
    actual_batch_size = length(idx)
    
    if normalization === nothing
        normalization = actual_batch_size
    end
    
    # Compute stochastic Hessian
    A_idx = lro.A[idx, :]
    Ax = A_idx * x
    if issparse(Ax)
        Ax = vec(Array(Ax))
    end
    
    activation = sigmoid(Ax)
    weights = activation .* (1.0 .- activation)
    
    A_weighted = safe_sparse_multiply(A_idx', weights)
    hess = A_weighted * A_idx / normalization
    
    # Add L2 regularization
    if lro.oracle.l2 > 0
        hess += lro.oracle.l2 * I(lro.d)
    end
    
    return hess
end

"""
    smoothness(lro::LogisticRegressionOracle)

Compute the smoothness constant (largest eigenvalue of the Hessian).

# Arguments
- `lro::LogisticRegressionOracle`: Logistic regression oracle

# Returns
- `Float64`: Smoothness constant

The smoothness constant is 0.25 * λ_max(A^T*A/n) + λ₂
"""
function smoothness(lro::LogisticRegressionOracle)::Float64
    if lro.oracle._smoothness === nothing
        if lro.d > 20000 && lro.n > 20000
            @warn "The matrix is too large to estimate the smoothness constant, so Frobenius estimate is used instead."
            if issparse(lro.A)
                lro.oracle._smoothness = 0.25 * norm(lro.A, :fro)^2 / lro.n + lro.oracle.l2
            else
                lro.oracle._smoothness = 0.25 * norm(lro.A, :fro)^2 / lro.n + lro.oracle.l2
            end
        else
            svd_result = svd(lro.A)
            sing_val_max = maximum(svd_result.S)
            lro.oracle._smoothness = 0.25 * sing_val_max^2 / lro.n + lro.oracle.l2
        end
    end
    return lro.oracle._smoothness
end

"""
    max_smoothness(lro::LogisticRegressionOracle)

Compute the maximum smoothness constant over all samples.

# Arguments
- `lro::LogisticRegressionOracle`: Logistic regression oracle

# Returns
- `Float64`: Maximum smoothness constant
"""
function max_smoothness(lro::LogisticRegressionOracle)::Float64
    if lro.oracle._max_smoothness === nothing
        max_squared_sum = maximum(sum(abs2, lro.A; dims=2))
        lro.oracle._max_smoothness = 0.25 * max_squared_sum + lro.oracle.l2
    end
    return lro.oracle._max_smoothness
end

"""
    average_smoothness(lro::LogisticRegressionOracle)

Compute the average smoothness constant over all samples.

# Arguments
- `lro::LogisticRegressionOracle`: Logistic regression oracle

# Returns
- `Float64`: Average smoothness constant
"""
function average_smoothness(lro::LogisticRegressionOracle)::Float64
    if lro.oracle._ave_smoothness === nothing
        ave_squared_sum = mean(sum(abs2, lro.A; dims=2))
        lro.oracle._ave_smoothness = 0.25 * ave_squared_sum + lro.oracle.l2
    end
    return lro.oracle._ave_smoothness
end

"""
    batch_smoothness(lro::LogisticRegressionOracle, batch_size::Int)

Compute the smoothness constant for stochastic gradients sampled in minibatches.

# Arguments
- `lro::LogisticRegressionOracle`: Logistic regression oracle
- `batch_size::Int`: Batch size

# Returns
- `Float64`: Batch smoothness constant
"""
function batch_smoothness(lro::LogisticRegressionOracle, batch_size::Int)::Float64
    L = smoothness(lro)
    L_max = max_smoothness(lro)
    L_batch = lro.n / (lro.n - 1) * (1 - 1/batch_size) * L + (lro.n/batch_size - 1) / (lro.n - 1) * L_max
    return L_batch
end

"""
    individ_smoothness(lro::LogisticRegressionOracle)

Compute individual smoothness constants for each sample.

# Arguments
- `lro::LogisticRegressionOracle`: Logistic regression oracle

# Returns
- `Vector{Float64}`: Individual smoothness constants
"""
function individ_smoothness(lro::LogisticRegressionOracle)::Vector{Float64}
    if lro._individ_smoothness === nothing
        lro._individ_smoothness = [norm(lro.A[i, :]) for i in 1:lro.n]
    end
    return lro._individ_smoothness
end

"""
    hessian_lipschitz(lro::LogisticRegressionOracle)

Compute the Hessian Lipschitz constant.

# Arguments
- `lro::LogisticRegressionOracle`: Logistic regression oracle

# Returns
- `Float64`: Hessian Lipschitz constant
"""
function hessian_lipschitz(lro::LogisticRegressionOracle)::Float64
    if !hasfield(typeof(lro.oracle), :_hessian_lipschitz)
        # Use a different field if _hessian_lipschitz doesn't exist
        # For now, compute it directly
        a_max = maximum([norm(lro.A[i, :]) for i in 1:lro.n])
        A_norm = (smoothness(lro) - lro.oracle.l2) * 4
        return A_norm * a_max / (6 * sqrt(3))
    end
    
    # Implementation would go here if the field exists
    return 0.0  # Placeholder
end

"""
    density(lro::LogisticRegressionOracle, x::Vector{Float64})

Compute the density (sparsity) of a vector.

# Arguments
- `lro::LogisticRegressionOracle`: Logistic regression oracle
- `x::Vector{Float64}`: Input vector

# Returns
- `Float64`: Density (proportion of non-zero elements)
"""
function density(lro::LogisticRegressionOracle, x::Vector{Float64})::Float64
    if issparse(x)
        return Float64(nnz(x)) / length(x)
    else
        return Float64(count(!iszero, x)) / length(x)
    end
end

# Export the main oracle type and functions
export LogisticRegressionOracle
export logsig, sigmoid
export mat_vec_product, partial_value, stochastic_gradient, stochastic_hessian
export smoothness, max_smoothness, average_smoothness, batch_smoothness
export individ_smoothness, hessian_lipschitz, density