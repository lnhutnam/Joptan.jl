"""
LogSumExp Oracle Implementation

Implements the logarithm of the sum of exponentials plus optional quadratic terms:
    log(sum_{i=1}^n exp(<a_i, x>-b_i)) + 1/2*||Ax - b||^2 + l2/2*||x||^2

See, for instance:
    https://arxiv.org/pdf/2002.00657.pdf
    https://arxiv.org/pdf/2002.09403.pdf
for examples of using the objective to benchmark second-order methods.

Due to the potential under- and overflow, log-sum-exp and softmax
functions might be unstable. This implementation uses LogExpFunctions.jl
for numerical stability.

Arguments:
    max_smoothing (Float64): the smoothing constant of the log-sum-exp term (default: 1.0)
    least_squares_term (Bool): add term 0.5*||Ax-b||^2 to the objective (default: false)
"""

using LinearAlgebra
using Random
using Statistics
using LogExpFunctions  # For numerically stable logsumexp and softmax
using SparseArrays

include("loss_oracle.jl")

"""
    LogSumExpOracle

LogSumExp oracle that returns loss values, gradients and Hessians.
Inherits from Oracle base class for regularization support.

# Fields
- `max_smoothing::Float64`: Smoothing constant for the log-sum-exp term
- `least_squares_term::Bool`: Whether to add 0.5*||Ax-b||^2 term
- `A::Matrix{Float64}`: Data matrix (n × d)
- `b::Vector{Float64}`: Target vector (n × 1)
- `n::Int`: Number of samples
- `d::Int`: Number of features
- `store_mat_vec_prod::Bool`: Whether to cache matrix-vector products
- `store_softmax::Bool`: Whether to cache softmax values
- `x_last_mv::Vector{Float64}`: Last x for which mat-vec product was computed
- `x_last_soft::Vector{Float64}`: Last x for which softmax was computed
- `mat_vec_prod::Vector{Float64}`: Cached matrix-vector product Ax
- `softmax_cache::Vector{Float64}`: Cached softmax values
"""
mutable struct LogSumExpOracle <: AbstractOracle
    oracle::Oracle  # Base oracle for regularization
    max_smoothing::Float64
    least_squares_term::Bool
    A::Matrix{Float64}
    b::Vector{Float64}
    n::Int
    d::Int
    store_mat_vec_prod::Bool
    store_softmax::Bool
    x_last_mv::Vector{Float64}
    x_last_soft::Vector{Float64}
    mat_vec_prod::Vector{Float64}
    softmax_cache::Vector{Float64}
    
    function LogSumExpOracle(A::Union{Matrix{Float64}, Nothing}=nothing, 
                            b::Union{Vector{Float64}, Nothing}=nothing;
                            max_smoothing::Float64=1.0,
                            least_squares_term::Bool=false,
                            n::Union{Int, Nothing}=nothing,
                            dim::Union{Int, Nothing}=nothing,
                            l1::Float64=0.0, 
                            l2::Float64=0.0,
                            store_mat_vec_prod::Bool=true,
                            store_softmax::Bool=true,
                            seed::Int=42)
        
        # Create base oracle with regularization
        base_oracle = Oracle(l1=l1, l2=l2, seed=seed)
        
        # Handle matrix and vector initialization
        if A === nothing
            if n === nothing || dim === nothing
                throw(ArgumentError("If A is not provided, both n and dim must be specified"))
            end
            # Generate random data
            Random.seed!(seed)
            A_temp = rand(base_oracle.rng, n, dim) * 2 .- 1  # Uniform[-1, 1]
        else
            A_temp = A
            n, dim = size(A_temp)
        end
        
        if b === nothing
            Random.seed!(seed)
            b_temp = randn(base_oracle.rng, n)  # Normal(-1, 1) - approximated as standard normal
        else
            b_temp = copy(b)
            if length(b_temp) != n
                throw(DimensionMismatch("Length of b ($(length(b_temp))) must match number of rows in A ($n)"))
            end
        end
        
        # Initialize cache variables
        x_last_mv = zeros(dim)
        x_last_soft = zeros(dim)
        mat_vec_prod = zeros(n)
        softmax_cache = zeros(n)
        
        # Create oracle
        oracle_obj = new(base_oracle, max_smoothing, least_squares_term, A_temp, b_temp, n, dim,
                        store_mat_vec_prod, store_softmax, x_last_mv, x_last_soft, 
                        mat_vec_prod, softmax_cache)
        
        # Handle the special initialization from Python code
        if A === nothing
            # Adjust A to make gradient at zero equal to zero
            oracle_obj.store_mat_vec_prod = false
            oracle_obj.store_softmax = false
            grad_at_zero = gradient(oracle_obj, zeros(dim))
            oracle_obj.A = oracle_obj.A - grad_at_zero'  # Subtract outer product
            
            # Re-evaluate to update caches
            _ = value(oracle_obj, zeros(dim))
            
            # Re-enable caching
            oracle_obj.store_mat_vec_prod = store_mat_vec_prod
            oracle_obj.store_softmax = store_softmax
        end
        
        return oracle_obj
    end
end

# Forward Oracle methods to base oracle
for method in [:set_seed!, :get_best_point, :reset_best!]
    @eval $method(lse::LogSumExpOracle, args...; kwargs...) = $method(lse.oracle, args...; kwargs...)
end

# Forward properties
function Base.getproperty(lse::LogSumExpOracle, prop::Symbol)
    if prop in fieldnames(LogSumExpOracle)
        return getfield(lse, prop)
    else
        # Forward to base oracle
        return getproperty(lse.oracle, prop)
    end
end

function Base.setproperty!(lse::LogSumExpOracle, prop::Symbol, value)
    if prop in fieldnames(LogSumExpOracle)
        setfield!(lse, prop, value)
    else
        # Forward to base oracle
        setproperty!(lse.oracle, prop, value)
    end
end

"""
    mat_vec_product(lse::LogSumExpOracle, x::Vector{Float64})

Compute matrix-vector product Ax with optional caching.

# Arguments
- `lse::LogSumExpOracle`: LogSumExp oracle
- `x::Vector{Float64}`: Input vector

# Returns
- `Vector{Float64}`: Matrix-vector product Ax
"""
function mat_vec_product(lse::LogSumExpOracle, x::Vector{Float64})::Vector{Float64}
    if !lse.store_mat_vec_prod || safe_sparse_norm(x - lse.x_last_mv) != 0
        z = lse.A * x
        if lse.store_mat_vec_prod
            lse.mat_vec_prod = z
            lse.x_last_mv = copy(x)
        end
        return z
    else
        return lse.mat_vec_prod
    end
end

"""
    softmax_values(lse::LogSumExpOracle, x::Union{Vector{Float64}, Nothing}=nothing, 
                   Ax::Union{Vector{Float64}, Nothing}=nothing)

Compute softmax values with optional caching.

# Arguments
- `lse::LogSumExpOracle`: LogSumExp oracle
- `x::Vector{Float64}`: Input vector (optional if Ax is provided)
- `Ax::Vector{Float64}`: Matrix-vector product (optional if x is provided)

# Returns
- `Vector{Float64}`: Softmax values
"""
function softmax_values(lse::LogSumExpOracle, 
                       x::Union{Vector{Float64}, Nothing}=nothing,
                       Ax::Union{Vector{Float64}, Nothing}=nothing)::Vector{Float64}
    
    if x === nothing && Ax === nothing
        throw(ArgumentError("Either x or Ax must be provided to compute softmax"))
    end
    
    if lse.store_softmax && x !== nothing && safe_sparse_norm(x - lse.x_last_soft) == 0
        return lse.softmax_cache
    end
    
    if Ax === nothing
        Ax = mat_vec_product(lse, x)
    end
    
    # Compute softmax: softmax((Ax - b) / max_smoothing)
    # Use LogExpFunctions.jl for numerical stability
    scaled_input = (Ax - lse.b) / lse.max_smoothing
    softmax_vals = softmax(scaled_input)
    
    if lse.store_softmax && x !== nothing
        lse.softmax_cache = softmax_vals
        lse.x_last_soft = copy(x)
    end
    
    return softmax_vals
end

"""
    _value(lse::LogSumExpOracle, x::Vector{Float64})

Compute the base LogSumExp loss (without regularization).

# Arguments
- `lse::LogSumExpOracle`: LogSumExp oracle
- `x::Vector{Float64}`: Parameter vector

# Returns
- `Float64`: Base loss function value

# Formula
f(x) = max_smoothing * log(sum(exp((Ax - b) / max_smoothing))) + [optional least squares term]
"""
function _value(lse::LogSumExpOracle, x::Vector{Float64})::Float64
    Ax = mat_vec_product(lse, x)
    
    # Compute log-sum-exp: max_smoothing * logsumexp((Ax - b) / max_smoothing)
    scaled_input = (Ax - lse.b) / lse.max_smoothing
    lse_term = lse.max_smoothing * logsumexp(scaled_input)
    
    # Add least squares term if enabled
    ls_term = 0.0
    if lse.least_squares_term
        ls_term = 0.5 * safe_sparse_norm(Ax)^2
    end
    
    return lse_term + ls_term
end

"""
    value(lse::LogSumExpOracle, x::Vector{Float64})

Compute the full LogSumExp loss including regularization.

# Arguments
- `lse::LogSumExpOracle`: LogSumExp oracle
- `x::Vector{Float64}`: Parameter vector

# Returns
- `Float64`: Full loss function value including regularization
"""
function value(lse::LogSumExpOracle, x::Vector{Float64})::Float64
    # Compute base value
    base_value = _value(lse, x)
    
    # Add L2 regularization
    total_value = base_value
    if lse.oracle.l2 > 0
        total_value += 0.5 * lse.oracle.l2 * safe_sparse_norm(x)^2
    end
    
    # Track best solution
    if total_value < lse.oracle.f_opt
        lse.oracle.x_opt = copy(x)
        lse.oracle.f_opt = total_value
    end
    
    return total_value
end

"""
    gradient(lse::LogSumExpOracle, x::Vector{Float64})

Compute the gradient of the LogSumExp loss.

# Arguments
- `lse::LogSumExpOracle`: LogSumExp oracle
- `x::Vector{Float64}`: Parameter vector

# Returns
- `Vector{Float64}`: Gradient vector

# Formula
∇f(x) = A^T * softmax + [optional least squares term] + regularization_gradient
"""
function gradient(lse::LogSumExpOracle, x::Vector{Float64})::Vector{Float64}
    Ax = mat_vec_product(lse, x)
    softmax_vals = softmax_values(lse, x, Ax)
    
    # Base gradient: A^T * softmax
    if lse.least_squares_term
        # grad = (softmax + Ax)^T @ A
        grad = lse.A' * (softmax_vals + Ax)
    else
        # grad = softmax^T @ A
        grad = lse.A' * softmax_vals
    end
    
    # Add L2 regularization gradient
    if lse.oracle.l2 > 0
        grad = safe_sparse_add(grad, lse.oracle.l2 * x)
    end
    
    # Add L1 regularization (subgradient)
    if lse.oracle.l1 > 0
        grad = safe_sparse_add(grad, lse.oracle.l1 * sign.(x))
    end
    
    return grad
end

"""
    hessian(lse::LogSumExpOracle, x::Vector{Float64})

Compute the Hessian of the LogSumExp loss.

# Arguments
- `lse::LogSumExpOracle`: LogSumExp oracle
- `x::Vector{Float64}`: Parameter vector

# Returns
- `Matrix{Float64}`: Hessian matrix

# Formula
∇²f(x) = A^T * diag(softmax/max_smoothing) * A - (A^T * softmax) * (A^T * softmax)^T / max_smoothing + λ₂ * I
"""
function hessian(lse::LogSumExpOracle, x::Vector{Float64})::Matrix{Float64}
    Ax = mat_vec_product(lse, x)
    softmax_vals = softmax_values(lse, x, Ax)
    
    # First term: A^T * diag(softmax/max_smoothing) * A
    scaled_softmax = softmax_vals / lse.max_smoothing
    hess1 = lse.A' * Diagonal(scaled_softmax) * lse.A
    
    # Second term: -outer_product(A^T * softmax) / max_smoothing
    grad_base = lse.A' * softmax_vals
    hess2 = -outer(grad_base, grad_base) / lse.max_smoothing
    
    # Combine terms
    hessian_matrix = hess1 + hess2
    
    # Add least squares term if enabled
    if lse.least_squares_term
        hessian_matrix += lse.A' * lse.A
    end
    
    # Add L2 regularization
    if lse.oracle.l2 > 0
        hessian_matrix += lse.oracle.l2 * I(lse.d)
    end
    
    return hessian_matrix
end

"""
    smoothness(lse::LogSumExpOracle)

Compute the smoothness constant (largest eigenvalue of the Hessian bound).

# Arguments
- `lse::LogSumExpOracle`: LogSumExp oracle

# Returns
- `Float64`: Smoothness constant
"""
function smoothness(lse::LogSumExpOracle)::Float64
    if lse.oracle._smoothness === nothing
        matrix_coef = 1 + (lse.least_squares_term ? 1 : 0)
        
        if lse.d > 20000 && lse.n > 20000
            @warn "The matrix is too large to estimate the smoothness constant, so Frobenius estimate is used instead."
            lse.oracle._smoothness = matrix_coef * norm(lse.A, :fro)^2 + lse.oracle.l2
        else
            # Use largest singular value
            svd_result = svd(lse.A)
            sing_val_max = maximum(svd_result.S)
            lse.oracle._smoothness = matrix_coef * sing_val_max^2 + lse.oracle.l2
        end
    end
    return lse.oracle._smoothness
end

"""
    hessian_lipschitz(lse::LogSumExpOracle)

Compute the Hessian Lipschitz constant.

# Arguments
- `lse::LogSumExpOracle`: LogSumExp oracle

# Returns
- `Float64`: Hessian Lipschitz constant
"""
function hessian_lipschitz(lse::LogSumExpOracle)::Float64
    if lse.oracle._max_smoothness === nothing  # Reuse this field for Hessian Lipschitz
        row_norms = [norm(lse.A[i, :]) for i in 1:lse.n]
        max_row_norm = maximum(row_norms)
        lse.oracle._max_smoothness = 2 * max_row_norm / lse.max_smoothing * smoothness(lse)
    end
    return lse.oracle._max_smoothness
end

# Helper function for outer product
function outer(x::Vector{Float64}, y::Vector{Float64})::Matrix{Float64}
    return x * y'
end

# Export the main oracle type and functions
export LogSumExpOracle
export mat_vec_product, softmax_values, hessian_lipschitz