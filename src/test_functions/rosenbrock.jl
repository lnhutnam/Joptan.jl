"""
Rosenbrock function implementation for optimization testing.

The Rosenbrock function is a non-convex function used as a performance test problem
for optimization algorithms. It is introduced by Howard H. Rosenbrock in 1960.
The function is defined as:

f(x,y) = (a-x)² + b(y-x²)²

where a = 1 and b = 100 are constants.

For n-dimensional case:
f(x) = Σ[100(x_{i+1} - x_i²)² + (1 - x_i)²] for i = 1 to n-1

The global minimum is at x = [1, 1, ..., 1] with f(x) = 0.
"""

"""
    rosenbrock(x)

Compute the Rosenbrock function value for vector x.

# Arguments
- `x::Vector{Float64}`: Input vector of length n ≥ 2

# Returns
- `Float64`: Function value at point x

# Example
```julia
x = [1.0, 1.0]
val = rosenbrock(x)  # Returns 0.0 (global minimum)
```
"""
function rosenbrock(x::Vector{Float64})::Float64
    n = length(x)
    if n < 2
        throw(ArgumentError("Rosenbrock function requires at least 2 dimensions"))
    end
    
    result = 0.0
    for i in 1:(n-1)
        result += 100.0 * (x[i+1] - x[i]^2)^2 + (1.0 - x[i])^2
    end
    
    return result
end

"""
    rosenbrock_gradient(x)

Compute the gradient of the Rosenbrock function.

# Arguments
- `x::Vector{Float64}`: Input vector of length n ≥ 2

# Returns
- `Vector{Float64}`: Gradient vector at point x

# Example
```julia
x = [1.0, 1.0]
grad = rosenbrock_gradient(x)  # Returns [0.0, 0.0] (at minimum)
```
"""
function rosenbrock_gradient(x::Vector{Float64})::Vector{Float64}
    n = length(x)
    if n < 2
        throw(ArgumentError("Rosenbrock function requires at least 2 dimensions"))
    end
    
    grad = zeros(Float64, n)
    
    # First element
    grad[1] = -400.0 * x[1] * (x[2] - x[1]^2) - 2.0 * (1.0 - x[1])
    
    # Middle elements
    for i in 2:(n-1)
        grad[i] = 200.0 * (x[i] - x[i-1]^2) - 400.0 * x[i] * (x[i+1] - x[i]^2) - 2.0 * (1.0 - x[i])
    end
    
    # Last element
    grad[n] = 200.0 * (x[n] - x[n-1]^2)
    
    return grad
end

"""
    rosenbrock_hessian(x)

Compute the Hessian matrix of the Rosenbrock function.

# Arguments
- `x::Vector{Float64}`: Input vector of length n ≥ 2

# Returns
- `Matrix{Float64}`: Hessian matrix at point x

# Example
```julia
x = [1.0, 1.0]
H = rosenbrock_hessian(x)  # Returns positive definite matrix at minimum
```
"""
function rosenbrock_hessian(x::Vector{Float64})::Matrix{Float64}
    n = length(x)
    if n < 2
        throw(ArgumentError("Rosenbrock function requires at least 2 dimensions"))
    end
    
    H = zeros(Float64, n, n)
    
    # Diagonal elements
    H[1, 1] = -400.0 * (x[2] - x[1]^2) + 800.0 * x[1]^2 + 2.0
    
    for i in 2:(n-1)
        H[i, i] = 200.0 - 400.0 * (x[i+1] - x[i]^2) + 800.0 * x[i]^2 + 2.0
    end
    
    H[n, n] = 200.0
    
    # Off-diagonal elements
    for i in 1:(n-1)
        H[i, i+1] = -400.0 * x[i]
        H[i+1, i] = -400.0 * x[i]  # Symmetric
    end
    
    return H
end