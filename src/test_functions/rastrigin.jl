"""
Rastrigin function implementation for optimization testing.

The Rastrigin function is a highly multimodal function used as a performance test 
problem for optimization algorithms. It was first proposed by Rastrigin in 1974.
The function is defined as:

f(x) = A*n + Σ[x_i² - A*cos(2π*x_i)] for i = 1 to n

where A = 10 is a constant, and n is the dimensionality.

The global minimum is at x = [0, 0, ..., 0] with f(x) = 0.
The function has many local minima, making it challenging for optimization algorithms.
"""

"""
    rastrigin(x; A=10.0)

Compute the Rastrigin function value for vector x.

# Arguments
- `x::Vector{Float64}`: Input vector of length n ≥ 1
- `A::Float64`: Parameter A (default: 10.0)

# Returns
- `Float64`: Function value at point x

# Example
```julia
x = [0.0, 0.0]
val = rastrigin(x)  # Returns 0.0 (global minimum)
```
"""
function rastrigin(x::Vector{Float64}; A::Float64=10.0)::Float64
    n = length(x)
    if n < 1
        throw(ArgumentError("Rastrigin function requires at least 1 dimension"))
    end
    
    result = A * n
    for i in 1:n
        result += x[i]^2 - A * cos(2π * x[i])
    end
    
    return result
end

"""
    rastrigin_gradient(x; A=10.0)

Compute the gradient of the Rastrigin function.

# Arguments
- `x::Vector{Float64}`: Input vector of length n ≥ 1
- `A::Float64`: Parameter A (default: 10.0)

# Returns
- `Vector{Float64}`: Gradient vector at point x

# Example
```julia
x = [0.0, 0.0]
grad = rastrigin_gradient(x)  # Returns [0.0, 0.0] (at minimum)
```
"""
function rastrigin_gradient(x::Vector{Float64}; A::Float64=10.0)::Vector{Float64}
    n = length(x)
    if n < 1
        throw(ArgumentError("Rastrigin function requires at least 1 dimension"))
    end
    
    grad = zeros(Float64, n)
    
    for i in 1:n
        grad[i] = 2.0 * x[i] + 2π * A * sin(2π * x[i])
    end
    
    return grad
end

"""
    rastrigin_hessian(x; A=10.0)

Compute the Hessian matrix of the Rastrigin function.

# Arguments
- `x::Vector{Float64}`: Input vector of length n ≥ 1
- `A::Float64`: Parameter A (default: 10.0)

# Returns
- `Matrix{Float64}`: Hessian matrix at point x

# Example
```julia
x = [0.0, 0.0]
H = rastrigin_hessian(x)  # Returns diagonal matrix with positive values
```
"""
function rastrigin_hessian(x::Vector{Float64}; A::Float64=10.0)::Matrix{Float64}
    n = length(x)
    if n < 1
        throw(ArgumentError("Rastrigin function requires at least 1 dimension"))
    end
    
    # Rastrigin function has a diagonal Hessian
    H = zeros(Float64, n, n)
    
    for i in 1:n
        H[i, i] = 2.0 + 4π^2 * A * cos(2π * x[i])
    end
    
    return H
end