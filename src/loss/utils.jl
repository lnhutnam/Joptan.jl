"""
Utility functions for loss functions in Joptan.jl

This module provides utility functions for safe operations with different
array types, including sparse arrays when available.
"""

using LinearAlgebra
using SparseArrays

"""
    safe_sparse_add(a, b)

Implement a+b compatible with different types of input.
Supports scalars, arrays and sparse arrays.

# Arguments
- `a`: First operand (scalar, array, or sparse array)
- `b`: Second operand (scalar, array, or sparse array)

# Returns
- Sum of a and b with appropriate type handling
"""
function safe_sparse_add(a, b)
    # Handle scalar cases
    if isa(a, Number) || isa(b, Number)
        return a + b
    end
    
    # Handle sparse matrix cases
    both_sparse = issparse(a) && issparse(b)
    if both_sparse
        return a + b
    end
    
    # Convert sparse to dense if needed
    if issparse(a) && !issparse(b)
        a_dense = Array(a)
        # Handle dimension mismatch
        if ndims(a_dense) == 2 && ndims(b) == 1
            b = reshape(b, :, 1)
        end
        return a_dense + b
    elseif !issparse(a) && issparse(b)
        b_dense = Array(b)
        # Handle dimension mismatch
        if ndims(b_dense) == 2 && ndims(a) == 1
            b_dense = vec(b_dense)
        end
        return a + b_dense
    end
    
    # Both are dense
    return a + b
end

"""
    safe_sparse_multiply(a, b)

Element-wise multiplication compatible with different array types.

# Arguments
- `a`: First operand
- `b`: Second operand

# Returns
- Element-wise product of a and b
"""
function safe_sparse_multiply(a, b)
    # Handle sparse cases
    if issparse(a) && issparse(b)
        return a .* b
    end
    
    # Convert to dense if needed
    if issparse(a)
        a = Array(a)
    elseif issparse(b)
        b = Array(b)
    end
    
    return a .* b
end

"""
    safe_sparse_norm(a; ord=nothing)

Compute norm compatible with different array types.

# Arguments
- `a`: Input array (dense or sparse)
- `ord`: Norm order (1, 2, Inf, etc.)

# Returns
- Norm of the array
"""
function safe_sparse_norm(a; ord=nothing)
    if issparse(a)
        if ord === nothing || ord == 2
            return LinearAlgebra.norm(a)
        elseif ord == 1
            return LinearAlgebra.norm(a, 1)
        elseif ord == Inf
            return LinearAlgebra.norm(a, Inf)
        else
            return LinearAlgebra.norm(a, ord)
        end
    else
        if ord === nothing
            return LinearAlgebra.norm(a)
        else
            return LinearAlgebra.norm(a, ord)
        end
    end
end

"""
    safe_sparse_inner_prod(a, b)

Compute inner product compatible with different array types.

# Arguments
- `a`: First vector/matrix
- `b`: Second vector/matrix

# Returns
- Inner product of a and b
"""
function safe_sparse_inner_prod(a, b)
    # Handle sparse cases
    if issparse(a) && issparse(b)
        if ndims(a) == 2 && size(a, 2) == size(b, 1)
            return (a * b)[1, 1]
        elseif size(a, 1) == size(b, 1)
            return (a' * b)[1, 1]
        else
            return (a * b')[1, 1]
        end
    end
    
    # Convert to dense if needed
    if issparse(a)
        a = Array(a)
    elseif issparse(b)
        b = Array(b)
    end
    
    return dot(a, b)
end

"""
    safe_outer_prod(a, b)

Compute outer product compatible with different array types.

# Arguments
- `a`: First vector
- `b`: Second vector

# Returns
- Outer product a * b'
"""
function safe_outer_prod(a, b)
    # Convert sparse to dense for outer product
    if issparse(a)
        a = Array(a)
    end
    if issparse(b)
        b = Array(b)
    end
    
    return a * b'
end

"""
    safe_is_equal(a, b; rtol=1e-5, atol=1e-8)

Check equality compatible with different array types.

# Arguments
- `a`: First array
- `b`: Second array
- `rtol`: Relative tolerance
- `atol`: Absolute tolerance

# Returns
- Boolean indicating if arrays are approximately equal
"""
function safe_is_equal(a, b; rtol=1e-5, atol=1e-8)
    # Handle different types
    if typeof(a) != typeof(b)
        # Convert to same type
        if issparse(a) && !issparse(b)
            a = Array(a)
        elseif !issparse(a) && issparse(b)
            b = Array(b)
        end
    end
    
    # Check dimensions
    if size(a) != size(b)
        return false
    end
    
    # Use isapprox for numerical comparison
    return isapprox(a, b, rtol=rtol, atol=atol)
end