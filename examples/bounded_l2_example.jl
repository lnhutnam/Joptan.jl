"""
Example: Bounded L2 Regularization with Linear Regression

This example demonstrates how to use the BoundedL2Regularizer with linear regression
for nonconvex regularized optimization.
"""

using LinearAlgebra
using Random
using Statistics

using Joptan

# Set random seed for reproducibility
Random.seed!(42)

"""
    generate_test_data(n::Int, d::Int; noise_std::Float64=0.1, sparsity::Float64=0.3)

Generate synthetic data for testing bounded L2 regularization.
"""
function generate_test_data(n::Int, d::Int; noise_std::Float64=0.1, sparsity::Float64=0.3)
    # Generate sparse true parameters
    x_true = randn(d)
    
    # Make it sparse
    n_zeros = Int(floor(d * sparsity))
    zero_indices = randperm(d)[1:n_zeros]
    x_true[zero_indices] .= 0.0
    
    # Generate design matrix
    A = randn(n, d) / sqrt(d)
    
    # Generate targets with noise
    b = A * x_true + noise_std * randn(n)
    
    return A, b, x_true
end

"""
    BoundedL2LinearRegression

A linear regression oracle with bounded L2 regularization.
Uses composition to combine LinearRegressionOracle with BoundedL2Regularizer.
"""
mutable struct BoundedL2LinearRegression
    base_oracle::LinearRegressionOracle
    bounded_l2_reg::BoundedL2Regularizer
    
    function BoundedL2LinearRegression(A::Matrix{Float64}, b::Vector{Float64}, coef::Float64)
        # Create base linear regression oracle without regularization
        base_oracle = LinearRegressionOracle(A, b, l1=0.0, l2=0.0)
        
        # Create bounded L2 regularizer
        bounded_l2_reg = BoundedL2Regularizer(coef)
        
        new(base_oracle, bounded_l2_reg)
    end
end

# Forward methods to base oracle where appropriate
for method in [:set_seed!, :get_best_point, :reset_best!]
    @eval $method(oracle::BoundedL2LinearRegression, args...; kwargs...) = $method(oracle.base_oracle, args...; kwargs...)
end

# Implement oracle interface
function Joptan.value(oracle::BoundedL2LinearRegression, x::Vector{Float64})
    # Linear regression loss (without regularization)
    base_loss = Joptan._value(oracle.base_oracle, x)
    
    # Add bounded L2 regularization
    reg_loss = Joptan.value(oracle.bounded_l2_reg, x)
    
    total_loss = base_loss + reg_loss
    
    # Track best solution
    if total_loss < oracle.base_oracle.f_opt
        oracle.base_oracle.x_opt = copy(x)
        oracle.base_oracle.f_opt = total_loss
    end
    
    return total_loss
end

function Joptan.gradient(oracle::BoundedL2LinearRegression, x::Vector{Float64})
    # Linear regression gradient (without regularization)
    residual = oracle.base_oracle.A * x - oracle.base_oracle.b
    base_grad = oracle.base_oracle.A' * residual / oracle.base_oracle.n
    
    # Add bounded L2 gradient
    reg_grad = Joptan.gradient(oracle.bounded_l2_reg, x)
    
    return base_grad + reg_grad
end

function Joptan.hessian(oracle::BoundedL2LinearRegression, x::Vector{Float64})
    # Linear regression Hessian (without regularization)
    base_hessian = oracle.base_oracle.A' * oracle.base_oracle.A / oracle.base_oracle.n
    
    # Add bounded L2 Hessian
    reg_hessian = Joptan.hessian(oracle.bounded_l2_reg, x)
    
    return base_hessian + reg_hessian
end

"""
    test_bounded_l2_properties()

Test the mathematical properties of the bounded L2 regularizer.
"""
function test_bounded_l2_properties()
    println("=== Testing Bounded L2 Properties ===")
    
    # Create regularizer
    reg = BoundedL2Regularizer(1.0)
    
    # Test points
    test_points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [2.0, -1.5, 0.5]
    ]
    
    println("Point\t\t\t\tValue\t\tGrad Norm\tSmooth Const")
    println("-" ^ 70)
    
    for x in test_points
        val = Joptan.value(reg, x)
        grad = Joptan.gradient(reg, x)
        grad_norm = LinearAlgebra.norm(grad)
        smooth_const = Joptan.smoothness(reg)
        
        point_str = string(x)
        if length(point_str) < 20
            point_str = point_str * " " ^ (20 - length(point_str))
        end
        
        println("$point_str\t$(round(val, digits=4))\t\t$(round(grad_norm, digits=4))\t\t$(smooth_const)")
    end
    
    # Test properties
    println("\nRegularizer Properties:")
    println("  Is zero: $(Joptan.is_zero(reg))")
    println("  Smoothness constant: $(Joptan.smoothness(reg))")
    
    # Test Hessian
    x_test = [1.0, -0.5, 2.0]
    H = Joptan.hessian(reg, x_test)
    H_diag = Joptan.hessian_diagonal(reg, x_test)
    
    println("\nHessian at $x_test:")
    println("  Diagonal: $H_diag")
    println("  Full diagonal: $(diag(H))")
    
    # Check if Hessian is valid
    eigenvals_H = eigvals(H)
    println("  Eigenvalue range: [$(round(minimum(eigenvals_H), digits=4)), $(round(maximum(eigenvals_H), digits=4))]")
end

"""
    compare_regularizers()

Compare bounded L2 with standard L1 and L2 regularization.
"""
function compare_regularizers()
    println("\n=== Comparing Regularizers ===")
    
    # Generate data
    n, d = 50, 20
    A, b, x_true = generate_test_data(n, d)
    
    println("True parameter sparsity: $(sum(abs.(x_true) .< 1e-6)) / $d")
    
    println("\nRegularizer\t\t\tLoss\t\tParam Norm\tSparsity\tError")
    println("-" ^ 75)
    
    results = []
    
    # No regularization
    oracle_none = LinearRegressionOracle(A, b)
    x_none = (A' * A) \ (A' * b)  # Analytical solution
    loss_none = Joptan.value(oracle_none, x_none)
    param_norm_none = LinearAlgebra.norm(x_none)
    sparsity_none = sum(abs.(x_none) .< 1e-3)
    error_none = LinearAlgebra.norm(x_none - x_true)
    
    println("No regularization\t\t$(round(loss_none, digits=4))\t\t$(round(param_norm_none, digits=4))\t\t$sparsity_none/$d\t\t$(round(error_none, digits=4))")
    push!(results, ("No regularization", x_none, loss_none, error_none))
    
    # L2 regularization
    coef = 0.1
    oracle_l2 = LinearRegressionOracle(A, b, l2=coef)
    x_l2 = (A' * A + coef * I(d)) \ (A' * b)  # Ridge solution
    loss_l2 = Joptan.value(oracle_l2, x_l2)
    param_norm_l2 = LinearAlgebra.norm(x_l2)
    sparsity_l2 = sum(abs.(x_l2) .< 1e-3)
    error_l2 = LinearAlgebra.norm(x_l2 - x_true)
    
    println("L2 (λ=0.1)\t\t\t$(round(loss_l2, digits=4))\t\t$(round(param_norm_l2, digits=4))\t\t$sparsity_l2/$d\t\t$(round(error_l2, digits=4))")
    push!(results, ("L2", x_l2, loss_l2, error_l2))
    
    # L1 regularization (using gradient descent)
    oracle_l1 = LinearRegressionOracle(A, b, l1=coef)
    x_l1 = optimize_with_gradient_descent(oracle_l1, zeros(d), lr=0.01, max_iter=2000)
    loss_l1 = Joptan.value(oracle_l1, x_l1)
    param_norm_l1 = LinearAlgebra.norm(x_l1)
    sparsity_l1 = sum(abs.(x_l1) .< 1e-3)
    error_l1 = LinearAlgebra.norm(x_l1 - x_true)
    
    println("L1 (λ=0.1)\t\t\t$(round(loss_l1, digits=4))\t\t$(round(param_norm_l1, digits=4))\t\t$sparsity_l1/$d\t\t$(round(error_l1, digits=4))")
    push!(results, ("L1", x_l1, loss_l1, error_l1))
    
    # Bounded L2 regularization
    oracle_bounded = BoundedL2LinearRegression(A, b, coef)
    x_bounded = optimize_with_gradient_descent(oracle_bounded, zeros(d), lr=0.01, max_iter=2000)
    loss_bounded = Joptan.value(oracle_bounded, x_bounded)
    param_norm_bounded = LinearAlgebra.norm(x_bounded)
    sparsity_bounded = sum(abs.(x_bounded) .< 1e-3)
    error_bounded = LinearAlgebra.norm(x_bounded - x_true)
    
    println("Bounded L2 (λ=0.1)\t\t$(round(loss_bounded, digits=4))\t\t$(round(param_norm_bounded, digits=4))\t\t$sparsity_bounded/$d\t\t$(round(error_bounded, digits=4))")
    push!(results, ("Bounded L2", x_bounded, loss_bounded, error_bounded))
    
    return results
end

"""
    optimize_with_gradient_descent(oracle, x0; lr=0.01, max_iter=1000, tol=1e-6)

Simple gradient descent optimization.
"""
function optimize_with_gradient_descent(oracle, x0; lr=0.01, max_iter=1000, tol=1e-6)
    x = copy(x0)
    
    for i in 1:max_iter
        grad = Joptan.gradient(oracle, x)
        grad_norm = LinearAlgebra.norm(grad)
        
        if grad_norm < tol
            break
        end
        
        x = x - lr * grad
    end
    
    return x
end

"""
    visualize_regularization_functions()

Print values for different regularization functions for comparison.
"""
function visualize_regularization_functions()
    println("\n=== Regularization Function Comparison ===")
    
    # Create regularizer
    reg_bounded = BoundedL2Regularizer(1.0)
    
    # Test range
    x_values = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
    
    println("x\t\tBounded L2\tL2\t\tL1\t\tBounded L2 Grad")
    println("-" ^ 65)
    
    for x_val in x_values
        x_vec = [x_val]
        
        # Bounded L2
        val_bounded = Joptan.value(reg_bounded, x_vec)
        grad_bounded = Joptan.gradient(reg_bounded, x_vec)[1]
        
        # Standard L2
        val_l2 = 0.5 * x_val^2
        
        # Standard L1
        val_l1 = abs(x_val)
        
        println("$(x_val)\t\t$(round(val_bounded, digits=4))\t\t$(round(val_l2, digits=4))\t\t$(round(val_l1, digits=4))\t\t$(round(grad_bounded, digits=4))")
    end
end

"""
    optimization_comparison()

Compare optimization performance with different regularizers.
"""
function optimization_comparison()
    println("\n=== Optimization Comparison ===")
    
    # Generate data
    n, d = 100, 30
    A, b, x_true = generate_test_data(n, d, sparsity=0.4)
    
    # Different regularization coefficients
    coefs = [0.01, 0.1, 1.0]
    
    println("Coef\tRegularizer\t\tFinal Loss\tIterations\tFinal Error")
    println("-" ^ 70)
    
    for coef in coefs
        # L2 regularization
        oracle_l2 = LinearRegressionOracle(A, b, l2=coef)
        x_l2, hist_l2 = optimize_with_history(oracle_l2, zeros(d))
        
        # Bounded L2 regularization
        oracle_bounded = BoundedL2LinearRegression(A, b, coef)
        x_bounded, hist_bounded = optimize_with_history(oracle_bounded, zeros(d))
        
        # Print results
        println("$(coef)\tL2\t\t\t$(round(hist_l2[end][2], digits=6))\t\t$(length(hist_l2))\t\t$(round(LinearAlgebra.norm(x_l2 - x_true), digits=4))")
        println("$(coef)\tBounded L2\t\t$(round(hist_bounded[end][2], digits=6))\t\t$(length(hist_bounded))\t\t$(round(LinearAlgebra.norm(x_bounded - x_true), digits=4))")
    end
end

"""
    optimize_with_history(oracle, x0; lr=0.01, max_iter=1000, tol=1e-6)

Gradient descent with convergence history.
"""
function optimize_with_history(oracle, x0; lr=0.01, max_iter=1000, tol=1e-6)
    x = copy(x0)
    history = []
    
    for i in 1:max_iter
        loss_val = Joptan.value(oracle, x)
        grad = Joptan.gradient(oracle, x)
        grad_norm = LinearAlgebra.norm(grad)
        
        push!(history, (i, loss_val, grad_norm))
        
        if grad_norm < tol
            break
        end
        
        x = x - lr * grad
    end
    
    return x, history
end

"""
    test_with_adagrad()

Test bounded L2 regularization with Adagrad optimizer.
"""
function test_with_adagrad()
    println("\n=== Testing with Adagrad Optimizer ===")
    
    # Generate data
    n, d = 50, 20
    A, b, x_true = generate_test_data(n, d)
    
    # Create bounded L2 oracle
    oracle = BoundedL2LinearRegression(A, b, 0.1)
    
    # Define loss and gradient functions for Adagrad
    loss_func(x) = Joptan.value(oracle, x)
    grad_func(x) = Joptan.gradient(oracle, x)
    
    # Create Adagrad optimizer
    optimizer = AdagradOptimizer(loss_func, grad_func, lr=0.1, delta=1e-8, label="Bounded L2 Adagrad")
    
    # Run optimization
    x0 = randn(d) * 0.1
    trace = run!(optimizer, x0, it_max=500, verbose=true)
    
    # Get final solution
    x_final = optimizer.x
    final_loss = loss_func(x_final)
    final_error = LinearAlgebra.norm(x_final - x_true)
    
    println("\nAdagrad Results:")
    println("  Final loss: $(round(final_loss, digits=6))")
    println("  Final error: $(round(final_error, digits=4))")
    println("  Final sparsity: $(sum(abs.(x_final) .< 1e-3)) / $d")
    
    return x_final, trace
end

"""
    main()

Run all bounded L2 regularization examples.
"""
function main()
    println("Bounded L2 Regularization Examples")
    println("=" ^ 50)
    
    # Test regularizer properties
    test_bounded_l2_properties()
    
    # Compare with other regularizers
    results = compare_regularizers()
    
    # Visualize the regularization function (text-based)
    visualize_regularization_functions()
    
    # Compare optimization performance
    optimization_comparison()
    
    # Test with Adagrad optimizer
    test_with_adagrad()
    
    println("\nExample completed successfully!")
end

main()