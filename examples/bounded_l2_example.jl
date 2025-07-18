"""
Example: Bounded L2 Regularization with Linear Regression

This example demonstrates how to use the BoundedL2Regularizer with linear regression
for nonconvex regularized optimization.
"""

using LinearAlgebra
using Plots
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

A simple struct that combines linear regression with bounded L2 regularization.
This avoids the type conversion issues with the Oracle system.
"""
struct BoundedL2LinearRegression
    A::Matrix{Float64}
    b::Vector{Float64}
    regularizer::BoundedL2Regularizer
    n::Int
    d::Int
end

function BoundedL2LinearRegression(A::Matrix{Float64}, b::Vector{Float64}, coef::Float64)
    n, d = size(A)
    regularizer = BoundedL2Regularizer(coef)
    return BoundedL2LinearRegression(A, b, regularizer, n, d)
end

# Implement oracle interface
function value(oracle::BoundedL2LinearRegression, x::Vector{Float64})
    # Linear regression loss
    residual = oracle.A * x - oracle.b
    base_loss = 0.5 * LinearAlgebra.norm(residual)^2 / oracle.n
    
    # Add bounded L2 regularization
    reg_loss = value(oracle.regularizer, x)
    
    return base_loss + reg_loss
end

function gradient(oracle::BoundedL2LinearRegression, x::Vector{Float64})
    # Linear regression gradient
    residual = oracle.A * x - oracle.b
    base_grad = oracle.A' * residual / oracle.n
    
    # Add bounded L2 gradient
    reg_grad = gradient(oracle.regularizer, x)
    
    return base_grad + reg_grad
end

function hessian(oracle::BoundedL2LinearRegression, x::Vector{Float64})
    # Linear regression Hessian
    base_hessian = oracle.A' * oracle.A / oracle.n
    
    # Add bounded L2 Hessian
    reg_hessian = hessian(oracle.regularizer, x)
    
    return base_hessian + reg_hessian
end

"""
    create_oracle_with_bounded_l2(A, b, coef::Float64)

Create a BoundedL2LinearRegression oracle.
"""
function create_oracle_with_bounded_l2(A, b, coef::Float64)
    return BoundedL2LinearRegression(A, b, coef)
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
    
    println("Point\t\t\tValue\t\tGrad Norm\tSmooth Const")
    println("-" ^ 65)
    
    for x in test_points
        val = value(reg, x)
        grad = gradient(reg, x)
        grad_norm = LinearAlgebra.norm(grad)
        smooth_const = smoothness(reg)
        
        println("$x\t$(round(val, digits=4))\t\t$(round(grad_norm, digits=4))\t\t$(smooth_const)")
    end
    
    # Test properties
    println("\nRegularizer Properties:")
    println("  Is zero: $(is_zero(reg))")
    println("  Smoothness constant: $(smoothness(reg))")
    
    # Test Hessian
    x_test = [1.0, -0.5, 2.0]
    H = hessian(reg, x_test)
    H_diag = hessian_diagonal(reg, x_test)
    
    println("\nHessian at $x_test:")
    println("  Diagonal: $H_diag")
    println("  Full diagonal: $(diag(H))")
    println("  Max eigenvalue: $(maximum(eigvals(H)))")
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
    
    # Test different regularizers
    regularizers = [
        ("No regularization", nothing),
        ("L2 (λ=0.1)", 0.1, "l2"),
        ("L1 (λ=0.1)", 0.1, "l1"),
        ("Bounded L2 (λ=0.1)", 0.1, "bounded_l2")
    ]
    
    println("\nRegularizer\t\tLoss\t\tParam Norm\tSparsity\tError")
    println("-" ^ 70)
    
    results = []
    
    for reg_info in regularizers
        if length(reg_info) == 2
            # No regularization
            oracle = LinearRegressionOracle(A, b)
            x_sol = (A' * A) \ (A' * b)  # Analytical solution
        else
            reg_name, coef, reg_type = reg_info
            
            if reg_type == "l2"
                oracle = LinearRegressionOracle(A, b, l2=coef)
                x_sol = (A' * A + coef * I(d)) \ (A' * b)  # Ridge solution
            elseif reg_type == "l1"
                oracle = LinearRegressionOracle(A, b, l1=coef)
                # Use gradient descent for L1
                x_sol = optimize_with_gradient_descent(oracle, zeros(d), lr=0.01, max_iter=2000)
            elseif reg_type == "bounded_l2"
                oracle = create_oracle_with_bounded_l2(A, b, coef)
                # Use gradient descent for bounded L2
                x_sol = optimize_with_gradient_descent(oracle, zeros(d), lr=0.01, max_iter=2000)
            end
        end
        
        # Compute metrics
        loss_val = value(oracle, x_sol)
        param_norm = LinearAlgebra.norm(x_sol)
        sparsity = sum(abs.(x_sol) .< 1e-3)
        error = LinearAlgebra.norm(x_sol - x_true)
        
        reg_name = length(reg_info) == 2 ? reg_info[1] : reg_info[1]
        println("$reg_name\t\t$(round(loss_val, digits=4))\t\t$(round(param_norm, digits=4))\t\t$sparsity/$d\t\t$(round(error, digits=4))")
        
        push!(results, (reg_name, x_sol, loss_val, error))
    end
    
    return results
end

"""
    optimize_with_gradient_descent(oracle, x0; lr=0.01, max_iter=1000, tol=1e-6)

Simple gradient descent optimization.
"""
function optimize_with_gradient_descent(oracle, x0; lr=0.01, max_iter=1000, tol=1e-6)
    x = copy(x0)
    
    for i in 1:max_iter
        grad = gradient(oracle, x)
        grad_norm = LinearAlgebra.norm(grad)
        
        if grad_norm < tol
            break
        end
        
        x = x - lr * grad
    end
    
    return x
end

"""
    visualize_bounded_l2()

Visualize the bounded L2 regularization function.
"""
function visualize_bounded_l2()
    println("\n=== Visualizing Bounded L2 ===")
    
    # Create different regularizers for comparison
    reg_bounded = BoundedL2Regularizer(1.0)
    
    # Test range
    x_range = -3:0.1:3
    
    # Compute values for 1D case
    values_bounded = []
    values_l2 = []
    values_l1 = []
    gradients_bounded = []
    
    for x_val in x_range
        x_vec = [x_val]
        
        # Bounded L2
        val_bounded = value(reg_bounded, x_vec)
        grad_bounded = gradient(reg_bounded, x_vec)[1]
        
        # Standard L2
        val_l2 = 0.5 * x_val^2
        
        # Standard L1
        val_l1 = abs(x_val)
        
        push!(values_bounded, val_bounded)
        push!(values_l2, val_l2)
        push!(values_l1, val_l1)
        push!(gradients_bounded, grad_bounded)
    end
    
    # Create plots
    p1 = plot(x_range, values_bounded, label="Bounded L2", linewidth=2, color=:red)
    plot!(p1, x_range, values_l2, label="L2", linewidth=2, color=:blue)
    plot!(p1, x_range, values_l1, label="L1", linewidth=2, color=:green)
    xlabel!(p1, "x")
    ylabel!(p1, "Regularization Value")
    title!(p1, "Regularization Functions")
    
    p2 = plot(x_range, gradients_bounded, label="Bounded L2 Gradient", linewidth=2, color=:red)
    plot!(p2, x_range, 2 .* collect(x_range), label="L2 Gradient", linewidth=2, color=:blue)
    xlabel!(p2, "x")
    ylabel!(p2, "Gradient")
    title!(p2, "Regularization Gradients")
    
    p_combined = plot(p1, p2, layout=(2, 1), size=(600, 600))
    
    try
        if !isdir("examples")
            mkdir("examples")
        end
        savefig(p_combined, "examples/bounded_l2_comparison.png")
        println("✓ Visualization saved to examples/bounded_l2_comparison.png")
    catch e
        println("Warning: Could not save plot: $e")
    end
    
    return p_combined
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
        oracle_bounded = create_oracle_with_bounded_l2(A, b, coef)
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
        loss_val = value(oracle, x)
        grad = gradient(oracle, x)
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
    
    # Visualize the regularization function
    plot_result = visualize_bounded_l2()
    
    # Compare optimization performance
    optimization_comparison()
    
    println("\n" * "=" * 50)
    println("Key insights about Bounded L2 regularization:")
    println("- Smooth (differentiable) everywhere, unlike L1")
    println("- Nonconvex, unlike L2")
    println("- Bounded penalty (asymptotes to 0.5 for large |x|)")
    println("- No closed-form proximal operator")
    println("- Smoothness constant equals the coefficient")
    println("- Useful for benchmarking nonconvex optimization algorithms")
    
    println("\nExample completed successfully!")
end

main()