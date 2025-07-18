"""
Example: Linear Regression with Joptan.jl

This example demonstrates how to use the linear regression loss functions
from Joptan.jl, including regularized variants and optimization examples.
"""

using Joptan
using LinearAlgebra
using Plots
using Random
using Statistics
using StatsBase

# Import norm explicitly to avoid conflicts
import LinearAlgebra: norm

# Set random seed for reproducibility
Random.seed!(42)

# Generate synthetic data
function generate_regression_data(n::Int, d::Int; noise_std::Float64=0.1, condition_number::Float64=1.0)
    """
    Generate synthetic linear regression data.
    
    Arguments:
    - n: Number of samples
    - d: Number of features
    - noise_std: Standard deviation of noise
    - condition_number: Condition number of the design matrix
    
    Returns:
    - A: Design matrix (n × d)
    - b: Target vector (n × 1)
    - x_true: True parameters used to generate data
    """
    println("Generating synthetic regression data...")
    println("  Samples: $n, Features: $d")
    println("  Noise std: $noise_std, Condition number: $condition_number")
    
    # Generate true parameters
    x_true = randn(d)
    
    # Generate design matrix with specified condition number
    if condition_number > 1.0
        # Create matrix with specified condition number
        U, _, V = svd(randn(n, d))
        min_dim = min(n, d)
        singular_values = exp.(range(0, log(condition_number), length=min_dim))
        if d <= n
            A = U[:, 1:d] * Diagonal(singular_values) * V'
        else
            # Handle case where d > n
            A = U * Diagonal(singular_values) * V[:, 1:min_dim]'
        end
    else
        A = randn(n, d)
    end
    
    # Generate targets with noise
    b = A * x_true + noise_std * randn(n)
    
    println("  True parameter norm: $(round(LinearAlgebra.norm(x_true), digits=4))")
    println("  Actual condition number: $(round(cond(A), digits=2))")
    
    return A, b, x_true
end

# Demonstrate basic linear regression
function demonstrate_basic_linear_regression()
    println("\n=== Basic Linear Regression ===")
    
    # Generate data
    n, d = 100, 5
    A, b, x_true = generate_regression_data(n, d)
    
    # Create linear regression oracle
    lro = Joptan.LinearRegressionOracle(A, b)
    
    # Test point
    x_test = zeros(d)
    
    # Compute loss, gradient, and Hessian using oracle methods
    loss_val = value(lro, x_test)
    grad_val = gradient(lro, x_test)
    hess_val = hessian(lro, x_test)
    
    println("At x = zeros:")
    println("  Loss: $(round(loss_val, digits=6))")
    println("  Gradient norm: $(round(LinearAlgebra.norm(grad_val), digits=6))")
    println("  Hessian condition number: $(round(cond(hess_val), digits=2))")
    
    # Analytical solution (ordinary least squares)
    x_ols = (A' * A) \ (A' * b)
    loss_ols = value(lro, x_ols)
    grad_ols = gradient(lro, x_ols)
    
    println("\nOLS solution:")
    println("  Loss: $(round(loss_ols, digits=6))")
    println("  Gradient norm: $(round(LinearAlgebra.norm(grad_ols), digits=6))")
    println("  Distance to true params: $(round(LinearAlgebra.norm(x_ols - x_true), digits=6))")
    
    return lro, x_ols, x_true
end

# Demonstrate regularized linear regression
function demonstrate_regularized_regression()
    println("\n=== Regularized Linear Regression ===")
    
    # Generate data (more features than samples for regularization demo)
    n, d = 50, 100
    A, b, x_true = generate_regression_data(n, d, noise_std=0.5)
    
    # Test different regularization strengths
    regularization_strengths = [0.0, 0.1, 1.0, 10.0]
    
    println("L2 Regularization comparison:")
    println("λ₂\t\tLoss\t\tParam Norm\tGrad Norm")
    println("-" ^ 55)
    
    solutions = []
    
    for l2 in regularization_strengths
        lro = LinearRegressionOracle(A, b, l2=l2)
        
        # Analytical solution for Ridge regression
        x_ridge = (A' * A + l2 * I(d)) \ (A' * b)
        
        loss_val = value(lro, x_ridge)
        grad_val = gradient(lro, x_ridge)
        
        println("$(l2)\t\t$(round(loss_val, digits=6))\t$(round(LinearAlgebra.norm(x_ridge), digits=4))\t\t$(round(LinearAlgebra.norm(grad_val), digits=6))")
        
        push!(solutions, (l2, x_ridge, loss_val))
    end
    
    # Test L1 regularization (Lasso)
    println("\nL1 Regularization (Lasso) - using simple gradient descent:")
    l1_strengths = [0.1, 1.0, 10.0]
    
    for l1 in l1_strengths
        lro = LinearRegressionOracle(A, b, l1=l1)
        
        # Simple gradient descent for Lasso (subgradient method)
        x = zeros(d)
        lr = 0.01
        
        for i in 1:1000
            grad = gradient(lro, x)
            x = x - lr * grad
            
            if i % 200 == 0
                loss_val = value(lro, x)
                println("  λ₁=$(l1), iter=$i: loss=$(round(loss_val, digits=6)), sparsity=$(sum(abs.(x) .< 1e-3))/$(d)")
            end
        end
    end
    
    return solutions
end

# Demonstrate gradient descent optimization
function demonstrate_gradient_descent()
    println("\n=== Gradient Descent Optimization ===")
    
    # Generate well-conditioned data
    n, d = 200, 10
    A, b, x_true = generate_regression_data(n, d, condition_number=5.0)
    
    lro = LinearRegressionOracle(A, b, l2=0.1)
    
    # Gradient descent
    x = randn(d)  # Random initialization
    lr = 0.01
    max_iter = 1000
    tol = 1e-6
    
    history = []
    
    println("Running gradient descent...")
    println("Iter\t\tLoss\t\tGrad Norm\tStep Size")
    println("-" ^ 50)
    
    for i in 1:max_iter
        loss_val = value(lro, x)
        grad_val = gradient(lro, x)
        grad_norm = LinearAlgebra.norm(grad_val)
        
        push!(history, (i, loss_val, grad_norm, LinearAlgebra.norm(x - x_true)))
        
        if i % 100 == 0 || i == 1
            println("$i\t\t$(round(loss_val, digits=6))\t$(round(grad_norm, digits=6))\t$(lr)")
        end
        
        # Check convergence
        if grad_norm < tol
            println("Converged at iteration $i")
            break
        end
        
        # Update
        x = x - lr * grad_val
    end
    
    # Final results
    final_loss = value(lro, x)
    final_grad_norm = LinearAlgebra.norm(gradient(lro, x))
    distance_to_true = LinearAlgebra.norm(x - x_true)
    
    println("\nFinal results:")
    println("  Loss: $(round(final_loss, digits=6))")
    println("  Gradient norm: $(round(final_grad_norm, digits=6))")
    println("  Distance to true params: $(round(distance_to_true, digits=6))")
    
    return history, x, x_true
end

# Demonstrate stochastic gradient descent
function demonstrate_stochastic_gradient_descent()
    println("\n=== Stochastic Gradient Descent ===")
    
    # Generate larger dataset
    n, d = 1000, 20
    A, b, x_true = generate_regression_data(n, d)
    
    lro = LinearRegressionOracle(A, b, l2=0.01)
    
    # SGD parameters
    x = randn(d)
    lr = 0.1
    batch_size = 32
    max_iter = 2000
    
    history_sgd = []
    
    println("Running stochastic gradient descent...")
    println("Batch size: $batch_size")
    println("Iter\t\tLoss\t\tGrad Norm")
    println("-" ^ 40)
    
    for i in 1:max_iter
        # Compute full loss and gradient for monitoring
        if i % 200 == 0 || i == 1
            loss_val = value(lro, x)
            grad_val = gradient(lro, x)
            grad_norm = LinearAlgebra.norm(grad_val)
            
            push!(history_sgd, (i, loss_val, grad_norm))
            println("$i\t\t$(round(loss_val, digits=6))\t$(round(grad_norm, digits=6))")
        end
        
        # Stochastic gradient step
        stoch_grad = stochastic_gradient(lro, x, nothing, batch_size=batch_size)
        x = x - lr * stoch_grad
        
        # Decay learning rate
        if i % 500 == 0
            lr *= 0.9
        end
    end
    
    # Final results
    final_loss = value(lro, x)
    distance_to_true = LinearAlgebra.norm(x - x_true)
    
    println("\nSGD final results:")
    println("  Loss: $(round(final_loss, digits=6))")
    println("  Distance to true params: $(round(distance_to_true, digits=6))")
    
    return history_sgd, x
end

# Visualize optimization convergence
function visualize_convergence(history, title_str)
    println("\n=== Creating Convergence Plot ===")
    
    iterations = [h[1] for h in history]
    losses = [h[2] for h in history]
    grad_norms = [h[3] for h in history]
    
    # Loss convergence
    p1 = plot(iterations, losses,
             title="$title_str - Loss Convergence",
             xlabel="Iteration",
             ylabel="Loss",
             yscale=:log10,
             linewidth=2,
             color=:blue,
             label="Loss")
    
    # Gradient norm convergence
    p2 = plot(iterations, grad_norms,
             title="$title_str - Gradient Norm",
             xlabel="Iteration",
             ylabel="||∇f||",
             yscale=:log10,
             linewidth=2,
             color=:red,
             label="Gradient Norm")
    
    # Combined plot
    p_combined = plot(p1, p2, layout=(2, 1), size=(600, 400))
    
    return p_combined
end

# Compare different optimization methods
function compare_optimization_methods()
    println("\n=== Optimization Methods Comparison ===")
    
    # Generate data
    n, d = 100, 10
    A, b, x_true = generate_regression_data(n, d, condition_number=10.0)
    
    methods = [
        ("Analytical (OLS)", nothing, 0.0),
        ("Gradient Descent", 0.01, 0.0),
        ("Ridge (λ=0.1)", 0.01, 0.1),
        ("Ridge (λ=1.0)", 0.01, 1.0)
    ]
    
    results = []
    
    println("Method\t\t\tLoss\t\tDistance to True\tTime (ms)")
    println("-" ^ 65)
    
    for (method_name, lr, l2) in methods
        lro = LinearRegressionOracle(A, b, l2=l2)
        
        time_taken = @elapsed begin
            if method_name == "Analytical (OLS)"
                if l2 == 0.0
                    x_sol = (A' * A) \ (A' * b)
                else
                    x_sol = (A' * A + l2 * I(d)) \ (A' * b)
                end
            else
                # Gradient descent
                x_sol = randn(d)
                for i in 1:1000
                    grad = gradient(lro, x_sol)
                    x_sol = x_sol - lr * grad
                    
                    if LinearAlgebra.norm(grad) < 1e-6
                        break
                    end
                end
            end
        end
        
        final_loss = value(lro, x_sol)
        distance_to_true = LinearAlgebra.norm(x_sol - x_true)
        
        println("$method_name\t\t$(round(final_loss, digits=6))\t$(round(distance_to_true, digits=6))\t\t$(round(time_taken*1000, digits=2))")
        
        push!(results, (method_name, x_sol, final_loss, distance_to_true))
    end
    
    return results
end

# Demonstrate smoothness properties
function demonstrate_smoothness_properties()
    println("\n=== Smoothness Properties ===")
    
    # Generate data with different condition numbers
    n, d = 50, 10
    condition_numbers = [1.0, 5.0, 10.0, 50.0]
    
    println("Condition Number\tSmoothness\tMax Smoothness\tAvg Smoothness")
    println("-" ^ 70)
    
    for cond_num in condition_numbers
        A, b, _ = generate_regression_data(n, d, condition_number=cond_num)
        lro = LinearRegressionOracle(A, b, l2=0.1)
        
        smooth_val = smoothness(lro)
        max_smooth = max_smoothness(lro)
        avg_smooth = average_smoothness(lro)
        
        println("$(cond_num)\t\t\t$(round(smooth_val, digits=4))\t\t$(round(max_smooth, digits=4))\t\t$(round(avg_smooth, digits=4))")
    end
end

# Test with real-world-like data
function test_with_realistic_data()
    println("\n=== Realistic Data Test ===")
    
    # Generate data with some realistic properties
    n, d = 200, 50
    
    # Create correlated features
    println("Creating correlated features...")
    X_base = randn(n, 10)
    X_corr = X_base * randn(10, d)  # Create correlations
    A = X_corr + 0.1 * randn(n, d)  # Add some noise
    
    # Create sparse true parameters
    x_true = zeros(d)
    sparse_indices = StatsBase.sample(1:d, 10, replace=false)
    x_true[sparse_indices] = randn(10)
    
    # Generate targets
    b = A * x_true + 0.1 * randn(n)
    
    println("  Effective rank of A: $(round(rank(A), digits=0))")
    println("  Condition number: $(round(cond(A), digits=2))")
    println("  True parameter sparsity: $(sum(abs.(x_true) .> 1e-6))/$(d)")
    
    # Test different regularization approaches
    methods = [
        ("No regularization", 0.0, 0.0),
        ("L2 only", 0.0, 1.0),
        ("L1 only", 1.0, 0.0),
        ("Elastic net", 0.5, 0.5)
    ]
    
    println("\nRegularization comparison:")
    println("Method\t\t\tLoss\t\tSparsity\tError")
    println("-" ^ 50)
    
    for (method_name, l1, l2) in methods
        lro = LinearRegressionOracle(A, b, l1=l1, l2=l2)
        
        # Use gradient descent (subgradient for L1)
        x = zeros(d)
        lr = 0.001
        
        for i in 1:2000
            grad = gradient(lro, x)
            x = x - lr * grad
            
            # Simple thresholding for L1 (crude but effective)
            if l1 > 0
                threshold = l1 * lr
                x = sign.(x) .* max.(abs.(x) .- threshold, 0)
            end
        end
        
        final_loss = value(lro, x)
        sparsity = sum(abs.(x) .< 1e-3)
        error = LinearAlgebra.norm(x - x_true)
        
        println("$method_name\t\t$(round(final_loss, digits=6))\t$(sparsity)/$(d)\t\t$(round(error, digits=4))")
    end
end

# Main function to run all demonstrations
function main()
    println("Linear Regression Loss Function Examples")
    println("=" ^ 50)
    
    # Basic demonstration
    lro, x_ols, x_true = demonstrate_basic_linear_regression()
    
    # Regularized regression
    solutions = demonstrate_regularized_regression()
    
    # Gradient descent
    history_gd, x_gd, x_true_gd = demonstrate_gradient_descent()
    
    # Stochastic gradient descent
    history_sgd, x_sgd = demonstrate_stochastic_gradient_descent()
    
    # Visualize convergence
    try
        p_gd = visualize_convergence(history_gd, "Gradient Descent")
        p_sgd = visualize_convergence(history_sgd, "Stochastic Gradient Descent")
        
        # Create examples directory if it doesn't exist
        if !isdir("examples")
            mkdir("examples")
        end
        
        # Save plots
        savefig(p_gd, "examples/linear_regression_gd_convergence.png")
        savefig(p_sgd, "examples/linear_regression_sgd_convergence.png")
        
        println("✓ Convergence plots saved!")
    catch e
        println("Warning: Could not create plots: $e")
    end
    
    # Compare optimization methods
    results = compare_optimization_methods()
    
    # Demonstrate smoothness properties
    demonstrate_smoothness_properties()
    
    # Test with realistic data
    test_with_realistic_data()
    
    println("\nExample completed successfully!")
end

# Test the simple functions
function test_simple_functions()
    println("\n=== Testing Simple Functions ===")
    
    # Generate small test data
    A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
    b = [1.0, 2.0, 3.0]
    x = [0.5, -0.5]
    
    # Test simple functions using the oracle
    lro_no_reg = LinearRegressionOracle(A, b)
    lro_reg = LinearRegressionOracle(A, b, l1=0.1, l2=0.1)
    
    loss_val = value(lro_no_reg, x)
    grad_val = gradient(lro_no_reg, x)
    hess_val = hessian(lro_no_reg, x)
    
    println("Simple function test:")
    println("  A = $A")
    println("  b = $b")
    println("  x = $x")
    println("  Loss: $(round(loss_val, digits=6))")
    println("  Gradient: $grad_val")
    println("  Hessian: $hess_val")
    
    # Test with regularization
    loss_reg = value(lro_reg, x)
    grad_reg = gradient(lro_reg, x)
    hess_reg = hessian(lro_reg, x)
    
    println("\nWith regularization (l1=0.1, l2=0.1):")
    println("  Loss: $(round(loss_reg, digits=6))")
    println("  Gradient: $grad_reg")
    println("  Hessian: $hess_reg")
    
    # Test convenience functions (if they exist)
    try
        loss_simple = linear_regression_loss(A, b, x, l1=0.0, l2=0.0)
        grad_simple = linear_regression_gradient(A, b, x, l1=0.0, l2=0.0)
        hess_simple = linear_regression_hessian(A, b, x, l1=0.0, l2=0.0)
        
        println("\nUsing convenience functions:")
        println("  Loss: $(round(loss_simple, digits=6))")
        println("  Gradient: $grad_simple")
        println("  Hessian: $hess_simple")
    catch e
        println("Convenience functions not available: $e")
    end
end

main()