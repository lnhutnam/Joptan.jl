"""
Example: Optimizing the Rosenbrock function

This example demonstrates how to use the Rosenbrock function from Joptan.jl
and optimize it using different approaches.
"""

using Joptan
using Plots
using LinearAlgebra

# Function to demonstrate Rosenbrock function properties
function demonstrate_rosenbrock()
    println("=== Rosenbrock Function Demonstration ===")
    
    # Test the global minimum
    x_min = [1.0, 1.0]
    f_min = rosenbrock(x_min)
    grad_min = rosenbrock_gradient(x_min)
    
    println("Global minimum:")
    println("  x* = $x_min")
    println("  f(x*) = $f_min")
    println("  ∇f(x*) = $grad_min")
    println()
    
    # Test some other points
    test_points = [
        [0.0, 0.0],
        [2.0, 4.0],
        [-1.0, 1.0],
        [1.0, 2.0]
    ]
    
    println("Function values at test points:")
    for x in test_points
        f_val = rosenbrock(x)
        grad_val = rosenbrock_gradient(x)
        println("  x = $x: f(x) = $(round(f_val, digits=4)), ||∇f(x)|| = $(round(norm(grad_val), digits=4))")
    end
    println()
end

# Simple gradient descent implementation for demonstration
function gradient_descent(f, grad_f, x0; lr=0.001, max_iter=1000, tol=1e-6)
    x = copy(x0)
    history = [copy(x)]
    
    for i in 1:max_iter
        g = grad_f(x)
        
        # Check convergence
        if norm(g) < tol
            println("Converged after $i iterations")
            break
        end
        
        # Update
        x = x - lr * g
        push!(history, copy(x))
        
        # Print progress
        if i % 100 == 0
            f_val = f(x)
            println("Iteration $i: f(x) = $(round(f_val, digits=6)), ||∇f(x)|| = $(round(norm(g), digits=6))")
        end
    end
    
    return x, history
end

# Optimize Rosenbrock function
function optimize_rosenbrock()
    println("=== Optimizing Rosenbrock Function ===")
    
    # Starting point
    x0 = [-1.0, 1.0]
    println("Starting point: $x0")
    println("Initial function value: $(rosenbrock(x0))")
    println()
    
    # Run optimization
    x_opt, history = gradient_descent(rosenbrock, rosenbrock_gradient, x0, lr=0.001, max_iter=5000)
    
    println()
    println("Final results:")
    println("  x_opt = $x_opt")
    println("  f(x_opt) = $(rosenbrock(x_opt))")
    println("  ||∇f(x_opt)|| = $(norm(rosenbrock_gradient(x_opt)))")
    println("  Distance to true minimum: $(norm(x_opt - [1.0, 1.0]))")
    
    return x_opt, history
end

# Visualize the optimization path (2D only)
function visualize_optimization(history)
    println("\n=== Creating Visualization ===")
    
    # Create contour plot of Rosenbrock function
    x_range = -2:0.1:2
    y_range = -1:0.1:3
    
    # Compute function values on grid
    Z = [rosenbrock([x, y]) for y in y_range, x in x_range]
    
    # Create contour plot
    p = contour(x_range, y_range, Z, levels=50, fill=true, 
                title="Rosenbrock Function Optimization Path",
                xlabel="x₁", ylabel="x₂")
    
    # Add optimization path
    x_path = [h[1] for h in history]
    y_path = [h[2] for h in history]
    
    plot!(p, x_path, y_path, linewidth=2, color=:red, label="Optimization Path")
    scatter!(p, [x_path[1]], [y_path[1]], color=:green, markersize=8, label="Start")
    scatter!(p, [x_path[end]], [y_path[end]], color=:blue, markersize=8, label="End")
    scatter!(p, [1.0], [1.0], color=:black, markersize=10, marker=:star, label="True Minimum")
    
    return p
end

# Higher dimensional example
function high_dimensional_example()
    println("\n=== High-Dimensional Rosenbrock Example ===")
    
    dimensions = [2, 5, 10, 20]
    
    for n in dimensions
        println("Testing $n-dimensional Rosenbrock:")
        
        # Random starting point
        x0 = randn(n)
        x_min = ones(n)  # True minimum
        
        f_start = rosenbrock(x0)
        f_min = rosenbrock(x_min)
        
        grad_start = rosenbrock_gradient(x0)
        grad_min = rosenbrock_gradient(x_min)
        
        println("  Starting point norm: $(round(norm(x0), digits=4))")
        println("  f(x₀) = $(round(f_start, digits=4))")
        println("  f(x*) = $(round(f_min, digits=4))")
        println("  ||∇f(x₀)|| = $(round(norm(grad_start), digits=4))")
        println("  ||∇f(x*)|| = $(round(norm(grad_min), digits=4))")
        println()
    end
end

# Main execution
function main()
    println("Rosenbrock Function Optimization Example")
    println("=" ^ 50)
    
    # Demonstrate function properties
    demonstrate_rosenbrock()
    
    # Optimize the function
    x_opt, history = optimize_rosenbrock()
    
    # Create visualization (only for 2D case)
    if length(history[1]) == 2
        plot_result = visualize_optimization(history)
        
        # Save plot
        try
            savefig(plot_result, "examples/rosenbrock_optimization.png")
            println("Plot saved as 'examples/rosenbrock_optimization.png'")
        catch
            println("Could not save plot (Plots.jl might not be properly configured)")
        end
    end
    
    # High-dimensional example
    high_dimensional_example()
    
    println("\nExample completed!")
end

main()