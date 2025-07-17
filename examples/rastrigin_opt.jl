"""
Example: Optimizing the Rastrigin function

This example demonstrates how to use the Rastrigin function from Joptan.jl
and shows the challenges of optimizing highly multimodal functions.
"""

using Joptan
using Plots
using LinearAlgebra
using Random

# Function to demonstrate Rastrigin function properties
function demonstrate_rastrigin()
    println("=== Rastrigin Function Demonstration ===")
    
    # Test the global minimum
    x_min = [0.0, 0.0]
    f_min = rastrigin(x_min)
    grad_min = rastrigin_gradient(x_min)
    
    println("Global minimum:")
    println("  x* = $x_min")
    println("  f(x*) = $f_min")
    println("  ∇f(x*) = $grad_min")
    println()
    
    # Test some other points to show multimodality
    test_points = [
        [1.0, 1.0],
        [2.0, 2.0],
        [π, π],
        [0.5, -0.5],
        [3.0, -2.0]
    ]
    
    println("Function values at test points (showing multimodality):")
    for x in test_points
        f_val = rastrigin(x)
        grad_val = rastrigin_gradient(x)
        println("  x = $x: f(x) = $(round(f_val, digits=4)), ||∇f(x)|| = $(round(norm(grad_val), digits=4))")
    end
    println()
end

# Simple gradient descent with random restarts
function gradient_descent_multistart(f, grad_f, dim; num_starts=5, lr=0.01, max_iter=1000, tol=1e-6, search_range=5.0)
    best_x = nothing
    best_f = Inf
    all_results = []
    
    println("Running gradient descent with $num_starts random starts...")
    
    for start in 1:num_starts
        # Random starting point
        x0 = (rand(dim) .- 0.5) * 2 * search_range
        println("Start $start: x₀ = $x0")
        
        x = copy(x0)
        
        for i in 1:max_iter
            g = grad_f(x)
            
            # Check convergence
            if norm(g) < tol
                break
            end
            
            # Update with gradient descent
            x = x - lr * g
        end
        
        f_val = f(x)
        push!(all_results, (copy(x), f_val))
        
        println("  Result: x = $x, f(x) = $(round(f_val, digits=6))")
        
        # Track best result
        if f_val < best_f
            best_f = f_val
            best_x = copy(x)
        end
    end
    
    return best_x, best_f, all_results
end

# Optimize Rastrigin function
function optimize_rastrigin()
    println("=== Optimizing Rastrigin Function ===")
    
    # 2D optimization
    dim = 2
    best_x, best_f, all_results = gradient_descent_multistart(
        rastrigin, rastrigin_gradient, dim, 
        num_starts=10, lr=0.01, max_iter=2000, search_range=5.0
    )
    
    println()
    println("Best result across all starts:")
    println("  x_best = $best_x")
    println("  f(x_best) = $(round(best_f, digits=6))")
    println("  Distance to true minimum: $(norm(best_x))")
    
    return best_x, best_f, all_results
end

# Visualize the Rastrigin function and optimization results
function visualize_rastrigin(all_results)
    println("\n=== Creating Visualization ===")
    
    # Create surface plot of Rastrigin function
    x_range = -5:0.1:5
    y_range = -5:0.1:5
    
    # Compute function values on grid
    Z = [rastrigin([x, y]) for y in y_range, x in x_range]
    
    # Create contour plot
    p = contour(x_range, y_range, Z, levels=50, fill=true, 
                title="Rastrigin Function with Optimization Results",
                xlabel="x₁", ylabel="x₂")
    
    # Add optimization results
    x_coords = [result[1][1] for result in all_results]
    y_coords = [result[1][2] for result in all_results]
    f_vals = [result[2] for result in all_results]
    
    scatter!(p, x_coords, y_coords, 
            color=:red, markersize=6, 
            label="Local Minima Found")
    
    scatter!(p, [0.0], [0.0], 
            color=:black, markersize=10, marker=:star, 
            label="Global Minimum")
    
    return p
end

# Compare different parameter A values
function parameter_comparison()
    println("\n=== Parameter A Comparison ===")
    
    A_values = [1.0, 5.0, 10.0, 20.0]
    x_test = [1.0, 1.0]
    
    println("Effect of parameter A on function value at x = $x_test:")
    for A in A_values
        f_val = rastrigin(x_test, A=A)
        grad_val = rastrigin_gradient(x_test, A=A)
        println("  A = $A: f(x) = $(round(f_val, digits=4)), ||∇f(x)|| = $(round(norm(grad_val), digits=4))")
    end
    println()
end

# Higher dimensional example
function high_dimensional_example()
    println("\n=== High-Dimensional Rastrigin Example ===")
    
    dimensions = [2, 5, 10, 20]
    
    for n in dimensions
        println("Testing $n-dimensional Rastrigin:")
        
        # Test points
        x_origin = zeros(n)
        x_random = randn(n)
        
        f_origin = rastrigin(x_origin)
        f_random = rastrigin(x_random)
        
        grad_origin = rastrigin_gradient(x_origin)
        grad_random = rastrigin_gradient(x_random)
        
        println("  At origin: f(0) = $(round(f_origin, digits=4)), ||∇f(0)|| = $(round(norm(grad_origin), digits=4))")
        println("  At random point: f(x) = $(round(f_random, digits=4)), ||∇f(x)|| = $(round(norm(grad_random), digits=4))")
        
        # Show how the number of local minima grows exponentially
        estimated_minima = 3^n  # Rough estimate
        println("  Estimated number of local minima: ~$estimated_minima")
        println()
    end
end

# Demonstrate the challenge of escaping local minima
function local_minima_challenge()
    println("\n=== Local Minima Challenge ===")
    
    # Start near a local minimum (not the global one)
    x_local = [π, π]  # Near a local minimum
    f_local = rastrigin(x_local)
    
    println("Starting near local minimum at x = $x_local")
    println("Function value: $(round(f_local, digits=4))")
    
    # Try gradient descent from this point
    x = copy(x_local)
    lr = 0.01
    
    println("Gradient descent steps:")
    for i in 1:10
        g = rastrigin_gradient(x)
        x_new = x - lr * g
        f_new = rastrigin(x_new)
        
        println("  Step $i: x = [$(round(x_new[1], digits=4)), $(round(x_new[2], digits=4))], f(x) = $(round(f_new, digits=4))")
        
        if norm(g) < 1e-6
            println("  Converged to local minimum!")
            break
        end
        
        x = x_new
    end
    
    println("Final distance to global minimum: $(round(norm(x), digits=4))")
    println()
end

# Main execution
function main()
    println("Rastrigin Function Optimization Example")
    println("=" ^ 50)
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Demonstrate function properties
    demonstrate_rastrigin()
    
    # Parameter comparison
    parameter_comparison()
    
    # Optimize the function
    best_x, best_f, all_results = optimize_rastrigin()
    
    # Create visualization (only for 2D case)
    if length(all_results[1][1]) == 2
        plot_result = visualize_rastrigin(all_results)
        
        # Save plot
        try
            savefig(plot_result, "examples/rastrigin_optimization.png")
            println("Plot saved as 'examples/rastrigin_optimization.png'")
        catch
            println("Could not save plot (Plots.jl might not be properly configured)")
        end
    end
    
    # Demonstrate local minima challenge
    local_minima_challenge()
    
    # High-dimensional example
    high_dimensional_example()
    
    println("\nExample completed!")
    println("Note: The Rastrigin function is highly multimodal, making it challenging")
    println("for local optimization methods. Global optimization techniques or")
    println("metaheuristic algorithms are typically needed for better results.")
end

main()