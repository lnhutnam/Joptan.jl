"""
Adagrad with New Optimizer Framework

This example demonstrates the new Adagrad implementation using the
generalized optimizer framework in Joptan.jl.
"""

using Joptan
using Plots
using LinearAlgebra
using Random
using Statistics

# Set random seed for reproducibility
Random.seed!(42)

# Test basic Adagrad functionality
function test_basic_adagrad()
    println("Adagrad Test")
    println("=" ^ 60)
    
    # Define problem
    loss_func = x -> rosenbrock(x)
    grad_func = x -> rosenbrock_gradient(x)
    x0 = [-1.0, 1.0]
    
    println("Problem: Rosenbrock function")
    println("Starting point: $x0")
    println("Initial loss: $(round(loss_func(x0), digits=6))")
    
    # Create Adagrad optimizer
    adagrad = AdagradOptimizer(loss_func, grad_func, 
                              primal_dual=false, lr=0.5, delta=1e-8,
                              label="Adagrad-Standard")
    
    println("\nRunning Adagrad optimization...")
    
    # Run optimization
    trace = run!(adagrad, x0, it_max=2000, verbose=true)
    
    # Analyze results
    println("\nResults Analysis:")
    println("Final parameters: [$(round(adagrad.x[1], digits=6)), $(round(adagrad.x[2], digits=6))]")
    println("Distance to optimum: $(round(norm(adagrad.x - [1.0, 1.0]), digits=6))")
    
    # Get convergence statistics
    stats = get_convergence_statistics(trace)
    println("\nConvergence Statistics:")
    for (key, value) in stats
        if isa(value, Number)
            println("  $key: $(round(value, digits=6))")
        else
            println("  $key: $value")
        end
    end
    
    return trace, adagrad
end

# Test stochastic Adagrad
function test_stochastic_adagrad()
    println("\n" * "=" ^ 60)
    println("Stochastic Adagrad Test")
    println("=" ^ 60)
    
    # Define problem
    loss_func = x -> rastrigin(x)
    grad_func = x -> rastrigin_gradient(x)
    x0 = [2.0, -2.0]
    
    println("Problem: Rastrigin function")
    println("Starting point: $x0")
    println("Initial loss: $(round(loss_func(x0), digits=6))")
    
    # Create stochastic Adagrad optimizer with multiple seeds
    adagrad_stoch = AdagradStochasticOptimizer(loss_func, grad_func,
                                              primal_dual=false, lr=0.1, delta=1e-8,
                                              n_seeds=5, label="Adagrad-Stochastic")
    
    println("\nRunning stochastic Adagrad optimization with 5 seeds...")
    
    # Run optimization
    trace = run!(adagrad_stoch, x0, it_max=1000, verbose=true)
    
    # Analyze results
    stats = get_convergence_statistics(trace)
    println("\nStochastic Results Analysis:")
    for (key, value) in stats
        if isa(value, Number)
            println("  $key: $(round(value, digits=6))")
        end
    end
    
    # Get best seed
    best_seed = get_best_seed(trace)
    println("\nBest seed: $best_seed")
    if best_seed !== nothing
        best_trace = trace.seed_traces[best_seed]
        final_x = best_trace.xs[end]
        println("Best final parameters: [$(round(final_x[1], digits=6)), $(round(final_x[2], digits=6))]")
        println("Distance to optimum: $(round(norm(final_x), digits=6))")
    end
    
    return trace, adagrad_stoch
end

# Compare standard vs dual averaging
function compare_adagrad_variants()
    println("\n" * "=" ^ 60)
    println("Comparing Adagrad Variants")
    println("=" ^ 60)
    
    # Test problems
    problems = [
        ("Rosenbrock", rosenbrock, rosenbrock_gradient, [-1.0, 1.0], [1.0, 1.0]),
        ("Rastrigin", rastrigin, rastrigin_gradient, [2.0, -2.0], [0.0, 0.0])
    ]
    
    results = Dict()
    
    for (prob_name, loss_func, grad_func, x0, x_true) in problems
        println("\n--- $prob_name Function ---")
        
        # Test both variants
        variants = [
            ("Standard", false, 0.5),
            ("Dual Averaging", true, 0.5)
        ]
        
        prob_results = Dict()
        
        for (variant_name, primal_dual, lr) in variants
            println("\nTesting $variant_name...")
            
            # Create optimizer
            adagrad = AdagradOptimizer(loss_func, grad_func,
                                     primal_dual=primal_dual, lr=lr, delta=1e-8,
                                     label="Adagrad-$variant_name")
            
            # Run optimization
            trace = run!(adagrad, x0, it_max=2000, verbose=false)
            
            # Store results
            final_loss = loss_func(adagrad.x)
            final_grad_norm = norm(grad_func(adagrad.x))
            distance_to_true = norm(adagrad.x - x_true)
            
            prob_results[variant_name] = Dict(
                "final_loss" => final_loss,
                "final_grad_norm" => final_grad_norm,
                "distance_to_true" => distance_to_true,
                "iterations" => adagrad.it,
                "trace" => trace,
                "optimizer" => adagrad
            )
            
            println("  Final loss: $(round(final_loss, digits=6))")
            println("  Distance to optimum: $(round(distance_to_true, digits=6))")
            println("  Iterations: $(adagrad.it)")
        end
        
        results[prob_name] = prob_results
    end
    
    return results
end

# Visualize optimization traces
function visualize_traces()
    println("\n" * "=" ^ 60)
    println("Visualizing Optimization Traces")
    println("=" ^ 60)
    
    try
        # Test on Rosenbrock
        loss_func = x -> rosenbrock(x)
        grad_func = x -> rosenbrock_gradient(x)
        x0 = [-1.0, 1.0]
        
        # Create optimizers
        adagrad_std = AdagradOptimizer(loss_func, grad_func,
                                      primal_dual=false, lr=0.5, delta=1e-8,
                                      label="Standard")
        
        adagrad_dual = AdagradOptimizer(loss_func, grad_func,
                                       primal_dual=true, lr=0.5, delta=1e-8,
                                       label="Dual Averaging")
        
        # Run optimizations
        trace_std = run!(adagrad_std, x0, it_max=2000, verbose=false)
        trace_dual = run!(adagrad_dual, x0, it_max=2000, verbose=false)
        
        # Get data for plotting
        losses_std = get_losses(trace_std)
        losses_dual = get_losses(trace_dual)
        
        grad_norms_std = get_grad_norms(trace_std)
        grad_norms_dual = get_grad_norms(trace_dual)
        
        # Create loss convergence plot
        p1 = plot(1:length(losses_std), losses_std,
                 label="Standard", linewidth=2, color=:blue,
                 title="Adagrad Loss Convergence",
                 xlabel="Iteration", ylabel="Loss",
                 yscale=:log10)
        
        plot!(p1, 1:length(losses_dual), losses_dual,
              label="Dual Averaging", linewidth=2, color=:red)
        
        # Create gradient norm plot
        p2 = plot(1:length(grad_norms_std), grad_norms_std,
                 label="Standard", linewidth=2, color=:blue,
                 title="Gradient Norm Convergence",
                 xlabel="Iteration", ylabel="||∇f||",
                 yscale=:log10)
        
        plot!(p2, 1:length(grad_norms_dual), grad_norms_dual,
              label="Dual Averaging", linewidth=2, color=:red)
        
        # Create optimization path plot
        x_path_std = [x[1] for x in trace_std.xs]
        y_path_std = [x[2] for x in trace_std.xs]
        x_path_dual = [x[1] for x in trace_dual.xs]
        y_path_dual = [x[2] for x in trace_dual.xs]
        
        # Rosenbrock contour
        x_range = -2:0.1:2
        y_range = -1:0.1:3
        Z = [rosenbrock([x, y]) for y in y_range, x in x_range]
        
        p3 = contour(x_range, y_range, Z', levels=50, fill=true, alpha=0.7,
                    title="Optimization Paths",
                    xlabel="x₁", ylabel="x₂", color=:viridis)
        
        plot!(p3, x_path_std, y_path_std, linewidth=2, color=:blue, label="Standard")
        plot!(p3, x_path_dual, y_path_dual, linewidth=2, color=:red, label="Dual Averaging")
        
        # Add markers
        scatter!(p3, [x0[1]], [x0[2]], color=:green, markersize=8, label="Start")
        scatter!(p3, [1.0], [1.0], color=:black, markersize=8, marker=:star, label="Optimum")
        
        # Combine plots
        p_combined = plot(p1, p2, p3, layout=(2, 2), size=(800, 600))
        
        # Save plots
        savefig(p1, "examples/adagrad_new_loss_convergence.png")
        savefig(p2, "examples/adagrad_new_gradient_convergence.png")
        savefig(p3, "examples/adagrad_new_optimization_paths.png")
        savefig(p_combined, "examples/adagrad_new_combined_analysis.png")
        
        println("✓ Plots saved successfully!")
        return p_combined
        
    catch e
        println("Warning: Could not create plots: $e")
        return nothing
    end
end

# Test learning rate sensitivity
function test_learning_rate_sensitivity()
    println("\n" * "=" ^ 60)
    println("Learning Rate Sensitivity Analysis")
    println("=" ^ 60)
    
    # Test on Rosenbrock
    loss_func = x -> rosenbrock(x)
    grad_func = x -> rosenbrock_gradient(x)
    x0 = [-1.0, 1.0]
    
    lr_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    
    println("Testing learning rates: $lr_values")
    println("\nLR\tFinal Loss\tGrad Norm\tDistance\tIterations")
    println("-" ^ 60)
    
    results = Dict()
    
    for lr in lr_values
        # Create optimizer
        adagrad = AdagradOptimizer(loss_func, grad_func,
                                 primal_dual=false, lr=lr, delta=1e-8,
                                 tolerance=1e-6, label="Adagrad-LR$lr")
        
        # Run optimization
        trace = run!(adagrad, x0, it_max=2000, verbose=false)
        
        # Extract results
        final_loss = loss_func(adagrad.x)
        final_grad_norm = norm(grad_func(adagrad.x))
        distance = norm(adagrad.x - [1.0, 1.0])
        
        results[lr] = Dict(
            "final_loss" => final_loss,
            "final_grad_norm" => final_grad_norm,
            "distance" => distance,
            "iterations" => adagrad.it,
            "trace" => trace
        )
        
        println("$(lr)\t$(round(final_loss, digits=6))\t$(round(final_grad_norm, digits=6))\t$(round(distance, digits=6))\t$(adagrad.it)")
    end
    
    # Find best learning rate
    best_lr = nothing
    best_loss = Inf
    
    for (lr, result) in results
        if result["final_loss"] < best_loss
            best_loss = result["final_loss"]
            best_lr = lr
        end
    end
    
    println("\nBest learning rate: $best_lr")
    println("Best final loss: $(round(best_loss, digits=6))")
    
    return results, best_lr
end

# Test trace functionality
function test_trace_functionality()
    println("\n" * "=" ^ 60)
    println("Testing Trace Functionality")
    println("=" ^ 60)
    
    # Create stochastic optimizer
    loss_func = x -> rastrigin(x)
    grad_func = x -> rastrigin_gradient(x)
    x0 = [1.0, 1.0]
    
    adagrad_stoch = AdagradStochasticOptimizer(loss_func, grad_func,
                                              primal_dual=false, lr=0.1, delta=1e-8,
                                              n_seeds=3, label="Trace-Test")
    
    # Run optimization
    trace = run!(adagrad_stoch, x0, it_max=500, verbose=false)
    
    # Test trace methods
    println("Trace functionality tests:")
    println("- Number of seeds: $(length(trace.seeds))")
    println("- Seeds: $(trace.seeds)")
    
    # Get mean and std losses
    mean_losses = get_mean_losses(trace)
    std_losses = get_std_losses(trace)
    
    println("- Mean losses computed: $(length(mean_losses)) points")
    println("- Std losses computed: $(length(std_losses)) points")
    
    # Get best seed
    best_seed = get_best_seed(trace)
    println("- Best seed: $best_seed")
    
    if best_seed !== nothing
        best_trace = trace.seed_traces[best_seed]
        println("- Best seed iterations: $(length(best_trace.xs))")
        println("- Best seed final loss: $(round(get_losses(best_trace)[end], digits=6))")
    end
    
    # Test adding seeds
    println("\nAdding 2 more seeds...")
    add_seeds!(adagrad_stoch, 2)
    println("- Total seeds now: $(length(adagrad_stoch.seeds))")
    
    return trace
end

# Test with linear regression
function test_linear_regression()
    println("\n" * "=" ^ 60)
    println("Testing Adagrad on Linear Regression")
    println("=" ^ 60)
    
    # Generate synthetic data
    n, d = 100, 10
    A = randn(n, d)
    x_true = randn(d)
    b = A * x_true + 0.1 * randn(n)
    
    println("Generated linear regression problem:")
    println("- Samples: $n, Features: $d")
    println("- True parameter norm: $(round(norm(x_true), digits=4))")
    
    # Create loss functions
    lrl = LinearRegressionLoss(A, b, l2=0.01)
    loss_func = x -> linear_regression_loss(lrl, x)
    grad_func = x -> linear_regression_gradient(lrl, x)
    
    # Starting point
    x0 = zeros(d)
    
    println("Initial loss: $(round(loss_func(x0), digits=6))")
    
    # Create and run Adagrad
    adagrad = AdagradOptimizer(loss_func, grad_func,
                              primal_dual=false, lr=1.0, delta=1e-8,
                              tolerance=1e-8, label="Adagrad-LinReg")
    
    trace = run!(adagrad, x0, it_max=1000, verbose=false)
    
    # Analyze results
    println("\nResults:")
    println("- Final loss: $(round(loss_func(adagrad.x), digits=6))")
    println("- Distance to true params: $(round(norm(adagrad.x - x_true), digits=6))")
    println("- Iterations: $(adagrad.it)")
    
    # Compare with analytical solution
    x_analytical = (A' * A + 0.01 * I(d)) \ (A' * b)
    analytical_loss = loss_func(x_analytical)
    
    println("\nComparison with analytical solution:")
    println("- Analytical loss: $(round(analytical_loss, digits=6))")
    println("- Adagrad vs analytical distance: $(round(norm(adagrad.x - x_analytical), digits=6))")
    
    return trace, adagrad, x_analytical
end

# Main function
function main()
    println("Adagrad")
    println("=" ^ 60)
    
    # Test basic functionality
    trace1, adagrad1 = test_basic_adagrad()
    
    # Test stochastic functionality
    trace2, adagrad2 = test_stochastic_adagrad()
    
    # Compare variants
    comparison_results = compare_adagrad_variants()
    
    # Visualize results
    plot_result = visualize_traces()
    
    # Test learning rate sensitivity
    lr_results, best_lr = test_learning_rate_sensitivity()
    
    # Test trace functionality
    trace_test = test_trace_functionality()
    
    # Test on linear regression
    trace_linreg, adagrad_linreg, x_analytical = test_linear_regression()
    
    println("\nBest learning rate found: $best_lr")
    
    # Show performance comparison
    println("\nPerformance comparison (Rosenbrock):")
    if haskey(comparison_results, "Rosenbrock")
        for (variant, result) in comparison_results["Rosenbrock"]
            println("- $variant: $(round(result["final_loss"], digits=6)) loss in $(result["iterations"]) iterations")
        end
    end
end

main()