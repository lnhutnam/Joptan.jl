"""
Adagrad with Optimizer Framework

This example demonstrates the new Adagrad implementation using the
generalized optimizer framework in Joptan.jl.
"""

using Joptan
using LinearAlgebra
using Random
using Statistics

# Only use Plots if available
try
    using Plots
    global PLOTS_AVAILABLE = true
catch
    global PLOTS_AVAILABLE = false
    println("Plots.jl not available - visualization will be skipped")
end

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
    
    if !PLOTS_AVAILABLE
        println("Plots.jl not available - skipping visualization")
        return nothing
    end
    
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
        p1 = Plots.plot(1:length(losses_std), losses_std,
                 label="Standard", linewidth=2, color=:blue,
                 title="Adagrad Loss Convergence",
                 xlabel="Iteration", ylabel="Loss",
                 yscale=:log10)
        
        Plots.plot!(p1, 1:length(losses_dual), losses_dual,
              label="Dual Averaging", linewidth=2, color=:red)
        
        # Create gradient norm plot
        p2 = Plots.plot(1:length(grad_norms_std), grad_norms_std,
                 label="Standard", linewidth=2, color=:blue,
                 title="Gradient Norm Convergence",
                 xlabel="Iteration", ylabel="||∇f||",
                 yscale=:log10)
        
        Plots.plot!(p2, 1:length(grad_norms_dual), grad_norms_dual,
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
        
        p3 = Plots.contour(x_range, y_range, Z', levels=50, fill=true, alpha=0.7,
                    title="Optimization Paths",
                    xlabel="x₁", ylabel="x₂", color=:viridis)
        
        Plots.plot!(p3, x_path_std, y_path_std, linewidth=2, color=:blue, label="Standard")
        Plots.plot!(p3, x_path_dual, y_path_dual, linewidth=2, color=:red, label="Dual Averaging")
        
        # Add markers
        Plots.scatter!(p3, [x0[1]], [x0[2]], color=:green, markersize=8, label="Start")
        Plots.scatter!(p3, [1.0], [1.0], color=:black, markersize=8, marker=:star, label="Optimum")
        
        # Combine plots
        p_combined = Plots.plot(p1, p2, p3, layout=(2, 2), size=(800, 600))
        
        # Save plots
        if !isdir("examples")
            mkdir("examples")
        end
        Plots.savefig(p1, "examples/adagrad_new_loss_convergence.png")
        Plots.savefig(p2, "examples/adagrad_new_gradient_convergence.png")
        Plots.savefig(p3, "examples/adagrad_new_optimization_paths.png")
        Plots.savefig(p_combined, "examples/adagrad_new_combined_analysis.png")
        
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

# Test with linear regression - FIXED VERSION
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
    
    # Create loss functions using the oracle interface
    lro = LinearRegressionOracle(A, b, l2=0.01)
    loss_func = x -> value(lro, x)
    grad_func = x -> gradient(lro, x)
    
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
    
    # Test different Adagrad variants
    println("\n--- Testing Adagrad Variants ---")
    variants = [
        ("Standard", false, 1.0),
        ("Dual Averaging", true, 1.0)
    ]
    
    variant_results = Dict()
    
    for (variant_name, primal_dual, lr) in variants
        # Create optimizer
        adagrad_variant = AdagradOptimizer(loss_func, grad_func,
                                          primal_dual=primal_dual, lr=lr, delta=1e-8,
                                          tolerance=1e-8, label="Adagrad-LinReg-$variant_name")
        
        # Run optimization
        trace_variant = run!(adagrad_variant, x0, it_max=1000, verbose=false)
        
        final_loss = loss_func(adagrad_variant.x)
        distance_to_true = norm(adagrad_variant.x - x_true)
        distance_to_analytical = norm(adagrad_variant.x - x_analytical)
        
        variant_results[variant_name] = Dict(
            "final_loss" => final_loss,
            "distance_to_true" => distance_to_true,
            "distance_to_analytical" => distance_to_analytical,
            "iterations" => adagrad_variant.it,
            "trace" => trace_variant
        )
        
        println("$variant_name:")
        println("  - Final loss: $(round(final_loss, digits=6))")
        println("  - Distance to true: $(round(distance_to_true, digits=6))")
        println("  - Distance to analytical: $(round(distance_to_analytical, digits=6))")
        println("  - Iterations: $(adagrad_variant.it)")
    end
    
    # Test regularization sensitivity
    println("\n--- Testing Regularization Sensitivity ---")
    l2_values = [0.0, 0.001, 0.01, 0.1, 1.0]
    
    println("L2\tFinal Loss\tDistance to True\tIterations")
    println("-" ^ 50)
    
    for l2_val in l2_values
        # Create oracle with different regularization
        lro_reg = LinearRegressionOracle(A, b, l2=l2_val)
        loss_reg = x -> value(lro_reg, x)
        grad_reg = x -> gradient(lro_reg, x)
        
        # Quick optimization
        adagrad_reg = AdagradOptimizer(loss_reg, grad_reg,
                                      primal_dual=false, lr=1.0, delta=1e-8,
                                      tolerance=1e-8, label="Adagrad-L2-$l2_val")
        
        _ = run!(adagrad_reg, x0, it_max=500, verbose=false)
        
        reg_loss = loss_reg(adagrad_reg.x)
        reg_distance = norm(adagrad_reg.x - x_true)
        
        println("$(l2_val)\t$(round(reg_loss, digits=6))\t$(round(reg_distance, digits=6))\t\t$(adagrad_reg.it)")
    end
    
    return trace, adagrad, x_analytical, variant_results
end

# Test with logistic regression
function test_logistic_regression()
    println("\n" * "=" ^ 60)
    println("Testing Adagrad on Logistic Regression")
    println("=" ^ 60)
    
    # Generate synthetic binary classification data
    n, d = 200, 15
    Random.seed!(42)  # For reproducibility
    
    # Generate data with some structure
    A = randn(n, d) / sqrt(d)
    x_true = randn(d)
    x_true[rand(1:d, d÷3)] .= 0.0  # Make some features irrelevant
    
    # Generate binary targets
    logits = A * x_true
    probabilities = sigmoid.(logits)
    b = Float64.(rand(n) .< probabilities)
    
    println("Generated logistic regression problem:")
    println("- Samples: $n, Features: $d")
    println("- True parameter norm: $(round(norm(x_true), digits=4))")
    println("- True parameter sparsity: $(sum(abs.(x_true) .< 1e-6)) / $d")
    println("- Class balance: $(round(mean(b), digits=3)) positive class")
    
    # Create logistic regression oracle
    lro = LogisticRegressionOracle(A, b, l2=0.01)
    
    # Define loss and gradient functions for optimizer
    loss_func = x -> value(lro, x)
    grad_func = x -> gradient(lro, x)
    
    # Starting point
    x0 = zeros(d)
    
    println("Initial loss: $(round(loss_func(x0), digits=6))")
    println("Initial accuracy: $(round(compute_accuracy(lro, x0), digits=4))")
    
    # Test both Adagrad variants
    variants = [
        ("Standard", false, 0.5),
        ("Dual Averaging", true, 0.5)
    ]
    
    results = Dict()
    
    for (variant_name, primal_dual, lr) in variants
        println("\n--- Testing $variant_name ---")
        
        # Create and run Adagrad
        adagrad = AdagradOptimizer(loss_func, grad_func,
                                  primal_dual=primal_dual, lr=lr, delta=1e-8,
                                  tolerance=1e-8, label="Adagrad-LogReg-$variant_name")
        
        trace = run!(adagrad, x0, it_max=1000, verbose=false)
        
        # Analyze results
        final_loss = loss_func(adagrad.x)
        final_grad_norm = norm(grad_func(adagrad.x))
        final_accuracy = compute_accuracy(lro, adagrad.x)
        distance_to_true = norm(adagrad.x - x_true)
        
        results[variant_name] = Dict(
            "final_loss" => final_loss,
            "final_grad_norm" => final_grad_norm,
            "final_accuracy" => final_accuracy,
            "distance_to_true" => distance_to_true,
            "iterations" => adagrad.it,
            "trace" => trace,
            "final_params" => copy(adagrad.x)
        )
        
        println("Results for $variant_name:")
        println("- Final loss: $(round(final_loss, digits=6))")
        println("- Final accuracy: $(round(final_accuracy, digits=4))")
        println("- Distance to true params: $(round(distance_to_true, digits=6))")
        println("- Final gradient norm: $(round(final_grad_norm, digits=8))")
        println("- Iterations: $(adagrad.it)")
        
        # Check feature selection (sparsity)
        learned_sparsity = sum(abs.(adagrad.x) .< 1e-3)
        println("- Learned sparsity: $learned_sparsity / $d")
    end
    
    # Compare with stochastic gradient descent
    println("\n--- Testing Stochastic Adagrad ---")
    
    # Create stochastic Adagrad
    adagrad_stoch = AdagradStochasticOptimizer(loss_func, grad_func,
                                              primal_dual=false, lr=0.3, delta=1e-8,
                                              n_seeds=3, label="Adagrad-LogReg-Stochastic")
    
    trace_stoch = run!(adagrad_stoch, x0, it_max=800, verbose=false)
    
    # Get best seed results
    best_seed = get_best_seed(trace_stoch)
    if best_seed !== nothing
        best_trace = trace_stoch.seed_traces[best_seed]
        best_x = best_trace.xs[end]
        best_loss = loss_func(best_x)
        best_accuracy = compute_accuracy(lro, best_x)
        
        println("Stochastic Adagrad (best of 3 seeds):")
        println("- Best seed: $best_seed")
        println("- Final loss: $(round(best_loss, digits=6))")
        println("- Final accuracy: $(round(best_accuracy, digits=4))")
        println("- Distance to true params: $(round(norm(best_x - x_true), digits=6))")
        
        results["Stochastic"] = Dict(
            "final_loss" => best_loss,
            "final_accuracy" => best_accuracy,
            "distance_to_true" => norm(best_x - x_true),
            "best_seed" => best_seed,
            "trace" => trace_stoch,
            "final_params" => copy(best_x)
        )
    end
    
    # Test with different regularization strengths
    println("\n--- Testing Regularization Sensitivity ---")
    l2_values = [0.0, 0.001, 0.01, 0.1, 1.0]
    
    println("L2\tFinal Loss\tAccuracy\tSparsity")
    println("-" ^ 45)
    
    for l2_val in l2_values
        # Create oracle with different regularization
        lro_reg = LogisticRegressionOracle(A, b, l2=l2_val)
        loss_reg = x -> value(lro_reg, x)
        grad_reg = x -> gradient(lro_reg, x)
        
        # Quick optimization
        adagrad_reg = AdagradOptimizer(loss_reg, grad_reg,
                                      primal_dual=false, lr=0.5, delta=1e-8,
                                      label="Adagrad-L2-$l2_val")
        
        _ = run!(adagrad_reg, x0, it_max=500, verbose=false)
        
        reg_loss = loss_reg(adagrad_reg.x)
        reg_accuracy = compute_accuracy(lro_reg, adagrad_reg.x)
        reg_sparsity = sum(abs.(adagrad_reg.x) .< 1e-3)
        
        println("$(l2_val)\t$(round(reg_loss, digits=6))\t$(round(reg_accuracy, digits=4))\t\t$reg_sparsity/$d")
    end
    
    # Performance comparison summary
    println("\n--- Performance Summary ---")
    best_variant = nothing
    best_accuracy = 0.0
    
    for (variant, result) in results
        if haskey(result, "final_accuracy") && result["final_accuracy"] > best_accuracy
            best_accuracy = result["final_accuracy"]
            best_variant = variant
        end
    end
    
    if best_variant !== nothing
        println("Best performing variant: $best_variant")
        println("Best accuracy: $(round(best_accuracy, digits=4))")
        println("Corresponding loss: $(round(results[best_variant]["final_loss"], digits=6))")
    end
    
    return results, lro, x_true
end

# Helper function to compute classification accuracy
function compute_accuracy(lro::LogisticRegressionOracle, x::Vector{Float64})
    # Compute predictions
    Ax = lro.A * x
    probabilities = sigmoid.(Ax)
    predictions = Float64.(probabilities .> 0.5)
    
    # Compute accuracy
    accuracy = mean(predictions .== lro.b)
    return accuracy
end

# Helper function for sigmoid (if not already defined)
function sigmoid(x::Real)
    if x >= 0
        exp_neg_x = exp(-x)
        return 1.0 / (1.0 + exp_neg_x)
    else
        exp_x = exp(x)
        return exp_x / (1.0 + exp_x)
    end
end

function sigmoid(x::AbstractVector)
    return [sigmoid(xi) for xi in x]
end

# Test with LogSumExp oracle - SAFER VERSION
function test_log_sum_exp()
    println("\n" * "=" ^ 60)
    println("Testing Adagrad on LogSumExp Oracle")
    println("=" ^ 60)
    
    # Start with a simple test to ensure basic functionality works
    try
        # Basic configuration test
        println("--- Basic LogSumExp Test ---")
        
        n, d = 100, 15
        Random.seed!(42)
        A = randn(n, d) / sqrt(d)
        b = randn(n)
        
        # Create LogSumExp oracle
        lse = LogSumExpOracle(A, b, max_smoothing=1.0, least_squares_term=false, l2=0.01)
        
        # Define loss and gradient functions
        loss_func = x -> value(lse, x)
        grad_func = x -> gradient(lse, x)
        
        x0 = zeros(d)
        initial_loss = loss_func(x0)
        
        println("Initial loss: $(round(initial_loss, digits=6))")
        
        # Test both Adagrad variants
        variants = [
            ("Standard", false, 0.1),
            ("Dual Averaging", true, 0.1)
        ]
        
        basic_results = Dict()
        
        for (variant_name, primal_dual, lr) in variants
            println("\nTesting $variant_name...")
            
            try
                # Create and run Adagrad
                adagrad = AdagradOptimizer(loss_func, grad_func,
                                          primal_dual=primal_dual, lr=lr, delta=1e-8,
                                          tolerance=1e-8, label="Adagrad-LSE-$variant_name")
                
                trace = run!(adagrad, x0, it_max=1000, verbose=false)
                
                # Safely extract results
                final_loss = loss_func(adagrad.x)
                final_grad_norm = norm(grad_func(adagrad.x))
                param_norm = norm(adagrad.x)
                loss_reduction = initial_loss - final_loss
                iterations = adagrad.it
                
                basic_results[variant_name] = Dict(
                    "final_loss" => final_loss,
                    "final_grad_norm" => final_grad_norm,
                    "param_norm" => param_norm,
                    "loss_reduction" => loss_reduction,
                    "iterations" => iterations,
                    "trace" => trace,
                    "final_params" => copy(adagrad.x),
                    "success" => true
                )
                
                println("Results for $variant_name:")
                println("- Final loss: $(round(final_loss, digits=6))")
                println("- Loss reduction: $(round(loss_reduction, digits=6))")
                println("- Parameter norm: $(round(param_norm, digits=6))")
                println("- Final gradient norm: $(round(final_grad_norm, digits=8))")
                println("- Iterations: $iterations")
                
            catch e
                println("✗ $variant_name failed: $e")
                basic_results[variant_name] = Dict(
                    "success" => false,
                    "error" => string(e)
                )
            end
        end
        
        # Test different smoothness values
        println("\n--- Smoothness Analysis ---")
        smoothness_values = [0.1, 0.5, 1.0, 2.0]
        smoothness_results = Dict()
        
        println("Max Smoothing\tFinal Loss\tGrad Norm\tIterations")
        println("-" ^ 55)
        
        for smooth_val in smoothness_values
            try
                lse_smooth = LogSumExpOracle(A, b, 
                                            max_smoothing=smooth_val, 
                                            least_squares_term=false,
                                            l2=0.01)
                
                loss_smooth = x -> value(lse_smooth, x)
                grad_smooth = x -> gradient(lse_smooth, x)
                
                adagrad_smooth = AdagradOptimizer(loss_smooth, grad_smooth,
                                                 primal_dual=false, lr=0.1, delta=1e-8,
                                                 label="Adagrad-Smooth-$smooth_val")
                
                _ = run!(adagrad_smooth, x0, it_max=500, verbose=false)
                
                smooth_loss = loss_smooth(adagrad_smooth.x)
                smooth_grad_norm = norm(grad_smooth(adagrad_smooth.x))
                smooth_iterations = adagrad_smooth.it
                
                smoothness_results[smooth_val] = Dict(
                    "final_loss" => smooth_loss,
                    "final_grad_norm" => smooth_grad_norm,
                    "iterations" => smooth_iterations,
                    "oracle_smoothness" => smoothness(lse_smooth),
                    "success" => true
                )
                
                println("$(smooth_val)\t\t$(round(smooth_loss, digits=6))\t$(round(smooth_grad_norm, digits=6))\t$smooth_iterations")
                
            catch e
                println("$(smooth_val)\t\tFAILED\t\tFAILED\t\tFAILED")
                smoothness_results[smooth_val] = Dict(
                    "success" => false,
                    "error" => string(e)
                )
            end
        end
        
        # Learning rate sensitivity
        println("\n--- Learning Rate Sensitivity ---")
        lr_values = [0.01, 0.05, 0.1, 0.2, 0.5]
        lr_results = Dict()
        
        println("LR\tFinal Loss\tGrad Norm\tIterations\tStable")
        println("-" ^ 50)
        
        for lr in lr_values
            try
                adagrad_lr = AdagradOptimizer(loss_func, grad_func,
                                             primal_dual=false, lr=lr, delta=1e-8,
                                             tolerance=1e-8, label="Adagrad-LR$lr")
                
                _ = run!(adagrad_lr, x0, it_max=500, verbose=false)
                
                lr_loss = loss_func(adagrad_lr.x)
                lr_grad_norm = norm(grad_func(adagrad_lr.x))
                lr_iterations = adagrad_lr.it
                stable = !any(isnan.(adagrad_lr.x)) && !any(isinf.(adagrad_lr.x))
                
                lr_results[lr] = Dict(
                    "final_loss" => lr_loss,
                    "final_grad_norm" => lr_grad_norm,
                    "iterations" => lr_iterations,
                    "stable" => stable,
                    "success" => true
                )
                
                stability_str = stable ? "Yes" : "No"
                println("$(lr)\t$(round(lr_loss, digits=6))\t$(round(lr_grad_norm, digits=6))\t$lr_iterations\t\t$stability_str")
                
            catch e
                println("$(lr)\tFAILED\t\tFAILED\t\tFAILED\t\tNo")
                lr_results[lr] = Dict(
                    "stable" => false, 
                    "error" => string(e),
                    "success" => false
                )
            end
        end
        
        # Numerical stability analysis
        println("\n--- Numerical Stability Analysis ---")
        
        extreme_tests = [
            ("Large Values", 5.0),
            ("Small Values", 0.01),
            ("Normal Values", 1.0)
        ]
        
        stability_results = Dict()
        
        for (test_name, scale_factor) in extreme_tests
            println("\nTesting $test_name (scale: $scale_factor):")
            
            # Generate scaled data
            A_scaled = scale_factor * randn(50, 10) / sqrt(10)
            b_scaled = scale_factor * randn(50)
            
            try
                lse_stable = LogSumExpOracle(A_scaled, b_scaled, 
                                            max_smoothing=1.0, 
                                            least_squares_term=false,
                                            l2=0.01)
                
                loss_stable = x -> value(lse_stable, x)
                grad_stable = x -> gradient(lse_stable, x)
                
                adagrad_stable = AdagradOptimizer(loss_stable, grad_stable,
                                                 primal_dual=false, lr=0.1, delta=1e-8,
                                                 label="Adagrad-Stable")
                
                x0 = zeros(10)
                _ = run!(adagrad_stable, x0, it_max=300, verbose=false)
                
                stable_loss = loss_stable(adagrad_stable.x)
                stable_grad_norm = norm(grad_stable(adagrad_stable.x))
                numerical_stable = !any(isnan.(adagrad_stable.x)) && !any(isinf.(adagrad_stable.x))
                
                stability_results[test_name] = Dict(
                    "final_loss" => stable_loss,
                    "final_grad_norm" => stable_grad_norm,
                    "iterations" => adagrad_stable.it,
                    "stable" => numerical_stable,
                    "success" => true
                )
                
                println("  - Final loss: $(round(stable_loss, digits=6))")
                println("  - Gradient norm: $(round(stable_grad_norm, digits=6))")
                println("  - Iterations: $(adagrad_stable.it)")
                println("  - Numerically stable: $numerical_stable")
                
            catch e
                println("  - Test failed: $e")
                stability_results[test_name] = Dict(
                    "stable" => false,
                    "error" => string(e),
                    "success" => false
                )
            end
        end
        
        # Performance summary
        println("\n--- Performance Summary ---")
        best_config = nothing
        best_loss_reduction = -Inf
        
        for (variant_name, result) in basic_results
            if haskey(result, "success") && result["success"] && 
               haskey(result, "loss_reduction") && result["loss_reduction"] > best_loss_reduction
                best_loss_reduction = result["loss_reduction"]
                best_config = ("Basic", variant_name)
            end
        end
        
        if best_config !== nothing
            config_name, variant = best_config
            best_result = basic_results[variant]
            println("Best performing variant: $variant")
            println("Best loss reduction: $(round(best_loss_reduction, digits=6))")
            println("Final loss: $(round(best_result["final_loss"], digits=6))")
            if haskey(best_result, "iterations")
                println("Iterations: $(best_result["iterations"])")
            end
        else
            println("No successful optimizations found")
        end
        
        # Stability assessment
        stable_lrs = sum(get(result, "stable", false) for result in values(lr_results))
        total_lrs = length(lr_values)
        stable_scales = sum(get(result, "stable", false) for result in values(stability_results))
        total_scales = length(extreme_tests)
        
        println("\nStability Assessment:")
        println("- Stable learning rates: $stable_lrs / $total_lrs")
        println("- Stable across scales: $stable_scales / $total_scales")
        
        return Dict(
            "configurations" => Dict("Basic" => basic_results),
            "smoothness_analysis" => smoothness_results,
            "lr_sensitivity" => lr_results,
            "stability_analysis" => stability_results,
            "best_config" => best_config,
            "success" => true
        )
        
    catch e
        println("✗ LogSumExp test failed with error: $e")
        return Dict(
            "success" => false,
            "error" => string(e)
        )
    end
end

# Main function
function main()
    println("Adagrad Performance Testing")
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
    trace_linreg, adagrad_linreg, x_analytical, variant_results_linreg = test_linear_regression()

    # Test on logistic regression
    logistic_results, lro, x_true_logistic = test_logistic_regression()

    # Test on LogSumExp oracle
    logsumexp_results = test_log_sum_exp()
    
    println("\n" * "=" ^ 60)
    println("SUMMARY OF ALL TESTS")
    println("=" ^ 60)
    
    println("\nBest learning rate found: $best_lr")
    
    # Show performance comparison
    println("\nPerformance comparison (Rosenbrock):")
    if haskey(comparison_results, "Rosenbrock")
        for (variant, result) in comparison_results["Rosenbrock"]
            println("- $variant: $(round(result["final_loss"], digits=6)) loss in $(result["iterations"]) iterations")
        end
    end

    # Show linear regression results
    println("\nLinear Regression Results:")
    for (variant, result) in variant_results_linreg
        println("- $variant: $(round(result["final_loss"], digits=6)) loss, distance to true: $(round(result["distance_to_true"], digits=6))")
    end

    # Show logistic regression results
    println("\nLogistic Regression Results:")
    for (variant, result) in logistic_results
        if haskey(result, "final_accuracy")
            println("- $variant: $(round(result["final_accuracy"], digits=4)) accuracy, $(round(result["final_loss"], digits=6)) loss")
        end
    end
    
     # Show LogSumExp results
    println("\nLogSumExp Oracle Results:")
    if haskey(logsumexp_results, "best_config") && logsumexp_results["best_config"] !== nothing
        config_name, variant = logsumexp_results["best_config"]
        best_lse_result = logsumexp_results["configurations"][config_name][variant]
        println("- Best configuration: $config_name with $variant")
        println("- Best loss reduction: $(round(best_lse_result["loss_reduction"], digits=6))")
        println("- Final loss: $(round(best_lse_result["final_loss"], digits=6))")
    end
    
    # Stability summary for LogSumExp
    stable_lrs = 0
    total_lrs = 0
    if haskey(logsumexp_results, "lr_sensitivity")
        for (lr, result) in logsumexp_results["lr_sensitivity"]
            total_lrs += 1
            if haskey(result, "stable") && result["stable"]
                stable_lrs += 1
            end
        end
    end
    println("- LogSumExp stability: $stable_lrs/$total_lrs learning rates stable")
end

main()