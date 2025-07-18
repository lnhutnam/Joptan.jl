"""
Utility functions for first-order optimization methods

This module contains common utility functions that are shared across
different first-order optimization algorithms.
"""

using LinearAlgebra
using Statistics

"""
    optimize_with_legacy_interface(optimizer_type, loss_func, grad_func, x0::Vector{Float64}; 
                                  max_iter::Int=1000, tol::Float64=1e-6, verbose::Bool=false, 
                                  optimizer_kwargs...)

Generic legacy interface for first-order optimizers.

# Arguments
- `optimizer_type`: Constructor for the optimizer (e.g., AdagradOptimizer)
- `loss_func::Function`: Loss function to minimize
- `grad_func::Function`: Gradient function
- `x0::Vector{Float64}`: Initial parameters
- `max_iter::Int`: Maximum number of iterations (default: 1000)
- `tol::Float64`: Convergence tolerance (default: 1e-6)
- `verbose::Bool`: Whether to print progress (default: false)
- `optimizer_kwargs...`: Additional arguments passed to optimizer constructor

# Returns
- `x_opt::Vector{Float64}`: Optimal parameters
- `history::Vector{Dict}`: Optimization history

# Example
```julia
x_opt, history = optimize_with_legacy_interface(
    AdagradOptimizer, loss_func, grad_func, x0,
    lr=0.1, delta=1e-8, max_iter=1000
)
```
"""
function optimize_with_legacy_interface(optimizer_type, loss_func::Function, grad_func::Function, 
                                       x0::Vector{Float64}; 
                                       max_iter::Int=1000, tol::Float64=1e-6, verbose::Bool=false,
                                       optimizer_kwargs...)
    
    # Create optimizer with provided kwargs
    optimizer = optimizer_type(loss_func, grad_func; 
                              tolerance=tol, label=string(optimizer_type), 
                              optimizer_kwargs...)
    
    # Run optimization
    trace = run!(optimizer, x0, it_max=max_iter, verbose=verbose)
    
    # Extract results
    x_opt = optimizer.x
    
    # Convert trace to legacy history format
    history = []
    for i in 1:length(trace.xs)
        push!(history, Dict(
            "iteration" => trace.its[i],
            "loss" => trace.losses_computed ? trace.losses[i] : loss_func(trace.xs[i]),
            "gradient_norm" => trace.grad_norms_computed ? trace.grad_norms[i] : norm(grad_func(trace.xs[i])),
            "parameters" => trace.xs[i]
        ))
    end
    
    return x_opt, history
end

"""
    compare_optimizer_variants(optimizer_type, loss_func, grad_func, x0::Vector{Float64};
                              parameter_grid::Dict, max_iter::Int=1000, tol::Float64=1e-6)

Compare different configurations of an optimizer.

# Arguments
- `optimizer_type`: Constructor for the optimizer
- `loss_func::Function`: Loss function to minimize
- `grad_func::Function`: Gradient function
- `x0::Vector{Float64}`: Initial parameters
- `parameter_grid::Dict`: Dictionary of parameter names to lists of values to test
- `max_iter::Int`: Maximum iterations per run (default: 1000)
- `tol::Float64`: Convergence tolerance (default: 1e-6)

# Returns
- `results::Vector{Dict}`: Results for each configuration

# Example
```julia
results = compare_optimizer_variants(
    AdagradOptimizer, loss_func, grad_func, x0,
    parameter_grid=Dict(
        :lr => [0.1, 1.0, 10.0],
        :delta => [0.0, 1e-8, 1e-6],
        :primal_dual => [false, true]
    )
)
```
"""
function compare_optimizer_variants(optimizer_type, loss_func::Function, grad_func::Function, 
                                   x0::Vector{Float64}; parameter_grid::Dict,
                                   max_iter::Int=1000, tol::Float64=1e-6)
    
    results = Dict[]
    
    # Generate all parameter combinations
    param_names = collect(keys(parameter_grid))
    param_values = collect(values(parameter_grid))
    
    # Get all combinations using Cartesian product
    for combination in Iterators.product(param_values...)
        # Create parameter dictionary for this combination
        params = Dict(zip(param_names, combination))
        
        println("Testing $(string(optimizer_type)) with parameters: $params")
        
        # Create and run optimizer
        optimizer = optimizer_type(loss_func, grad_func; 
                                  tolerance=tol, label=string(optimizer_type),
                                  params...)
        
        trace = run!(optimizer, x0, it_max=max_iter, verbose=false)
        
        # Extract results
        x_opt = optimizer.x
        final_loss = loss_func(x_opt)
        final_grad_norm = norm(grad_func(x_opt))
        converged = final_grad_norm < tol
        iterations = optimizer.it
        
        result = Dict(
            "optimizer_type" => string(optimizer_type),
            "parameters" => params,
            "final_loss" => final_loss,
            "final_grad_norm" => final_grad_norm,
            "converged" => converged,
            "iterations" => iterations,
            "x_opt" => x_opt,
            "trace" => trace
        )
        
        push!(results, result)
        
        println("  Final loss: $(round(final_loss, digits=6))")
        println("  Gradient norm: $(round(final_grad_norm, digits=6))")
        println("  Converged: $converged")
        println("  Iterations: $iterations")
        println()
    end
    
    return results
end

"""
    get_best_result(results::Vector{Dict}; metric::String="final_loss", minimize::Bool=true)

Find the best result from a comparison of optimizer variants.

# Arguments
- `results::Vector{Dict}`: Results from compare_optimizer_variants
- `metric::String`: Metric to optimize (default: "final_loss")
- `minimize::Bool`: Whether to minimize the metric (default: true)

# Returns
- `best_result::Dict`: Best result based on the specified metric
"""
function get_best_result(results::Vector{Dict}; metric::String="final_loss", minimize::Bool=true)
    if isempty(results)
        error("No results provided")
    end
    
    best_result = results[1]
    best_value = best_result[metric]
    
    for result in results[2:end]
        current_value = result[metric]
        
        if minimize && current_value < best_value
            best_value = current_value
            best_result = result
        elseif !minimize && current_value > best_value
            best_value = current_value
            best_result = result
        end
    end
    
    return best_result
end

"""
    print_results_summary(results::Vector{Dict}; 
                         sort_by::String="final_loss", 
                         show_params::Vector{String}=String[],
                         max_rows::Int=20)

Print a summary table of optimization results.

# Arguments
- `results::Vector{Dict}`: Results from compare_optimizer_variants
- `sort_by::String`: Metric to sort by (default: "final_loss")
- `show_params::Vector{String}`: Parameter names to show in table (default: all)
- `max_rows::Int`: Maximum number of rows to display (default: 20)
"""
function print_results_summary(results::Vector{Dict}; 
                              sort_by::String="final_loss", 
                              show_params::Vector{String}=String[],
                              max_rows::Int=20)
    
    if isempty(results)
        println("No results to display")
        return
    end
    
    # Sort results
    sorted_results = sort(results, by=r -> r[sort_by])
    
    # Determine which parameters to show
    if isempty(show_params)
        # Extract all parameter names from first result
        first_params = results[1]["parameters"]
        show_params = collect(string(k) for k in keys(first_params))
    end
    
    # Print header
    println("Optimization Results Summary (sorted by $sort_by)")
    println("=" ^ 80)
    
    # Create header row
    header_parts = ["Rank", show_params..., "Final Loss", "Grad Norm", "Converged", "Iters"]
    header = join(header_parts, "\t")
    println(header)
    println("-" ^ 80)
    
    # Print results
    num_to_show = min(length(sorted_results), max_rows)
    
    for (rank, result) in enumerate(sorted_results[1:num_to_show])
        row_parts = [string(rank)]
        
        # Add parameter values
        params = result["parameters"]
        for param_name in show_params
            param_key = Symbol(param_name)
            if haskey(params, param_key)
                push!(row_parts, string(params[param_key]))
            else
                push!(row_parts, "N/A")
            end
        end
        
        # Add metrics
        push!(row_parts, string(round(result["final_loss"], digits=6)))
        push!(row_parts, string(round(result["final_grad_norm"], digits=6)))
        push!(row_parts, result["converged"] ? "Yes" : "No")
        push!(row_parts, string(result["iterations"]))
        
        row = join(row_parts, "\t")
        println(row)
    end
    
    if length(sorted_results) > max_rows
        println("... and $(length(sorted_results) - max_rows) more results")
    end
    
    println("-" ^ 80)
    
    # Highlight best result
    best_result = sorted_results[1]
    println("Best result: Rank 1")
    println("Parameters: $(best_result["parameters"])")
    println("Final loss: $(round(best_result["final_loss"], digits=6))")
end

"""
    create_convergence_plot(results::Vector{Dict}; 
                           show_top_n::Int=5, 
                           metric::String="final_loss")

Create convergence plots for the top N results.

# Arguments
- `results::Vector{Dict}`: Results from compare_optimizer_variants
- `show_top_n::Int`: Number of top results to plot (default: 5)
- `metric::String`: Metric to rank results by (default: "final_loss")

# Returns
- Plot object (requires Plots.jl to be loaded)

# Note
This function requires Plots.jl to be available. It will return `nothing` 
if Plots.jl is not loaded.
"""
function create_convergence_plot(results::Vector{Dict}; 
                                show_top_n::Int=5, 
                                metric::String="final_loss")
    
    # Check if Plots is available
    if !isdefined(Main, :Plots)
        @warn "Plots.jl not available. Cannot create convergence plot."
        return nothing
    end
    
    # Sort results and take top N
    sorted_results = sort(results, by=r -> r[metric])
    top_results = sorted_results[1:min(show_top_n, length(sorted_results))]
    
    # Create plot
    p = Main.Plots.plot(title="Convergence Comparison", 
                       xlabel="Iteration", 
                       ylabel="Loss",
                       yscale=:log10)
    
    for (i, result) in enumerate(top_results)
        trace = result["trace"]
        losses = get_losses(trace)
        iterations = 0:(length(losses)-1)
        
        # Create label from parameters
        params = result["parameters"]
        label_parts = [string(k) * "=" * string(v) for (k, v) in params]
        label = "Rank $i: " * join(label_parts[1:min(2, length(label_parts))], ", ")
        
        Main.Plots.plot!(p, iterations, losses, 
                        label=label, linewidth=2)
    end
    
    return p
end

"""
    save_results_to_file(results::Vector{Dict}, filename::String; format::String="json")

Save optimization results to file.

# Arguments
- `results::Vector{Dict}`: Results to save
- `filename::String`: Output filename
- `format::String`: File format ("json" or "csv", default: "json")
"""
function save_results_to_file(results::Vector{Dict}, filename::String; format::String="json")
    if format == "json"
        # Save as JSON (requires JSON.jl)
        if isdefined(Main, :JSON)
            Main.JSON.print(open(filename, "w"), results, 4)
            println("Results saved to $filename (JSON format)")
        else
            @warn "JSON.jl not available. Cannot save as JSON."
        end
    elseif format == "csv"
        # Save as CSV (simplified version)
        open(filename, "w") do file
            if !isempty(results)
                # Write header
                first_result = results[1]
                headers = ["final_loss", "final_grad_norm", "converged", "iterations"]
                
                # Add parameter headers
                param_headers = ["param_" * string(k) for k in keys(first_result["parameters"])]
                all_headers = [headers..., param_headers...]
                println(file, join(all_headers, ","))
                
                # Write data
                for result in results
                    row_data = [
                        result["final_loss"],
                        result["final_grad_norm"],
                        result["converged"],
                        result["iterations"]
                    ]
                    
                    # Add parameter values
                    param_values = [v for v in values(result["parameters"])]
                    all_data = [row_data..., param_values...]
                    println(file, join(all_data, ","))
                end
            end
        end
        println("Results saved to $filename (CSV format)")
    else
        error("Unsupported format: $format. Use 'json' or 'csv'.")
    end
end