"""
3D Visualization of the Rosenbrock Function

This example creates various 3D visualizations of the Rosenbrock function,
including surface plots, contour plots, and interactive visualizations.
"""

using Joptan
using Plots
using LinearAlgebra

# Set up the plotting backend for better 3D support
plotlyjs()

# Create 3D surface plot of Rosenbrock function
function create_rosenbrock_surface()
    println("Creating Rosenbrock 3D surface plot...")
    
    # Define the range for visualization
    x_range = -2:0.1:2
    y_range = -1:0.1:3
    
    # Create meshgrid
    X = [x for x in x_range, y in y_range]
    Y = [y for x in x_range, y in y_range]
    
    # Compute function values
    Z = [rosenbrock([x, y]) for x in x_range, y in y_range]
    
    # Create surface plot
    p1 = surface(x_range, y_range, Z',
                title="Rosenbrock Function - 3D Surface",
                xlabel="x₁", ylabel="x₂", zlabel="f(x₁, x₂)",
                camera=(45, 30),
                colorbar_title="Function Value",
                color=:viridis)
    
    # Add the global minimum point
    scatter!(p1, [1.0], [1.0], [0.0], 
            markersize=8, color=:red, 
            label="Global Minimum")
    
    return p1
end

# Create wireframe plot
function create_rosenbrock_wireframe()
    println("Creating Rosenbrock wireframe plot...")
    
    x_range = -2:0.2:2
    y_range = -1:0.2:3
    
    Z = [rosenbrock([x, y]) for x in x_range, y in y_range]
    
    p2 = wireframe(x_range, y_range, Z',
                  title="Rosenbrock Function - Wireframe",
                  xlabel="x₁", ylabel="x₂", zlabel="f(x₁, x₂)",
                  camera=(45, 30),
                  linewidth=1,
                  color=:blue)
    
    return p2
end

# Create contour plot with 3D effect
function create_rosenbrock_contour3d()
    println("Creating Rosenbrock 3D contour plot...")
    
    x_range = -2:0.05:2
    y_range = -1:0.05:3
    
    Z = [rosenbrock([x, y]) for x in x_range, y in y_range]
    
    # Create filled contour plot
    p3 = contour(x_range, y_range, Z',
                levels=50,
                fill=true,
                title="Rosenbrock Function - Contour Plot",
                xlabel="x₁", ylabel="x₂",
                colorbar_title="Function Value",
                color=:plasma)
    
    # Add global minimum
    scatter!(p3, [1.0], [1.0], 
            markersize=10, color=:red, 
            markershape=:star,
            label="Global Minimum")
    
    # Add some level curves
    contour!(p3, x_range, y_range, Z',
            levels=[1, 10, 50, 100, 200, 500],
            color=:black,
            linewidth=2,
            linestyle=:dash,
            label="Level Curves")
    
    return p3
end

# Create cross-section plots
function create_rosenbrock_cross_sections()
    println("Creating Rosenbrock cross-section plots...")
    
    # Cross-section along x₁ axis (x₂ = 1)
    x1_range = -2:0.01:2
    z1 = [rosenbrock([x, 1.0]) for x in x1_range]
    
    p4 = plot(x1_range, z1,
             title="Rosenbrock Cross-section: x₂ = 1",
             xlabel="x₁", ylabel="f(x₁, 1)",
             linewidth=3, color=:blue,
             label="f(x₁, 1)")
    
    # Mark the minimum
    scatter!(p4, [1.0], [0.0], 
            markersize=8, color=:red,
            label="Minimum")
    
    # Cross-section along x₂ axis (x₁ = 1)
    x2_range = -1:0.01:3
    z2 = [rosenbrock([1.0, y]) for y in x2_range]
    
    p5 = plot(x2_range, z2,
             title="Rosenbrock Cross-section: x₁ = 1",
             xlabel="x₂", ylabel="f(1, x₂)",
             linewidth=3, color=:green,
             label="f(1, x₂)")
    
    # Mark the minimum
    scatter!(p5, [1.0], [0.0], 
            markersize=8, color=:red,
            label="Minimum")
    
    return p4, p5
end

# Create gradient visualization
function create_rosenbrock_gradient_field()
    println("Creating Rosenbrock gradient field...")
    
    # Coarser grid for gradient field
    x_range = -2:0.3:2
    y_range = -1:0.3:3
    
    # Create arrays for quiver plot
    X = [x for x in x_range, y in y_range]
    Y = [y for x in x_range, y in y_range]
    
    # Compute gradients
    U = zeros(size(X))
    V = zeros(size(Y))
    
    for i in 1:length(x_range)
        for j in 1:length(y_range)
            grad = rosenbrock_gradient([x_range[i], y_range[j]])
            # Normalize for better visualization
            grad_norm = norm(grad)
            if grad_norm > 0
                U[i, j] = -grad[1] / grad_norm  # Negative for gradient descent direction
                V[i, j] = -grad[2] / grad_norm
            end
        end
    end
    
    # Create base contour plot
    x_fine = -2:0.1:2
    y_fine = -1:0.1:3
    Z = [rosenbrock([x, y]) for x in x_fine, y in y_fine]
    
    p6 = contour(x_fine, y_fine, Z',
                levels=30,
                fill=true,
                title="Rosenbrock Function - Gradient Field",
                xlabel="x₁", ylabel="x₂",
                color=:viridis,
                alpha=0.7)
    
    # Add gradient field
    quiver!(p6, X, Y, quiver=(U, V),
           color=:white,
           arrow=:closed,
           linewidth=2,
           label="Gradient Direction")
    
    # Add global minimum
    scatter!(p6, [1.0], [1.0], 
            markersize=10, color=:red, 
            markershape=:star,
            label="Global Minimum")
    
    return p6
end

# Create animated rotation of 3D surface
function create_rosenbrock_animation()
    println("Creating Rosenbrock 3D animation...")
    
    x_range = -2:0.15:2
    y_range = -1:0.15:3
    
    Z = [rosenbrock([x, y]) for x in x_range, y in y_range]
    
    # Create animation
    anim = @animate for angle in 0:5:360
        surface(x_range, y_range, Z',
               title="Rosenbrock Function - Rotating View",
               xlabel="x₁", ylabel="x₂", zlabel="f(x₁, x₂)",
               camera=(angle, 30),
               colorbar_title="Function Value",
               color=:viridis,
               legend=false)
        
        # Add the global minimum point
        scatter!([1.0], [1.0], [0.0], 
                markersize=8, color=:red)
    end
    
    return anim
end

# Create log-scale visualization for better detail
function create_rosenbrock_log_scale()
    println("Creating Rosenbrock log-scale visualization...")
    
    x_range = -2:0.05:2
    y_range = -1:0.05:3
    
    Z = [rosenbrock([x, y]) for x in x_range, y in y_range]
    
    # Apply log transformation (add small value to avoid log(0))
    Z_log = log.(Z .+ 1e-10)
    
    p7 = surface(x_range, y_range, Z_log',
                title="Rosenbrock Function - Log Scale",
                xlabel="x₁", ylabel="x₂", zlabel="log(f(x₁, x₂))",
                camera=(45, 30),
                colorbar_title="log(Function Value)",
                color=:inferno)
    
    # Add the global minimum point
    scatter!(p7, [1.0], [1.0], [log(1e-10)], 
            markersize=8, color=:cyan, 
            label="Global Minimum")
    
    return p7
end

# Create multiple view angles
function create_rosenbrock_multiple_views()
    println("Creating Rosenbrock multiple view angles...")
    
    x_range = -2:0.1:2
    y_range = -1:0.1:3
    
    Z = [rosenbrock([x, y]) for x in x_range, y in y_range]
    
    # Different camera angles
    angles = [(30, 30), (60, 45), (90, 30), (120, 60)]
    plots = []
    
    for (i, (azimuth, elevation)) in enumerate(angles)
        p = surface(x_range, y_range, Z',
                   title="View $i: ($(azimuth)°, $(elevation)°)",
                   xlabel="x₁", ylabel="x₂", zlabel="f(x₁, x₂)",
                   camera=(azimuth, elevation),
                   color=:viridis,
                   legend=false)
        
        # Add the global minimum point
        scatter!(p, [1.0], [1.0], [0.0], 
                markersize=6, color=:red)
        
        push!(plots, p)
    end
    
    return plot(plots..., layout=(2, 2), size=(800, 600))
end

# Main visualization function
function main()
    println("Rosenbrock Function 3D Visualization")
    println("=" ^ 40)
    
    try
        # Create all visualizations
        println("\n1. Creating 3D surface plot...")
        p1 = create_rosenbrock_surface()
        
        println("\n2. Creating wireframe plot...")
        p2 = create_rosenbrock_wireframe()
        
        println("\n3. Creating contour plot...")
        p3 = create_rosenbrock_contour3d()
        
        println("\n4. Creating cross-sections...")
        p4, p5 = create_rosenbrock_cross_sections()
        
        println("\n5. Creating gradient field...")
        p6 = create_rosenbrock_gradient_field()
        
        println("\n6. Creating log-scale visualization...")
        p7 = create_rosenbrock_log_scale()
        
        println("\n7. Creating multiple view angles...")
        p8 = create_rosenbrock_multiple_views()
        
        # Save all plots
        println("\nSaving plots...")
        
        try
            savefig(p1, "examples/rosenbrock_3d_surface.html")
            savefig(p2, "examples/rosenbrock_wireframe.html")
            savefig(p3, "examples/rosenbrock_contour.png")
            savefig(p4, "examples/rosenbrock_cross_section_x1.png")
            savefig(p5, "examples/rosenbrock_cross_section_x2.png")
            savefig(p6, "examples/rosenbrock_gradient_field.png")
            savefig(p7, "examples/rosenbrock_log_scale.html")
            savefig(p8, "examples/rosenbrock_multiple_views.png")
            
            println("✓ All plots saved successfully!")
            
        catch e
            println("Warning: Could not save some plots: $e")
        end
        
        # Create animation (optional, can be time-consuming)
        println("\n8. Creating animation (this may take a while)...")
        try
            anim = create_rosenbrock_animation()
            gif(anim, "rosenbrock_rotation.gif", fps=10)
            println("✓ Animation saved as rosenbrock_rotation.gif")
        catch e
            println("Warning: Could not create animation: $e")
        end
        
        # Display summary
        println("\n" * "=" * 40)
        println("Rosenbrock Function Properties:")
        println("- Global minimum: [1.0, 1.0] with f(x*) = 0")
        println("- Banana-shaped valley")
        println("- Unimodal (single minimum)")
        println("- Ill-conditioned (slow convergence)")
        println("- Good test for second-order methods")
        println("\nVisualization complete!")
        
        # Show the main surface plot
        display(p1)
        
    catch e
        println("Error during visualization: $e")
        println("Make sure you have Plots.jl installed and properly configured.")
    end
end

main()