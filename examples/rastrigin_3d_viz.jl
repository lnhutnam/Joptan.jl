"""
3D Visualization of the Rastrigin Function

This example creates various 3D visualizations of the Rastrigin function,
showcasing its highly multimodal nature with many local minima.
"""

using Joptan
using Plots
using LinearAlgebra

# Set up the plotting backend for better 3D support
plotlyjs()

# Create 3D surface plot of Rastrigin function
function create_rastrigin_surface()
    println("Creating Rastrigin 3D surface plot...")
    
    # Define the range for visualization
    x_range = -5:0.1:5
    y_range = -5:0.1:5
    
    # Compute function values
    Z = [rastrigin([x, y]) for x in x_range, y in y_range]
    
    # Create surface plot
    p1 = surface(x_range, y_range, Z',
                title="Rastrigin Function - 3D Surface",
                xlabel="x₁", ylabel="x₂", zlabel="f(x₁, x₂)",
                camera=(45, 30),
                colorbar_title="Function Value",
                color=:plasma)
    
    # Add the global minimum point
    scatter!(p1, [0.0], [0.0], [0.0], 
            markersize=8, color=:red, 
            label="Global Minimum")
    
    return p1
end

# Create wireframe plot to show the multimodal structure
function create_rastrigin_wireframe()
    println("Creating Rastrigin wireframe plot...")
    
    x_range = -5:0.2:5
    y_range = -5:0.2:5
    
    Z = [rastrigin([x, y]) for x in x_range, y in y_range]
    
    p2 = wireframe(x_range, y_range, Z',
                  title="Rastrigin Function - Wireframe",
                  xlabel="x₁", ylabel="x₂", zlabel="f(x₁, x₂)",
                  camera=(45, 30),
                  linewidth=1,
                  color=:blue)
    
    return p2
end

# Create contour plot showing local minima
function create_rastrigin_contour3d()
    println("Creating Rastrigin 3D contour plot...")
    
    x_range = -5:0.05:5
    y_range = -5:0.05:5
    
    Z = [rastrigin([x, y]) for x in x_range, y in y_range]
    
    # Create filled contour plot
    p3 = contour(x_range, y_range, Z',
                levels=50,
                fill=true,
                title="Rastrigin Function - Contour Plot",
                xlabel="x₁", ylabel="x₂",
                colorbar_title="Function Value",
                color=:turbo)
    
    # Add global minimum
    scatter!(p3, [0.0], [0.0], 
            markersize=10, color=:red, 
            markershape=:star,
            label="Global Minimum")
    
    # Find and mark some local minima (approximate)
    local_minima = []
    for i in -2:1:2
        for j in -2:1:2
            if i != 0 || j != 0  # Skip global minimum
                x_local = [Float64(i), Float64(j)]
                f_val = rastrigin(x_local)
                if f_val < 5.0  # Only mark shallow local minima
                    push!(local_minima, (x_local[1], x_local[2], f_val))
                end
            end
        end
    end
    
    if !isempty(local_minima)
        x_locals = [lm[1] for lm in local_minima]
        y_locals = [lm[2] for lm in local_minima]
        scatter!(p3, x_locals, y_locals,
                markersize=6, color=:yellow,
                markershape=:circle,
                label="Local Minima")
    end
    
    return p3
end

# Create cross-section plots showing the oscillatory nature
function create_rastrigin_cross_sections()
    println("Creating Rastrigin cross-section plots...")
    
    # Cross-section along x₁ axis (x₂ = 0)
    x1_range = -5:0.01:5
    z1 = [rastrigin([x, 0.0]) for x in x1_range]
    
    p4 = plot(x1_range, z1,
             title="Rastrigin Cross-section: x₂ = 0",
             xlabel="x₁", ylabel="f(x₁, 0)",
             linewidth=2, color=:blue,
             label="f(x₁, 0)")
    
    # Mark the global minimum
    scatter!(p4, [0.0], [0.0], 
            markersize=8, color=:red,
            label="Global Minimum")
    
    # Cross-section along x₂ axis (x₁ = 0)
    x2_range = -5:0.01:5
    z2 = [rastrigin([0.0, y]) for y in x2_range]
    
    p5 = plot(x2_range, z2,
             title="Rastrigin Cross-section: x₁ = 0",
             xlabel="x₂", ylabel="f(0, x₂)",
             linewidth=2, color=:green,
             label="f(0, x₂)")
    
    # Mark the global minimum
    scatter!(p5, [0.0], [0.0], 
            markersize=8, color=:red,
            label="Global Minimum")
    
    # Diagonal cross-section (x₁ = x₂)
    x_diag = -5:0.01:5
    z_diag = [rastrigin([x, x]) for x in x_diag]
    
    p6 = plot(x_diag, z_diag,
             title="Rastrigin Cross-section: x₁ = x₂",
             xlabel="x₁ = x₂", ylabel="f(x, x)",
             linewidth=2, color=:purple,
             label="f(x, x)")
    
    # Mark the global minimum
    scatter!(p6, [0.0], [0.0], 
            markersize=8, color=:red,
            label="Global Minimum")
    
    return p4, p5, p6
end

# Create gradient field visualization
function create_rastrigin_gradient_field()
    println("Creating Rastrigin gradient field...")
    
    # Coarser grid for gradient field
    x_range = -4:0.5:4
    y_range = -4:0.5:4
    
    # Create arrays for quiver plot
    X = [x for x in x_range, y in y_range]
    Y = [y for x in x_range, y in y_range]
    
    # Compute gradients
    U = zeros(size(X))
    V = zeros(size(Y))
    
    for i in 1:length(x_range)
        for j in 1:length(y_range)
            grad = rastrigin_gradient([x_range[i], y_range[j]])
            # Normalize for better visualization
            grad_norm = norm(grad)
            if grad_norm > 0
                U[i, j] = -grad[1] / grad_norm  # Negative for gradient descent direction
                V[i, j] = -grad[2] / grad_norm
            end
        end
    end
    
    # Create base contour plot
    x_fine = -4:0.1:4
    y_fine = -4:0.1:4
    Z = [rastrigin([x, y]) for x in x_fine, y in y_fine]
    
    p7 = contour(x_fine, y_fine, Z',
                levels=30,
                fill=true,
                title="Rastrigin Function - Gradient Field",
                xlabel="x₁", ylabel="x₂",
                color=:plasma,
                alpha=0.7)
    
    # Add gradient field
    quiver!(p7, X, Y, quiver=(U, V),
           color=:white,
           arrow=:closed,
           linewidth=1.5,
           label="Gradient Direction")
    
    # Add global minimum
    scatter!(p7, [0.0], [0.0], 
            markersize=10, color=:red, 
            markershape=:star,
            label="Global Minimum")
    
    return p7
end

# Create zoom-in view around the global minimum
function create_rastrigin_zoom()
    println("Creating Rastrigin zoom-in view...")
    
    # Smaller range around the global minimum
    x_range = -2:0.05:2
    y_range = -2:0.05:2
    
    Z = [rastrigin([x, y]) for x in x_range, y in y_range]
    
    p8 = surface(x_range, y_range, Z',
                title="Rastrigin Function - Zoom Around Global Minimum",
                xlabel="x₁", ylabel="x₂", zlabel="f(x₁, x₂)",
                camera=(45, 30),
                colorbar_title="Function Value",
                color=:viridis)
    
    # Add the global minimum point
    scatter!(p8, [0.0], [0.0], [0.0], 
            markersize=8, color=:red, 
            label="Global Minimum")
    
    return p8
end

# Create parameter comparison (different A values)
function create_rastrigin_parameter_comparison()
    println("Creating Rastrigin parameter comparison...")
    
    A_values = [5.0, 10.0, 15.0, 20.0]
    plots = []
    
    x_range = -3:0.1:3
    y_range = -3:0.1:3
    
    for A in A_values
        Z = [rastrigin([x, y], A=A) for x in x_range, y in y_range]
        
        p = surface(x_range, y_range, Z',
                   title="Rastrigin Function (A = $A)",
                   xlabel="x₁", ylabel="x₂", zlabel="f(x₁, x₂)",
                   camera=(45, 30),
                   color=:plasma,
                   legend=false)
        
        # Add the global minimum point
        scatter!(p, [0.0], [0.0], [0.0], 
                markersize=6, color=:red)
        
        push!(plots, p)
    end
    
    return plot(plots..., layout=(2, 2), size=(800, 600))
end

# Create animated rotation
function create_rastrigin_animation()
    println("Creating Rastrigin 3D animation...")
    
    x_range = -5:0.15:5
    y_range = -5:0.15:5
    
    Z = [rastrigin([x, y]) for x in x_range, y in y_range]
    
    # Create animation
    anim = @animate for angle in 0:5:360
        surface(x_range, y_range, Z',
               title="Rastrigin Function - Rotating View",
               xlabel="x₁", ylabel="x₂", zlabel="f(x₁, x₂)",
               camera=(angle, 30),
               colorbar_title="Function Value",
               color=:plasma,
               legend=false)
        
        # Add the global minimum point
        scatter!([0.0], [0.0], [0.0], 
                markersize=8, color=:red)
    end
    
    return anim
end

# Create heatmap showing local minima density
function create_rastrigin_heatmap()
    println("Creating Rastrigin heatmap...")
    
    x_range = -5:0.1:5
    y_range = -5:0.1:5
    
    Z = [rastrigin([x, y]) for x in x_range, y in y_range]
    
    p9 = heatmap(x_range, y_range, Z',
                title="Rastrigin Function - Heatmap",
                xlabel="x₁", ylabel="x₂",
                colorbar_title="Function Value",
                color=:hot,
                aspect_ratio=:equal)
    
    # Add global minimum
    scatter!(p9, [0.0], [0.0], 
            markersize=10, color=:cyan, 
            markershape=:star,
            label="Global Minimum")
    
    return p9
end

# Create multiple view angles
function create_rastrigin_multiple_views()
    println("Creating Rastrigin multiple view angles...")
    
    x_range = -5:0.15:5
    y_range = -5:0.15:5
    
    Z = [rastrigin([x, y]) for x in x_range, y in y_range]
    
    # Different camera angles
    angles = [(30, 30), (60, 45), (90, 30), (120, 60)]
    plots = []
    
    for (i, (azimuth, elevation)) in enumerate(angles)
        p = surface(x_range, y_range, Z',
                   title="View $i: ($(azimuth)°, $(elevation)°)",
                   xlabel="x₁", ylabel="x₂", zlabel="f(x₁, x₂)",
                   camera=(azimuth, elevation),
                   color=:plasma,
                   legend=false)
        
        # Add the global minimum point
        scatter!(p, [0.0], [0.0], [0.0], 
                markersize=6, color=:red)
        
        push!(plots, p)
    end
    
    return plot(plots..., layout=(2, 2), size=(800, 600))
end

# Main visualization function
function main()
    println("Rastrigin Function 3D Visualization")
    println("=" ^ 40)
    
    try
        # Create all visualizations
        println("\n1. Creating 3D surface plot...")
        p1 = create_rastrigin_surface()
        
        println("\n2. Creating wireframe plot...")
        p2 = create_rastrigin_wireframe()
        
        println("\n3. Creating contour plot...")
        p3 = create_rastrigin_contour3d()
        
        println("\n4. Creating cross-sections...")
        p4, p5, p6 = create_rastrigin_cross_sections()
        
        println("\n5. Creating gradient field...")
        p7 = create_rastrigin_gradient_field()
        
        println("\n6. Creating zoom-in view...")
        p8 = create_rastrigin_zoom()
        
        println("\n7. Creating parameter comparison...")
        p9 = create_rastrigin_parameter_comparison()
        
        println("\n8. Creating heatmap...")
        p10 = create_rastrigin_heatmap()
        
        println("\n9. Creating multiple view angles...")
        p11 = create_rastrigin_multiple_views()
        
        # Save all plots
        println("\nSaving plots...")
        
        try
            savefig(p1, "examples/rastrigin_3d_surface.html")
            savefig(p2, "examples/rastrigin_wireframe.html")
            savefig(p3, "examples/rastrigin_contour.png")
            savefig(p4, "examples/rastrigin_cross_section_x1.png")
            savefig(p5, "examples/rastrigin_cross_section_x2.png")
            savefig(p6, "examples/rastrigin_cross_section_diagonal.png")
            savefig(p7, "examples/rastrigin_gradient_field.png")
            savefig(p8, "examples/rastrigin_zoom.html")
            savefig(p9, "examples/rastrigin_parameter_comparison.png")
            savefig(p10, "examples/rastrigin_heatmap.png")
            savefig(p11, "examples/rastrigin_multiple_views.png")
            
            println("✓ All plots save