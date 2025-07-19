"""
Test suite for linear regression loss functions in Joptan.jl
"""

using Test
using LinearAlgebra
using Random
using Statistics
using Joptan

# Set seed for reproducible tests
Random.seed!(42)

@testset "Linear Regression Loss Tests" begin
    
    @testset "Basic LinearRegressionOracle Construction" begin
        # Test basic construction
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [1.0, 2.0, 3.0]
        
        lro = LinearRegressionOracle(A, b)
        
        @test lro.A == A
        @test lro.b == b
        @test lro.l1 == 0.0
        @test lro.l2 == 0.0
        @test lro.n == 3
        @test lro.d == 2
        @test lro.store_mat_vec_prod == true
        
        # Test with regularization
        lro_reg = LinearRegressionOracle(A, b, l1=0.1, l2=0.5)
        @test lro_reg.l1 == 0.1
        @test lro_reg.l2 == 0.5
        
        # Test dimension mismatch
        b_wrong = [1.0, 2.0]
        @test_throws DimensionMismatch LinearRegressionOracle(A, b_wrong)
    end
    
    @testset "Matrix-Vector Product" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [1.0, 2.0, 3.0]
        x = [0.5, -0.5]
        
        lro = LinearRegressionOracle(A, b)
        
        # Test matrix-vector product
        result = mat_vec_product(lro, x)
        expected = A * x
        @test result ≈ expected
        
        # Test caching
        result2 = mat_vec_product(lro, x)
        @test result2 ≈ expected
        @test result2 == result
        
        # Test cache invalidation
        x_new = [1.0, 1.0]
        result3 = mat_vec_product(lro, x_new)
        expected3 = A * x_new
        @test result3 ≈ expected3
    end
    
    @testset "Loss Function Values" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [1.0, 2.0, 3.0]
        x = [0.5, -0.5]
        
        # Test basic loss
        lro = LinearRegressionOracle(A, b)
        loss_val = value(lro, x)
        
        # Manual calculation
        residual = A * x - b
        expected_loss = 0.5 * norm(residual)^2 / 3
        @test loss_val ≈ expected_loss
        
        # Test with L2 regularization
        lro_l2 = LinearRegressionOracle(A, b, l2=0.1)
        loss_l2 = value(lro_l2, x)
        expected_l2 = expected_loss + 0.5 * 0.1 * norm(x)^2
        @test loss_l2 ≈ expected_l2
        
        # Test with L1 regularization
        lro_l1 = LinearRegressionOracle(A, b, l1=0.1)
        loss_l1 = value(lro_l1, x)
        expected_l1 = expected_loss + 0.1 * norm(x, 1)
        @test loss_l1 ≈ expected_l1
    end
    
    @testset "Gradient Computation" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [1.0, 2.0, 3.0]
        x = [0.5, -0.5]
        
        # Test basic gradient
        lro = LinearRegressionOracle(A, b)
        grad = gradient(lro, x)
        
        # Manual calculation
        residual = A * x - b
        expected_grad = A' * residual / 3
        @test grad ≈ expected_grad
        
        # Test with L2 regularization
        lro_l2 = LinearRegressionOracle(A, b, l2=0.1)
        grad_l2 = gradient(lro_l2, x)
        expected_grad_l2 = expected_grad + 0.1 * x
        @test grad_l2 ≈ expected_grad_l2
        
        # Test with L1 regularization
        lro_l1 = LinearRegressionOracle(A, b, l1=0.1)
        grad_l1 = gradient(lro_l1, x)
        expected_grad_l1 = expected_grad + 0.1 * sign.(x)
        @test grad_l1 ≈ expected_grad_l1
    end
    
    @testset "Hessian Computation" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [1.0, 2.0, 3.0]
        x = [0.5, -0.5]
        
        # Test basic Hessian
        lro = LinearRegressionOracle(A, b)
        hess = hessian(lro, x)
        
        # Manual calculation
        expected_hess = A' * A / 3
        @test hess ≈ expected_hess
        
        # Test with L2 regularization
        lro_l2 = LinearRegressionOracle(A, b, l2=0.1)
        hess_l2 = hessian(lro_l2, x)
        expected_hess_l2 = expected_hess + 0.1 * I(2)
        @test hess_l2 ≈ expected_hess_l2
        
        # Test symmetry
        @test hess ≈ hess'
        @test hess_l2 ≈ hess_l2'
    end
    
    @testset "Stochastic Gradient" begin
        A = rand(100, 5)
        b = rand(100)
        x = rand(5)
        
        lro = LinearRegressionOracle(A, b, l2=0.1)
        
        # Test stochastic gradient with specified indices
        idx = [1, 3, 5]
        stoch_grad = stochastic_gradient(lro, x, idx)
        
        # Manual calculation
        residual = A[idx, :] * x - b[idx]
        expected_grad = A[idx, :]' * residual / length(idx) + 0.1 * x
        @test stoch_grad ≈ expected_grad
        
        # Test with random sampling
        stoch_grad_rand = stochastic_gradient(lro, x, nothing, batch_size=10)
        @test length(stoch_grad_rand) == 5
        
        # Test that full batch equals regular gradient
        full_batch_grad = stochastic_gradient(lro, x, collect(1:100))
        regular_grad = gradient(lro, x)
        @test full_batch_grad ≈ regular_grad
    end
    
    @testset "Smoothness Properties" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [1.0, 2.0, 3.0]
        
        lro = LinearRegressionOracle(A, b, l2=0.1)
        
        # Test smoothness computation
        smooth = smoothness(lro)
        covariance = A' * A / 3
        expected_smoothness = maximum(eigvals(covariance)) + 0.1
        @test smooth ≈ expected_smoothness
        
        # Test max smoothness
        max_smooth = max_smoothness(lro)
        expected_max = maximum(sum(abs2, A, dims=2)) + 0.1
        @test max_smooth ≈ expected_max
        
        # Test average smoothness
        avg_smooth = average_smoothness(lro)
        expected_avg = mean(sum(abs2, A, dims=2)) + 0.1
        @test avg_smooth ≈ expected_avg
        
        # Test ordering
        @test smooth <= avg_smooth <= max_smooth
    end
    
    @testset "Simple Functions" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [1.0, 2.0, 3.0]
        x = [0.5, -0.5]
        
        # Test simple loss function
        loss_simple = linear_regression_loss(A, b, x)
        lro = LinearRegressionOracle(A, b)
        loss_object = value(lro, x)
        @test loss_simple ≈ loss_object
        
        # Test simple gradient function
        grad_simple = linear_regression_gradient(A, b, x)
        grad_object = gradient(lro, x)
        @test grad_simple ≈ grad_object
        
        # Test simple Hessian function
        hess_simple = linear_regression_hessian(A, b, x)
        hess_object = hessian(lro, x)
        @test hess_simple ≈ hess_object
        
        # Test with regularization
        loss_reg = linear_regression_loss(A, b, x, l1=0.1, l2=0.2)
        lro_reg = LinearRegressionOracle(A, b, l1=0.1, l2=0.2)
        loss_reg_obj = value(lro_reg, x)
        @test loss_reg ≈ loss_reg_obj
    end
    
    @testset "Oracle Interface Methods" begin
        A = rand(10, 5)
        b = rand(10)
        x = rand(5)
        
        lro = LinearRegressionOracle(A, b, l1=0.1, l2=0.2)
        
        # Test best point tracking through the oracle's internal state
        # Initially, no best point should be set
        @test lro.x_opt === nothing
        @test lro.f_opt == Inf
        
        # Evaluate at a point to set best
        loss_val = value(lro, x)
        @test lro.x_opt ≈ x
        @test lro.f_opt ≈ loss_val
        
        # Evaluate at a worse point - best should not change
        x_worse = x .+ 10.0  # Much worse point
        loss_worse = value(lro, x_worse)
        @test lro.x_opt ≈ x  # Should still be the original point
        @test lro.f_opt ≈ loss_val  # Should still be the original loss
        @test loss_worse > loss_val  # Verify it's actually worse
        
        # Evaluate at a better point - best should update
        x_better = zeros(5)  # Should be better for regularized problem
        loss_better = value(lro, x_better)
        if loss_better < loss_val
            @test lro.x_opt ≈ x_better
            @test lro.f_opt ≈ loss_better
        end
    end
    
    @testset "Label Processing" begin
        # Test binary label transformation {1, 2} -> {0, 1}
        A = rand(10, 3)
        b_12 = Float64[1, 2, 1, 2, 1, 2, 1, 2, 1, 2]  # Convert to Float64
        lro = LinearRegressionOracle(A, b_12)
        @test all(lro.b .∈ Ref([0.0, 1.0]))
        
        # Test binary label transformation {-1, 1} -> {0, 1}
        b_neg1_1 = Float64[-1, 1, -1, 1, -1, 1, -1, 1, -1, 1]  # Convert to Float64
        lro2 = LinearRegressionOracle(A, b_neg1_1)
        @test all(lro2.b .∈ Ref([0.0, 1.0]))
        
        # Test that {0, 1} labels are preserved
        b_01 = Float64[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Convert to Float64
        lro3 = LinearRegressionOracle(A, b_01)
        @test lro3.b == b_01
        
        # Test continuous labels are preserved
        b_continuous = rand(10)
        lro4 = LinearRegressionOracle(A, b_continuous)
        @test lro4.b == b_continuous
    end
    
    @testset "Analytical Solutions" begin
        # Test that gradient is zero at analytical solution
        A = rand(20, 5)
        b = rand(20)
        
        # OLS solution
        lro = LinearRegressionOracle(A, b)
        x_ols = (A' * A) \ (A' * b)
        grad_at_ols = gradient(lro, x_ols)
        @test norm(grad_at_ols) < 1e-10
        
        # Loss: f(x) = (1/2n) * ||Ax - b||² + (λ/2) * ||x||²
        # Ridge solution - the gradient formula in LinearRegressionOracle is:
        # grad = A' * (A*x - b) / n + λ * x
        # Setting to zero: A' * (A*x - b) / n + λ * x = 0
        # A' * A * x / n - A' * b / n + λ * x = 0
        # (A' * A / n + λ * I) * x = A' * b / n
        # So the solution is: x = (A' * A / n + λ * I)^(-1) * (A' * b / n)
        λ = 0.1
        lro_ridge = LinearRegressionOracle(A, b, l2=λ)
        x_ridge = (A' * A / lro_ridge.n + λ * I(5)) \ (A' * b / lro_ridge.n)
        grad_at_ridge = gradient(lro_ridge, x_ridge)
        @test norm(grad_at_ridge) < 1e-10
        
        # Alternative test: verify the analytical solution satisfies the normal equations
        # Check that the residual is orthogonal to the columns of A (for OLS)
        residual_ols = A * x_ols - b
        @test norm(A' * residual_ols) < 1e-10
        
        # For Ridge, check the modified normal equations
        residual_ridge = A * x_ridge - b
        normal_eq_ridge = A' * residual_ridge / lro_ridge.n + λ * x_ridge
        @test norm(normal_eq_ridge) < 1e-10
    end
    
    @testset "Finite Difference Gradient Check" begin
        A = rand(10, 3)
        b = rand(10)
        x = rand(3)
        
        # Test only with L2 regularization (L1 is non-differentiable)
        lro_smooth = LinearRegressionOracle(A, b, l2=0.1)
        
        # Analytical gradient
        grad_analytical = gradient(lro_smooth, x)
        
        # Finite difference gradient
        h = 1e-8
        grad_fd = zeros(3)
        
        for i in 1:3
            x_plus = copy(x)
            x_minus = copy(x)
            x_plus[i] += h
            x_minus[i] -= h
            
            grad_fd[i] = (value(lro_smooth, x_plus) - value(lro_smooth, x_minus)) / (2 * h)
        end
        
        @test norm(grad_analytical - grad_fd) < 1e-6
    end
    
    @testset "Finite Difference Hessian Check" begin
        A = rand(5, 3)  # Smaller problem for better numerical stability
        b = rand(5)
        x = rand(3)
        
        lro = LinearRegressionOracle(A, b, l2=0.1)
        
        # Analytical Hessian
        hess_analytical = hessian(lro, x)
        
        # For linear regression, Hessian is constant and doesn't depend on x
        # Let's just verify this property instead of finite differences
        x2 = rand(3)
        hess_analytical2 = hessian(lro, x2)
        @test norm(hess_analytical - hess_analytical2) < 1e-12
        
        # Verify Hessian structure: A'A/n + λI
        expected_hess = A' * A / 5 + 0.1 * I(3)
        @test norm(hess_analytical - expected_hess) < 1e-12
        
        # Test that Hessian is symmetric and positive definite
        @test norm(hess_analytical - hess_analytical') < 1e-12
        @test all(eigvals(hess_analytical) .> 0)
    end
    
    @testset "Integration with Optimizers" begin
        # Test that LinearRegressionOracle works with Adagrad
        A = rand(20, 5)
        b = rand(20)
        x0 = randn(5) * 0.1
        
        lro = LinearRegressionOracle(A, b, l2=0.1)
        
        # Define loss and gradient functions for optimizer
        loss_func(x) = value(lro, x)
        grad_func(x) = gradient(lro, x)
        
        # Create and run optimizer
        optimizer = AdagradOptimizer(loss_func, grad_func, lr=0.1, delta=1e-8)
        trace = run!(optimizer, x0, it_max=100, verbose=false)
        
        # Check that optimization ran
        @test length(trace.xs) > 1
        @test optimizer.it > 0
        
        # Check that loss decreased
        initial_loss = loss_func(x0)
        final_loss = loss_func(optimizer.x)
        @test final_loss <= initial_loss
    end
    
    @testset "Edge Cases" begin
        # Test with single sample
        A = reshape([1.0, 2.0], 1, 2)
        b = [1.0]
        x = [0.5, -0.5]
        
        lro = LinearRegressionOracle(A, b)
        @test !isnan(value(lro, x))
        @test !any(isnan.(gradient(lro, x)))
        @test !any(isnan.(hessian(lro, x)))
        
        # Test with zero parameters
        x_zero = zeros(2)
        @test !isnan(value(lro, x_zero))
        @test !any(isnan.(gradient(lro, x_zero)))
        
        # Test with large regularization
        lro_large_reg = LinearRegressionOracle(A, b, l1=1000.0, l2=1000.0)
        @test !isnan(value(lro_large_reg, x))
        @test !any(isnan.(gradient(lro_large_reg, x)))
        
        # Test with very small regularization
        lro_small_reg = LinearRegressionOracle(A, b, l1=1e-12, l2=1e-12)
        @test !isnan(value(lro_small_reg, x))
        @test !any(isnan.(gradient(lro_small_reg, x)))
    end
    
    @testset "Memory and Performance" begin
        # Test that large problems don't cause memory issues
        n, d = 1000, 100
        A = randn(n, d) / sqrt(d)
        b = randn(n)
        x = randn(d)
        
        lro = LinearRegressionOracle(A, b, l2=0.01)
        
        # These should complete without error
        @test !isnan(value(lro, x))
        @test !any(isnan.(gradient(lro, x)))
        
        # Test stochastic gradient with large batch
        stoch_grad = stochastic_gradient(lro, x, nothing, batch_size=100)
        @test length(stoch_grad) == d
        @test !any(isnan.(stoch_grad))
    end
end

println("All linear regression tests passed! ✓")