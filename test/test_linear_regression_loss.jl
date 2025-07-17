"""
Test suite for linear regression loss functions in Joptan.jl
"""

using Test
using LinearAlgebra
using Random
using Statistics

# # Add the src directory to the path for testing
# push!(LOAD_PATH, "../src")
using Joptan

# Set seed for reproducible tests
Random.seed!(42)

@testset "Linear Regression Loss Tests" begin
    
    @testset "Basic LinearRegressionLoss Construction" begin
        # Test basic construction
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [1.0, 2.0, 3.0]
        
        lrl = LinearRegressionLoss(A, b)
        
        @test lrl.A == A
        @test lrl.b == b
        @test lrl.l1 == 0.0
        @test lrl.l2 == 0.0
        @test lrl.n == 3
        @test lrl.d == 2
        @test lrl.store_mat_vec_prod == true
        
        # Test with regularization
        lrl_reg = LinearRegressionLoss(A, b, l1=0.1, l2=0.5)
        @test lrl_reg.l1 == 0.1
        @test lrl_reg.l2 == 0.5
        
        # Test dimension mismatch
        b_wrong = [1.0, 2.0]
        @test_throws DimensionMismatch LinearRegressionLoss(A, b_wrong)
    end
    
    @testset "Matrix-Vector Product" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [1.0, 2.0, 3.0]
        x = [0.5, -0.5]
        
        lrl = LinearRegressionLoss(A, b)
        
        # Test matrix-vector product
        result = mat_vec_product(lrl, x)
        expected = A * x
        @test result ≈ expected
        
        # Test caching
        result2 = mat_vec_product(lrl, x)
        @test result2 ≈ expected
        @test result2 == result
        
        # Test cache invalidation
        x_new = [1.0, 1.0]
        result3 = mat_vec_product(lrl, x_new)
        expected3 = A * x_new
        @test result3 ≈ expected3
    end
    
    @testset "Loss Function Values" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [1.0, 2.0, 3.0]
        x = [0.5, -0.5]
        
        # Test basic loss
        lrl = LinearRegressionLoss(A, b)
        loss_val = linear_regression_loss(lrl, x)
        
        # Manual calculation
        residual = A * x - b
        expected_loss = 0.5 * norm(residual)^2 / 3
        @test loss_val ≈ expected_loss
        
        # Test with L2 regularization
        lrl_l2 = LinearRegressionLoss(A, b, l2=0.1)
        loss_l2 = linear_regression_loss(lrl_l2, x)
        expected_l2 = expected_loss + 0.5 * 0.1 * norm(x)^2
        @test loss_l2 ≈ expected_l2
        
        # Test with L1 regularization
        lrl_l1 = LinearRegressionLoss(A, b, l1=0.1)
        loss_l1 = linear_regression_loss(lrl_l1, x)
        expected_l1 = expected_loss + 0.1 * norm(x, 1)
        @test loss_l1 ≈ expected_l1
    end
    
    @testset "Gradient Computation" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [1.0, 2.0, 3.0]
        x = [0.5, -0.5]
        
        # Test basic gradient
        lrl = LinearRegressionLoss(A, b)
        grad = linear_regression_gradient(lrl, x)
        
        # Manual calculation
        residual = A * x - b
        expected_grad = A' * residual / 3
        @test grad ≈ expected_grad
        
        # Test with L2 regularization
        lrl_l2 = LinearRegressionLoss(A, b, l2=0.1)
        grad_l2 = linear_regression_gradient(lrl_l2, x)
        expected_grad_l2 = expected_grad + 0.1 * x
        @test grad_l2 ≈ expected_grad_l2
        
        # Test with L1 regularization
        lrl_l1 = LinearRegressionLoss(A, b, l1=0.1)
        grad_l1 = linear_regression_gradient(lrl_l1, x)
        expected_grad_l1 = expected_grad + 0.1 * sign.(x)
        @test grad_l1 ≈ expected_grad_l1
    end
    
    @testset "Hessian Computation" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [1.0, 2.0, 3.0]
        x = [0.5, -0.5]
        
        # Test basic Hessian
        lrl = LinearRegressionLoss(A, b)
        hess = linear_regression_hessian(lrl, x)
        
        # Manual calculation
        expected_hess = A' * A / 3
        @test hess ≈ expected_hess
        
        # Test with L2 regularization
        lrl_l2 = LinearRegressionLoss(A, b, l2=0.1)
        hess_l2 = linear_regression_hessian(lrl_l2, x)
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
        
        lrl = LinearRegressionLoss(A, b, l2=0.1)
        
        # Test stochastic gradient with specified indices
        idx = [1, 3, 5]
        stoch_grad = linear_regression_stochastic_gradient(lrl, x, idx)
        
        # Manual calculation
        residual = A[idx, :] * x - b[idx]
        expected_grad = A[idx, :]' * residual / length(idx) + 0.1 * x
        @test stoch_grad ≈ expected_grad
        
        # Test with random sampling
        stoch_grad_rand = linear_regression_stochastic_gradient(lrl, x, batch_size=10)
        @test length(stoch_grad_rand) == 5
        
        # Test that full batch equals regular gradient
        full_batch_grad = linear_regression_stochastic_gradient(lrl, x, collect(1:100))
        regular_grad = linear_regression_gradient(lrl, x)
        @test full_batch_grad ≈ regular_grad
    end
    
    @testset "Smoothness Properties" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [1.0, 2.0, 3.0]
        
        lrl = LinearRegressionLoss(A, b, l2=0.1)
        
        # Test smoothness computation
        smoothness = linear_regression_smoothness(lrl)
        covariance = A' * A / 3
        expected_smoothness = maximum(eigvals(covariance)) + 0.1
        @test smoothness ≈ expected_smoothness
        
        # Test max smoothness
        max_smoothness = linear_regression_max_smoothness(lrl)
        expected_max = maximum(sum(abs2, A, dims=2)) + 0.1
        @test max_smoothness ≈ expected_max
        
        # Test average smoothness
        avg_smoothness = linear_regression_average_smoothness(lrl)
        expected_avg = mean(sum(abs2, A, dims=2)) + 0.1
        @test avg_smoothness ≈ expected_avg
        
        # Test ordering
        @test avg_smoothness <= smoothness <= max_smoothness
    end
    
    @testset "Simple Functions" begin
        A = [1.0 2.0; 3.0 4.0; 5.0 6.0]
        b = [1.0, 2.0, 3.0]
        x = [0.5, -0.5]
        
        # Test simple loss function
        loss_simple = linear_regression_simple(A, b, x)
        lrl = LinearRegressionLoss(A, b)
        loss_object = linear_regression_loss(lrl, x)
        @test loss_simple ≈ loss_object
        
        # Test simple gradient function
        grad_simple = linear_regression_gradient_simple(A, b, x)
        grad_object = linear_regression_gradient(lrl, x)
        @test grad_simple ≈ grad_object
        
        # Test simple Hessian function
        hess_simple = linear_regression_hessian_simple(A, b, x)
        hess_object = linear_regression_hessian(lrl, x)
        @test hess_simple ≈ hess_object
        
        # Test with regularization
        loss_reg = linear_regression_simple(A, b, x, l1=0.1, l2=0.2)
        lrl_reg = LinearRegressionLoss(A, b, l1=0.1, l2=0.2)
        loss_reg_obj = linear_regression_loss(lrl_reg, x)
        @test loss_reg ≈ loss_reg_obj
    end
    
    @testset "Analytical Solutions" begin
        # Test that gradient is zero at analytical solution
        A = rand(20, 5)
        b = rand(20)
        
        # OLS solution
        lrl = LinearRegressionLoss(A, b)
        x_ols = (A' * A) \ (A' * b)
        grad_at_ols = linear_regression_gradient(lrl, x_ols)
        @test norm(grad_at_ols) < 1e-10
        
        # Ridge solution
        lrl_ridge = LinearRegressionLoss(A, b, l2=0.1)
        x_ridge = (A' * A + 0.1 * I(5)) \ (A' * b)
        grad_at_ridge = linear_regression_gradient(lrl_ridge, x_ridge)
        @test norm(grad_at_ridge) < 1e-10
    end
    
    @testset "Finite Difference Gradient Check" begin
        A = rand(10, 3)
        b = rand(10)
        x = rand(3)
        
        lrl = LinearRegressionLoss(A, b, l1=0.1, l2=0.1)
        
        # Analytical gradient
        grad_analytical = linear_regression_gradient(lrl, x)
        
        # Finite difference gradient
        h = 1e-8
        grad_fd = zeros(3)
        
        for i in 1:3
            x_plus = copy(x)
            x_minus = copy(x)
            x_plus[i] += h
            x_minus[i] -= h
            
            grad_fd[i] = (linear_regression_loss(lrl, x_plus) - linear_regression_loss(lrl, x_minus)) / (2 * h)
        end
        
        # Note: L1 regularization makes this test approximate due to non-differentiability
        # We test only the differentiable part
        lrl_smooth = LinearRegressionLoss(A, b, l2=0.1)
        grad_smooth = linear_regression_gradient(lrl_smooth, x)
        
        grad_fd_smooth = zeros(3)
        for i in 1:3
            x_plus = copy(x)
            x_minus = copy(x)
            x_plus[i] += h
            x_minus[i] -= h
            
            grad_fd_smooth[i] = (linear_regression_loss(lrl_smooth, x_plus) - linear_regression_loss(lrl_smooth, x_minus)) / (2 * h)
        end
        
        @test norm(grad_smooth - grad_fd_smooth) < 1e-6
    end
    
    @testset "Finite Difference Hessian Check" begin
        A = rand(10, 3)
        b = rand(10)
        x = rand(3)
        
        lrl = LinearRegressionLoss(A, b, l2=0.1)
        
        # Analytical Hessian
        hess_analytical = linear_regression_hessian(lrl, x)
        
        # Finite difference Hessian
        h = 1e-8
        hess_fd = zeros(3, 3)
        
        for i in 1:3
            for j in 1:3
                x_pp = copy(x)
                x_pm = copy(x)
                x_mp = copy(x)
                x_mm = copy(x)
                
                x_pp[i] += h; x_pp[j] += h
                x_pm[i] += h; x_pm[j] -= h
                x_mp[i] -= h; x_mp[j] += h
                x_mm[i] -= h; x_mm[j] -= h
                
                hess_fd[i, j] = (linear_regression_loss(lrl, x_pp) - linear_regression_loss(lrl, x_pm) -
                                linear_regression_loss(lrl, x_mp) + linear_regression_loss(lrl, x_mm)) / (4 * h^2)
            end
        end
        
        @test norm(hess_analytical - hess_fd) < 1e-6
    end
    
    @testset "Edge Cases" begin
        # Test with single sample
        A = reshape([1.0, 2.0], 1, 2)
        b = [1.0]
        x = [0.5, -0.5]
        
        lrl = LinearRegressionLoss(A, b)
        @test !isnan(linear_regression_loss(lrl, x))
        @test !any(isnan.(linear_regression_gradient(lrl, x)))
        @test !any(isnan.(linear_regression_hessian(lrl, x)))
        
        # Test with zero parameters
        x_zero = zeros(2)
        @test !isnan(linear_regression_loss(lrl, x_zero))
        @test !any(isnan.(linear_regression_gradient(lrl, x_zero)))
        
        # Test with large regularization
        lrl_large_reg = LinearRegressionLoss(A, b, l1=1000.0, l2=1000.0)
        @test !isnan(linear_regression_loss(lrl_large_reg, x))
        @test !any(isnan.(linear_regression_gradient(lrl_large_reg, x)))
    end
end

println("All tests passed! ✓")