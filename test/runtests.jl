using Joptan
using Test

@testset "Joptan.jl" begin
    # Write your tests here.
    # @test Joptan.greet_your_package_name() == "Hello YourPackageName!"
    # @test Joptan.greet_your_package_name() != "Hello world!"


    @testset "test_linear_regression_loss" begin
      include("test_linear_regression_loss.jl")
    end
end
