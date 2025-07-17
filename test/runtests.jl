using Joptan
using Test

@testset "Joptan.jl" begin
    # Write your tests here.
    @test Joptan.greet_your_package_name() == "Hello YourPackageName!"
    @test Joptan.greet_your_package_name() != "Hello world!"
end
