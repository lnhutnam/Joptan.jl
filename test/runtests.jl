using joptan
using Test

@testset "joptan.jl" begin
    # Write your tests here.
    @test joptan.greet_your_package_name() == "Hello YourPackageName!"
    @test joptan.greet_your_package_name() != "Hello world!"
end
