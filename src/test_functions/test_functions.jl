include("rastrigin.jl")
include("rosenbrock.jl")

# Export loss functions
export rosenbrock, rosenbrock_gradient, rosenbrock_hessian
export rastrigin, rastrigin_gradient, rastrigin_hessian