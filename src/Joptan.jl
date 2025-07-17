module Joptan

# Include all submodules
include("functions.jl")

# Include loss functions
include("loss/loss.jl")

# Export loss functions
export rosenbrock, rosenbrock_gradient, rosenbrock_hessian
export rastrigin, rastrigin_gradient, rastrigin_hessian

end
