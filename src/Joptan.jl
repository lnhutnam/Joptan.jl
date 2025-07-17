module Joptan

# Include all submodules
include("functions.jl")

# Include loss functions
include("loss/loss.jl")

# Export loss functions
export rosenbrock, rosenbrock_gradient, rosenbrock_hessian
export rastrigin, rastrigin_gradient, rastrigin_hessian

# Export linear regression functions
export LinearRegressionLoss
export linear_regression_loss, linear_regression_gradient, linear_regression_hessian
export linear_regression_stochastic_gradient
export linear_regression_smoothness, linear_regression_max_smoothness, linear_regression_average_smoothness
export linear_regression_simple, linear_regression_gradient_simple, linear_regression_hessian_simple

end
