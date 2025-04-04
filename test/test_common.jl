@testset "Kernel gradients" begin
    k_∇k = NonparametricVI.kernel_and_gradient_fn(KernelFunctions.SqExponentialKernel(), AutoForwardDiff())
    k, ∇k = k_∇k([1.0], [0.5])
end