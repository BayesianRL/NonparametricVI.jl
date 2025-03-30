
import DifferentiationInterface as ADI
import LogDensityProblems
import LogDensityProblemsAD as LogAD
import ForwardDiff

import KernelFunctions


"""
    kernel_and_gradient_fn(K::KernelFunctions.Kernel, ad_backend)

Returns a function that computes the kernel value and its gradient with respect to the first argument.

# Arguments
- `K::KernelFunctions.Kernel`: The kernel function from KernelFunctions.jl
- `ad_backend`: The automatic differentiation backend to use (e.g., `AbstractDifferentiation.ForwardDiffBackend()`).

# Returns
- A function `k_∇k(x, a)` that takes two arguments `x` and `a` (of compatible types for the kernel `K`) and returns a tuple containing:
    - The kernel value `K(x, a)`.
    - The gradient of the kernel with respect to `x`, evaluated at `x`.

"""
function kernel_and_gradient_fn(K::KernelFunctions.Kernel, ad_backend=ADI.AutoForwardDiff())
    function k_∇k(x, a)
        k, ∇k = ADI.value_and_gradient(t -> K(t, a), ad_backend, x)
        return k, ∇k
    end
    return k_∇k
end