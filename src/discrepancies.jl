"""
    kernelized_stein_discrepancy(P, q, K::KernelFunctions.Kernel; ad_backend)

Computes the Kernelized Stein Discrepancy (KSD) between a set of samples `P` and a distribution `q`.

The KSD measures the discrepancy between two probability distributions by evaluating the expectation of two Stein operators applied to a kernel function.

# Arguments
- `P`: A matrix of samples from the empirical distribution. Each column represents a sample.
- `q`: A `LogDensityProblems.LogDensityProblem` representing the target distribution.
- `K`: A kernel function from `KernelFunctions.Kernel`.
- `ad_backend`: An automatic differentiation backend from `DifferentiationInterface`.

# Returns
The Kernelized Stein Discrepancy (KSD) as a scalar value.

# Details
The function calculates the KSD using the following formula:

```math
\\text{KSD}(P, q) = \\frac{1}{n(n-1)} \\sum_{i=1}^n \\sum_{j=1}^n u(P_i, P_j)
```

```math
u(x, y) = \\nabla s(x)^T k(x, y) \\nabla s(y) + \\nabla s(x)^T \\nabla_y k(x, y) + \\nabla_x k(x, y)^T \\nabla s(y) + \\text{tr}(\\nabla_{x,y} k(x, y))
```

For more details see :  
- A Kernelized Stein Discrepancy for Goodness-of-fit Tests, Qiang Liu, Jason Lee, Michael Jordan

"""
function kernelized_stein_discrepancy(P, q, K::KernelFunctions.Kernel; ad_backend)
    
    ∇_y(u,v) = ADI.gradient(t->K(u, t), ad_backend, v)
    ∇_x_y(u,v) = ADI.jacobian(t->∇_y(t,v), ad_backend, u)

    function u(x,y)
        k, ∇x_k = ADI.value_and_gradient(t->K(t,y), ad_backend, x)
        ∇y_k = ADI.gradient(t->K(x,t), ad_backend, y)    

        s_x, ∇s_x = LogDensityProblems.logdensity_and_gradient(q, x)
        s_y, ∇s_y = LogDensityProblems.logdensity_and_gradient(q, y)

        ∇s_x'*k*∇s_y + ∇s_x'*∇y_k + ∇x_k'*∇s_y + LinearAlgebra.tr(∇_x_y(x,y))
    end

    D = 0
    n = size(P)[2]

    for i in 1:n
        for j in 1:i
            D += u(P[:, i], P[:, j])
        end
    end

    return D/(n*(n-1))
end
