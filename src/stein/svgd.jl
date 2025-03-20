import KernelFunctions

mutable struct SVGD
    K::KernelFunctions.Kernel
    η
    batchsize
end

# function stein_∇(P, x, h)
#     S = sample(1:N, batchsize; replace=false)

#     Δ = [k(P[:, S[i]], x, h) * ∇logρ(P[:, S[i]]) + ∇k(P[:, S[i]], x, h) for i ∈ 1:M]
#     mean(Δ)
# end


# function stein_variational_gradient_descent!(P, η, h)
#     for i ∈ 1:N
#         P[:, i] += η * stein_∇(P, P[:, i], h)
#     end
# end