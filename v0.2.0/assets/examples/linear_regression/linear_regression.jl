using Pkg
Pkg.activate("../examples_env")

Pkg.add(url="https://github.com/BayesianRL/NonparametricVI.jl.git")

Pkg.add(["Distributions", "Turing", "KernelFunctions", "CairoMakie", "LogDensityProblems"])

using NonparametricVI
using Turing
using LinearAlgebra
using KernelFunctions
using CairoMakie

n = 100
X = 2rand(n) .- 1.0
y = 3X .+ 1 + randn(n)



fig = Figure(size=(700,700))
ax = Axis(fig[1,1])
scatter!(X,y)
save("data.png", fig)


using Turing
using NonparametricVI
using CairoMakie

@model function bayesian_regression(X, y)
    α ~ Normal(0.0, 1.0)
    β ~ Normal(0.0, 1.0)

    for i in eachindex(y)
        y[i] ~ Normal(α * X[i] + β, 0.5)
    end
end

model = bayesian_regression(X, y)


kernel = SqExponentialKernel() ∘ ScaleTransform(0.3)
dynamics = SVGD(K=kernel, η=0.003, batchsize=32)
pc, state = init(model, dynamics; n_particles=128)

samples = get_samples(pc, state)
α_samples = [s[@varname(α)] for s in samples]
β_samples = [s[@varname(β)] for s in samples];

fig = Figure(size=(700,700))
ax = Axis(fig[1,1])
x_rng = collect(-1:0.1:1)
for i in eachindex(α_samples)
  lines!(x_rng, α_samples[i] * x_rng .+ β_samples[i], color=:gray, alpha=0.5)
end
scatter!(X,y)
save("particles_before_inference.png", fig)



infer!(pc, state; iters=200)


samples = get_samples(pc, state)
α_samples = [s[@varname(α)] for s in samples]
β_samples = [s[@varname(β)] for s in samples];

fig = Figure(size=(700,700))
ax = Axis(fig[1,1])
x_rng = collect(-1:0.1:1)
for i in eachindex(α_samples)
  lines!(x_rng, α_samples[i] * x_rng .+ β_samples[i], color=:gray, alpha=0.5)
end
scatter!(X,y)
save("particles_after_inference.png", fig)