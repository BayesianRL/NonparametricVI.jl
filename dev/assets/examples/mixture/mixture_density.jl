using Pkg
Pkg.activate("../examples_env")

Pkg.add(url="https://github.com/BayesianRL/NonparametricVI.jl.git")

Pkg.add(["Distributions", "Turing", "KernelFunctions", "CairoMakie", "LogDensityProblems"])

using NonparametricVI
using LogDensityProblems
using LinearAlgebra
using KernelFunctions
using CairoMakie

struct MixtureDensity end

function LogDensityProblems.capabilities(::Type{<:MixtureDensity})
    LogDensityProblems.LogDensityOrder{0}()
end

LogDensityProblems.dimension(::MixtureDensity) = 2

function LogDensityProblems.logdensity(::MixtureDensity, x)
    log(0.25 * exp(-1/0.5 * norm(x-[-1.5, -1.5])^2) +
        0.25 * exp(-1/0.5 * norm(x-[-1.5,  1.5])^2) +
        0.25 * exp(-1/0.5 * norm(x-[ 1.5, -1.5])^2) +
        0.25 * exp(-1/0.5 * norm(x-[ 1.5,  1.5])^2))
end

ρ = MixtureDensity()


kernel = SqExponentialKernel() ∘ ScaleTransform(2.0)
dynamics = SVGD(K=kernel, η=0.5, batchsize=16)

pc, state = init(ρ, dynamics; n_particles=512)

S = get_samples(pc)

# plot samples
fig = Figure(size=(700,700))
ax = Axis(fig[1,1])

M = 3.5
heatmap!(-M:0.01:M, -M:0.01:M,
         (x,y)->exp(LogDensityProblems.logdensity(ρ, [x,y])), colormap=:ice)

scatter!(S, color=:deepskyblue, markersize=7)

xlims!(-M, M)
ylims!(-M, M)

save("particles_before_inference.png", fig)


infer!(pc, state; iters=100, verbose=true);

S = get_samples(pc)


# plot samples
fig = Figure(size=(700,700))
ax = Axis(fig[1,1])

M = 3.5
heatmap!(-M:0.01:M, -M:0.01:M,
         (x,y)->exp(LogDensityProblems.logdensity(ρ, [x,y])), colormap=:ice)

scatter!(S, color=:deepskyblue, markersize=7)

xlims!(-M, M)
ylims!(-M, M)

save("particles_after_inference.png", fig)
