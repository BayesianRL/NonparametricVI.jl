![NonparametricVI.jl logo](logo/logo-light-typo-1200.png)
[![Build Status](https://github.com/BayesianRL/NonparametricVI.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BayesianRL/NonparametricVI.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://bayesianrl.github.io/NonparametricVI.jl/dev/)


# NonparametricVI.jl


## Getting Started
### Installation
NonparametricVI.jl is under development, you can install the latest version using Pkg:
```julia
Pkg.add(url="https://github.com/BayesianRL/NonparametricVI.jl.git")
```


### Using with Turing.jl Probabilistic Programs
We start by defining a Turing.jl model and instantiate it:
```julia
using Turing
using NonparametricVI

@model function beta_binomial(x)
    θ ~ Beta(1.0, 1.0)
    for i in eachindex(x)
        x[i] ~ Bernoulli(θ)
    end
end

model = beta_binomial([1, 1, 1, 0, 1, 1])
```
In this example we use `SVGD` for inference so first we need a positive-definite kernel. You can use kernels provided by [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl). Here we simply use a squared exponential kernel:
```julia
using KernelFunctions
kernel = SqExponentialKernel()
```
Next we define the desired particle dynamics for inference which in this case is `SVGD`
```julia
dynamics = SVGD(K=kernel, η=0.08, batchsize=16)
```
The `init` method creates the particles in addition to an internal state which will be used by in the inference procedure.
```julia
pc, state = init(model, dynamics; n_particles=128)
```
Finally we can perform inference. `infer!` will modify particles in-place.
```julia
infer!(pc, state; iters=10)
```

### Using with `LogDensityProblems`
In addtion to Turing programs, you can use NonparametricVI for a custom Bayesian inference problem by implementing the [`LogDensityProblems.jl`](https://github.com/tpapp/LogDensityProblems.jl) interface. For example here we define a toy unnormalized mixture density:
```julia
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
```

Next we define the inference dynamics by choosing a custom kernel. It can be any kernel provided by [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl). Here we use a scaled version of the squared exponential kernel:
```julia
kernel = SqExponentialKernel() ∘ ScaleTransform(2.0)
dynamics = SVGD(K=kernel, η=0.5, batchsize=16)
```

Now we create a set of particles that represent samples:
```julia
pc, state = init(ρ, dynamics; n_particles=512)
```
We can access particle positions by `get_samples` and visualize the their current position:
```julia
S = get_samples(pc)
```

<p align="center">
    <img src="examples/mixture/particles_before_inference.png" width="512">
</p>

Obviously the initial samples does not match the target density. Now we run the `SVGD` dynamics to adjust the samples:

```julia
infer!(pc, state; iters=100, verbose=true)
S = get_samples(pc)
```  

Finally we can check the terminal position of particles:
<p align="center">
    <img src="examples/mixture/particles_after_inference.png" width="512">
</p>

## Implemented Methods

| Method            | 📝 Paper                                            | Support       | Notes               |
|----------------------------|---------------------------------------------------------|---------------|---------------------|
| Stein Variational Gradient Descent | [Paper](https://arxiv.org/abs/1608.04471)          | ✅ Basic functionality           |  |
| Stein Variational Newton method | [Paper](https://arxiv.org/abs/1806.03085)          | 🚧 todo           |  |
| Projected Stein Variational Newton | [Paper](https://arxiv.org/abs/1901.08659)          | 🚧 todo           |  |
| Stein Self-Repulsive Dynamics | [Paper](https://arxiv.org/abs/2002.09070)          | 🚧 todo           |  |
| SPH-ParVI | [Paper](https://arxiv.org/abs/2407.09186)          | 🚧 todo           |  |
| MPM-ParVI | [Paper](https://arxiv.org/abs/2407.20287)          | 🚧 todo           |  |
| EParVI | [Paper](https://arxiv.org/abs/2406.20044)          | 🚧 todo           |  |


