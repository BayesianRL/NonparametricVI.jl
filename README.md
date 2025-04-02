![NonparametricVI.jl logo](logo/logo-light-typo-1200.png)
[![Build Status](https://github.com/BayesianRL/NonparametricVI.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BayesianRL/NonparametricVI.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://bayesianrl.github.io/NonparametricVI.jl/dev/)


# NonparametricVI.jl


## Getting Started
### Installation
NonparametricVI.jl is under development, you can install the latest version using Pkg:
```julia
Pkg.add("https://github.com/BayesianRL/NonparametricVI.jl.git")
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


