![NonparametricVI.jl logo](logo/logo-light-typo-1200.png)
[![Build Status](https://github.com/BayesianRL/NonparametricVI.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/BayesianRL/NonparametricVI.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](https://bayesianrl.github.io/NonparametricVI.jl/dev/)


# NonparametricVI.jl
NonparametricVI.jl is a collection of particle-based and nonparametric variational methods for approximate Bayesian inference in Julia. You can use it either with [Turing](https://turinglang.org/) probabilistic programming language or other custom sampling problems like Energy-Based Models. In comparison to parametric variational inference, particle-based approaches do not require a parametric family of distibution to approximate posterior. 

https://github.com/user-attachments/assets/3dc29684-2642-4dd2-8be3-3e402de744d2

## Getting Started
### Installation
NonparametricVI.jl is under development, you can install the latest version from this repository using Pkg:
```julia
Pkg.add(url="https://github.com/BayesianRL/NonparametricVI.jl.git")
```


### Using with Turing.jl Probabilistic Programs
#### Example: Linear Regression
Let's craft a toy regression problem:
```julia
using NonparametricVI
using Turing
using LinearAlgebra
using KernelFunctions
using CairoMakie

n = 100
X = 2rand(n) .- 1.0
y = 3X .+ 1 + randn(n)
```

The generated problem looks like this:
<p align="center">
    <img src="examples/linear_regression/data.png" width="400">
</p>
We start by defining a simple Turing.jl model for regression and instantiate it:

```julia
@model function bayesian_regression(X, y)
    α ~ Normal(0.0, 1.0)
    β ~ Normal(0.0, 1.0)

    for i in eachindex(y)
        y[i] ~ Normal(α * X[i] + β, 0.5)
    end
end

model = bayesian_regression(X, y)
```

To define the dynamics of Stein Variational Gradient Descent (SVGD), we need a positive-definite kernel. You can use all provided by [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl). We use a scaled squared exponential kernel. for more details on designing more complex kernels, check out [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl) documentation:  
```julia
using KernelFunctions
kernel = SqExponentialKernel() ∘ ScaleTransform(0.3)
```

Next we define the parameters of `SVGD`:  

```julia
dynamics = SVGD(K=kernel, η=0.003, batchsize=32)
```

Nonparametric Variational Inference methods use a set of particles instead of a parametric family of distribution to approximate posterior (or any target) distribution. The `init` method creates the particles `pc`, in addition to an internal state `state` which will be used by the inference procedure.
```julia
pc, state = init(model, dynamics; n_particles=128)
```

`pc` is a simple struct containing position of particles. Using `get_samples` we can access the particles and plot them:
```julia
samples = get_samples(pc, state)
α_samples = [s[@varname(α)] for s in samples]
β_samples = [s[@varname(β)] for s in samples];
```
Note that some Turing models may contain constrained parameters (e.g. positive, bounded, ...) while most inference methods are performed on an unconstrained space obtained by transforming the original denisty of parameters. The `get_samples` method transforms the particle positions back to the contrained space. Before running SVGD we can visualize the currest state of particles:  

<p align="center">
    <img src="examples/linear_regression/particles_before_inference.png" width="400">
</p>

Finally we can perform inference. Note the `infer!` method modifies the particles in-place.
```julia
infer!(pc, state; iters=200)
```

After collecting samples with `get_samples` we can visualize the final result:
<p align="center">
    <img src="examples/linear_regression/particles_after_inference.png" width="400">
</p>


### Using with `LogDensityProblems`
#### Example: A Mixture Density
In addtion to Turing programs, you can use NonparametricVI for a custom Bayesian inference problem by implementing the [`LogDensityProblems`](https://github.com/tpapp/LogDensityProblems.jl) interface. For example here we define a toy unnormalized mixture density:
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
dynamics = SVGD(K=kernel, η=0.4, batchsize=16)
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
report = infer!(pc, state; iters=150, track=Dict(
    "KSD" => KernelizedSteinDiscrepancy(kernel, 64)
));
S = get_samples(pc)
```  

The above code also tracks the value of Kernelized Stein Discrepancy(KSD) during inference. Since KSD can be expensive to compute, we use a Monte Carlo estimation with `64` particles sampled at each step. After inference we can access the tracked values using `report.metrics["KSD"]` and plot it:

<p align="center">
    <img src="examples/mixture/KSD.png" width="512">
</p>

Finally we can check the terminal position of particles:
<p align="center">
    <img src="examples/mixture/particles_after_inference.png" width="512">
</p>

## Implemented Methods

| Method            | 📝 Paper                                            | Support       | Notes               |
|----------------------------|---------------------------------------------------------|---------------|---------------------|
| Stein Variational Gradient Descent | 📔 [Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm](https://arxiv.org/abs/1608.04471) ✏️ Qiang Liu, Dilin Wang         | ✅ Basic functionality           |  Accuracy is sensitive to kernel choice. see [`SVGD`](https://bayesianrl.github.io/NonparametricVI.jl/dev/api/#NonparametricVI.SVGD)  |
| Stein Variational Newton method | 📔 [A Stein variational Newton method](https://arxiv.org/abs/1806.03085) ✏️ Gianluca Detommaso, Tiangang Cui, Alessio Spantini, Youssef Marzouk, Robert Scheichl         | 🚧 todo           |  |
| Projected Stein Variational Newton | 📔 [Projected Stein Variational Newton: A Fast and Scalable Bayesian Inference Method in High Dimensions](https://arxiv.org/abs/1901.08659) ✏️ Peng Chen, Keyi Wu, Joshua Chen, Thomas O'Leary-Roseberry, Omar Ghattas         | 🚧 todo           |  |
| Stein Self-Repulsive Dynamics | 📔 [Stein Self-Repulsive Dynamics: Benefits From Past Samples](https://arxiv.org/abs/2002.09070) ✏️ Mao Ye, Tongzheng Ren, Qiang Liu         | 🚧 todo           |  |
| SPH-ParVI | 📔 [Variational Inference via Smoothed Particle Hydrodynamics](https://arxiv.org/abs/2407.09186) ✏️ Yongchao Huang         | 🚧 todo           |  |
| MPM-ParVI | 📔 [Variational Inference Using Material Point Method](https://arxiv.org/abs/2407.20287)  ✏️ Yongchao Huang        | 🚧 todo           |  |
| EParVI | 📔 [Electrostatics-based particle sampling and approximate inference](https://arxiv.org/abs/2406.20044) ✏️ Yongchao Huang         | 🚧 todo           |  |


