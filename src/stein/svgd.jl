import Base.@kwdef

import DynamicPPL.LogDensityFunction

import Statistics
import StatsBase




"""
    SVGD <: ParticleDynamics

Stein Variational Gradient Descent (SVGD) particle dynamics.

# Fields
- `K::KernelFunctions.Kernel`: The kernel function used to define the interaction between particles.
- `η`: The step size or learning rate for updating particle positions.
- `batchsize`: The number of particles to use in each update step (for mini-batching). If `nothing`, all particles are used.

# Examples
```julia
using KernelFunctions
```

## Define a squared exponential kernel
```julia
sqexp_kernel = SqExponentialKernel()
```

## Create an SVGD dynamics object with a fixed step size and full batch
```julia
svgd_fullbatch = SVGD(K=sqexp_kernel, η=0.1, batchsize=nothing)
```

## Create an SVGD dynamics object with a smaller step size and a batch size of 100
```julia
svgd_minibatch = SVGD(K=sqexp_kernel, η=0.05, batchsize=100)
```

"""
@kwdef mutable struct SVGD <: ParticleDynamics
    K::KernelFunctions.Kernel
    η
    batchsize
end


mutable struct SVGDInferenceState <: InferenceState
    ρ
    dynamics::SVGD
    model::Union{DynamicPPL.Model, Nothing}
end

function init_state(ρ, dynamics::SVGD, model::DynamicPPL.Model)
    return SVGDInferenceState(ρ, dynamics, model)
end

function init_state(ρ, dynamics::SVGD)
    return SVGDInferenceState(ρ, dynamics, nothing)
end


"""
    particle_velocity(pc::ParticleContainer, ρ, pi, k_∇k, dynamics::SVGD)

Computes the velocity of a single particle based on the Stein Variational Gradient Descent (SVGD) update rule, potentially using a mini-batch of other particles.

# Arguments
- `pc::ParticleContainer`: The container holding the particles.
- `ρ`: The log-density function (a `LogDensityProblem`) which must be differentiable.
- `pi::Int`: The index of the particle for which to compute the velocity.
- `k_∇k`: A function that takes two particle positions (as vectors) and returns a tuple containing the kernel value and the gradient of the kernel with respect to the first argument. This is generated by `kernel_and_gradient_fn`.
- `dynamics::SVGD`: The `SVGD` dynamics object containing the kernel, step size, and batch size.

# Returns
- `velocity::Vector`: The computed velocity vector for the particle with index `pi`.

"""
function particle_velocity(pc::ParticleContainer,
                           ρ,
                           pi, k_∇k, dynamics::SVGD)

    P = pc.P
    batchsize = dynamics.batchsize
    if isnothing(batchsize)
        batchsize = pc.size
    end
    
    # sample a mini-batch
    S = StatsBase.sample(1:pc.size, batchsize; replace=false)
    # compute velocity
    minibtach_∇ = [zeros(pc.dim) for i in 1:batchsize]

    for bi in 1:batchsize
        k, ∇k = k_∇k(P[:, S[bi]], P[:, pi])
        ∇logρ = LogDensityProblems.logdensity_and_gradient(ρ, P[:, S[bi]])[2]
        minibtach_∇[bi] = k * ∇logρ + ∇k
    end

    return sum(minibtach_∇)/batchsize
end


"""
    update_particles!(ρ, pc::ParticleContainer, dynamics::SVGD)

Updates the positions of all particles in the `ParticleContainer` according to the Stein Variational Gradient Descent (SVGD) update rule.

# Arguments
- `ρ`: The log-density function (a `LogDensityProblem`) that the particles aim to sample from.
- `pc::ParticleContainer`: The container holding the current positions of the particles. The particle positions are updated in-place.
- `dynamics::SVGD`: The `SVGD` dynamics object specifying the kernel, step size (`η`), and batch size for the update.

"""
function update_particles!(ρ, pc::ParticleContainer, dynamics::SVGD, ad_backend)
    N = pc.size
    # kernel value and gradient
    k_∇k = kernel_and_gradient_fn(dynamics.K, ad_backend)

    for i ∈ 1:N
        pc.P[:, i] += dynamics.η * particle_velocity(pc, ρ, i, k_∇k, dynamics)
    end
    return nothing
end

mutable struct SVGDInferenceReport <: InferenceReport
    success::Bool
end




"""
    infer!(
        pc::ParticleContainer,
        state::SVGDInferenceState;
        iters::Integer,
        ad_backend=ADTypes.AutoForwardDiff(),
        verbose::Bool=false
    )

Perform inference using Stein Variational Gradient Descent (SVGD).

# Arguments
- `pc::ParticleContainer`: The particle container holding the current set of particles. 
- `state::SVGDInferenceState`: The internal state object for the SVGD algorithm

# Keyword Arguments
- `iters::Integer`: The number of SVGD iterations to perform.
- `ad_backend=ADTypes.AutoForwardDiff()`: The automatic differentiation backend to use for computing gradients. Defaults to `ADTypes.AutoForwardDiff()`.
- `verbose::Bool=false`: A boolean flag indicating whether to print progress information during the inference. Defaults to `false`.

# Returns
- `SVGDInferenceReport`: An `SVGDInferenceReport` object 

# Details
This function modifies the `pc` in-place, updating the positions of the particles.
"""
function infer!(
    pc::ParticleContainer,
    state::SVGDInferenceState;
    iters::Integer=10,
    ad_backend=ADTypes.AutoForwardDiff(),
    verbose::Bool=false
)
    for i in 1:iters
        update_particles!(state.ρ, pc, state.dynamics, ad_backend)
    end
    return SVGDInferenceReport(true)
end