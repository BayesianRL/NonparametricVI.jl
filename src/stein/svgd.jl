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

"""
    SVGDInferenceState <: InferenceState

A mutable struct representing the internal state of the Stein Variational Gradient Descent (SVGD) inference algorithm.

# Fields
- `ρ`: A `LogDensityProblem` representing the target distribution's log-density function.
- `dynamics::SVGD`: An `SVGD` object defining the dynamics used to update the particles.
- `model::Union{DynamicPPL.Model, Nothing}`: The probabilistic model associated with the inference, or `nothing` if no model is provided.

# Description
This struct encapsulates the state information required by the SVGD algorithm. It stores the target log-density, the SVGD dynamics, and optionally, a reference to the DynamicPPL model used. This state is passed between iterations of the SVGD algorithm to maintain and update the necessary information.
"""
mutable struct SVGDInferenceState <: InferenceState
    ρ
    dynamics::SVGD
    model::Union{DynamicPPL.Model, Nothing}
end

"""
    init_state(ρ, dynamics::SVGD, model::DynamicPPL.Model)

Initialize the `SVGDInferenceState` for using `SVGD` dynamics on Turing models 

# Arguments
- `ρ`: A `LogDensityProblem` representing the target distribution.
- `dynamics::SVGD`: An `SVGD` object defining the dynamics used to update the particles.
- `model::DynamicPPL.Model`: The probabilistic model associated with the inference.

# Returns
- An `SVGDInferenceState` object initialized with the provided arguments.

"""
function init_state(ρ, dynamics::SVGD, model::DynamicPPL.Model)
    return SVGDInferenceState(ρ, dynamics, model)
end


"""
    init_state(ρ, dynamics::SVGD)

Initialize the `SVGDInferenceState` for using `SVGD` dynamics with LogDensityProblem ρ

# Arguments
- `ρ`: A `LogDensityProblem` representing the target distribution's log-density function.
- `dynamics::SVGD`: An `SVGD` object defining the dynamics used to update the particles.

# Returns
- An `SVGDInferenceState` object initialized with the provided arguments.

"""
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

    Base.Threads.@threads for bi in 1:batchsize
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


"""
    SVGDInferenceReport <: InferenceReport

A mutable struct representing the report generated after running Stein Variational Gradient Descent (SVGD) inference.

# Fields
- `metrics::Dict{String, Vector{Any}}`: A dictionary storing various metrics tracked during the SVGD inference. Keys are metric names (strings), and values are vectors of metric values recorded at each iteration or relevant time step.
- `success::Bool`: A boolean flag indicating whether the inference process completed successfully.

"""
mutable struct SVGDInferenceReport <: InferenceReport
    metrics::Dict{String, Vector{Any}}
    success::Bool
end




"""
    infer!(
        pc::ParticleContainer,
        state::SVGDInferenceState;
        iters::Integer,
        ad_backend=ADTypes.AutoForwardDiff(),
        verbose::Bool=false,
        track::Dict{String, Metric}=Dict()
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
    verbose::Bool=false,
    track=Dict{String, Any}()
)
    metrics = Dict()

    # initial values of metrics     
    for (metric_name, metric_type) in track
        metrics[metric_name] = [compute_metric(metric_type, pc, state.ρ; ad_backend=ad_backend)]
    end

    for i in 1:iters
        update_particles!(state.ρ, pc, state.dynamics, ad_backend)
        for (metric_name, metric_type) in track
            metric_value = compute_metric(metric_type, pc, state.ρ; ad_backend=ad_backend)
            push!(metrics[metric_name], metric_value)
        end
    end

    inference_report = SVGDInferenceReport(metrics, true)
    
    return inference_report
end