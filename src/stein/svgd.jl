import Base.@kwdef
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
    struct SVGDInferenceContext <: AbstractInferenceContext
        dynamics::SVGD
    end

A struct representing the inference context for SVGD.

# Fields
- `dynamics::SVGD`: An instance of the `SVGD` dynamics
"""
struct SVGDInferenceContext <: AbstractInferenceContext 
    dynamics::SVGD
end



"""
    init_inference_context(ρ, dynamics::SVGD)

Initializes an `SVGDInferenceContext` with the provided SVGD dynamics.
For now ρ is not used.

# Arguments
- `ρ`: The target LogDensityProblem
- `dynamics::SVGD`: The SVGD dynamics to be used for inference.

# Returns
- An instance of `SVGDInferenceContext` initialized with the given `dynamics`.
"""
function init_inference_context(ρ, dynamics::SVGD)
    return SVGDInferenceContext(dynamics)
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
        ctx::Context{<:AbstractProblemContext, SVGDInferenceContext};
        iters::Integer=10,
        ad_backend::ADTypes.AbstractADType=ADTypes.AutoForwardDiff(),
        verbose::Bool=false,
        track=Dict{String, Any}()
    )

Performs Stein Variational Gradient Descent (SVGD) inference to update a particle container.

# Arguments
- `pc::ParticleContainer`: The container holding the particles to be updated.
- `ctx::Context{<:AbstractProblemContext, SVGDInferenceContext}`: The context containing the problem definition and the SVGD inference settings.
- `iters::Integer=10`: The number of SVGD iterations to perform.
- `ad_backend::ADTypes.AbstractADType=ADTypes.AutoForwardDiff()`: The automatic differentiation backend to use for gradient computations.
- `verbose::Bool=false`: A flag to enable verbose output during inference (currently not implemented).
- `track::Dict{String, Any}()`: A dictionary specifying metrics to compute and track during inference. The keys are metric names (strings), and the values are metric types (e.g., functions or structs that can be passed to `compute_metric`).

# Returns
- An `SVGDInferenceReport` containing the tracked metrics and a success flag.

# Notes
- This function retrieves the log-density function from the problem context.
- It initializes and updates the particles using the SVGD dynamics specified in the inference context.
- Optionally, it computes and tracks specified metrics at each iteration.
"""
function infer!(
    pc::ParticleContainer,
    ctx::Context{<:AbstractProblemContext, SVGDInferenceContext};
    iters::Integer=10,
    ad_backend::ADTypes.AbstractADType=ADTypes.AutoForwardDiff(),
    verbose::Bool=false,
    track=Dict{String, Any}()
)
    ρ = get_problem(ctx.problem)

    metrics = Dict()    
    # initial values of metrics     
    for (metric_name, metric_type) in track
        metrics[metric_name] = [compute_metric(metric_type, pc, ρ; ad_backend=ad_backend)]
    end

    for i in 1:iters
        update_particles!(ρ, pc, ctx.inference.dynamics, ad_backend)
        for (metric_name, metric_type) in track
            metric_value = compute_metric(metric_type, pc, ρ; ad_backend=ad_backend)
            push!(metrics[metric_name], metric_value)
        end
    end

    inference_report = SVGDInferenceReport(metrics, true)
    
    return inference_report
end