module NonparametricVITuringExt

import NonparametricVI as NVI
import DynamicPPL
import LogDensityProblems
import LogDensityProblemsAD
import DifferentiationInterface
import ADTypes

mutable struct DynamicPPLProblemContext <: NVI.AbstractProblemContext
    ρ
    model::DynamicPPL.Model
end

function NVI.get_problem(ctx::DynamicPPLProblemContext)
    return ctx.ρ
end

"""
    logdensityproblem_from_turing(model::DynamicPPL.Model, ad_backend::ADTypes.AbstractADType)

Constructs a differentiable `LogDensityProblem` from a Turing `DynamicPPL.Model`

This function takes a Turing model and an automatic differentiation backend,
 performs necessary transformations to the model's parameter space, and returns a `LogDensityProblem` that can be used with optimization or sampling algorithms that require gradient information.

# Arguments
- `model::DynamicPPL.Model`: The Turing probabilistic model.
- `ad_backend::ADTypes.AbstractADType`: The automatic differentiation backend to use (e.g., `ADTypes.AutoForwardDiff()`, `ADTypes.AutoZygote()`).

# Returns
- `∇ρ::LogDensityProblemsAD.ADgradient`: A `LogDensityProblem` object that wraps the log-density function of the Turing model and provides gradient computation capabilities using the specified `ad_backend`. The parameters of this problem are in the unconstrained space.

"""
function logdensityproblem_from_turing(
    model::DynamicPPL.Model,
    ad_backend::ADTypes.AbstractADType
)
    ρ = DynamicPPL.LogDensityFunction(model)
    # transforming to unconstrained space
    DynamicPPL.link!!(ρ.varinfo, model)
    # enabling automated differentiation
    ∇ρ = LogDensityProblemsAD.ADgradient(ad_backend, ρ)

    return ∇ρ
end

"""
    constrained_particles(pc::ParticleContainer, model::DynamicPPL.Model)

Tranforming the position of samples in `pc` back to the constrained space

# Arguments
- `pc::ParticleContainer`: The container holding the unconstrained positions of the particles.
- `model::DynamicPPL.Model`: The target Turing program

# Returns
- `samples::Vector{<:DynamicPPL.VarInfo}`: A vector of `VarInfo` objects, where each `VarInfo` contains a sample from the constrained parameter space of the `model`, corresponding to a particle in the `ParticleContainer`.

"""
function constrained_particles(
    pc::NVI.ParticleContainer,
    model::DynamicPPL.Model
)
    varinfo = DynamicPPL.VarInfo(model)
    DynamicPPL.link!!(varinfo, model)
    samples = Vector{typeof(varinfo)}(undef, pc.size)
    # transforming each sample back to the constrained space
    for si in 1:pc.size
        unconstrained_varinfo = DynamicPPL.unflatten(varinfo, pc.P[:, si])
        samples[si] = DynamicPPL.invlink(unconstrained_varinfo, model)
    end
    return samples
end




function NVI.get_samples(
    pc::NVI.ParticleContainer,
    ctx::NVI.Context{
        DynamicPPLProblemContext,
        <:NVI.AbstractInferenceContext
    }
)
    return constrained_particles(pc, ctx.problem.model)
end




function NVI.init_problem_context(ρ, model)
    return DynamicPPLProblemContext(ρ, model)
end

function NVI.init_context(
    ρ,
    model::DynamicPPL.Model,
    dynamics::NVI.ParticleDynamics
)
    problem_ctx = NVI.init_problem_context(ρ, model)
    inference_ctx = NVI.init_inference_context(ρ, dynamics)
    return NVI.Context(problem_ctx, inference_ctx)
end




function NVI.init(
    model::DynamicPPL.Model,
    dynamics::NVI.ParticleDynamics;
    particle_initializer=NVI.NormalInitializer(),
    n_particles::Integer=16,
    ad_backend::ADTypes.AbstractADType=ADTypes.AutoForwardDiff()
)
    # create LogDensityProblem from Truing model
    ρ = logdensityproblem_from_turing(model, ad_backend)
    dim = LogDensityProblems.dimension(ρ)
    # initial position of particles
    pc = NVI.init_particle(dim, n_particles, particle_initializer)
    ctx = NVI.init_context(ρ, model, dynamics)
    return pc, ctx
end

end