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
    linked_vis::Vector{DynamicPPL.VarInfo}
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


function generate_prior_samples(model::DynamicPPL.Model, n::Integer)
    linked_vis = Vector{DynamicPPL.VarInfo}(undef, n)
    for i in 1:n 
      linked_vis[i] = DynamicPPL.invlink(DynamicPPL.VarInfo(model), model)
    end
    return linked_vis
end
  

function samples_to_matrix(samples_varinfo)
    M = (t->DynamicPPL.values_as(t, Vector)).(samples_varinfo)
    return stack(M)
end


function constrained_particle(
    θ,
    idx,
    ctx::NVI.Context{
        DynamicPPLProblemContext,
        <:NVI.AbstractInferenceContext
    }
)
    vil = DynamicPPL.unflatten(ctx.problem.linked_vis[idx], θ)
    # constrained = DynamicPPL.values_as_in_model(ctx.problem.model, true, vil)
    constrained = DynamicPPL.invlink(vil, ctx.problem.model)
    return constrained
end



function NVI.get_samples(
    pc::NVI.ParticleContainer,
    ctx::NVI.Context{
        DynamicPPLProblemContext,
        <:NVI.AbstractInferenceContext
    }
)
    constrained_particles = [constrained_particle(pc.P[:, i], i, ctx) for i in 1:pc.size]
    return constrained_particles
end


function init_context(
    ρ,
    model::DynamicPPL.Model,
    dynamics::NVI.ParticleDynamics,
    linked_vis::Vector{DynamicPPL.VarInfo}
)
    inference_ctx = NVI.init_inference_context(ρ, dynamics)
    problem_ctx = DynamicPPLProblemContext(ρ, model, linked_vis)
    return NVI.Context(problem_ctx, inference_ctx)
end


function NVI.init_particles(
    n_particles::Integer,
    particle_initializer::NVI.PriorInitializer,
    ctx::NVI.Context{
        DynamicPPLProblemContext,
        <:NVI.AbstractInferenceContext
    }  
)
    return NVI.ParticleContainer(samples_to_matrix(ctx.problem.linked_vis))
end

"""
    NVI.init(
        model::DynamicPPL.Model,
        dynamics::NVI.ParticleDynamics;
        particle_initializer=NVI.PriorInitializer(),
        n_particles::Integer=16,
        ad_backend::ADTypes.AbstractADType=ADTypes.AutoForwardDiff()
    )

Initializes a particle container and the corresponding inference context for a DynamicPPL model.

# Arguments
- `model::DynamicPPL.Model`: The DynamicPPL model for which to perform inference.
- `dynamics::NVI.ParticleDynamics`: The particle dynamics to be used for inference.
- `particle_initializer`: An object that initializes the particle positions (default: `NVI.PriorInitializer()`).
- `n_particles::Integer=16`: The number of particles to initialize (default: 16).
- `ad_backend::ADTypes.AbstractADType=ADTypes.AutoForwardDiff()`: The automatic differentiation backend to use (default: `ADTypes.AutoForwardDiff()`).

# Returns
- A tuple containing:
    - `pc::NVI.ParticleContainer`: The initialized particle container.
    - `ctx::NVI.Context`: The initialized inference context.

# Notes
- This function converts the DynamicPPL model into a `LogDensityProblem` using `logdensityproblem_from_turing`.
- It generates initial particle positions by drawing samples from the prior distribution defined by the `model` using `generate_prior_samples`.
"""
function NVI.init(
    model::DynamicPPL.Model,
    dynamics::NVI.ParticleDynamics;
    particle_initializer::NVI.ParticleInitializer=NVI.PriorInitializer(),
    n_particles::Integer=16,
    ad_backend::ADTypes.AbstractADType=ADTypes.AutoForwardDiff()
)
    # create LogDensityProblem from Truing model
    ρ = logdensityproblem_from_turing(model, ad_backend)
    # initial position of particles

    linked_vis = generate_prior_samples(model, n_particles)

    ctx = init_context(ρ, model, dynamics, linked_vis)
    pc = NVI.init_particles(n_particles, particle_initializer, ctx)
    
    return pc, ctx
end

end