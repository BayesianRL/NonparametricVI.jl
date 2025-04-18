struct ZeroInitializer <: ParticleInitializer end
struct NormalInitializer <: ParticleInitializer end
struct PriorInitializer <: ParticleInitializer end


"""
    LangevinInitializer(ϵ, steps) <: ParticleInitializer

Initializes particles using a short Langevin dynamics simulation.

This initializer applies a specified number of Langevin dynamics steps to particles.

# Fields
- `ϵ::Float64`: The step size  used in the Langevin dynamics simulation. A smaller `ϵ` generally leads to more stable but slower exploration of the configuration space.
- `steps::Int`: The number of Langevin dynamics steps to perform for each particle. More steps allow for greater exploration and potentially better sampling of the target distribution.
"""
struct LangevinInitializer <: ParticleInitializer
    ϵ
    steps
end

LangevinInitializer() = LangevinInitializer(0.002, 100)

function init_particles(
    n_particles::Integer,
    particle_initializer::NormalInitializer,
    ctx::Context{
        <:AbstractProblemContext,
        <:AbstractInferenceContext
    }  
)
    ρ = get_problem(ctx.problem)
    dim = LogDensityProblems.dimension(ρ)

    return ParticleContainer(randn((dim, n_particles)))
end

function init_particles(
    n_particles::Integer,
    particle_initializer::ZeroInitializer,
    ctx::Context{
        <:AbstractProblemContext,
        <:AbstractInferenceContext
    }  
)
    ρ = get_problem(ctx.problem)
    dim = LogDensityProblems.dimension(ρ)

    return ParticleContainer(zeros((dim, n_particles)))
end


function init_particles(
    n_particles::Integer,
    particle_initializer::LangevinInitializer,
    ctx::Context{
        <:AbstractProblemContext,
        <:AbstractInferenceContext
    }  
)
    ρ = get_problem(ctx.problem)
    dim = LogDensityProblems.dimension(ρ)

    P = randn((dim, n_particles))
    ϵ = particle_initializer.ϵ
    rϵ = sqrt(ϵ)

    for step in 1:particle_initializer.steps
        for i in 1:n_particles
            ∇logρ = LogDensityProblems.logdensity_and_gradient(ρ, P[:, i])[2]
            P[:, i] = P[:, i] .+ ϵ*∇logρ .+ rϵ * randn(dim)
        end
    end

    return ParticleContainer(P)

end