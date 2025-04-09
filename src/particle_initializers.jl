struct ZeroInitializer <: ParticleInitializer end
struct NormalInitializer <: ParticleInitializer end
struct PriorInitializer <: ParticleInitializer end

function init_particles(
    n_particles::Integer,
    particle_initializer::NormalInitializer,
    ctx::Context{
        <:AbstractProblemContext,
        <:AbstractInferenceContext
    }  
)
    dim = LogDensityProblems.dimension(ctx.problem.ρ)

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
    dim = LogDensityProblems.dimension(ctx.problem.ρ)

    return ParticleContainer(zeros((dim, n_particles)))
end