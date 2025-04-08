import LogDensityProblems
import LogDensityProblemsAD
import DifferentiationInterface
import ADTypes


init_problem_context(ρ) = LogDensityProblemContext(ρ)

function init_inference_context(ρ, dynamics::ParticleDynamics)
    error("context initialization is not implemented for the dynamics")
end

function init_context(
    ρ,
    dynamics::ParticleDynamics
) 
    problem_ctx = init_problem_context(ρ)
    inference_ctx = init_inference_context(ρ, dynamics)
    return Context(problem_ctx, inference_ctx)
end

"""
    init(
        ρ,
        dynamics::ParticleDynamics;
        particle_initializer=NormalInitializer(),
        n_particles::Integer,
        ad_backend=ADTypes.AutoForwardDiff()
    )

Initialize the particle container and the internal state for a particle-based inference algorithm, given a log-density problem.

# Arguments
- `ρ`: A `LogDensityProblem` representing the target distribution's log-density function.
- `dynamics::ParticleDynamics`: The particle dynamics object that governs how particles evolve.

# Keyword Arguments
- `particle_initializer`: An object that initializes the positions of the particles. Defaults to `NormalInitializer()`.
- `n_particles::Integer`: The number of particles to initialize. 
- `ad_backend=ADTypes.AutoForwardDiff()`: The automatic differentiation backend to use if the provided `ρ` is not differentiable. Defaults to `ADTypes.AutoForwardDiff()`.

# Returns
- `pc`: The initialized particle container, holding the initial positions of all particles. 
- `state`: The initialized internal state associated with the provided `dynamics` and `ρ`. 

"""
function init(
    ρ,
    dynamics::ParticleDynamics;
    particle_initializer=NormalInitializer(),
    n_particles::Integer=16,
    ad_backend::ADTypes.AbstractADType=ADTypes.AutoForwardDiff()
)
    dim = LogDensityProblems.dimension(ρ)
    # ensuring logdensity is differentiable
    if LogDensityProblems.capabilities(ρ) == LogDensityProblems.LogDensityOrder{0}()
        ρ = LogDensityProblemsAD.ADgradient(ad_backend, ρ)
    end
    # initial position of particles
    pc = init_particle(dim, n_particles, particle_initializer)
    ctx = init_context(ρ, dynamics)
    return pc, ctx
end