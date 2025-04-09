import LogDensityProblems
import LogDensityProblemsAD
import DifferentiationInterface
import ADTypes


init_problem_context(ρ) = LogDensityProblemContext(ρ)

function init_inference_context(ρ, dynamics::ParticleDynamics)
    error("context initialization is not implemented for the dynamics")
end

"""
    init_context(
        ρ,
        dynamics::ParticleDynamics
    )

Initializes a `Context` struct containing a problem context and an inference context.

# Arguments
- `ρ`: The target log-density function.
- `dynamics::ParticleDynamics`: The particle dynamics to be used for inference.

# Returns
- An instance of `Context` containing:
    - A problem context initialized from `ρ`.
    - An inference context initialized from `ρ` and `dynamics`.
"""
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
        n_particles::Integer=16,
        ad_backend::ADTypes.AbstractADType=ADTypes.AutoForwardDiff()
    )

Initializes a particle container and the corresponding inference context.

# Arguments
- `ρ`: The target log-density function.
- `dynamics::ParticleDynamics`: The particle dynamics to be used for inference.
- `particle_initializer`: An object that initializes the particle positions (default: `NormalInitializer()`).
- `n_particles::Integer=16`: The number of particles to initialize (default: 16).
- `ad_backend::ADTypes.AbstractADType=ADTypes.AutoForwardDiff()`: The automatic differentiation backend to use for gradient computations if needed (default: `ADTypes.AutoForwardDiff()`).

# Returns
- A tuple containing:
    - `pc::ParticleContainer`: The initialized particle container.
    - `ctx::Context`: The initialized inference context.

# Notes
- This function determines the dimensionality of the problem from the log-density function `ρ`.
- It ensures that the log-density function is differentiable by wrapping it with an AD backend.
- It initializes the particle positions using the provided `particle_initializer`.
- It initializes the inference context based on the provided `dynamics`.
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
    ctx = init_context(ρ, dynamics)
    # initial position of particles
    pc = init_particles(n_particles, particle_initializer, ctx)
    return pc, ctx
end