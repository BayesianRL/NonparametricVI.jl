module NonparametricVI

export SVGD
export ParticleContainer


abstract type ParticleDynamics end

abstract type InferenceState end

abstract type InferenceReport end

abstract type ParticleInitializer end





include("common.jl")
include("particle_containers.jl")
include("turing.jl")
include("particle_initializers.jl")
include("stein/svgd.jl")


"""
    init(
        model::DynamicPPL.Model,
        dynamics::ParticleDynamics;
        particle_initializer=NormalInitializer(),
        n_particles::Integer=16
    )

Initialize the particle container and the internal state for an inference algorithm.

# Arguments
- `model::DynamicPPL.Model`: The probabilistic model defined using DynamicPPL.
- `dynamics::ParticleDynamics`: The inference dynamics.

# Keyword Arguments
- `particle_initializer=NormalInitializer()`: An object that initializes the positions of the particles. Defaults to `NormalInitializer()`, which initializes particles from a standard normal distribution.
- `n_particles::Integer=16`: The number of particles to initialize. Defaults to 16.

# Returns
- `pc`: The initialized particle container, holding the initial positions of all particles. The initialization method depends on the chosen `particle_initializer`.
- `state`: The initialized internal state associated with the provided `dynamics`. The exact type depends on the specific `ParticleDynamics` implementation.
"""
function init(
    model::DynamicPPL.Model,
    dynamics::ParticleDynamics;
    particle_initializer=NormalInitializer(),
    n_particles::Integer=16
)
    # create LogDensityProblem from Truing model
    ρ = logdensityproblem_from_turing(model, ADTypes.AutoForwardDiff())
    dim = LogDensityProblems.dimension(ρ)
    # initial position of particles
    pc = init_particle(dim, n_particles, particle_initializer)
    state = init_state(ρ, dynamics)
    return pc, state
end



function infer!(
    pc::ParticleContainer,
    state::SVGDInferenceState;
    iters::Integer=10,
    verbose::Bool=false
)
    for i in 1:iters
        update_particles!(state.ρ, pc, state.dynamics)
        println(pc.P[1, 1])
    end
    return SVGDInferenceReport(true)
end


end