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
    return SVGDInferenceReport(true)
end


end