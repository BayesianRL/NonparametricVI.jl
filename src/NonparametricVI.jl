module NonparametricVI

export SVGD
export ParticleContainer


abstract type ParticleDynamics end

include("common.jl")
include("particle_containers.jl")
include("turing.jl")
include("stein/svgd.jl")





end
