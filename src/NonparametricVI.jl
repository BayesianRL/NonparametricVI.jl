module NonparametricVI

export init
export infer!

export get_samples

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
include("inference_initializers.jl")
include("sample_utils.jl")

end