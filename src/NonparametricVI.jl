module NonparametricVI

# main inference functionalities
export init
export infer!

# metrics
export KernelizedSteinDiscrepancy

# sample accessors
export get_samples

# inference dynamics
export SVGD

# particle containers
export ParticleContainer


abstract type ParticleDynamics end

abstract type InferenceState end

abstract type Metric end

abstract type InferenceReport end

abstract type ParticleInitializer end



include("common.jl")
include("particle_containers.jl")
include("turing.jl")
include("discrepancies.jl")
include("metrics.jl")
include("particle_initializers.jl")
include("stein/svgd.jl")
include("inference_initializers.jl")
include("sample_utils.jl")

end