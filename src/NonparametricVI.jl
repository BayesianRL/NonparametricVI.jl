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

# particle initialization methods
export ZeroInitializer,
       NormalInitializer,
       PriorInitializer


abstract type ParticleDynamics end

abstract type AbstractProblemContext end
abstract type AbstractInferenceContext end

"""
    mutable struct LogDensityProblemContext <: AbstractProblemContext
        ρ
    end

A mutable struct representing the problem context for LogDensityProblem.jl

# Fields
- `ρ`: should implement LogDensityProblem interface
"""
mutable struct LogDensityProblemContext <: AbstractProblemContext
    ρ
end

"""
    Context{T<:AbstractProblemContext, U<:AbstractInferenceContext}

A mutable struct that holds instances of a problem context and an inference context.

# Fields
- `problem::T`: An instance of a subtype of `AbstractProblemContext`.
- `inference::U`: An instance of a subtype of `AbstractInferenceContext`.
"""
mutable struct Context{T<:AbstractProblemContext, U<:AbstractInferenceContext}
    problem::T
    inference::U
end


abstract type InferenceState end

abstract type Metric end

abstract type InferenceReport end

abstract type ParticleInitializer end



include("common.jl")
include("particle_containers.jl")
include("discrepancies.jl")
include("metrics.jl")
include("particle_initializers.jl")
include("stein/svgd.jl")
include("inference_initializers.jl")
include("sample_utils.jl")



end