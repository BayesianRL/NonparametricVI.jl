abstract type AbstractParticleContainer{T} end

mutable struct ParticleContainer{T} <: AbstractParticleContainer{T}
    P::Matrix{T}
    dim
    size
end


ParticleContainer(P) = ParticleContainer(P, size(P)[1], size(P)[2])