abstract type AbstractParticleContainer{T} end

"""
    mutable struct ParticleContainer{T} <: AbstractParticleContainer{T}
        P::Matrix{T}
        dim
        size
    end

A mutable struct that holds a collection of particles.

# Fields
- `P::Matrix{T}`: A matrix where each column represents a particle.
- `dim`: The dimensionality of each particle (number of rows in `P`).
- `size`: The number of particles (number of columns in `P`).
"""
mutable struct ParticleContainer{T} <: AbstractParticleContainer{T}
    P::Matrix{T}
    dim
    size
end


ParticleContainer(P) = ParticleContainer(P, size(P)[1], size(P)[2])