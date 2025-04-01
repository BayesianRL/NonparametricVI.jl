struct ZeroInitializer <: ParticleInitializer end

function init_particle(dim, particles, initializer::ZeroInitializer)
    return ParticleContainer(zeros((dim, particles)))
end

struct NormalInitializer <: ParticleInitializer end

function init_particle(dim, particles, initializer::NormalInitializer)
    return ParticleContainer(randn((dim, particles)))
end

struct PriorInitializer <: ParticleInitializer end
# [todo] `init_particle` needs access to the Turing model