function get_samples(pc::ParticleContainer, state::InferenceState)
    return get_samples(pc)
end

"""
    get_samples(pc::ParticleContainer, state::SVGDInferenceState)

Extract constrained samples from the particle container for Turing models

# Arguments
- `pc::ParticleContainer`: The particle container holding the current set of particles.
- `state::SVGDInferenceState`: The internal state for the SVGD algorithm
# Returns
- `Vector{DynamicPPL.VarInfo}` a vector of varinfo containing parameters in the constrained space
"""
function get_samples(pc::ParticleContainer, state::SVGDInferenceState)
    return constrained_particles(pc, state.model)
end


function get_samples(pc::ParticleContainer)
    return pc.P
end