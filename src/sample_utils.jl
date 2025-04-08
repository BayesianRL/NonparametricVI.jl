function get_samples(
    pc::ParticleContainer,
    ctx::Context{
        <:AbstractProblemContext,
        <:AbstractInferenceContext
    }
)
    return pc.P
end

get_samples(pc::ParticleContainer) = pc.P