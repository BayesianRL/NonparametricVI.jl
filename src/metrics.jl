"""
    KernelizedSteinDiscrepancy <: Metric

A struct representing the Kernelized Stein Discrepancy (KSD) metric.

# Fields
- `K::KernelFunctions.Kernel`: The kernel function used in the KSD computation.
- `samplesize::Integer`: the size of sample used to evaluate KSD

# Description
This struct encapsulates the kernel function needed to compute the Kernelized Stein Discrepancy. The KSD is a measure of discrepancy between two probability distributions, and it relies on a kernel function to define the feature space in which the discrepancy is measured.

By holding the kernel function, this struct provides a convenient way to pass the necessary information for KSD computation to functions that track or evaluate the discrepancy between a particle distribution and a target distribution.
"""
struct KernelizedSteinDiscrepancy <: Metric
    K::KernelFunctions.Kernel
    samplesize::Integer
end

KernelizedSteinDiscrepancy(K::KernelFunctions.Kernel) = KernelizedSteinDiscrepancy(K, 64)

"""
    compute_metric(
        metric::KernelizedSteinDiscrepancy,
        pc::ParticleContainer,
        ρ;
        ad_backend::ADTypes.AbstractADType
    )

Compute the Kernelized Stein Discrepancy (KSD) between the particles in the `ParticleContainer` and the target log-density.

# Arguments
- `metric::KernelizedSteinDiscrepancy`: A `KernelizedSteinDiscrepancy` object that defines the kernel and other parameters used for KSD computation.
- `pc::ParticleContainer`: The particle container holding the current set of particles.
- `ρ`: A `LogDensityProblem` representing the target distribution's log-density function.

# Keyword Arguments
- `ad_backend`: The automatic differentiation backend to use for computing gradients required by the KSD.

# Returns
- The computed Kernelized Stein Discrepancy (KSD) value, a scalar representing the discrepancy between the particle distribution and the target distribution.

# Details
This function calculates the KSD, a measure of the discrepancy between the distribution represented by the particles in `pc` and the target distribution defined by `ρ`. It utilizes the kernel specified in the `metric` object and the provided automatic differentiation backend `ad_backend` to compute the necessary gradients.

The function extracts the particle positions from `pc.P` and calls the `kernelized_stein_discrepancy` function to perform the KSD computation.

This function serves as a metric tracking tool during particle-based inference, allowing monitoring of the convergence of the particle distribution to the target.
"""
function compute_metric(
    metric::KernelizedSteinDiscrepancy,
    pc::ParticleContainer,
    ρ;
    ad_backend::ADTypes.AbstractADType)
    
    return kernelized_stein_discrepancy(pc.P, ρ, metric.K; samplesize=metric.samplesize, ad_backend=ad_backend)
end