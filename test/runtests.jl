using NonparametricVI
using Test


using Distributions
using DynamicPPL
using LogDensityProblems, LogDensityProblemsAD, ForwardDiff

using KernelFunctions

using DifferentiationInterface

@testset verbose = true "NonparametricVI.jl" begin


    @testset "Kernel gradients" begin
        k_∇k = NonparametricVI.kernel_and_gradient_fn(KernelFunctions.SqExponentialKernel(), AutoForwardDiff())
        k, ∇k = k_∇k([1.0], [0.5])
    end

    include("turing_compatibility.jl")


    @testset "SVGD" begin

        @model function constrained_program(x)
            σ ~ Exponential(1.0)
            λ ~ Exponential(0.4)
            x ~ Normal(0.0, σ)
        end

        trace = constrained_program(0.0)

        ρ = LogDensityFunction(trace)
        DynamicPPL.link!!(ρ.varinfo, trace)
        ∇ρ = ADgradient(AutoForwardDiff(), ρ)

        n = 10
        d = 2
        pc = ParticleContainer(trace, n)
        pc.P = randn((d, n))
        pd = NonparametricVI.SVGD(K=KernelFunctions.SqExponentialKernel(), η=0.02, batchsize=n)
        NonparametricVI.update_particles!(∇ρ, pc, pd)
        

    end

end
