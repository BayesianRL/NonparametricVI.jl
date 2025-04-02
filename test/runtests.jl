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

        using LinearAlgebra

        struct Density end

        function LogDensityProblems.capabilities(::Type{<:Density})
            LogDensityProblems.LogDensityOrder{0}()
        end
        
        LogDensityProblems.dimension(::Density) = 2
        
        function LogDensityProblems.logdensity(::Density, x)
            log(0.5 * exp(-1/0.5 * norm(x-[ 1.0,  1.0])^2) +
                0.5 * exp(-1/0.5 * norm(x-[-1.0, -1.0])^2))
        end

        ρ = Density()

        kernel = KernelFunctions.SqExponentialKernel()
        dynamics = NonparametricVI.SVGD(K=kernel, η=0.06, batchsize=nothing)
        pc, state = NonparametricVI.init(ρ, dynamics; n_particles=16)
        report = NonparametricVI.infer!(pc, state; iters=10, verbose=true)

    end

end
