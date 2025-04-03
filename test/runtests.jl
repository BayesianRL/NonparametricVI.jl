using NonparametricVI
using Test

using LinearAlgebra
using Distributions
using DynamicPPL
using LogDensityProblems, LogDensityProblemsAD, ForwardDiff

using KernelFunctions

using DifferentiationInterface

@testset verbose = true "NonparametricVI.jl" begin

    @testset "Stein discrepancy" begin
        

        struct TargetDensity end

        function LogDensityProblems.capabilities(::Type{<:TargetDensity})
            LogDensityProblems.LogDensityOrder{0}()
        end
        
        LogDensityProblems.dimension(::TargetDensity) = 2
        
        function LogDensityProblems.logdensity(::TargetDensity, x)
            logpdf(MvNormal([0.5, 0.5], 1.0*I), x)
        end

        ρ = TargetDensity()
        ∇ρ = LogDensityProblemsAD.ADgradient(AutoForwardDiff(), ρ)
         
        P1 = rand(MvNormal([0.0, 0.0], 1.0*I), 100)
        P2 = rand(MvNormal([0.5, 0.5], 1.0*I), 100)
        P3 = rand(MvNormal([-0.5, 0.5], 1.0*I), 100)
        
        kernel = KernelFunctions.SqExponentialKernel()
        D_stein_P1 = NonparametricVI.kernelized_stein_discrepancy(P1, ∇ρ, kernel; ad_backend=AutoForwardDiff())
        D_stein_P2 = NonparametricVI.kernelized_stein_discrepancy(P2, ∇ρ, kernel; ad_backend=AutoForwardDiff())
        D_stein_P3 = NonparametricVI.kernelized_stein_discrepancy(P3, ∇ρ, kernel; ad_backend=AutoForwardDiff())

        @test D_stein_P2 < D_stein_P1
        @test D_stein_P2 < D_stein_P3

    end


    @testset "Kernel gradients" begin
        k_∇k = NonparametricVI.kernel_and_gradient_fn(KernelFunctions.SqExponentialKernel(), AutoForwardDiff())
        k, ∇k = k_∇k([1.0], [0.5])
    end

    include("turing_compatibility.jl")


    @testset "SVGD" begin

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

        samples = NonparametricVI.get_samples(pc)

    end

end
