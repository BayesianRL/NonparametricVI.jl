using NonparametricVI
using Test


using Distributions
using DynamicPPL

using KernelFunctions

using LogDensityProblems


@testset verbose=true "NonparametricVI.jl" begin

    

    @testset "Stein" begin
        container = ParticleContainer(ones(2,10))
        @test container.dim == 2
        @test container.size == 10
    end

    # @testset "Turing.jl integration" begin
    #     @model function constrained_program(x)
    #         σ ~ Exponential(1.0)
    #         λ ~ Exponential(0.4)
    #         x ~ Normal(0.0, σ)
    #     end
        
    #     trace = constrained_program(0.0)

    #     ρ = LogDensityFunction(trace)

    #     DynamicPPL.link!!(ρ.varinfo, trace)

    #     unconstrained_params = [0.0, -1.0]
        
        

    #     println("constrained : $(constrained_params)")


    # end
end
