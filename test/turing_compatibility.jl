@testset "Turng compatibility" begin
    @testset "log density of probabilistic programs" begin
        @model function program(x)
            σ ~ Exponential(1.0)
            λ ~ Exponential(0.4)
            x ~ Normal(0.0, σ + λ)
        end

        trace = program(0.0)

        ∇ρ = NonparametricVI.logdensityproblem_from_turing(trace, AutoForwardDiff())
        
        logρ, ∇logρ = LogDensityProblems.logdensity_and_gradient(∇ρ, zeros(2))
        @test logρ < 0.0
        @test all(isfinite, ∇logρ)

    end

    
    @testset "Handling constrained parameters" begin
        @model function constrained_program(x)
            σ ~ Exponential(1.0)
            λ ~ Exponential(0.4)
            x ~ Normal(0.0, σ + λ)
        end

        model = constrained_program(0.0)

        n_particles = 10
        pc = ParticleContainer(model, n_particles)

        @test pc.dim == 2
        @test pc.size == n_particles
        
        pc.P[:, 1] = [-1.0, -2.0]

        constrained_parameters = NonparametricVI.constrained_particles(pc, model)

    end

end