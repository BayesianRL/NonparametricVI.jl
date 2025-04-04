@testset "SVGD" begin

    struct Density end

    function LogDensityProblems.capabilities(::Type{<:Density})
        LogDensityProblems.LogDensityOrder{0}()
    end

    LogDensityProblems.dimension(::Density) = 2

    function LogDensityProblems.logdensity(::Density, x)
        log(0.5 * exp(-1 / 0.5 * norm(x - [1.0, 1.0])^2) +
            0.5 * exp(-1 / 0.5 * norm(x - [-1.0, -1.0])^2))
    end

    ρ = Density()
    ∇ρ = LogDensityProblemsAD.ADgradient(AutoForwardDiff(), ρ)

    kernel = KernelFunctions.SqExponentialKernel()
    dynamics = NonparametricVI.SVGD(K=kernel, η=0.06, batchsize=nothing)
    pc, state = NonparametricVI.init(ρ, dynamics; n_particles=64)
    
    
    report = NonparametricVI.infer!(pc, state; iters=10, verbose=true, track=Dict(
        "KSD" => KernelizedSteinDiscrepancy(kernel)
    ))

    @test report.metrics["KSD"][1] > 0.0
    samples = NonparametricVI.get_samples(pc)

end