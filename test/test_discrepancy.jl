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