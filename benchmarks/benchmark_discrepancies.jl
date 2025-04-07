using Pkg
Pkg.activate("benchmarks_env")

using NonparametricVI
using KernelFunctions
using ADTypes
using ForwardDiff
using ReverseDiff
using LogDensityProblems
using LogDensityProblemsAD
using BenchmarkTools


include("benchmark_problems/mixture_density.jl")


ad_backends = [("ForwardDiff", AutoForwardDiff()), ("ReverseDiff", AutoReverseDiff())]

BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 100.0


suite = BenchmarkGroup()
suite["KSD"] = BenchmarkGroup(["discrepancies"])

n = 1024
samplesize_rng = collect(10:10:100)
d = 2

kernel = SqExponentialKernel()
ρ = MixtureDensity()
P = randn((d, n))
    
println("Benchmarking...")

for (ad_name, ad_backend) in ad_backends   
    ∇ρ = LogDensityProblemsAD.ADgradient(ad_backend, ρ)
    for samplesize in samplesize_rng
        suite["KSD"]["S_$(samplesize)"][ad_name] = @benchmarkable NonparametricVI.kernelized_stein_discrepancy(
            P,
            $∇ρ,
            kernel;
            samplesize=$samplesize,
            ad_backend=$ad_backend)
    end

end

# tune!(suite)

result = run(suite, verbose=true)

v = pkgversion(NonparametricVI)
v_major = Int32(v.major)
v_minor = Int32(v.minor)
v_patch = Int32(v.patch)

BenchmarkTools.save("result/v$(v_major).$(v_minor).$(v_patch)/benchmark_KSD.json", result)