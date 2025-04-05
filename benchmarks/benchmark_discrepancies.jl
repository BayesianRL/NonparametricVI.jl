using Pkg
Pkg.activate("benchmarks_env")

using NonparametricVI
using KernelFunctions
using ADTypes
using ForwardDiff
using LogDensityProblems
using LogDensityProblemsAD
using BenchmarkTools


include("benchmark_problems/mixture_density.jl")


ad_backend = AutoForwardDiff()

kernel = SqExponentialKernel()
ρ = MixtureDensity()
∇ρ = LogDensityProblemsAD.ADgradient(ad_backend, ρ)

n = 1024
samplesize_rng = [16, 32, 64, 128]
d = 2

P = randn((2, d))


suite = BenchmarkGroup()

suite["KSD"] = BenchmarkGroup(["discrepancies"])

for samplesize in samplesize_rng
    suite["KSD"]["S_$(samplesize)"]["ForwardDiff"] = @benchmarkable NonparametricVI.kernelized_stein_discrepancy(
        P,
        ∇ρ,
        kernel;
        samplesize=$samplesize,
        ad_backend=ad_backend)
end

tune!(suite)

result = run(suite, verbose=true)

v = pkgversion(NonparametricVI)
v_major = Int32(v.major)
v_minor = Int32(v.minor)
v_patch = Int32(v.patch)

BenchmarkTools.save("result/benchmark_discrepancies_v.$(v_major).$(v_minor).json", result)