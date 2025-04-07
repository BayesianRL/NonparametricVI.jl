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
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 10000.0


suite = BenchmarkGroup()
suite["SVGD"] = BenchmarkGroup(["inference"])

batchsize_rng = collect(10:10:100)

ρ = MixtureDensity()




    
println("Benchmarking...")

for (ad_name, ad_backend) in ad_backends   
    for batchsize in batchsize_rng
        kernel = SqExponentialKernel()
        dynamics = SVGD(K=kernel, η=0.02, batchsize=batchsize)
        pc, state = init(ρ, dynamics; n_particles=512)
        suite["SVGD"]["B_$(batchsize)"][ad_name] = @benchmarkable NonparametricVI.infer!($pc, $state; iters=1, ad_backend=$ad_backend)
    end

end

# tune!(suite)

result = run(suite, verbose=true)

v = pkgversion(NonparametricVI)
v_major = Int32(v.major)
v_minor = Int32(v.minor)
v_patch = Int32(v.patch)

BenchmarkTools.save("result/v$(v_major).$(v_minor).$(v_patch)/benchmark_SVGD_2D_mixture_4_Normal.json", result)