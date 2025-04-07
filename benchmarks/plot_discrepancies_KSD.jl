using Pkg
Pkg.activate("benchmarks_env")

using BenchmarkTools
using CairoMakie
using LaTeXStrings

versions = [
    "v0.1.0" # latest version should be first 
]

plotname = "KSD.png"
plot_data = []
scale = 10^6
ad_backends = ["ForwardDiff", "ReverseDiff"]
samplesize_rng = collect(10:10:100)

for version in versions
    report = BenchmarkTools.load("result/$(version)/benchmark_KSD.json")
    for ad_backend in ad_backends
        data = [median(report[1]["KSD"]["S_$(samplesize)"][ad_backend].times) for samplesize in samplesize_rng]
        push!(plot_data, Dict("data"=>data/scale, "ad_backend"=>ad_backend, "version"=>version))
    end
end


fig = Figure(size=(1000,500))

ax = Axis(fig[1,1],
          title="Kernelized Stein Discrepancy",
          xticks=samplesize_rng,
          xlabel="sample size",
          ylabel="median runtime (ms)")

for data in plot_data
    ad_backend = data["ad_backend"]
    version = data["version"]
    scatterlines!(samplesize_rng, data["data"], label="$(version) | AD Backend : $(ad_backend)")
end

axislegend(backgroundcolor=(:white, 0.7))

save("result/$(versions[1])/$(plotname)", fig)
    