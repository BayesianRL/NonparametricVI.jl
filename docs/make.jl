using Documenter, NonparametricVI

makedocs(sitename="NonparametricVI.jl", remotes=nothing)

deploydocs(
    repo = "github.com/BayesianRL/NonparametricVI.jl.git",
)