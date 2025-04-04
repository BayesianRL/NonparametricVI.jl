using NonparametricVI
using Test

using LinearAlgebra
using Distributions
using DynamicPPL
using LogDensityProblems, LogDensityProblemsAD, ForwardDiff

using KernelFunctions

using DifferentiationInterface

@testset verbose = true "NonparametricVI.jl" begin

    include("test_common.jl")
    include("test_discrepancy.jl")
    include("test_turing_compatibility.jl")
    include("test_svgd.jl")

end
