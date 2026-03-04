using LinearAlgebra
using Graphs: Graphs, connected_components
using NautyGraphs: NautyGraph, canonize!

if !isdefined(Main, :logcoversym_ref)
    include("testutils.jl")
end

valrange = 1:20
comps = Dict{Vector{Vector{Int}}, Matrix{Int}}()
for i = 1:10^5
    A = Symmetric([rand(valrange) for _ in 1:5, _ in 1:5])
    logA = log.(abs.(A))
    α = try
            logcoversym_ref(A, logA)
        catch err
            @warn "Optimization failed for matrix $A: $err"
            continue
        end
    dA = logA .- α .- α'
    active = abs.(dA) .< 1e-8
    g = NautyGraph(active)
    canonize!(g)
    cc = connected_components(g)
    if !haskey(comps, cc)
        comps[cc] = A
    end
    length(comps) == 24 && break  # there are 24 distinct patterns for 5x5 symmetric matrices
end

open("testmatrices.jl", "w") do io
    ccomps = sort(collect(comps), by = x -> length.(x[1]))
    println(io, "const symmetric_matrices = [")
    for (cc, A) in ccomps
        println(io, "    $(repr(cc)) => $(repr(A)),")
    end
    println(io, "]")
end
