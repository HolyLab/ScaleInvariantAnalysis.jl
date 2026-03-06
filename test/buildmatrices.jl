using LinearAlgebra
using Graphs: Graphs, connected_components
using NautyGraphs: NautyGraph, canonize!

if !isdefined(Main, :logsymcover_ref)
    include("testutils.jl")
end

function accumulate_examples_symmetric!(comps, A, model, αvariable, ps)
    logA = log.(abs.(A))
    logsymcover_ref_reset!(ps, logA)
    α = try
            JuMP.optimize!(model)
            JuMP.value.(αvariable)
        catch err
            @warn "Optimization failed for matrix $A: $err"
            return
        end
    dA = logA .- α .- α'
    active = abs.(dA) .< 1e-8
    g = NautyGraph(active)
    p = canonize!(g)
    edges = Tuple{Int,Int}[]
    for e in Graphs.edges(g)
        push!(edges, (min(e.src, e.dst), max(e.src, e.dst)))
    end
    if !haskey(comps, edges)
        comps[edges] = A[p, p]
    end
end

valrange = 1:20
comps = Dict{Vector{Tuple{Int,Int}}, Matrix{Int}}()
Adummy = Symmetric([rand(valrange) for _ in 1:5, _ in 1:5])
model, αvariable, ps = logsymcover_ref_setup(log.(abs.(Adummy)))
for i = 1:10^5
    A = Symmetric([rand(valrange) for _ in 1:5, _ in 1:5])
    accumulate_examples_symmetric!(comps, A, model, αvariable, ps)
end

open("testmatrices.jl", "w") do io
    ccomps = sort(collect(comps), by = x -> x[1])
    println(io, "const symmetric_matrices = [")
    for (cc, A) in ccomps
        println(io, "    $(repr(cc)) => $(repr(A)),")
    end
    println(io, "]")
end
