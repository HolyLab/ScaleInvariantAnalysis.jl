using LinearAlgebra
using Graphs: Graphs, connected_components
using NautyGraphs: NautyGraphs, NautyGraph, NautyDiGraph, canonize!

# The following partially duplicate "testutils.jl", but with an emphasis on performance in generating
# many examples of symmetric matrices and their covers
using JuMP: JuMP, @variable, @objective, @constraint, @expression, Parameter
using HiGHS: HiGHS
import ParametricOptInterface as POI

# This version is safe only if A has no zeros
function logsymcover_ref_setup(logA)
    n = size(logA, 1)
    model = JuMP.Model(() -> POI.Optimizer(HiGHS.Optimizer))
    JuMP.set_silent(model)
    @variable(model, α[1:n])
    @variable(model, ps[i=1:n, j=1:n] in Parameter(logA[i, j]))
    @objective(model, Min, sum(abs2, α[i] + α[j] - ps[i, j] for i in 1:n, j in 1:n))
    for i in 1:n
        for j in i:n
            @constraint(model, α[i] + α[j] - ps[i, j] >= 0)
        end
    end
    return model, α, ps
end

function logcover_ref_setup(logA)
    m, n = size(logA)
    model = JuMP.Model(() -> POI.Optimizer(HiGHS.Optimizer))
    JuMP.set_silent(model)
    @variable(model, α[1:m])
    @variable(model, β[1:n])
    @variable(model, ps[i=1:m, j=1:n] in Parameter(logA[i, j]))
    @objective(model, Min, sum(abs2, α[i] + β[j] - ps[i, j] for i in 1:m, j in 1:n))
    for i in 1:m
        for j in 1:n
            @constraint(model, α[i] + β[j] - ps[i, j] >= 0)
        end
    end
    nza, nzb = sum(!iszero, logA; dims=2)[:], sum(!iszero, logA; dims=1)[:]
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    return model, α, β, ps
end

function logA_reset!(ps, logA)
    n = size(logA, 1)
    for i in 1:n
        for j in 1:n
            JuMP.set_parameter_value(ps[i, j], logA[i, j])
        end
    end
    return ps
end


function accumulate_examples!(fdA, NG, comps, A, model, ps, vars...)
    logA = log.(abs.(A))
    logA_reset!(ps, logA)
    dA = fdA(model, A, logA, vars...)
    dA === nothing && return
    active = abs.(dA) .< 1e-8
    g = NG(active)
    p = canonize!(g)
    edges = Tuple{Int,Int}[]
    for e in Graphs.edges(g)
        push!(edges, (min(e.src, e.dst), max(e.src, e.dst)))
    end
    if !haskey(comps, edges)
        comps[edges] = A[p, p]
    end
end

function dA_symmetric(model, A, logA, αvariable)
    α = try
            JuMP.optimize!(model)
            JuMP.value.(αvariable)
        catch err
            @warn "Optimization failed for matrix $A: $err"
            return
        end
    return logA .- α .- α'
end

function dA_general(model, A, logA, αvariable, βvariable)
    α, β = try
            JuMP.optimize!(model)
            JuMP.value.(αvariable), JuMP.value.(βvariable)
        catch err
            @warn "Optimization failed for matrix $A: $err"
            return
        end
    return logA .- α .- β'
end

valrange = 1:20
comps_sym = Dict{Vector{Tuple{Int,Int}}, Matrix{Int}}()
Adummy = Symmetric([rand(valrange) for _ in 1:5, _ in 1:5])
model, αvariable, ps = logsymcover_ref_setup(log.(abs.(Adummy)))
for i = 1:10^5
    A = Symmetric([rand(valrange) for _ in 1:5, _ in 1:5])
    accumulate_examples!(dA_symmetric, NautyGraph, comps_sym, A, model, ps, αvariable)
end

comps = Dict{Vector{Tuple{Int,Int}}, Matrix{Int}}()
Adummy = [rand(valrange) for _ in 1:5, _ in 1:5]
model, αvariable, βvariable, ps = logcover_ref_setup(log.(abs.(Adummy)))
for i = 1:10^5
    A = [rand(valrange) for _ in 1:5, _ in 1:5]
    accumulate_examples!(dA_general, NautyDiGraph, comps, A, model, ps, αvariable, βvariable)
end


open("testmatrices.jl", "w") do io
    ccomps = sort(collect(comps_sym), by = x -> x[1])
    println(io, "const symmetric_matrices = [")
    for (cc, A) in ccomps
        println(io, "    $(repr(cc)) => $(repr(A)),")
    end
    println(io, "]")

    ccomps = sort(collect(comps), by = x -> x[1])
    println(io, "const general_matrices = [")
    for (cc, A) in ccomps
        println(io, "    $(repr(cc)) => $(repr(A)),")
    end
    println(io, "]")
end
