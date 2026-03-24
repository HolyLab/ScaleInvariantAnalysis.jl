module SIAJuMP

using JuMP: JuMP, @variable, @objective, @constraint
using HiGHS: HiGHS
using ScaleInvariantAnalysis

# Reference implementation for a symmetric matrix cover
function ScaleInvariantAnalysis.symcover_qmin(A)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_qmin requires a square matrix"))
    logA = log.(abs.(A))
    n = size(logA, 1)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:n])
    @objective(model, Min, sum(abs2, α[i] + α[j] - logA[i, j] for i in 1:n, j in 1:n if A[i, j] != 0))
    for i in 1:n
        for j in i:n
            if A[i, j] != 0
                @constraint(model, α[i] + α[j] - logA[i, j] >= 0)
            end
        end
    end
    JuMP.optimize!(model)
    return exp.(JuMP.value.(α))
end

function ScaleInvariantAnalysis.symcover_lmin(A)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_lmin_ref requires a square matrix"))
    logA = log.(abs.(A))
    n = size(logA, 1)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:n])
    @objective(model, Min, sum(α))
    for i in 1:n
        for j in i:n
            if A[i, j] != 0
                @constraint(model, α[i] + α[j] - logA[i, j] >= 0)
            end
        end
    end
    JuMP.optimize!(model)
    return exp.(JuMP.value.(α))
end

# Reference implementation for a general matrix cover
function ScaleInvariantAnalysis.cover_qmin(A)
    logA = log.(abs.(A))
    m, n = size(logA)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:m])
    @variable(model, β[1:n])
    @objective(model, Min, sum(abs2, α[i] + β[j] - logA[i, j] for i in 1:m, j in 1:n if A[i, j] != 0))
    for i in 1:m
        for j in 1:n
            if A[i, j] != 0
                @constraint(model, α[i] + β[j] - logA[i, j] >= 0)
            end
        end
    end
    nza, nzb = sum(!iszero, A; dims=2)[:], sum(!iszero, A; dims=1)[:]
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    JuMP.optimize!(model)
    return exp.(JuMP.value.(α)), exp.(JuMP.value.(β))
end

function ScaleInvariantAnalysis.cover_lmin(A)
    logA = log.(abs.(A))
    m, n = size(logA)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:m])
    @variable(model, β[1:n])
    @objective(model, Min, sum(α) + sum(β))
    for i in 1:m
        for j in 1:n
            if A[i, j] != 0
                @constraint(model, α[i] + β[j] - logA[i, j] >= 0)
            end
        end
    end
    nza, nzb = sum(!iszero, A; dims=2)[:], sum(!iszero, A; dims=1)[:]
    @constraint(model, sum(nza[i] * α[i] for i in 1:m) == sum(nzb[j] * β[j] for j in 1:n))
    JuMP.optimize!(model)
    return exp.(JuMP.value.(α)), exp.(JuMP.value.(β))
end

end
