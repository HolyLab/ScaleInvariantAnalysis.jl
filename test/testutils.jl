using JuMP: JuMP, @variable, @objective, @constraint, @expression, Parameter
using HiGHS: HiGHS
using LinearAlgebra
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

function logsymcover_ref(logA, A)
    n = size(logA, 1)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:n])
    @objective(model, Min, sum(abs2, α[i] + α[j] - logA[i, j] for i in 1:n, j in 1:n if !iszero(A[i, j])))
    for i in 1:n
        for j in i:n
            if iszero(A[i, j])
                continue
            end
            @constraint(model, α[i] + α[j] - logA[i, j] >= 0)
        end
    end
    JuMP.optimize!(model)
    return JuMP.value.(α)
end

function logsymcover_ref_reset!(ps, logA)
    n = size(logA, 1)
    for i in 1:n
        for j in 1:n
            JuMP.set_parameter_value(ps[i, j], logA[i, j])
        end
    end
    return ps
end

function logsymcover_ref(logA)
    model, α, ps = logsymcover_ref_setup(logA)
    JuMP.optimize!(model)
    return JuMP.value.(α)
end

function symcover_ref(A)
    issymmetric(A) || error("Matrix A must be symmetric")
    α = logsymcover_ref(log.(abs.(A)), A)
    return exp.(α)
end

function logcover_ref(logA, A)
    m, n = size(logA, 1), size(logA, 2)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    nz1, nz2 = vec(sum(!iszero, A; dims=2)), vec(sum(!iszero, A; dims=1))
    @variable(model, α[1:m])
    @variable(model, β[1:n])
    @objective(model, Min, sum(abs2, α[i] + β[j] - logA[i, j] for i in 1:m, j in 1:n if !iszero(A[i, j])))
    for i in 1:m
        for j in 1:n
            if iszero(A[i, j])
                continue
            end
            @constraint(model, α[i] + β[j] - logA[i, j] >= 0)
        end
    end
    @constraint(model, sum(nz1[i] * α[i] for i in 1:m) == sum(nz2[j] * β[j] for j in 1:n))
    JuMP.optimize!(model)
    return JuMP.value.(α), JuMP.value.(β)
end

function cover_ref(A)
    α, β = logcover_ref(log.(abs.(A)), A)
    return exp.(α), exp.(β)
end
