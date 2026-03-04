using JuMP: JuMP, @variable, @objective, @constraint, @expression, Parameter
using HiGHS: HiGHS
using LinearAlgebra
import ParametricOptInterface as POI

function logcoversym_ref_setup(logA)
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

function logcoversym_ref_reset!(ps, logA)
    n = size(logA, 1)
    for i in 1:n
        for j in 1:n
            JuMP.set_parameter_value(ps[i, j], logA[i, j])
        end
    end
    return ps
end

function logcoversym_ref(logA)
    model, α, ps = logcoversym_ref_setup(logA)
    JuMP.optimize!(model)
    return JuMP.value.(α)
end

function coversym_ref(A)
    issymmetric(A) || error("Matrix A must be symmetric")
    α = logcoversym_ref(log.(abs.(A)))
    return exp.(α)
end
