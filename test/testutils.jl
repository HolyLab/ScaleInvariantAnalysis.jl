using JuMP: JuMP, @variable, @objective, @constraint
using HiGHS: HiGHS

function coversym_ref(A)
    issymmetric(A) || error("Matrix A must be symmetric")
    α = logcoversym_ref(A, log.(abs.(A)))
    return exp.(α)
end
function logcoversym_ref(A, logA)
    n = size(A, 1)
    model = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(model)
    @variable(model, α[1:n])
    @objective(model, Min, sum((logA[i, j] - α[i] - α[j])^2 for i in 1:n for j in 1:n if A[i, j] != 0))
    for i in 1:n
        for j in i:n
            if A[i, j] != 0
                @constraint(model, logA[i, j] - α[i] - α[j] <= 0)
            end
        end
    end
    JuMP.optimize!(model)
    @show JuMP.termination_status(model)
    return JuMP.value.(α)
end
