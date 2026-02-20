module ScaleInvariantAnalysis

using LinearAlgebra
using SparseArrays
using PrimalDualLinearAlgebra
using LinearOperators
using Krylov
using PDMats

export condscale, divmag, dotabs, cover, symcover

include("linalg.jl")
include("utils.jl")

"""
    dotabs(x, y)

Compute the sum of absolute values of elementwise products of `x` and `y`:

    ∑_i |x[i] * y[i]|
"""
function dotabs(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    s = zero(eltype(x)) * zero(eltype(y))
    for i in eachindex(x, y)
        s += abs(x[i] * y[i])
    end
    return s
end


"""
    a = symcover(A; exact=false, regularize=false)

Given a matrix `A` assumed to be symmetric, return a vector `a` serving as a
 symmetric matrix cover, so that `|A[i,j]| <= a[i] * a[j]` for all `i, j`. `a[i]`
is nonnegative, and is zero only if `A[i, j] = 0` for all `j`.

With `exact=true`, `a` minimizes the objective function

    ∑_{i,j : A[i,j] ≠ 0} (log(|A[i,j]| / (a[i] * a[j])))²

and is therefore covariant under changes of scale but not general linear
transformations.

With `exact=false`, the pattern of nonzeros in `A` is approximated as `u * u'`,
where `sum(u) * u[i] = nz[i]` is the number of nonzero in row `i`. This results
in an `O(n^2)` rather than `O(n^3)` algorithm. `regularize=true` adds a small
offset to the diagonal (relevant only when `exact=true`), which handles all-zero
rows of `A`.
"""
function symcover(A::AbstractMatrix; exact::Bool=false, regularize::Bool=false)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover requires a square matrix"))
    sumlogA, nz = _symcover(A, ax)
    n = length(ax)
    if !exact || all(==(n), nz)
        # Sherman-Morrison formula for efficiency
        offset = sum(sumlogA) / (2 * sum(nz))
        divsafe!(sumlogA, nz)
        return exp.(sumlogA ./ nz .- offset)
    end
    τ = regularize ? sqrt(eps(eltype(sumlogA))) : zero(eltype(sumlogA))
    W = isnz(A)
    divsafe!(sumlogA, vec(sum(W; dims=2)); sentinel=-1/τ)
    return exp.(cholesky(Diagonal(nz) + isnz(A) + τ * I) \ sumlogA)
end

function symcover_barrier(A::AbstractMatrix; exact::Bool=false, τ=1.0, τminfrac = 1//8, itermax=20, btmax=8)
    @assert issymmetric(A)  # will generalize later
    ax = axes(A, 1)
    n = length(ax)
    W = isnz(A)
    z = log(oneunit(eltype(A)))
    T = typeof(z)
    logA = [iszero(aij) ? z : log(abs(aij)) for aij in A]
    nz = vec(sum(W; dims=2))
    B = if !exact || all(==(n), nz)
        u = nz / sqrt(sum(nz))
        ShermanMorrisonMatrix{T}(Diagonal(nz), u, u)
    else
        Diagonal(nz) + W
    end

    sumlogA = vec(sum(logA; dims=2))
    α = B \ sumlogA
    display(exp.(α) .* exp.(α)')
    r = residual(logA, α, W)
    N = length(r)
    s = -r
    means = sum(s) / N
    vars = sum(s.^2) / N - means^2
    δ = sqrt(vars) / N   # a small positive value
    s .= max.(s, δ)
    λ = τ ./ s
    xs = XS(α, s)
    λν = LN(λ, T[])
    H = HessXS(B, xs, λν)
    # Hinv = HessXS(ShermanMorrisonInverse(B), xs.s ./ λν.λ)  #
    J = JacLNXS(jopsym(A, N), zeros(T, 0, length(α)))
    gxs = TopBottomVector(J.Ji'*r, -τ ./ s)
    cviol = TopBottomVector(r + s, T[])
    Δxs0 = zero(TopBottomVector(xs))
    ws = TrimrWorkspace(KrylovConstructor(gxs, cviol))
    wslsqr = LsqrWorkspace(KrylovConstructor(gxs, cviol))
    wslnlq = LnlqWorkspace(KrylovConstructor(cviol, gxs))
    totalviol = sum(abs2, cviol) + sum(abs2, gxs + J' * λν)
    iter = 0
    while iter < itermax
        println("Iteration $iter:")
        # @show H.λsratio
        trimr!(ws, J', -gxs, -cviol#=, Δxs0, λν=#; ν=0.0, τ=1.0, M=H, ldiv=true) #itmax=2*(length(gxs) + length(cviol)))
        @show ws.stats.solved
        lsqr!(wslsqr, J', -gxs; M=H, ldiv=true)
        lnlq!(wslnlq, J, cviol; N=H, ldiv=true) #, λ=sqrt(eps(eltype(g))))
        @show wslsqr.stats.solved wslnlq.stats.solved
        λνpar = wslsqr.x
        Δxspar = - (H \ (gxs + J' * λνpar))
        Δxsperp, λνperp = -wslnlq.x, wslnlq.y
        @show ws.x Δxspar + Δxsperp
        @show ws.y λνpar + λνperp
        error("stop")
        # S = [AbstractMatrix(H) AbstractMatrix(J)'; AbstractMatrix(J) zeros(T, size(J, 1), size(J, 1))]
        # soldirect = S \ vcat(-gxs, -cviol)
        # @show minres(S, vcat(-gxs, -cviol); verbose=1)
        # jactλ!(ξ, W, λ)
        # @show ξ sumlogA - ξ - B*α
        # divsm!(Δα, sumlogA - ξ - B*α, nz)
        # @show Δα
        # solveΔs!(Δs, W, Δα)
        # @show Δs
        Δαs, λνnew = ws.x, ws.y
        # @show Δαs soldirect[1:length(Δαs)] λνnew soldirect[length(Δαs)+1:end]
        # copyto!(Δαs, soldirect[1:length(Δαs)])
        # copyto!(λνnew, soldirect[length(Δαs)+1:end])
        # @show H * Δαs + J' * λνnew + gxs
        # @show J * Δαs + cviol
        # error("stop")
        Δα, Δs = top(Δαs), bottom(Δαs)
        λnew = top(λνnew)
        Δλ = λnew - λ
        γ, γλ = maxstep(Δs, s), maxstep(Δλ, λ)
        @show sum(abs2, gxs + J' * λν) sum(abs2, cviol)
        # @show Δα Δs λnew γ γλ
        iterbt = 0
        local αnew, snew, λnew
        while iterbt < btmax
            αnew = α + γ * Δα
            snew = s + γ * Δs
            λnew = λ + γλ * Δλ
            residual!(r, logA, αnew, W)
            mul!(top(gxs), J.Ji', r)
            bottom(gxs) .= -τ ./ snew
            top(cviol) .= r + snew
            totalviolnew = sum(abs2, cviol) + sum(abs2, gxs + J' * λνnew)
            totalviolnew < totalviol && break
            γ /= 2
            γλ /= 2
            iterbt += 1
        end
        @show iterbt
        iterbt == btmax && break
        copyto!(α, αnew)
        copyto!(s, snew)
        copyto!(λ, λnew)
        τ *= sqrt(1 - min(γ, γλ, 1 - τminfrac^2))
        # @show α s λ τ
        iter += 1
        bottom(gxs) .= -τ ./ s
        H.λsratio .= λ ./ s
        totalviolnew = sum(abs2, cviol) + sum(abs2, gxs + J' * λν)
        # totalviolnew > totalviol && break
        totalviol = totalviolnew
    end
    @show τ
    return exp.(α)
end

function symnz(W)
    ax = axes(W, 1)
    k = 0
    for j in ax
        for i in j:last(ax)
            W[i, j] && (k += 1)
        end
    end
    return k
end

function residual!(r, logA, α::AbstractVector, W::AbstractMatrix{Bool})
    ax = axes(logA, 1)
    @assert axes(logA, 2) == ax
    k = firstindex(r) - 1
    for j in ax
        for i in j:last(ax)
            W[i, j] || continue
            r[k += 1] = logA[i, j] - α[i] - α[j]
        end
    end
    @assert k == lastindex(r)
    return r
end
function residual!(r, logA, α::AbstractVector, β::AbstractVector, W::AbstractMatrix{Bool})
    k = firstindex(r) - 1
    for j in axes(logA, 2)
        βj = β[j]
        for i in axes(logA, 1)
            W[i, j] || continue
            r[k += 1] = logA[i, j] - α[i] - βj
        end
    end
    @assert k == lastindex(r)
    return r
end

function residual(logA, α::AbstractVector, W::AbstractMatrix{Bool}, N::Int=symnz(W))
    T = promote_type(eltype(logA), eltype(α))
    return residual!(Vector{T}(undef, N), logA, α, W)
end
function residual(logA, α::AbstractVector, β::AbstractVector, W::AbstractMatrix{Bool}, N::Int=sum(W))
    T = promote_type(eltype(logA), eltype(α), eltype(β))
    return residual!(Vector{T}(undef, N), logA, α, β, W)
end

function jopsym(A, N)
    ax = axes(A, 1)
    n = length(ax)
    T = typeof(log(oneunit(eltype(A))))
    return LinearOperator(T, N, n, false, false,
        function(out, v, α, β)    # mul!(out, J, v, α, β)
            if iszero(β)
                fill!(out, zero(T))
            elseif β != 1
                out .*= β
            end
            k = 0
            for j in ax
                for i in j:last(ax)
                    iszero(A[i, j]) && continue
                    out[k += 1] -= v[i] + v[j]
                end
            end
            @assert k == length(out)
            return out
        end,
        function(out, v, α, β)    # mul!(out, J', v, α, β)
            if iszero(β)
                fill!(out, zero(T))
            elseif β != 1
                out .*= β
            end
            k = 0
            for j in ax
                for i in j:last(ax)
                    iszero(A[i, j]) && continue
                    k += 1
                    out[i] -= v[k]
                    out[j] -= v[k]
                end
            end
            @assert k == length(v)
            return out
        end)
end




# Sherman-Morrison division
function divsm!(result::AbstractVector{T}, v::AbstractVector, nz::AbstractVector{Int}) where T<:Real
    sumnz = sum(nz)
    @assert sumnz > 0 "Cannot divide by zero: all rows are zero"
    offset = sum(v) / (2 * sumnz)
    for i in eachindex(result, v, nz)
        nzi = nz[i]
        result[i] = nzi == 0 ? typemin(T) : v[i] / nzi - offset
    end
    return result
end

function jactλ!(ξ, W, λ)  # ξ = J' * λ
    @assert issymmetric(W)  # will generalize later
    fill!(ξ, zero(eltype(ξ)))
    ax = axes(W, 1)
    for j in ax
        λj = λ[j]
        for i in ax
            wij = W[i, j]
            ξ[i] -= wij * λj
            ξ[j] -= wij * λ[i]
        end
    end
    return ξ
end

function solveΔs!(Δs, W, Δα)   # Δs = -J * Δα
    @assert issymmetric(W)  # will generalize later
    fill!(Δs, zero(eltype(Δs)))
    ax = axes(W, 1)
    for j in ax
        Δαj = Δα[j]
        for i in ax
            wij = W[i, j]
            Δs[i] += wij * Δαj
            Δs[j] += wij * Δα[i]
        end
    end
    return Δs
end

function maxstep(Δs, s, steplen=one(eltype(Δs)); η=1//16)
    for i in eachindex(s, Δs)
        Δsi = Δs[i]
        if Δsi < 0
            steplen = min(steplen, -(1 - η) * s[i] / Δsi)
        end
    end
    return steplen
end

"""
    a, b = cover(A; exact=false, regularize=false)

Given a matrix `A`, return vectors `a` and `b` representing the "scale of each
axis," so that `|A[i,j]| ~ a[i] * b[j]` for all `i, j`. `a[i]` and `b[j]` are
nonnegative, and are zero only if `A[i, j] = 0` for all `j` or all `i`,
respectively.

With `exact=true`, `a` and `b` solve the optimization problem

    min ∑_{i,j : A[i,j] ≠ 0} (log(|A[i,j]| / (a[i] * b[j])))²
    s.t. ∑_i nA[i] * log(a[i]) = ∑_j mA[j] * log(b[j])

where `nA` and `mA` are the number of nonzeros in each row and column,
respectively. Up to multiplication by a scalar, these vectors are covariant
under changes of scale but not general linear transformations.

With `exact=false`, the pattern of nonzeros in `A` is approximated as `u * v'`,
where `sum(u) * v[j] = mA[j]` and `sum(v) * u[i] = nA[i]`. This results in an
`O(m*n)` rather than `O((m+n)^3)` algorithm.
"""
function cover(A::AbstractMatrix; exact::Bool=false, regularize::Bool=false)
    Base.require_one_based_indexing(A)
    ax1, ax2 = axes(A, 1), axes(A, 2)
    (s, ns), (t, mt) = _cover(A, ax1, ax2)
    m, n = length(ax1), length(ax2)
    if !exact || (all(==(n), ns) && all(==(m), mt))
        z = sum(ns)
        @assert sum(mt) == z "Inconsistent nonzero counts in rows and columns"
        offsets, offsett = sum(s) / (2z), sum(t) / (2z)
        divsafe!(s, ns)
        divsafe!(t, mt)
        a = exp.(s ./ ns .- offsets)
        b = exp.(t ./ mt .- offsett)
        return a, b
    end
    p = vcat(ns, -mt)
    W = isnz(A)
    T = promote_type(eltype(s), eltype(t))
    τ = regularize ? sqrt(eps(T)) : zero(T)
    divsafe!(s, vec(sum(W; dims=2)); sentinel=-1/τ)
    divsafe!(t, vec(sum(W; dims=1)); sentinel=-1/τ)
    a12 = exp.(cholesky(Diagonal(vcat(ns, mt)) + odblocks(W) + p * p' + τ * I) \ vcat(s, t))
    return a12[begin:begin+m-1], a12[m+begin:end]
end


ratio_nz(n, d) = iszero(d) ? zero(n) / oneunit(d) : n / d

"""
    κ = condscale(A; exact=true)

Given a symmetric matrix `A`, return the condition number of the scaled matrix

    A ./ (a .* a')

where `a = symcover(A; exact)`.

This is a scale-invariant estimate of the condition number of `A`.
"""
function condscale(A; exact=true)
    a = symcover(A; exact)
    return cond(A ./ (a .* a'))
end

"""
    a, mag = divmag(A, b; cond::Bool=false, kwargs...)

Given a symmetric matrix `A` and vector `b`, for `x = A \\ b` return a pair
where `mag` is a naive estimate of the magnitude of `sum(abs.(x .* a))`. `a` and
`mag` are scale-covariant in circumstances where `A \\ b` is contravariant. With
`cond=false`, the estimate is based only on the magnitudes of the numbers in `A`
and `b`, and does not account for the conditioning of `A` or cancellation in the
solution process. Any `kwargs` are passed to [`symcover`](@ref).

This can be used to form scale-invariant estimates of roundoff errors in
computations involving `A`, `b`, and `x`.
"""
function divmag(A, b; cond::Bool=false, exact=cond)
    a = symcover(A; exact)
    κ = cond ? LinearAlgebra.cond(A ./ (a .* a')) : 1
    return a, κ * sum(abs ∘ splat(ratio_nz), zip(b, a))
end

# function diagapprox(A; iter=30)
#     ax = axes(A, 1)
#     axes(A, 2) == ax || throw(ArgumentError("diagapprox requires a square matrix"))
#     first(ax) == 1 || throw(ArgumentError("diagapprox requires 1-based indexing"))
#     n = length(ax)
#     T = float(eltype(A))
#     d = zeros(T, ax)
#     for _ in 1:iter
#         x = log.(rand(T, n)) .* (rand(Bool, n) .- T(0.5))
#         y = A * x
#         d .+= abs.(y) ./ abs.(x)
#     end
#     return d / iter
# end

end # module
