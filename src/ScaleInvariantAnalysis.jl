module ScaleInvariantAnalysis

using LinearAlgebra
using SparseArrays

export condscale, divmag, dotabs, matrixscale, symscale

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
    a = symscale(A; exact=false, regularize=false)

Given a matrix `A` assumed to be symmetric, return a vector `a` representing the
"scale of each axis," so that `|A[i,j]| ~ a[i] * a[j]` for all `i, j`. `a[i]` is
nonnegative, and is zero only if `A[i, j] = 0` for all `j`.

With `exact=true`, `a` minimizes the objective function

    ∑_{i,j : A[i,j] ≠ 0} (log(|A[i,j]| / (a[i] * a[j])))²

and is therefore covariant under changes of scale but not general linear
transformations.

With `exact=false`, the pattern of nonzeros in `A` is approximated as `u * u'`,
where `sum(u) * u[i] = nz[i]` is the number of nonzero in row `i`. This results in an
`O(n^2)` rather than `O(n^3)` algorithm. `regularize=true` adds a small offset to the
diagonal (relevant only when `exact=true`), which handles all-zero rows of `A`.
"""
function symscale(A::AbstractMatrix; exact::Bool=false, regularize::Bool=false)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symscale requires a square matrix"))
    sumlogA, nz = _symscale(A, ax)
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

function symscale_barrier(A::AbstractMatrix{T}) where T<:Real
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symscale requires a square matrix"))
    W = isnz(A)
    z = log(oneunit(T))
    logA = [iszero(aij) ? z : log(aij) for aij in A]
    sumlogA = vec(sum(logA; dims=2))
    nz = vec(sum(W; dims=2))
    alpha = solvesm(sumlogA, nz)
    s = similar(logA)
    sbar, nA = zero(eltype(s)), 0
    for j in ax
        for i in j:last(ax)
            sij = alpha[i] + alpha[j] - logA[i, j]
            s[i, j] = sij
            if sij < zero(sij)
                sbar -= sij    # constraint violation
                nA += 1
            end
        end
    end
    iszero(nA) && return exp.(alpha)
    sbar /= nA
    δ = solveδ(sbar)
    s .= max.(s, δ)
    λ = τ ./ s
    # TODO: barrier method iterations
    ξ = similar(α)
    while iter < itermax
        jactλ!(ξ, W, λ)
        divsm!(Δα, sumlogA - ξ - B*α, nz)
        solveΔs!(Δs, W, Δα)
        γ = maxstep(Δs, s)
        α .+= γ * Δα
        s .+= γ * Δs
        τ /= β
        λ .*= τ ./ s
        check_convergence(Δs, Δα, sbar)
    end
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
        for i in j:last(ax)
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
        for i in j:last(ax)
            wij = W[i, j]
            Δs[i] += wij * Δαj
            Δs[j] += wij * Δα[i]
        end
    end
    return Δs
end

"""
    a, b = matrixscale(A; exact=false, regularize=false)

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
function matrixscale(A::AbstractMatrix; exact::Bool=false, regularize::Bool=false)
    Base.require_one_based_indexing(A)
    ax1, ax2 = axes(A, 1), axes(A, 2)
    (s, ns), (t, mt) = _matrixscale(A, ax1, ax2)
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

where `a = symscale(A; exact)`.

This is a scale-invariant estimate of the condition number of `A`.
"""
function condscale(A; exact=true)
    a = symscale(A; exact)
    return cond(A ./ (a .* a'))
end

"""
    a, mag = divmag(A, b; cond::Bool=false, kwargs...)

Given a symmetric matrix `A` and vector `b`, for `x = A \\ b` return a pair
where `mag` is a naive estimate of the magnitude of `sum(abs.(x .* a))`. `a` and
`mag` are scale-covariant in circumstances where `A \\ b` is contravariant. With
`cond=false`, the estimate is based only on the magnitudes of the numbers in `A`
and `b`, and does not account for the conditioning of `A` or cancellation in the
solution process. Any `kwargs` are passed to [`symscale`](@ref).

This can be used to form scale-invariant estimates of roundoff errors in
computations involving `A`, `b`, and `x`.
"""
function divmag(A, b; cond::Bool=false, exact=cond)
    a = symscale(A; exact)
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
