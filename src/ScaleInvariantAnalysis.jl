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
    a = symscale(A; exact=false)

Given a matrix `A` assumed to be symmetric, return a vector `a` representing the
"scale of each axis," so that `|A[i,j]| ~ a[i] * a[j]` for all `i, j`. `a[i]` is
nonnegative, and is zero only if `A[i, j] = 0` for all `j`.

With `exact=true`, `a` minimizes the objective function

    ∑_{i,j : A[i,j] ≠ 0} (log(|A[i,j]| / (a[i] * a[j])))²

and is therefore covariant under changes of scale but not general linear
transformations.

With `exact=false`, the pattern of nonzeros in `A` is approximated as `u * u'`,
where `sum(u) * u[i] = nz[i]` is the number of nonzero in row `i`. This results in an
`O(n^2)` rather than `O(n^3)` algorithm.
"""
function symscale(A::AbstractMatrix; exact::Bool=false)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symscale requires a square matrix"))
    sumlogA, nz = _symscale(A, ax)
    n = length(ax)
    if !exact || all(==(n), nz)
        # Sherman-Morrison formula for efficiency
        offset = sum(sumlogA) / (2 * sum(nz))
        return exp.(sumlogA ./ nz .- offset)
    end
    return exp.(cholesky(Diagonal(nz) + isnz(A)) \ sumlogA)
end

"""
    a1, a2 = matrixscale(A; exact=false)

Given a matrix `A`, return vectors `a1` and `a2` representing the "scale of each
axis," so that `|A[i,j]| ~ a1[i] * a2[j]` for all `i, j`. `a1[i]` and `a2[j]`
are nonnegative, and are zero only if `A[i, j] = 0` for all `j` or all `i`,
respectively.

With `exact=true`, `a1` and `a2` solve the optimization problem

    min ∑_{i,j : A[i,j] ≠ 0} (log(|A[i,j]| / (a1[i] * a2[j])))²
    s.t. n * ∑_i log(a1[i]) = m * ∑_j log(a2[j])

where `m, n = size(A)`. Up to multiplication by a scalar, these vectors are
covariant under changes of scale but not general linear transformations.

With `exact=false`, the pattern of nonzeros in `A` is approximated as `u * v'`,
where `sum(u) * v[j]` is the number of nonzeros in column `j` and `sum(v) *
u[i]` is the number of nonzeros in row `i`. This results in an `O(m*n)` rather
than `O((m+n)^3)` algorithm.
"""
function matrixscale(A::AbstractMatrix; exact::Bool=false)
    Base.require_one_based_indexing(A)
    ax1, ax2 = axes(A, 1), axes(A, 2)
    (sumlogA1, nz1), (sumlogA2, nz2) = _matrixscale(A, ax1, ax2)
    m, n = length(ax1), length(ax2)
    if !exact || (all(==(n), nz1) && all(==(m), nz2))
        z = sum(nz1)
        u, v = nz1 / sqrt(z), nz2 / sqrt(z)
        uᵀ1, vᵀ1 = sum(u), sum(v)
        s′, t′ = sumlogA1 ./ u, sumlogA2 ./ v
        αoffset, βoffset = s′ ./ vᵀ1, t′ ./ uᵀ1
    end
    p = [fill(n, m); fill(-m, n)]   # used to enforce the constraint
    W = isnz(A)
    a12 = exp.(cholesky(Diagonal(vcat(nz1, nz2)) + odblocks(W) + p * p') \ vcat(sumlogA1, sumlogA2))
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
