"""
    lobjective(a, b, A)
    lobjective(a, A)

Compute the sum of log-domain excesses over nonzero entries of `A`:
    ∑_{i,j : A[i,j] ≠ 0} log(a[i] * b[j] / |A[i,j]|)

The two-argument form is for symmetric matrices where the cover is `a*a'`.`

See also: [`qobjective`](@ref) for the sum of squared log-domain excesses.
"""
lobjective(a, b, A) = sum(log(a[i] * b[j] / abs(A[i, j])) for i in axes(a, 1), j in axes(b, 1) if A[i, j] != 0)
lobjective(a, A) = lobjective(a, a, A)

"""
    qobjective(a, b, A)
    qobjective(a, A)

Compute the sum of squared log-domain excesses over nonzero entries of `A`:
    ∑_{i,j : A[i,j] ≠ 0} log(a[i] * b[j] / |A[i,j]|)²

The two-argument form is for symmetric matrices where the cover is `a*a'`.`

See also: [`lobjective`](@ref) for the sum of log-domain excesses.
"""
qobjective(a, b, A) = sum(log(a[i] * b[j] / abs(A[i, j]))^2 for i in axes(a, 1), j in axes(b, 1) if A[i, j] != 0)
qobjective(a, A) = qobjective(a, a, A)

"""
    a = symcover(A; iter=3)

Given a square matrix `A` assumed to be symmetric, return a vector `a`
representing the symmetric cover of `A`, so that `a[i] * a[j] >= abs(A[i, j])`
for all `i`, `j`.

`a` is not minimal, but with increasing `iter` it is increasingly tight.
`symcover` is fast and generally recommended for production use.

See also: [`symcover_lmin`](@ref), [`symcover_qmin`](@ref), [`cover`](@ref).
"""
function symcover(A::AbstractMatrix; kwargs...)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover requires a square matrix"))
    a = similar(A, float(eltype(A)), ax)
    for j in ax
        a[j] = sqrt(abs(A[j, j]))
    end
    # Iterate over the diagonals of A, and update a[i] and a[j] to satisfy |A[i, j]| ≤ a[i] * a[j] whenever this constraint is violated
    # Iterating over the diagonals gives a more "balanced" result and typically results in lower loss than iterating in a triangular pattern.
    for k in 1:length(ax)-1
        for j in first(ax):last(ax)-k
            i = j + k
            Aij, ai, aj = abs(A[i, j]), a[i], a[j]
            if iszero(aj)
                if !iszero(ai)
                    a[j] = Aij / ai
                end
            elseif iszero(ai)
                a[i] = Aij / aj
            else
                aprod = ai * aj
                aprod >= Aij && continue
                s = sqrt(Aij / sqrt(aprod))
                a[i] = s * ai
                a[j] = s * aj
            end
        end
    end
    return tighten_cover!(a, A; kwargs...)
end

"""
    a, b = cover(A; iter=3)

Given a matrix `A`, return vectors `a` and `b` such that `a[i] * b[j] >= abs(A[i, j])`
for all `i`, `j`.

`a .* b'` is not minimal, but with increasing `iter` it is increasingly tight.
`cover` is fast and generally recommended for production use.

See also: [`cover_lmin`](@ref), [`cover_qmin`](@ref), [`symcover`](@ref).
"""
function cover(A::AbstractMatrix; kwargs...)
    T = float(eltype(A))
    a, b = zeros(T, axes(A, 1)), zeros(T, axes(A, 2))
    @turbo for j in axes(A, 2)
        bj = zero(T)
        for i in axes(A, 1)
            Aij = abs(A[i, j])
            a[i] = max(a[i], Aij)
            bj = max(bj, Aij)
        end
        b[j] = bj
    end
    map!(sqrt, a, a)
    map!(sqrt, b, b)
    return tighten_cover!(a, b, A; kwargs...)
end

function tighten_cover!(a::AbstractVector{T}, A::AbstractMatrix; iter::Int=3) where T
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("`tighten_cover!(a, A)` requires a square matrix `A`"))
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`"))
    aratio = similar(a)
    for _ in 1:iter
        fill!(aratio, typemax(T))
        @turbo for j in eachindex(a)
            aratioj, aj = aratio[j], a[j]
            for i in eachindex(a)
                Aij = T(abs(A[i, j]))
                r = a[i] * aj / Aij
                aratio[i] = min(aratio[i], r)
                aratioj = min(aratioj, r)
            end
            aratio[j] = aratioj
        end
        a ./= sqrt.(aratio)
    end
    return a
end

function tighten_cover!(a::AbstractVector{T}, b::AbstractVector{T}, A::AbstractMatrix; iter::Int=3) where T
    aratio = fill(typemax(T), eachindex(a))
    bratio = fill(typemax(T), eachindex(b))
    eachindex(a) == axes(A, 1) || throw(DimensionMismatch("indices of a must match row-indexing of A"))
    eachindex(b) == axes(A, 2) || throw(DimensionMismatch("indices of b must match column-indexing of A"))
    for _ in 1:iter
        fill!(aratio, typemax(T))
        fill!(bratio, typemax(T))
        @turbo for j in eachindex(b)
            bratioj, bj = bratio[j], b[j]
            for i in eachindex(a)
                Aij = T(abs(A[i, j]))
                r = a[i] * bj / Aij
                aratio[i] = min(aratio[i], r)
                bratioj = min(bratioj, r)
            end
            bratio[j] = bratioj
        end
        a ./= sqrt.(aratio)
        b ./= sqrt.(bratio)
    end
    return a, b
end


# The next four have methods that are defined in the extension module, but we
# define the function and docstring here to avoid circular dependencies.

"""
    a = symcover_qmin(A)

Given a square matrix `A` assumed to be symmetric, return a vector `a` representing the
symmetric q-minimal (quadratic-minimal) cover of `A`. This solves the optimization problem

    min ∑_{i,j : A[i,j] ≠ 0} log(a[i] * a[j] / |A[i,j]|)²
    s.t.                     a[i] * a[j] ≥ |A[i, j]| for all i, j

This implementation is *slow*. See also:
- [`symcover_lmin`](@ref) for a much more efficient option that is not quadratically-optimal
- [`cover_qmin`](@ref) for a generalization to non-symmetric matrices.

!!! note
    This function requires loading the JuMP and HiGHS packages, which are not dependencies of this package.
"""
function symcover_qmin end

"""
    a = symcover_lmin(A)

Similar to [`symcover_qmin`](@ref), but returns a symmetric l-minimal (linear-minimal) cover of `A`.
"""
function symcover_lmin end

"""
    a, b = cover_qmin(A)

Given a matrix `A`, return vectors `a` and `b` representing the q-minimal (quadratic-minimal) cover of `A`. This solves the optimization problem
    min ∑_{i,j : A[i,j] ≠ 0} log(a[i] * b[j] / |A[i,j]|)²
    s.t.                     a[i] * b[j] ≥ |A[i, j]| for all i, j

This implementation is *slow*. See also:
- [`cover_lmin`](@ref) for a much more efficient option that is not quadratically-optimal
- [`symcover_qmin`](@ref) for a specialization to symmetric matrices
"""
function cover_qmin end

"""
    a, b = cover_lmin(A)

Similar to [`cover_qmin`](@ref), but returns an l-minimal (linear-minimal) cover of `A`.
"""
function cover_lmin end
