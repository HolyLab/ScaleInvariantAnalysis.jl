"""
    cover_lobjective(a, b, A)
    cover_lobjective(a, A)

Compute the sum of log-domain excesses over nonzero entries of `A`:

    ∑_{i,j : A[i,j] ≠ 0} log(a[i] * b[j] / |A[i,j]|)

The two-argument form is for symmetric matrices where the cover is `a*a'`.

See also: [`cover_qobjective`](@ref) for the sum of squared log-domain excesses.
"""
cover_lobjective(a, b, A) = sum(log(a[i] * b[j] / abs(A[i, j])) for i in axes(a, 1), j in axes(b, 1) if A[i, j] != 0)
cover_lobjective(a, A) = cover_lobjective(a, a, A)

"""
    cover_qobjective(a, b, A)
    cover_qobjective(a, A)

Compute the sum of squared log-domain excesses over nonzero entries of `A`:

    ∑_{i,j : A[i,j] ≠ 0} log(a[i] * b[j] / |A[i,j]|)²

The two-argument form is for symmetric matrices where the cover is `a*a'`.`

See also: [`cover_lobjective`](@ref) for the sum of log-domain excesses.
"""
cover_qobjective(a, b, A) = sum(log(a[i] * b[j] / abs(A[i, j]))^2 for i in axes(a, 1), j in axes(b, 1) if A[i, j] != 0)
cover_qobjective(a, A) = cover_qobjective(a, a, A)

"""
    a = symcover(A; prioritize::Symbol=:quality, iter=3)

Given a square matrix `A` assumed to be symmetric, return a vector `a`
representing the symmetric cover of `A`, so that `a[i] * a[j] >= abs(A[i, j])`
for all `i`, `j`.

`prioritize=:quality` yields a cover that is typically closer to being
quadratically optimal, though there are exceptions.
`prioritize=:speed` is often about twice as fast (with default `iter=3`). In
either case, after initialization `a` is tightened iteratively, with `iter`
specifying the number of iterations (more iterations make tighter covers).

Regardless of which `prioritize` option is chosen, `symcover` is fast and
generally recommended for production use.

See also: [`symcover_lmin`](@ref), [`symcover_qmin`](@ref), [`cover`](@ref).

# Examples

```jldoctest; filter = r"(\\d+\\.\\d{6})\\d+" => s"\\1"
julia> A = [4 -1; -1 0];

julia> a = symcover(A; prioritize=:speed)
2-element Vector{Float64}:
 2.0
 0.5

julia> a * a'
2×2 Matrix{Float64}:
 4.0  1.0
 1.0  0.25

julia> A = [0 12 9; 12 7 12; 9 12 0];

julia> a = symcover(A; prioritize=:quality)
3-element Vector{Float64}:
 3.4021999694928753
 3.54528705924512
 3.3847752803845172

julia> a * a'
3×3 Matrix{Float64}:
 11.575   12.0618  11.5157
 12.0618  12.5691  12.0
 11.5157  12.0     11.4567

julia> a = symcover(A; prioritize=:speed)
3-element Vector{Float64}:
 4.535573676110727
 2.6457513110645907
 4.535573676110727

julia> a * a'
3×3 Matrix{Float64}:
 20.5714  12.0  20.5714
 12.0      7.0  12.0
 20.5714  12.0  20.5714
```
"""
function symcover(A::AbstractMatrix; exclude_diagonal::Bool=false, prioritize::Symbol=:quality, kwargs...)
    prioritize in (:quality, :speed) || throw(ArgumentError("prioritize must be :quality or :speed"))
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover requires a square matrix"))
    a = similar(A, float(eltype(A)), ax)
    if prioritize == :quality
        _symcover_init_quadratic!(a, A; exclude_diagonal)
    else
        _symcover_init_fast!(a, A; exclude_diagonal)
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
                else
                    a[i] = a[j] = sqrt(Aij)
                end
            elseif iszero(ai)
                a[i] = Aij / aj
            else
                aprod = ai * aj
                aprod >= Aij && continue
                s = sqrt(Aij / aprod)
                a[i] = s * ai
                a[j] = s * aj
            end
        end
    end
    return tighten_cover!(a, A; exclude_diagonal, kwargs...)
end

function _symcover_init_quadratic!(a::AbstractVector{T}, A::AbstractMatrix; exclude_diagonal::Bool=false) where T
    ax = eachindex(a)
    loga = fill!(similar(a), zero(T))
    nza  = fill(0, ax)
    for j in ax
        for i in first(ax):j - exclude_diagonal
            Aij = abs(A[i, j])
            iszero(Aij) && continue
            lAij = log(Aij)
            loga[i] += lAij
            nza[i] += 1
            if j != i
                loga[j] += lAij
                nza[j] += 1
            end
        end
    end
    nztotal = sum(nza)
    halfmu = iszero(nztotal) ? zero(T) : sum(loga) / (2 * nztotal)
    for i in ax
        a[i] = ai = iszero(nza[i]) ? zero(T) : exp(loga[i] / nza[i] - halfmu)
        if !exclude_diagonal
            # The rest of the algorithm will ensure the initialization is a valid cover, but we have to do the diagonal here.
            Aii = abs(A[i, i])
            if ai^2 < Aii
                a[i] = sqrt(Aii)
            end
        end
    end
    return a
end

function _symcover_init_fast!(a::AbstractVector{T}, A::AbstractMatrix; exclude_diagonal::Bool=false) where T
    if exclude_diagonal
        fill!(a, zero(T))
    else
        for j in eachindex(a)
            a[j] = sqrt(abs(A[j, j]))
        end
    end
    return a
end

"""
    d, a = symdiagcover(A; kwargs...)

Given a square matrix `A` assumed to be symmetric, return vectors `d` and `a`
representing a symmetric diagonalized cover `Diagonal(d) + a * a'` of `A` with
the diagonal as tight as possible given `A` and `a`. In particular,

    abs(A[i, j]) ≤ a[i] * a[j] for all i ≠ j, and
    abs(A[i, i]) ≤ a[i]^2 + d[i] for all i.

# Examples

```jldoctest; setup=:(using LinearAlgebra), filter = r"(\\d+\\.\\d{6})\\d+" => s"\\1"
julia> A = [4 1e-8; 1e-8 1];

julia> a = symcover(A)
2-element Vector{Float64}:
 2.0
 1.0

julia> a * a'
2×2 Matrix{Float64}:
 4.0  2.0
 2.0  1.0

julia> d, a = symdiagcover(A)
([3.99999999, 0.99999999], [0.0001, 0.0001])

julia> Diagonal(d) + a * a'
2×2 Matrix{Float64}:
 4.0     1.0e-8
 1.0e-8  1.0
```
For this case, one sees much tighter covering with `symdiagcover`.
"""
function symdiagcover(A::AbstractMatrix; kwargs...)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover requires a square matrix"))
    a = symcover(A; exclude_diagonal=true, kwargs...)
    d = map(ax) do i
        Aii, ai = abs(A[i, i]), a[i]
        max(zero(Aii), Aii - ai^2)
    end
    return d, a
end

"""
    a, b = cover(A; iter=3)

Given a matrix `A`, return vectors `a` and `b` such that `a[i] * b[j] >= abs(A[i, j])`
for all `i`, `j`.

`a .* b'` may not be minimal, but it is tightened iteratively, with `iter` specifying
the number of iterations (more iterations make tighter covers).
`cover` is fast and generally recommended for production use.

See also: [`cover_lmin`](@ref), [`cover_qmin`](@ref), [`symcover`](@ref).

# Examples

```jldoctest; filter = r"(\\d+\\.\\d{6})\\d+" => s"\\1"
julia> A = [1 2 3; 6 5 4];

julia> a, b = cover(A)
([1.2674308473260654, 3.4759059767492304], [1.7261686708831454, 1.61137045961268, 2.366993044495631])

julia> a * b'
2×3 Matrix{Float64}:
 2.1878  2.0423   3.0
 6.0     5.60097  8.22745
```
"""
function cover(A::AbstractMatrix; kwargs...)
    T = float(eltype(A))
    a = zeros(T, axes(A, 1))
    b = zeros(T, axes(A, 2))
    loga = fill(zero(T), axes(A, 1))
    logb = fill(zero(T), axes(A, 2))
    nza  = fill(0, axes(A, 1))
    nzb  = fill(0, axes(A, 2))
    logmu = zero(T)
    nztotal = 0
    for j in axes(A, 2)
        for i in axes(A, 1)
            Aij = abs(A[i, j])
            iszero(Aij) && continue
            lAij = log(Aij)
            loga[i] += lAij
            logb[j] += lAij
            nza[i] += 1
            nzb[j] += 1
            logmu += lAij
            nztotal += 1
        end
    end
    halfmu = iszero(nztotal) ? zero(T) : T(logmu / (2 * nztotal))
    for i in axes(A, 1)
        a[i] = iszero(nza[i]) ? zero(T) : exp(loga[i] / nza[i] - halfmu)
    end
    for j in axes(A, 2)
        b[j] = iszero(nzb[j]) ? zero(T) : exp(logb[j] / nzb[j] - halfmu)
    end
    # Now we have sums of (log(a[i]) + log(b[j]) - log(A[i, j])) to be zero across rows or columns.
    # Now it needs to be boosted to cover A.
    for j in axes(A, 2)
        for i in axes(A, 1)
            Aij, ai, bj = abs(A[i, j]), a[i], b[j]
            aprod = ai * bj
            aprod >= Aij && continue
            s = sqrt(Aij / aprod)
            a[i] = s * ai
            b[j] = s * bj
        end
    end
    return tighten_cover!(a, b, A; kwargs...)
end

function tighten_cover!(a::AbstractVector{T}, A::AbstractMatrix; iter::Int=3, exclude_diagonal::Bool=false) where T
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("`tighten_cover!(a, A)` requires a square matrix `A`"))
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`"))
    aratio = similar(a)
    for _ in 1:iter
        fill!(aratio, typemax(T))
        if exclude_diagonal
            for j in eachindex(a)
                aratioj, aj = aratio[j], a[j]
                for i in first(ax):j-1
                    Aij = T(abs(A[i, j]))
                    r = ifelse(iszero(Aij), typemax(T), a[i] * aj / Aij)
                    aratio[i] = min(aratio[i], r)
                    aratioj = min(aratioj, r)
                end
                for i in j+1:last(ax)
                    Aij = T(abs(A[i, j]))
                    r = ifelse(iszero(Aij), typemax(T), a[i] * aj / Aij)
                    aratio[i] = min(aratio[i], r)
                    aratioj = min(aratioj, r)
                end
                aratio[j] = aratioj
            end
        else
            for j in eachindex(a)
                aratioj, aj = aratio[j], a[j]
                for i in eachindex(a)
                    Aij = T(abs(A[i, j]))
                    r = ifelse(iszero(Aij), typemax(T), a[i] * aj / Aij)
                    aratio[i] = min(aratio[i], r)
                    aratioj = min(aratioj, r)
                end
                aratio[j] = aratioj
            end
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
        for j in eachindex(b)
            bratioj, bj = bratio[j], b[j]
            for i in eachindex(a)
                Aij = T(abs(A[i, j]))
                r = ifelse(iszero(Aij), typemax(T), a[i] * bj / Aij)
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
symmetric quadratic-minimal cover of `A`. This solves the optimization problem

    min ∑_{i,j : A[i,j] ≠ 0} log(a[i] * a[j] / |A[i,j]|)²
    s.t.                     a[i] * a[j] ≥ |A[i, j]| for all i, j

This implementation is *slow*. See also:
- [`cover_qobjective`](@ref) for the objective minimized by this function
- [`symcover`](@ref) for a much more efficient option that is not quadratically-optimal
- [`cover_qmin`](@ref) for a generalization to non-symmetric matrices

!!! note
    This function requires loading the JuMP and HiGHS packages, which are not dependencies of this package.
"""
function symcover_qmin end

"""
    a = symcover_lmin(A)

Similar to [`symcover_qmin`](@ref), but returns a symmetric linear-minimal cover of `A`.
"""
function symcover_lmin end

"""
    a, b = cover_qmin(A)

Given a matrix `A`, return vectors `a` and `b` representing the quadratic-minimal cover of `A`.
This solves the optimization problem

    min ∑_{i,j : A[i,j] ≠ 0} log(a[i] * b[j] / |A[i,j]|)²
    s.t.                     a[i] * b[j] ≥ |A[i, j]| for all i, j

This implementation is *slow*. See also:
- [`cover_qobjective`](@ref) for the objective minimized by this function
- [`cover`](@ref) for a much more efficient option that is not quadratically-optimal
- [`symcover_qmin`](@ref) for a specialization to symmetric matrices

!!! note
    This function requires loading the JuMP and HiGHS packages, which are not dependencies of this package.
"""
function cover_qmin end

"""
    a, b = cover_lmin(A)

Similar to [`cover_qmin`](@ref), but returns a linear-minimal cover of `A`.
"""
function cover_lmin end
