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
    cover_qobjective(a::AbstractVector, b::AbstractVector, A::AbstractMatrix)
    cover_qobjective(a::AbstractVector, A::AbstractMatrix)
    cover_qobjective(alpha::AbstractVector, logA::AbstractMatrix, A::AbstractMatrix)   # alpha = log.(a)

Compute the sum of squared log-domain excesses over nonzero entries of `A`:

    ∑_{i,j : A[i,j] ≠ 0} log(a[i] * b[j] / |A[i,j]|)²

The two-argument form is for symmetric matrices where the cover is `a*a'`.
The `alpha/logA/A` form is used when you have `log.(abs.(A))` pre-computed and want to avoid taking additional logarithms.

See also: [`cover_lobjective`](@ref) for the sum of log-domain excesses.
"""
cover_qobjective(a::AbstractVector, b::AbstractVector, A::AbstractMatrix) = sum(log(a[i] * b[j] / abs(A[i, j]))^2 for i in axes(a, 1), j in axes(b, 1) if A[i, j] != 0)
cover_qobjective(a::AbstractVector, A::AbstractMatrix) = cover_qobjective(a, a, A)

function cover_qobjective(alpha::AbstractVector{T}, logA::AbstractMatrix, A::AbstractMatrix) where T
    # This is used internally and needs to be higher performance than the version above
    objval = zero(T)
    ax = eachindex(alpha)
    for j in ax
        alphaj = alpha[j]
        for i in j:last(ax)
            iszero(A[i, j]) && continue
            Δobjval = (alpha[i] + alphaj - logA[i, j])^2
            objval += Δobjval * (1 + (i != j))
        end
    end
    return objval
end

"""
    a = symcover(A; iter=3)

Given a square matrix `A` assumed to be symmetric, return a vector `a`
representing the symmetric cover of `A`, so that `a[i] * a[j] >= abs(A[i, j])`
for all `i`, `j`.

`symcover` is fast and generally recommended for production use.

See also: [`symcover_lmin`](@ref), [`symcover_qmin`](@ref), [`cover`](@ref).

# Examples

```jldoctest; filter = r"(\\d+\\.\\d{6})\\d+" => s"\\1"
julia> A = [4 -1; -1 0];

julia> a = symcover(A)
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
function symcover(A::AbstractMatrix; exclude_diagonal::Bool=false, kwargs...)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover requires a square matrix"))
    logA = log.(abs.(A))
    alpha, afrak, B = symcover_init_unconstrained(logA, A; exclude_diagonal)
    alphastar = copy(alpha)
    symcover_makefeasible_diags!(alpha, logA, A; exclude_diagonal)
    tighten_cover!(alpha, logA, A; exclude_diagonal, kwargs...)
    return exp.(simplex!(alpha, logA, A, alphastar, B, afrak; exclude_diagonal, kwargs...))
end

"""
    a = symcover_heuristic(A; iter=3)

Given a square matrix `A` assumed to be symmetric, return a vector `a`
representing the symmetric cover of `A`, so that `a[i] * a[j] >= abs(A[i, j])`
for all `i`, `j`.

This is a particularly fast implementation: it avoids taking logarithms and does
not attempt to achieve quadratic optimality. After initialization, `a` is
tightened iteratively, with `iter` specifying the number of iterations (more
iterations make tighter covers).

See also: [`symcover`](@ref), [`symcover_lmin`](@ref), [`symcover_qmin`](@ref), [`cover`](@ref).

# Examples

```jldoctest; filter = r"(\\d+\\.\\d{6})\\d+" => s"\\1"
julia> A = [4 -1; -1 0];

julia> a = symcover_heuristic(A)
2-element Vector{Float64}:
 2.0
 0.5

julia> a * a'
2×2 Matrix{Float64}:
 4.0  1.0
 1.0  0.25

julia> A = [0 12 9; 12 7 12; 9 12 0];

julia> a = symcover(A)
3-element Vector{Float64}:
 3.4021999694928753
 3.54528705924512
 3.3847752803845172

julia> a * a'
3×3 Matrix{Float64}:
 11.575   12.0618  11.5157
 12.0618  12.5691  12.0
 11.5157  12.0     11.4567

julia> a = symcover_heuristic(A)
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
function symcover_heuristic(A::AbstractMatrix; exclude_diagonal::Bool=false, kwargs...)
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover_heuristic requires a square matrix"))
    a = similar(A, float(eltype(A)), ax)
    if exclude_diagonal
        fill!(a, zero(T))
    else
        for j in eachindex(a)
            a[j] = sqrt(abs(A[j, j]))
        end
    end
    symcover_makefeasible_diags!(a, A)   # this excludes the diagonal (we already handled if, if we are supposed to)
    return tighten_cover!(a, A; exclude_diagonal, kwargs...)
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

## Internals

function symcover_init_unconstrained(logA::AbstractMatrix, A::AbstractMatrix; exclude_diagonal::Bool=false)
    ax = axes(A, 1)
    alpha = fill!(similar(logA, ax), zero(eltype(logA)))
    nza = zeros(Int, ax)
    for j in ax
        for i in j+exclude_diagonal:last(ax)
            iszero(A[i, j]) && continue
            lAij = logA[i, j]
            alpha[i] += lAij
            nza[i] += 1
            if j != i
                alpha[j] += lAij
                nza[j] += 1
            end
        end
    end
    afrak = copy(alpha)
    u = nza ./ sqrt(sum(nza))
    B = ShermanMorrisonMatrix(Diagonal(nza), u, u)
    ldiv!(B, alpha)
    return alpha, afrak, B
end

function symcover_makefeasible_diags!(a::AbstractVector, A::AbstractMatrix)
    ax = axes(A, 1)
    # Iterate over the diagonals of A, and update a[i] and a[j] to satisfy
    #     |A[i, j]| ≤ a[i] * a[j]
    # whenever this constraint is violated.
    #
    # Iterating over the diagonals gives a more "balanced" result and typically
    # results in lower loss than iterating in a triangular pattern.
    #
    # Unlike the version for alpha below, the caller will have already handled
    # the diagonal, if it is used at all.
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
    return a
end

# Similar to the method above, but operates on alpha, logA, and A
function symcover_makefeasible_diags!(alpha::AbstractVector, logA::AbstractMatrix, A::AbstractMatrix; exclude_diagonal::Bool=false)
    ax = eachindex(alpha)
    if !exclude_diagonal
        for j in ax
            iszero(A[j, j]) && continue
            alpha[j] = max(alpha[j], logA[j, j]/2)
        end
    end
    for k = 1:length(alpha)-1
         for j in first(ax):last(ax)-k
            i = j + k
            iszero(A[i, j]) && continue
            Δα = alpha[i] + alpha[j] - logA[i, j]
            if Δα < zero(Δα)
                alpha[i] -= Δα/2
                alpha[j] -= Δα/2
            end
        end
    end
    return alpha
end

# An alternative to the above, moving along the `e` axis until all constraints are satisfied
function symcover_makefeasible_shift!(alpha, logA, A)
    dalpha = zero(eltype(alpha))
    ax = eachindex(alpha)
    for j in ax
        alphaj = alpha[j]
        for i in j:last(ax)
            Aij = abs(A[i, j])
            iszero(Aij) && continue
            Δα = alpha[i] + alphaj - logA[i, j]
            dalpha = max(dalpha, -Δα/2)
        end
    end
    alpha .+= dalpha
    return alpha
end

function simplex!(
        alpha::AbstractVector{T}, logA, A, alphastar, B, afrak;
        exclude_diagonal::Bool=false, itersimplex=length(alpha), activemax=length(alpha)+1, tol=sqrt(eps(T))
    ) where T
    ax = eachindex(alpha)
    p, g = similar(alpha), similar(alpha)
    actives = sizehint!(Vector{Tuple{Int, Int}}(), activemax)
    # We'll declare J to be of its largest possible size, but many rows may be all-zeros
    J = LinearOperator{T, Vector{T}}(activemax, length(ax), false, false,
        (y, x) -> begin
            fill!(y, zero(T))
            for (k, (i, j)) in enumerate(actives)
                y[k] = x[i] + x[j]
            end
            return y
        end,
        (y, x) -> begin
            fill!(y, zero(T))
            for (k, (i, j)) in enumerate(actives)
                y[i] += x[k]
                y[j] += x[k]
            end
            return y
        end
    )
    # Declare the workspace for lslq
    ws = LslqWorkspace(length(ax), activemax, Vector{T})
    # Initialization and convergence check
    alphatol = tol * sum(isfinite(αstari) ? abs(αi - αstari) : zero(αi) for (αi, αstari) in zip(alpha, alphastar))
    objval = cover_qobjective(alpha, logA, A)
    # Simplex method
    for it = 1:itersimplex
        # @show objval
        # Compute the step direction
        copyto!(p, alphastar)
        p .-= alpha
        # Find the active constraints
        empty!(actives)
        mul!(g, B, alpha)
        for j in eachindex(alpha)
            alphaj = alpha[j]
            for i in j+exclude_diagonal:last(eachindex(alpha))
                iszero(A[i, j]) && continue
                Δα = alpha[i] + alphaj - logA[i, j]
                if Δα < tol
                    push!(actives, (i, j))
                end
            end
        end
        length(actives) > activemax && break
        g .-= afrak
        lslq!(ws, transpose(J), g; M=B, ldiv=true)
        nu, stats = Krylov.results(ws)
        Krylov.issolved(ws) || @warn "LSLQ did not solve successfully (status = $stats)"
        mul!(g, J', nu)   # re-use `g`, which no longer needs to hold the gradient and can just be temporary storage
        p .+= B \ g
        norm(p) < alphatol && break
        γ = symcover_maxstep(p, alpha, logA, A; tol)
        γ = min(γ, 1)
        g .= alpha .+ γ .* p
        objvalnew = cover_qobjective(g, logA, A)
        objvalnew > objval && break
        alpha .+= γ .* p
        objval = objvalnew
    end
    return alpha
end

function symcover_maxstep(p, alpha::AbstractVector{T}, logA, A; tol=sqrt(eps(T))) where T
    ax = eachindex(alpha)
    step = typemax(eltype(alpha))
    for j in eachindex(alpha)
        alphaj = alpha[j]
        pj = p[j]
        for i in j:last(ax)
            iszero(A[i, j]) && continue
            Δα = alpha[i] + alphaj - logA[i, j]
            Δα < tol && continue # should already be handled by the projection
            Δp = p[i] + pj
            if Δp < zero(Δp)
                step = min(step, abs(Δα) / abs(Δp))
            end
        end
    end
    return step
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

function tighten_cover!(alpha::AbstractVector{T}, logA::AbstractMatrix, A::AbstractMatrix; iter::Int=3, exclude_diagonal::Bool=false) where T
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("`tighten_cover!(alpha, logA, A)` requires a square matrix `A`"))
    eachindex(alpha) == ax || throw(DimensionMismatch("indices of `alpha` must match the indexing of `A`"))
    alphagap = similar(alpha)
    for _ in 1:iter
        fill!(alphagap, typemax(T))
        for j in ax
            alphagapj, alphaj = alphagap[j], alpha[j]
            for i in ax
                exclude_diagonal && i == j && continue
                iszero(A[i, j]) && continue
                gap = alpha[i] + alphaj - logA[i, j]
                alphagap[i] = min(alphagap[i], gap)
                alphagapj = min(alphagapj, gap)
            end
            alphagap[j] = alphagapj
        end
        alpha .-= alphagap ./ 2
    end
    return alpha
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


# ============================================================
# Adjoint and Transpose wrappers
#
# If B = adjoint(P) or transpose(P), then |B[i,j]| = |P[j,i]|, so a cover
# (a, b) of B satisfies a[i]*b[j] >= |P[j,i]|, which is exactly a cover
# (b, a) of P.  Therefore: unwrap the parent, compute its cover, and swap.
# ============================================================

cover_lobjective(a::AbstractVector, b::AbstractVector, A::Adjoint)   = cover_lobjective(b, a, parent(A))
cover_lobjective(a::AbstractVector, b::AbstractVector, A::Transpose) = cover_lobjective(b, a, parent(A))

cover_qobjective(a::AbstractVector, b::AbstractVector, A::Adjoint)   = cover_qobjective(b, a, parent(A))
cover_qobjective(a::AbstractVector, b::AbstractVector, A::Transpose) = cover_qobjective(b, a, parent(A))

function tighten_cover!(a::AbstractVector{T}, b::AbstractVector{T}, A::Adjoint; kwargs...) where T
    tighten_cover!(b, a, parent(A); kwargs...)
    return a, b
end
function tighten_cover!(a::AbstractVector{T}, b::AbstractVector{T}, A::Transpose; kwargs...) where T
    tighten_cover!(b, a, parent(A); kwargs...)
    return a, b
end

function cover(A::Adjoint; kwargs...)
    a, b = cover(parent(A); kwargs...)
    return b, a
end
function cover(A::Transpose; kwargs...)
    a, b = cover(parent(A); kwargs...)
    return b, a
end
