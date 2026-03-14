module SIASparseArrays

using LinearAlgebra
using SparseArrays
using ScaleInvariantAnalysis

# ============================================================
# Private helpers (operate on a plain SparseMatrixCSC parent)
# ============================================================

# Quadratic initialisation of the symmetric cover.
# `uplo` controls which triangle is treated as canonical ('U' → i ≤ j, 'L' → i ≥ j).
# For a fully-stored SparseMatrixCSC, use 'U' to avoid double-counting.
function _sparse_symcover_init_quadratic!(a::AbstractVector{T}, P::SparseMatrixCSC, uplo::Char; exclude_diagonal::Bool=false) where T
    rv = rowvals(P)
    nzs = nonzeros(P)
    loga = zeros(T, size(P, 1))
    nza  = zeros(Int, size(P, 1))
    for j in axes(P, 2)
        for k in nzrange(P, j)
            i = rv[k]
            (uplo == 'U' ? i > j : i < j) && continue   # canonical triangle only
            exclude_diagonal && i == j && continue
            Aij = abs(nzs[k])
            iszero(Aij) && continue
            lAij = log(Aij)
            loga[i] += lAij
            nza[i] += 1
            if i != j
                loga[j] += lAij
                nza[j] += 1
            end
        end
    end
    nztotal = sum(nza)
    halfmu = iszero(nztotal) ? zero(T) : sum(loga) / (2 * nztotal)
    for i in eachindex(a)
        a[i] = iszero(nza[i]) ? zero(T) : exp(loga[i] / nza[i] - halfmu)
    end
    if !exclude_diagonal
        # Enforce a[i]^2 >= |A[i,i]| for stored diagonal entries
        for j in axes(P, 2)
            for k in nzrange(P, j)
                i = rv[k]
                i != j && continue
                Aii = abs(nzs[k])
                if a[i]^2 < Aii
                    a[i] = sqrt(Aii)
                end
                break
            end
        end
    end
    return a
end

# Fast (diagonal-based) initialisation of the symmetric cover.
function _sparse_symcover_init_fast!(a::AbstractVector{T}, P::SparseMatrixCSC; exclude_diagonal::Bool=false) where T
    fill!(a, zero(T))
    exclude_diagonal && return a
    rv = rowvals(P)
    nzs = nonzeros(P)
    for j in axes(P, 2)
        for k in nzrange(P, j)
            i = rv[k]
            if i == j
                a[j] = sqrt(abs(nzs[k]))
                break
            end
        end
    end
    return a
end

# Boost: for each stored off-diagonal entry in the canonical triangle, ensure a[i]*a[j] >= |A[i,j]|.
function _sparse_symcover_boost!(a::AbstractVector, P::SparseMatrixCSC, uplo::Char)
    rv = rowvals(P)
    nzs = nonzeros(P)
    for j in axes(P, 2)
        for k in nzrange(P, j)
            i = rv[k]
            i == j && continue                          # skip diagonal
            (uplo == 'U' ? i > j : i < j) && continue  # canonical off-diagonal only
            Aij = abs(nzs[k])
            ai, aj = a[i], a[j]
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
end

# Tighten the symmetric cover, iterating only stored nonzeros.
# Works for fully-stored symmetric matrices and for one-triangle-stored Symmetric wrappers:
# for each column j, updating both aratio[i] (directly) and aratio[j] (via aratioj) correctly
# propagates the constraint a[i]*a[j] >= |A[i,j]| to both sides.
function _tighten_cover_sym_sparse!(a::AbstractVector{T}, P::SparseMatrixCSC; iter::Int, exclude_diagonal::Bool) where T
    rv = rowvals(P)
    nzs = nonzeros(P)
    aratio = similar(a)
    for _ in 1:iter
        fill!(aratio, typemax(T))
        for j in axes(P, 2)
            aratioj = aratio[j]
            aj = a[j]
            for k in nzrange(P, j)
                i = rv[k]
                (exclude_diagonal && i == j) && continue
                Aij = T(abs(nzs[k]))
                r = ifelse(iszero(Aij), typemax(T), a[i] * aj / Aij)
                aratio[i] = min(aratio[i], r)
                aratioj = min(aratioj, r)
            end
            aratio[j] = aratioj
        end
        a ./= sqrt.(aratio)
    end
    return a
end

# Tighten the asymmetric cover, iterating only stored nonzeros.
function _tighten_cover_asym_sparse!(a::AbstractVector{T}, b::AbstractVector{T}, P::SparseMatrixCSC; iter::Int) where T
    rv = rowvals(P)
    nzs = nonzeros(P)
    aratio = fill(typemax(T), eachindex(a))
    bratio = fill(typemax(T), eachindex(b))
    for _ in 1:iter
        fill!(aratio, typemax(T))
        fill!(bratio, typemax(T))
        for j in eachindex(b)
            bratioj = bratio[j]
            bj = b[j]
            for k in nzrange(P, j)
                i = rv[k]
                Aij = T(abs(nzs[k]))
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

# Accumulate objective sums over the canonical triangle of a symmetric matrix.
# For each off-diagonal stored entry (i,j), both (i,j) and (j,i) contribute to the sum.
function _cover_lobjective_sym_sparse(a, b, P::SparseMatrixCSC, uplo::Char)
    rv = rowvals(P)
    nzs = nonzeros(P)
    s = zero(float(promote_type(eltype(a), eltype(b), eltype(P))))
    for j in axes(P, 2)
        bj = b[j]
        for k in nzrange(P, j)
            i = rv[k]
            (uplo == 'U' ? i > j : i < j) && continue
            Aij = abs(nzs[k])
            iszero(Aij) && continue
            s += log(a[i] * bj / Aij)
            if i != j
                s += log(a[j] * b[i] / Aij)
            end
        end
    end
    return s
end

function _cover_qobjective_sym_sparse(a, b, P::SparseMatrixCSC, uplo::Char)
    rv = rowvals(P)
    nzs = nonzeros(P)
    s = zero(float(promote_type(eltype(a), eltype(b), eltype(P))))
    for j in axes(P, 2)
        bj = b[j]
        for k in nzrange(P, j)
            i = rv[k]
            (uplo == 'U' ? i > j : i < j) && continue
            Aij = abs(nzs[k])
            iszero(Aij) && continue
            s += log(a[i] * bj / Aij)^2
            if i != j
                s += log(a[j] * b[i] / Aij)^2
            end
        end
    end
    return s
end

# Extract the diagonal correction vector for symdiagcover.
function _symdiagcover_d!(d::AbstractVector{T}, a::AbstractVector, P::SparseMatrixCSC) where T
    rv = rowvals(P)
    nzs = nonzeros(P)
    for j in axes(P, 2)
        for k in nzrange(P, j)
            i = rv[k]
            if i == j
                Aii = abs(nzs[k])
                d[i] = max(zero(T), Aii - a[i]^2)
                break
            end
        end
    end
end

# ============================================================
# SparseMatrixCSC methods
# ============================================================

function ScaleInvariantAnalysis.cover_lobjective(a::AbstractVector, b::AbstractVector, A::SparseMatrixCSC)
    rv = rowvals(A)
    nzs = nonzeros(A)
    s = zero(float(promote_type(eltype(a), eltype(b), eltype(A))))
    for j in axes(A, 2)
        bj = b[j]
        for k in nzrange(A, j)
            Aij = abs(nzs[k])
            iszero(Aij) && continue
            s += log(a[rv[k]] * bj / Aij)
        end
    end
    return s
end

function ScaleInvariantAnalysis.cover_qobjective(a::AbstractVector, b::AbstractVector, A::SparseMatrixCSC)
    rv = rowvals(A)
    nzs = nonzeros(A)
    s = zero(float(promote_type(eltype(a), eltype(b), eltype(A))))
    for j in axes(A, 2)
        bj = b[j]
        for k in nzrange(A, j)
            Aij = abs(nzs[k])
            iszero(Aij) && continue
            s += log(a[rv[k]] * bj / Aij)^2
        end
    end
    return s
end

function ScaleInvariantAnalysis.tighten_cover!(a::AbstractVector{T}, A::SparseMatrixCSC; iter::Int=3, exclude_diagonal::Bool=false) where T
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("`tighten_cover!(a, A)` requires a square matrix `A`"))
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`"))
    return _tighten_cover_sym_sparse!(a, A; iter, exclude_diagonal)
end

function ScaleInvariantAnalysis.tighten_cover!(a::AbstractVector{T}, b::AbstractVector{T}, A::SparseMatrixCSC; iter::Int=3) where T
    eachindex(a) == axes(A, 1) || throw(DimensionMismatch("indices of a must match row-indexing of A"))
    eachindex(b) == axes(A, 2) || throw(DimensionMismatch("indices of b must match column-indexing of A"))
    return _tighten_cover_asym_sparse!(a, b, A; iter)
end

function ScaleInvariantAnalysis.symcover(A::SparseMatrixCSC; exclude_diagonal::Bool=false, prioritize::Symbol=:quality, kwargs...)
    prioritize in (:quality, :speed) || throw(ArgumentError("prioritize must be :quality or :speed"))
    ax = axes(A, 1)
    axes(A, 2) == ax || throw(ArgumentError("symcover requires a square matrix"))
    T = float(eltype(A))
    a = zeros(T, size(A, 1))
    if prioritize == :quality
        _sparse_symcover_init_quadratic!(a, A, 'U'; exclude_diagonal)
    else
        _sparse_symcover_init_fast!(a, A; exclude_diagonal)
    end
    _sparse_symcover_boost!(a, A, 'U')
    return _tighten_cover_sym_sparse!(a, A; iter=get(kwargs, :iter, 3), exclude_diagonal)
end

function ScaleInvariantAnalysis.symdiagcover(A::SparseMatrixCSC; kwargs...)
    axes(A, 1) == axes(A, 2) || throw(ArgumentError("symcover requires a square matrix"))
    a = ScaleInvariantAnalysis.symcover(A; exclude_diagonal=true, kwargs...)
    T = float(eltype(A))
    d = zeros(T, size(A, 1))
    _symdiagcover_d!(d, a, A)
    return d, a
end

function ScaleInvariantAnalysis.cover(A::SparseMatrixCSC; kwargs...)
    T = float(eltype(A))
    a = zeros(T, size(A, 1))
    b = zeros(T, size(A, 2))
    loga = zeros(T, size(A, 1))
    logb = zeros(T, size(A, 2))
    nza  = zeros(Int, size(A, 1))
    nzb  = zeros(Int, size(A, 2))
    logmu = zero(T)
    nztotal = 0
    rv = rowvals(A)
    nzs = nonzeros(A)
    for j in axes(A, 2)
        for k in nzrange(A, j)
            Aij = abs(nzs[k])
            iszero(Aij) && continue
            i = rv[k]
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
    for j in axes(A, 2)
        bj = b[j]
        for k in nzrange(A, j)
            i = rv[k]
            Aij, ai = abs(nzs[k]), a[i]
            aprod = ai * bj
            aprod >= Aij && continue
            s = sqrt(Aij / aprod)
            a[i] = s * ai
            b[j] = bj = s * bj
        end
    end
    return _tighten_cover_asym_sparse!(a, b, A; iter=get(kwargs, :iter, 3))
end

# ============================================================
# Symmetric{<:Any, <:SparseMatrixCSC} methods
# ============================================================

function ScaleInvariantAnalysis.cover_lobjective(a::AbstractVector, b::AbstractVector, S::Symmetric{<:Any, <:SparseMatrixCSC})
    return _cover_lobjective_sym_sparse(a, b, parent(S), S.uplo)
end

function ScaleInvariantAnalysis.cover_qobjective(a::AbstractVector, b::AbstractVector, S::Symmetric{<:Any, <:SparseMatrixCSC})
    return _cover_qobjective_sym_sparse(a, b, parent(S), S.uplo)
end

function ScaleInvariantAnalysis.tighten_cover!(a::AbstractVector{T}, S::Symmetric{<:Any, <:SparseMatrixCSC}; iter::Int=3, exclude_diagonal::Bool=false) where T
    P = parent(S)
    ax = axes(P, 1)
    axes(P, 2) == ax || throw(ArgumentError("`tighten_cover!(a, A)` requires a square matrix `A`"))
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`"))
    return _tighten_cover_sym_sparse!(a, P; iter, exclude_diagonal)
end

function ScaleInvariantAnalysis.symcover(S::Symmetric{<:Any, <:SparseMatrixCSC}; exclude_diagonal::Bool=false, prioritize::Symbol=:quality, kwargs...)
    prioritize in (:quality, :speed) || throw(ArgumentError("prioritize must be :quality or :speed"))
    P = parent(S)
    axes(P, 1) == axes(P, 2) || throw(ArgumentError("symcover requires a square matrix"))
    T = float(eltype(P))
    a = zeros(T, size(P, 1))
    uplo = S.uplo
    if prioritize == :quality
        _sparse_symcover_init_quadratic!(a, P, uplo; exclude_diagonal)
    else
        _sparse_symcover_init_fast!(a, P; exclude_diagonal)
    end
    _sparse_symcover_boost!(a, P, uplo)
    return _tighten_cover_sym_sparse!(a, P; iter=get(kwargs, :iter, 3), exclude_diagonal)
end

function ScaleInvariantAnalysis.symdiagcover(S::Symmetric{<:Any, <:SparseMatrixCSC}; kwargs...)
    P = parent(S)
    axes(P, 1) == axes(P, 2) || throw(ArgumentError("symcover requires a square matrix"))
    a = ScaleInvariantAnalysis.symcover(S; exclude_diagonal=true, kwargs...)
    T = float(eltype(P))
    d = zeros(T, size(P, 1))
    _symdiagcover_d!(d, a, P)
    return d, a
end

# ============================================================
# Hermitian{<:Real, <:SparseMatrixCSC} methods
# (Real-valued Hermitian is equivalent to Symmetric)
# ============================================================

function ScaleInvariantAnalysis.cover_lobjective(a::AbstractVector, b::AbstractVector, H::Hermitian{<:Real, <:SparseMatrixCSC})
    return _cover_lobjective_sym_sparse(a, b, parent(H), H.uplo)
end

function ScaleInvariantAnalysis.cover_qobjective(a::AbstractVector, b::AbstractVector, H::Hermitian{<:Real, <:SparseMatrixCSC})
    return _cover_qobjective_sym_sparse(a, b, parent(H), H.uplo)
end

function ScaleInvariantAnalysis.tighten_cover!(a::AbstractVector{T}, H::Hermitian{<:Real, <:SparseMatrixCSC}; iter::Int=3, exclude_diagonal::Bool=false) where T
    P = parent(H)
    ax = axes(P, 1)
    axes(P, 2) == ax || throw(ArgumentError("`tighten_cover!(a, A)` requires a square matrix `A`"))
    eachindex(a) == ax || throw(DimensionMismatch("indices of `a` must match the indexing of `A`"))
    return _tighten_cover_sym_sparse!(a, P; iter, exclude_diagonal)
end

function ScaleInvariantAnalysis.symcover(H::Hermitian{<:Real, <:SparseMatrixCSC}; exclude_diagonal::Bool=false, prioritize::Symbol=:quality, kwargs...)
    prioritize in (:quality, :speed) || throw(ArgumentError("prioritize must be :quality or :speed"))
    P = parent(H)
    axes(P, 1) == axes(P, 2) || throw(ArgumentError("symcover requires a square matrix"))
    T = float(eltype(P))
    a = zeros(T, size(P, 1))
    uplo = H.uplo
    if prioritize == :quality
        _sparse_symcover_init_quadratic!(a, P, uplo; exclude_diagonal)
    else
        _sparse_symcover_init_fast!(a, P; exclude_diagonal)
    end
    _sparse_symcover_boost!(a, P, uplo)
    return _tighten_cover_sym_sparse!(a, P; iter=get(kwargs, :iter, 3), exclude_diagonal)
end

function ScaleInvariantAnalysis.symdiagcover(H::Hermitian{<:Real, <:SparseMatrixCSC}; kwargs...)
    P = parent(H)
    axes(P, 1) == axes(P, 2) || throw(ArgumentError("symcover requires a square matrix"))
    a = ScaleInvariantAnalysis.symcover(H; exclude_diagonal=true, kwargs...)
    T = float(eltype(P))
    d = zeros(T, size(P, 1))
    _symdiagcover_d!(d, a, P)
    return d, a
end

end  # module SIASparseArrays
