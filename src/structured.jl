# Cover methods for structured LinearAlgebra matrix types.
#
# Diagonal is handled individually.
# SymTridiagonal, Bidiagonal, and Tridiagonal share a single implementation
# via the PlusMinus1Banded union: all are square with entries only on the main
# diagonal and the ±1 off-diagonals.  Structural zeros (e.g. the sub-diagonal
# of an upper Bidiagonal) are returned as exact zero by getindex, so they are
# skipped cheaply by the iszero checks.

const PlusMinus1Banded = Union{SymTridiagonal, Bidiagonal, Tridiagonal}

# ============================================================
# Diagonal
# ============================================================

function cover_lobjective(a::AbstractVector, b::AbstractVector, D::Diagonal)
    T = float(promote_type(eltype(a), eltype(b), eltype(D)))
    s = zero(T)
    for i in eachindex(D.diag)
        Dii = abs(D.diag[i])
        iszero(Dii) && continue
        s += log(T(a[i]) * T(b[i]) / T(Dii))
    end
    return s
end

function cover_qobjective(a::AbstractVector, b::AbstractVector, D::Diagonal)
    T = float(promote_type(eltype(a), eltype(b), eltype(D)))
    s = zero(T)
    for i in eachindex(D.diag)
        Dii = abs(D.diag[i])
        iszero(Dii) && continue
        s += log(T(a[i]) * T(b[i]) / T(Dii))^2
    end
    return s
end

function tighten_cover!(a::AbstractVector{T}, D::Diagonal; iter::Int=3, exclude_diagonal::Bool=false) where T
    exclude_diagonal && return a
    for i in eachindex(a, D.diag)
        Dii = T(abs(D.diag[i]))
        iszero(Dii) || (a[i] = sqrt(Dii))
    end
    return a
end

function tighten_cover!(a::AbstractVector{T}, b::AbstractVector{T}, D::Diagonal; iter::Int=3) where T
    for i in eachindex(a, b, D.diag)
        Dii = T(abs(D.diag[i]))
        iszero(Dii) && continue
        aprod = a[i] * b[i]
        iszero(aprod) && continue
        s = sqrt(aprod / Dii)
        a[i] /= s
        b[i] /= s
    end
    return a, b
end

function symcover(D::Diagonal; exclude_diagonal::Bool=false, kwargs...)
    T = float(eltype(D))
    a = zeros(T, length(D.diag))
    exclude_diagonal && return a
    for i in eachindex(a, D.diag)
        a[i] = sqrt(T(abs(D.diag[i])))
    end
    return a
end

function symdiagcover(D::Diagonal; kwargs...)
    T = float(eltype(D))
    return T.(abs.(D.diag)), zeros(T, length(D.diag))
end

function cover(D::Diagonal; kwargs...)
    a = symcover(D)
    return tighten_cover!(a, copy(a), D; kwargs...)
end

# ============================================================
# PlusMinus1Banded  (SymTridiagonal, Bidiagonal, Tridiagonal)
# ============================================================

function cover_lobjective(a::AbstractVector, b::AbstractVector, A::PlusMinus1Banded)
    T = float(promote_type(eltype(a), eltype(b), eltype(A)))
    s = zero(T)
    n = size(A, 1)
    for i in 1:n
        Aii = abs(A[i, i])
        iszero(Aii) || (s += log(T(a[i]) * T(b[i]) / T(Aii)))
    end
    for i in 1:n-1
        Aij = abs(A[i, i+1])
        iszero(Aij) || (s += log(T(a[i]) * T(b[i+1]) / T(Aij)))
        Aij = abs(A[i+1, i])
        iszero(Aij) || (s += log(T(a[i+1]) * T(b[i]) / T(Aij)))
    end
    return s
end

function cover_qobjective(a::AbstractVector, b::AbstractVector, A::PlusMinus1Banded)
    T = float(promote_type(eltype(a), eltype(b), eltype(A)))
    s = zero(T)
    n = size(A, 1)
    for i in 1:n
        Aii = abs(A[i, i])
        iszero(Aii) || (s += log(T(a[i]) * T(b[i]) / T(Aii))^2)
    end
    for i in 1:n-1
        Aij = abs(A[i, i+1])
        iszero(Aij) || (s += log(T(a[i]) * T(b[i+1]) / T(Aij))^2)
        Aij = abs(A[i+1, i])
        iszero(Aij) || (s += log(T(a[i+1]) * T(b[i]) / T(Aij))^2)
    end
    return s
end

# Symmetric tighten: both A[i,i+1] and A[i+1,i] are checked independently.
# For SymTridiagonal they are equal (no-op redundancy); for a Tridiagonal that
# the caller asserts is symmetric, using both is consistent with the general
# tighten_cover!(a, A::AbstractMatrix) which iterates over every entry.
function tighten_cover!(a::AbstractVector{T}, A::PlusMinus1Banded; iter::Int=3, exclude_diagonal::Bool=false) where T
    n = size(A, 1)
    aratio = similar(a)
    for _ in 1:iter
        fill!(aratio, typemax(T))
        if !exclude_diagonal
            for i in 1:n
                Aii = T(abs(A[i, i]))
                iszero(Aii) && continue
                aratio[i] = min(aratio[i], a[i]^2 / Aii)
            end
        end
        for i in 1:n-1
            Aij = T(abs(A[i, i+1]))
            if !iszero(Aij)
                r = a[i] * a[i+1] / Aij
                aratio[i]   = min(aratio[i],   r)
                aratio[i+1] = min(aratio[i+1], r)
            end
            Aij = T(abs(A[i+1, i]))
            if !iszero(Aij)
                r = a[i] * a[i+1] / Aij
                aratio[i]   = min(aratio[i],   r)
                aratio[i+1] = min(aratio[i+1], r)
            end
        end
        a ./= sqrt.(aratio)
    end
    return a
end

# Asymmetric tighten for all ±1-banded types.
function tighten_cover!(a::AbstractVector{T}, b::AbstractVector{T}, A::PlusMinus1Banded; iter::Int=3) where T
    n = size(A, 1)
    aratio = similar(a)
    bratio = similar(b)
    for _ in 1:iter
        fill!(aratio, typemax(T))
        fill!(bratio, typemax(T))
        for i in 1:n
            Aii = T(abs(A[i, i]))
            iszero(Aii) && continue
            r = a[i] * b[i] / Aii
            aratio[i] = min(aratio[i], r)
            bratio[i] = min(bratio[i], r)
        end
        for i in 1:n-1
            Aij = T(abs(A[i, i+1]))
            if !iszero(Aij)
                r = a[i] * b[i+1] / Aij
                aratio[i]   = min(aratio[i],   r)
                bratio[i+1] = min(bratio[i+1], r)
            end
            Aij = T(abs(A[i+1, i]))
            if !iszero(Aij)
                r = a[i+1] * b[i] / Aij
                aratio[i+1] = min(aratio[i+1], r)
                bratio[i]   = min(bratio[i],   r)
            end
        end
        a ./= sqrt.(aratio)
        b ./= sqrt.(bratio)
    end
    return a, b
end

# symcover and symdiagcover apply to any PlusMinus1Banded matrix; the caller
# asserts that A is symmetric in value (or accepts that only the upper triangle
# is used for initialization).
function symcover(A::PlusMinus1Banded; exclude_diagonal::Bool=false, prioritize::Symbol=:quality, kwargs...)
    prioritize in (:quality, :speed) || throw(ArgumentError("prioritize must be :quality or :speed"))
    n = size(A, 1)
    T = float(eltype(A))
    a = zeros(T, n)
    if prioritize == :quality
        loga = zeros(T, n)
        nza  = zeros(Int, n)
        if !exclude_diagonal
            for i in 1:n
                Aii = abs(A[i, i])
                iszero(Aii) && continue
                loga[i] += log(Aii)
                nza[i]  += 1
            end
        end
        for i in 1:n-1
            Aij = abs(A[i, i+1])   # upper triangle; caller asserts A[i+1,i] matches
            iszero(Aij) && continue
            lAij = log(Aij)
            loga[i]   += lAij;  nza[i]   += 1
            loga[i+1] += lAij;  nza[i+1] += 1
        end
        nztotal = sum(nza)
        halfmu = iszero(nztotal) ? zero(T) : sum(loga) / (2 * nztotal)
        for i in 1:n
            a[i] = iszero(nza[i]) ? zero(T) : exp(loga[i] / nza[i] - halfmu)
        end
        if !exclude_diagonal
            for i in 1:n
                Aii = T(abs(A[i, i]))
                a[i]^2 < Aii && (a[i] = sqrt(Aii))
            end
        end
    else  # :speed
        if !exclude_diagonal
            for i in 1:n
                a[i] = sqrt(T(abs(A[i, i])))
            end
        end
    end
    # Boost from off-diagonal entries (upper triangle)
    for i in 1:n-1
        Aij = T(abs(A[i, i+1]))
        iszero(Aij) && continue
        ai, aj = a[i], a[i+1]
        if iszero(aj)
            iszero(ai) ? (a[i] = a[i+1] = sqrt(Aij)) : (a[i+1] = Aij / ai)
        elseif iszero(ai)
            a[i] = Aij / aj
        else
            aprod = ai * aj
            if aprod < Aij
                s = sqrt(Aij / aprod)
                a[i] *= s;  a[i+1] *= s
            end
        end
    end
    return tighten_cover!(a, A; iter=get(kwargs, :iter, 3), exclude_diagonal)
end

function symdiagcover(A::PlusMinus1Banded; kwargs...)
    a = symcover(A; exclude_diagonal=true, kwargs...)
    T = float(eltype(A))
    d = map(1:size(A, 1)) do i
        Aii = T(abs(A[i, i]))
        max(zero(T), Aii - a[i]^2)
    end
    return d, a
end

function cover(A::PlusMinus1Banded; kwargs...)
    T = float(eltype(A))
    n = size(A, 1)
    a = zeros(T, n)
    b = zeros(T, n)
    loga = zeros(T, n)
    logb = zeros(T, n)
    nza  = zeros(Int, n)
    nzb  = zeros(Int, n)
    logmu   = zero(T)
    nztotal = 0
    for i in 1:n
        Aii = abs(A[i, i])
        iszero(Aii) && continue
        lAii = log(T(Aii))
        loga[i] += lAii;  logb[i] += lAii
        nza[i]  += 1;     nzb[i]  += 1
        logmu   += lAii;  nztotal += 1
    end
    for i in 1:n-1
        Aij = abs(A[i, i+1])
        if !iszero(Aij)
            lAij = log(T(Aij))
            loga[i]   += lAij;  logb[i+1] += lAij
            nza[i]    += 1;     nzb[i+1]  += 1
            logmu     += lAij;  nztotal   += 1
        end
        Aij = abs(A[i+1, i])
        if !iszero(Aij)
            lAij = log(T(Aij))
            loga[i+1] += lAij;  logb[i] += lAij
            nza[i+1]  += 1;     nzb[i]  += 1
            logmu     += lAij;  nztotal += 1
        end
    end
    halfmu = iszero(nztotal) ? zero(T) : T(logmu / (2 * nztotal))
    for i in 1:n
        a[i] = iszero(nza[i]) ? zero(T) : exp(loga[i] / nza[i] - halfmu)
        b[i] = iszero(nzb[i]) ? zero(T) : exp(logb[i] / nzb[i] - halfmu)
    end
    # Boost diagonal
    for i in 1:n
        Aii = T(abs(A[i, i]))
        aprod = a[i] * b[i]
        aprod >= Aii && continue
        s = sqrt(Aii / aprod)
        a[i] *= s;  b[i] *= s
    end
    # Boost off-diagonal
    for i in 1:n-1
        Aij = T(abs(A[i, i+1]))
        if !iszero(Aij)
            aprod = a[i] * b[i+1]
            if aprod < Aij
                s = sqrt(Aij / aprod)
                a[i] *= s;  b[i+1] *= s
            end
        end
        Aij = T(abs(A[i+1, i]))
        if !iszero(Aij)
            aprod = a[i+1] * b[i]
            if aprod < Aij
                s = sqrt(Aij / aprod)
                a[i+1] *= s;  b[i] *= s
            end
        end
    end
    return tighten_cover!(a, b, A; kwargs...)
end
