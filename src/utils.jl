function _symscale(A, ax)
    sumlogA, nz = fill!(similar(A, Float64, ax), 0), fill!(similar(A, Int, ax), 0)
    for j in ax
        for i in j:last(ax)
            Aij = abs(A[i, j])
            iszero(Aij) && continue
            sumlogA[j] += log(Aij)
            nz[j] += 1
            if i > j
                sumlogA[i] += log(Aij)
                nz[i] += 1
            end
        end
    end
    return sumlogA, nz
end

function _matrixscale(A, ax1, ax2)
    sumlogA1, nz1 = fill!(similar(A, Float64, ax1), 0), fill!(similar(A, Int, ax1), 0)
    sumlogA2, nz2 = fill!(similar(A, Float64, ax2), 0), fill!(similar(A, Int, ax2), 0)
    for j in ax2
        for i in ax1
            Aij = abs(A[i, j])
            iszero(Aij) && continue
            logAij = log(Aij)
            sumlogA1[i] += logAij
            nz1[i] += 1
            sumlogA2[j] += logAij
            nz2[j] += 1
        end
    end
    return (sumlogA1, nz1), (sumlogA2, nz2)
end

isnz(A) = .!iszero.(A)

function divsafe!(sumlog, nz; sentinel=-Inf)
    for i in eachindex(sumlog, nz)
        if iszero(nz[i])
           nz[i] = 1
           sumlog[i] = sentinel
        end
    end
    return sumlog, nz
end

function odblocks(Anz::AbstractMatrix{T}) where T
    m, n = size(Anz)
    return [zeros(T, m, m) Anz; Anz' zeros(T, n, n)]
end

# TODO: implementations for SparseMatrixCSC
