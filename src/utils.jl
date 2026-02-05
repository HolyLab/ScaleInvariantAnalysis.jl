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

# TODO: implement _symscale for SparseMatrixCSC
