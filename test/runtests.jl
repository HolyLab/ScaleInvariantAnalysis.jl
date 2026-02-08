using ScaleInvariantAnalysis
using LinearAlgebra
using Test

function test_scaleinv(f, A::AbstractMatrix, p::Int; iter=10, rtol=sqrt(eps(float(eltype(A)))))
    npass = 0
    n = size(A, 1)
    @assert size(A, 2) == n "Matrix A must be square"
    a = f(A)
    for _ in 1:iter
        d = -log.(rand(n))
        ad = f(A .* d .* d')
        npass += a .* d .^ p ≈ ad ? npass + 1 : npass
    end
    @test npass ≥ iter - 1
end

function test_sumlog(A, a, b; rtol=1e-6)
    α, β = log.(a), log.(b)
    Aref = sum(abs(log(abs(A[i, j]))) for i in axes(A, 1), j in axes(A, 2) if A[i, j] != 0)
    for j in axes(A, 2)
        s = 0.0
        for i in axes(A, 1)
            if A[i, j] != 0
                s += log(abs(A[i, j])) - α[i] - β[j]
            end
        end
        @test abs(s) ≤ rtol * Aref
    end
    for i in axes(A, 1)
        s = 0.0
        for j in axes(A, 2)
            if A[i, j] != 0
                s += log(abs(A[i, j])) - α[i] - β[j]
            end
        end
        @test abs(s) ≤ rtol * Aref
    end
end

@testset "ScaleInvariantAnalysis.jl" begin
    @test symscale([2.0 1.0; 1.0 3.0]) ≈ symscale([2.0 1.0; 1.0 3.0]; exact=true) ≈ exp.([3 1; 1 3] \ [log(2.0); log(3.0)])
    @test symscale([1.0 -0.2; -0.2 0]; exact=true) ≈ [1, 0.2]
    @test symscale([1.0 0; 0 2]; exact=true) ≈ [1, sqrt(2)]
    test_scaleinv(A -> symscale(A; exact=true), [2.0 1.0; 1.0 3.0], 1)
    a, b = matrixscale([2.0 1.0; 1.0 3.0]; exact=true)
    @test a ≈ b ≈ symscale([2.0 1.0; 1.0 3.0]; exact=true)
    a′, b′ = matrixscale([2.0 1.0; 1.0 3.0])
    @test a′ ≈ a && b′ ≈ b
    A = [0.0 1.0; -2.0 0.0]
    a, b = matrixscale(A; exact=true)
    test_sumlog(A, a, b)
    a′, b′ = matrixscale(A)
    @test sum(log, a) ≈ sum(log, b) ≈ sum(log, a′) ≈ sum(log, b′)

    @test condscale([1 0; 0 1e-8]) ≈ 1
    A = [1.0 -0.2; -0.2 0]
    b = [0.75, 7.0]
    a, mag = divmag(A, b)
    @test abs(dotabs(A \ b, a) - dotabs(big.(A) \ big.(b), a)) <= 2 * eps(mag)
    asc, magsc = divmag(1000*A, 1000*b)
    @test asc ≈ sqrt(1000) .* a
    @test magsc ≈ sqrt(1000) * mag
    for _ = 1:10
        d = -log.(rand(2))
        Ad = A .* d .* d'
        bd = b .* d
        a_d, mag_d = divmag(Ad, bd)
        @test abs(dotabs(Ad \ bd, a_d) - dotabs(big.(Ad) \ big.(bd), a_d)) <= 100 * eps(mag_d)
    end
    A = [1.0 -0.9999; -0.9999 1]   # nearly singular
    @test condscale(A) ≈ 19999
    a, mag = divmag(A, b)
    @test abs(dotabs(A \ b, a) - dotabs(big.(A) \ big.(b), a)) > 10^6 * eps(mag)
    a, mag = divmag(A, b; cond=true)
    @test abs(dotabs(A \ b, a) - dotabs(big.(A) \ big.(b), a)) <= 10^3 * eps(mag)

    # @test diagapprox([1.0 0; 0 2]) ≈ [1, 2]
    # od = 1e-8
    # A = Diagonal([1, 2, 3]) + od * randn(3, 3)
    # d = diagapprox(A; iter=100)
    # @test all(abs.(d .- diag(A)) .< 100 * od)
    # A = Diagonal([1, 2, 0]) + od * randn(3, 3)
    # d = diagapprox(A; iter=100)
    # @test all(abs.(d .- diag(A)) .< 100 * od)
    # test_scaleinv(diagapprox, A, 2; rtol=1e-6)
end
