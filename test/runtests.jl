using ScaleInvariantAnalysis
using ScaleInvariantAnalysis: divmag, dotabs
using JuMP, HiGHS   # triggers the SIAJuMP extension (symcover_lmin, cover_lmin, etc.)
using LinearAlgebra
using Statistics: median
using Test

@testset "ScaleInvariantAnalysis.jl" begin
    @testset "symcover" begin
        # Cover property: a[i]*a[j] >= abs(A[i,j]) for all i, j
        for A in ([2.0 1.0; 1.0 3.0], [1.0 -0.2; -0.2 0.0], [1.0 0.0; 0.0 0.0],
                  [100.0 1.0; 1.0 0.01])
            a = symcover(A)
            @test all(a[i] * a[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
        end
        # Zero diagonal gives zero scale
        a = symcover([1.0 0; 0 0])
        @test a[2] == 0
        # Diagonal scaling covariance: symcover(D*A*D') ≈ d .* symcover(A)
        A = [2.0 1.0; 1.0 3.0]
        d = [2.0, 0.5]
        @test symcover(A .* d .* d') ≈ d .* symcover(A)
        # Non-square input is rejected
        @test_throws ArgumentError symcover([1.0 2.0; 3.0 4.0; 5.0 6.0])
    end

    @testset "cover" begin
        # Cover property: a[i]*b[j] >= abs(A[i,j]) for all i, j
        for A in ([2.0 1.0; 1.0 3.0], [0.0 1.0; -2.0 0.0], [1.0 0.0; 0.0 0.0],
                  [100.0 1.0; 1.0 0.01])
            a, b = cover(A)
            @test all(a[i] * b[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
        end
        # Zero column gives zero b
        a, b = cover([1.0 0; 0 0])
        @test b[2] == 0
        # Rectangular matrix
        A = [1.0 2.0 3.0; 4.0 5.0 6.0]
        a, b = cover(A)
        @test all(a[i] * b[j] >= abs(A[i, j]) - 1e-12 for i in axes(A, 1), j in axes(A, 2))
        # Diagonal scaling covariance
        A = [2.0 1.0; 1.0 3.0]
        dr, dc = [2.0, 0.5], [3.0, 0.25]
        a, b = cover(A .* dr .* dc')
        a0, b0 = cover(A)
        c = a \ (dr .* a0)
        @test c * a ≈ dr .* a0
        @test b / c ≈ dc .* b0
    end

    @testset "cover_lobjective and cover_qobjective" begin
        A = [4.0 2.0; 2.0 1.0]
        a = symcover(A)
        # Non-negative (since a[i]*a[j] >= |A[i,j]|)
        @test cover_lobjective(a, A) >= 0
        @test cover_qobjective(a, A) >= 0
        # Two-argument form equals one-argument form
        @test cover_lobjective(a, a, A) == cover_lobjective(a, A)
        @test cover_qobjective(a, a, A) == cover_qobjective(a, A)
        # Explicit formula check
        a2, b2 = cover(A)
        @test cover_lobjective(a2, b2, A) ≈ sum(log(a2[i] * b2[j] / abs(A[i, j])) for i in 1:2, j in 1:2 if A[i, j] != 0)
        @test cover_qobjective(a2, b2, A) ≈ sum(log(a2[i] * b2[j] / abs(A[i, j]))^2 for i in 1:2, j in 1:2 if A[i, j] != 0)
        # Zeros in A are skipped; tight diagonal cover gives zero cover_lobjective
        A0 = [1.0 0.0; 0.0 4.0]
        a0 = symcover(A0)
        @test cover_lobjective(a0, A0) == 0.0
    end

    @testset "dotabs" begin
        @test dotabs([1.0, -2.0, 3.0], [4.0, 5.0, -6.0]) ≈ 4.0 + 10.0 + 18.0
        @test dotabs([0.0, 1.0], [1.0, 0.0]) == 0.0
        @test dotabs(big.([1.0, 2.0]), big.([3.0, 4.0])) ≈ 3.0 + 8.0
    end

    @testset "divmag" begin
        A = [1.0 -0.2; -0.2 0]
        b = [0.75, 7.0]
        a, mag = divmag(A, b)
        @test abs(dotabs(A \ b, a) - dotabs(big.(A) \ big.(b), a)) <= 2 * eps(mag)
        # Uniform scaling covariance
        asc, magsc = divmag(1000 * A, 1000 * b)
        @test asc ≈ sqrt(1000) .* a
        @test magsc ≈ sqrt(1000) * mag
        # Diagonal scaling covariance
        for _ in 1:10
            d = -log.(rand(2))
            Ad = A .* d .* d'
            bd = b .* d
            a_d, mag_d = divmag(Ad, bd)
            @test abs(dotabs(Ad \ bd, a_d) - dotabs(big.(Ad) \ big.(bd), a_d)) <= 100 * eps(mag_d)
        end
        # Ill-conditioned matrix: unscaled divmag gives poor accuracy
        A_ill = [1.0 -0.9999; -0.9999 1]
        a_ill, mag_ill = divmag(A_ill, b)
        @test abs(dotabs(A_ill \ b, a_ill) - dotabs(big.(A_ill) \ big.(b), a_ill)) > 10^6 * eps(mag_ill)
        # With cond=true, accuracy is recovered
        a_ill2, mag_ill2 = divmag(A_ill, b; cond=true)
        @test abs(dotabs(A_ill \ b, a_ill2) - dotabs(big.(A_ill) \ big.(b), a_ill2)) <= 10^3 * eps(mag_ill2)
    end

    @testset "symcover_qmin and cover_qmin" begin
        for A in ([2.0 1.0; 1.0 3.0], [100.0 1.0; 1.0 0.01], [4.0 2.0 1.0; 2.0 3.0 2.0; 1.0 2.0 5.0])
            a_fast = symcover(A)
            a_lmin = symcover_lmin(A)
            a_qmin = symcover_qmin(A)
            # qmin is a valid cover
            @test all(a_qmin[i] * a_qmin[j] >= abs(A[i, j]) - 1e-10 for i in axes(A, 1), j in axes(A, 2))
            # qmin achieves lower or equal cover_qobjective than both symcover and symcover_lmin
            @test cover_qobjective(a_qmin, A) <= cover_qobjective(a_fast, A) + 1e-8
            @test cover_qobjective(a_qmin, A) <= cover_qobjective(a_lmin, A) + 1e-8
        end
        for A in ([2.0 1.0; 1.0 3.0], [100.0 1.0; 0.5 0.01], [1.0 2.0 3.0; 4.0 5.0 6.0])
            a_fast, b_fast = cover(A)
            a_lmin, b_lmin = cover_lmin(A)
            a_qmin, b_qmin = cover_qmin(A)
            # qmin is a valid cover
            @test all(a_qmin[i] * b_qmin[j] >= abs(A[i, j]) - 1e-10 for i in axes(A, 1), j in axes(A, 2))
            # qmin achieves lower or equal cover_qobjective than both cover and cover_lmin
            @test cover_qobjective(a_qmin, b_qmin, A) <= cover_qobjective(a_fast, b_fast, A) + 1e-8
            @test cover_qobjective(a_qmin, b_qmin, A) <= cover_qobjective(a_lmin, b_lmin, A) + 1e-8
        end
    end

    @testset "quality vs optimal (testmatrices)" begin
        if !isdefined(@__MODULE__, :symmetric_matrices) || !isdefined(@__MODULE__, :general_matrices)
            include("testmatrices.jl")   # defines symmetric_matrices and general_matrices
        end

        # symcover cover_lobjective should be close to optimal (symcover_lmin)
        sym_ratios = Float64[]
        for (_, A) in symmetric_matrices
            Af = Float64.(A)
            lopt  = cover_lobjective(symcover_lmin(Af), Af)
            lfast = cover_lobjective(symcover(Af; iter=10), Af)
            iszero(lopt) || push!(sym_ratios, lfast / lopt)
        end
        @test median(sym_ratios) < 1.01

        # cover cover_lobjective should be close to optimal (cover_lmin)
        gen_ratios = Float64[]
        for (_, A) in general_matrices
            Af = Float64.(A)
            al, bl = cover_lmin(Af)
            a,  b  = cover(Af; iter=10)
            lopt  = cover_lobjective(al, bl, Af)
            lfast = cover_lobjective(a,  b,  Af)
            iszero(lopt) || push!(gen_ratios, lfast / lopt)
        end
        @test median(gen_ratios) < 1.1
    end
end
