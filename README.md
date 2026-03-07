# ScaleInvariantAnalysis

<!--- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/ScaleInvariantAnalysis.jl/stable/) --->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/ScaleInvariantAnalysis.jl/dev/)
[![Build Status](https://github.com/HolyLab/ScaleInvariantAnalysis.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/ScaleInvariantAnalysis.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/ScaleInvariantAnalysis.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HolyLab/ScaleInvariantAnalysis.jl)

This package computes **covers** of matrices: non-negative vectors `a` (and `b`)
such that `a[i] * b[j] >= abs(A[i, j])` for all `i`, `j`.  Covers are the
natural scale-covariant representation of a matrix — under row/column diagonal
scaling they transform exactly as the matrix entries do — making them a useful
building block for scale-invariant numerical analysis.

Fast O(n²) heuristics (`symcover`, `cover`) are provided for everyday use.
Globally optimal covers minimizing a log-domain linear or quadratic objective
(`symcover_lmin`, `cover_lmin`, `symcover_qmin`, `cover_qmin`) are available
as an extension when JuMP and HiGHS are loaded.

See the [documentation](https://HolyLab.github.io/ScaleInvariantAnalysis.jl/dev/)
for motivation, examples, and a full API reference.
