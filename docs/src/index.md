```@meta
CurrentModule = ScaleInvariantAnalysis
```

# ScaleInvariantAnalysis

This package computes **covers** of matrices.  Given a matrix `A`, a cover is a
pair of non-negative vectors `a`, `b` satisfying

```math
a_i \cdot b_j \;\geq\; |A_{ij}| \quad \text{for all } i, j.
```

For a symmetric matrix the cover is symmetric (`b = a`), so a single vector
suffices: `a[i] * a[j] >= abs(A[i, j])`.

## Why covers?

Covers are the natural **scale-covariant** representation of a matrix.  If you
rescale rows by a positive diagonal factor `D_r` and columns by `D_c`, the
optimal cover transforms as `a → D_r * a`, `b → D_c * b` — exactly mirroring
how the matrix entries change.  Scalar summaries like `norm(A)` or
`maximum(abs, A)` do not have this property and therefore implicitly encode an
arbitrary choice of units.

A concrete example: a 3×3 matrix whose rows and columns correspond to physical
variables at very different scales (position in metres, velocity in m/s, force
in N):

```jldoctest coverones; filter = r"\d+\.\d+" => "≈"
julia> using ScaleInvariantAnalysis

julia> A = [1e6 1e3 1.0; 1e3 1.0 1e-3; 1.0 1e-3 1e-6];

julia> a = symcover(A)
3-element Vector{Float64}:
 1000.0
    1.0
    0.001
```

The cover `a` captures the natural per-variable scale.  The normalised matrix
`A ./ (a .* a')` is all-ones and is scale-invariant.

## Measuring cover quality

A cover is valid as long as every constraint is satisfied, but tighter covers
better capture the scaling of `A`.  The *log-excess* of an entry is
`log(a[i] * b[j] / abs(A[i, j])) >= 0`; it is zero when the constraint is
exactly tight.  Two summary statistics aggregate these excesses:

- [`lobjective`](@ref) — sum of log-excesses (L1 in log space).
- [`qobjective`](@ref) — sum of squared log-excesses (L2 in log space).

Both equal zero if and only if every constraint is tight.

```jldoctest quality; filter = r"\d+\.\d+" => "≈"
julia> using ScaleInvariantAnalysis

julia> A = [4.0 2.0; 2.0 9.0];

julia> a = symcover(A)
2-element Vector{Float64}:
 2.0
 3.0

julia> lobjective(a, A)
2.1972245773362196

julia> qobjective(a, A)
2.413897921625164
```

## Choosing a cover algorithm

| Function | Symmetric | Minimizes | Requires |
|---|---|---|---|
| [`symcover`](@ref) | yes | (fast heuristic) | — |
| [`cover`](@ref) | no | (fast heuristic) | — |
| [`symcover_lmin`](@ref) | yes | `lobjective` | JuMP + HiGHS |
| [`cover_lmin`](@ref) | no | `lobjective` | JuMP + HiGHS |
| [`symcover_qmin`](@ref) | yes | `qobjective` | JuMP + HiGHS |
| [`cover_qmin`](@ref) | no | `qobjective` | JuMP + HiGHS |

**`symcover` and `cover` are recommended for production use.**  They run in
O(n²) time and often land within a few percent of the `lobjective`-optimal
cover (see the quality tests in `test/testmatrices.jl`).

The `*_lmin` and `*_qmin` variants solve a convex program (via
[JuMP](https://jump.dev/) and [HiGHS](https://highs.dev/)) and return a
global optimum of the respective objective.  They are loaded on demand as a
package extension — simply load both libraries before calling them:

```julia
using JuMP, HiGHS
using ScaleInvariantAnalysis

a      = symcover_lmin(A)     # globally l-minimal symmetric cover
a, b   = cover_qmin(A)        # globally q-minimal general cover
```

## Scale-invariant magnitude estimation

[`divmag`](@ref) combines `symcover` with a right-hand side vector to produce a
scale-covariant estimate of the magnitude of `A \ b` without solving the system:

```julia
a, mag = divmag(A, b)
```

`a` is `symcover(A)` and `mag` estimates `dotabs(A \ b, a)`.  Both transform
covariantly when `A` and `b` are rescaled together, so `mag` serves as a
reliable unit for assessing roundoff in the solution.  Pass `cond=true` to fold
in the scale-invariant condition number for ill-conditioned systems.

## Index of available tools

```@index
```

## Reference documentation

```@autodocs
Modules = [ScaleInvariantAnalysis]
```
