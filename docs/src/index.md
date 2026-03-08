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
variables with different units (position in meters, velocity in m/s, force
in N):

```jldoctest coverones
julia> using ScaleInvariantAnalysis

julia> A = [1e6 1e3 1.0; 1e3 1.0 1e-3; 1.0 1e-3 1e-6];

julia> a = symcover(A)
3-element Vector{Float64}:
 1000.0
    1.0
    0.001
```

The cover `a` captures the natural per-variable scale.  The normalized matrix
`A ./ (a .* a')` is all-ones and scale-invariant.

## Measuring cover quality

A cover is valid as long as every constraint is satisfied, but tighter covers
better capture the scaling of `A`.  The *log-excess* of an entry is
`log(a[i] * b[j] / abs(A[i, j])) >= 0`; it is zero when the constraint is
exactly tight.  Two summary statistics aggregate these excesses:

- [`cover_lobjective`](@ref) — sum of log-excesses (L1 in log space).
- [`cover_qobjective`](@ref) — sum of squared log-excesses (L2 in log space).

Both equal zero if and only if every constraint is tight.

```jldoctest quality; filter = r"(\d+\.\d{6})\d+" => s"\1"
julia> using ScaleInvariantAnalysis

julia> A = [4.0 2.0; 2.0 9.0];

julia> a = symcover(A)
2-element Vector{Float64}:
 2.0
 3.0

julia> cover_lobjective(a, A)
2.1972245773362196

julia> cover_qobjective(a, A)
2.413897921625164
```

Here's an example where the quadratically-optimal cover differs slightly from the one returned by `cover`:

```jldoctest quality2; filter = r"(\d+\.\d{6})\d+" => s"\1"
julia> using ScaleInvariantAnalysis, JuMP, HiGHS

julia> A = [1 2 3; 6 5 4];

julia> a, b = cover(A)
([1.2674308473260654, 3.4759059767492304], [1.7261686708831454, 1.61137045961268, 2.366993044495631])

julia> aq, bq = cover_qmin(A)
([1.1986299970143055, 3.25358233351279], [1.8441211516912772, 1.6685716234216104, 2.5028574351324164])

julia> a * b'
2×3 Matrix{Float64}:
 2.1878  2.0423   3.0
 6.0     5.60097  8.22745

julia> aq * bq'
2×3 Matrix{Float64}:
 2.21042  2.0      3.0
 6.0      5.42884  8.14325
```

## Choosing a cover algorithm

| Function | Symmetric | Minimizes | Requires |
|---|---|---|---|
| [`symcover`](@ref) | yes | (fast heuristic) | — |
| [`cover`](@ref) | no | (fast heuristic) | — |
| [`symcover_lmin`](@ref) | yes | `cover_lobjective` | JuMP + HiGHS |
| [`cover_lmin`](@ref) | no | `cover_lobjective` | JuMP + HiGHS |
| [`symcover_qmin`](@ref) | yes | `cover_qobjective` | JuMP + HiGHS |
| [`cover_qmin`](@ref) | no | `cover_qobjective` | JuMP + HiGHS |

**`symcover` and `cover` are recommended for production use.**  They run in
O(n²) time and often land within a few percent of the `cover_lobjective`-optimal
cover (see the quality tests involving `test/testmatrices.jl`).

The `*_lmin` and `*_qmin` variants solve a convex program (via
[JuMP](https://jump.dev/) and [HiGHS](https://highs.dev/)) and return a
global optimum of the respective objective.  They are loaded on demand as a
package extension — simply load both libraries before calling them:

```julia
using JuMP, HiGHS
using ScaleInvariantAnalysis

a      = symcover_lmin(A)     # globally linear-minimal symmetric cover
a, b   = cover_qmin(A)        # globally quadratic-minimal general cover
```

## Index of available tools

```@index
Modules = [ScaleInvariantAnalysis]
Private = false
```

## Reference documentation

```@autodocs
Modules = [ScaleInvariantAnalysis]
Private = false
```
