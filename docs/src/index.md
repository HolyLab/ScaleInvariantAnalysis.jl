```@meta
CurrentModule = ScaleInvariantAnalysis
```

# ScaleInvariantAnalysis

This small package provides a number of tools for numerical analysis under
conditions where the underlying problems are scale-invariant. At present it is
oriented toward the types of problems that appear in mathematical optimization.
Under scaling transformations ``x \rightarrow s \odot x`` (``\odot`` is the
[Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))),
a Hessian matrix ``H`` transforms as ``H \rightarrow H \oslash (s \otimes s)``.
Therefore, operations like `norm(H)` are non-sensical. Nevertheless, we work on
computers with finite precision, so operations like ``Hx`` and ``H^{-1} g`` are
expected to have some error. This package provides tools for calculating and
estimating errors in a scale-covariant manner.

## Example

Suppose you have a diagonal Hessian matrix and you estimate its condition number
using tools that are not scale-invariant:

```jldoctest example
julia> using LinearAlgebra

julia> H = [1 0; 0 1e-8];

julia> cond(H)
1.0e8
```

You might declare this matrix to be "poorly scaled." However, the operations
`H * x` and `H \ g` both have coordinatewise relative errors of size `eps()`: there
are no delicate cancelations and thus operations involving `H` reach the full
machine precision. This does not seem entirely consistent with common
expectations of working with matrices with large condition numbers.

Under a coordinate transformation `x → [x[1], x[2]/10^4]`, `H` becomes the
identity matrix which has a condition number of 1, and this better reflects our
actual experience with operations involving `H`. This package provides a
scale-invariant analog of the condition number:

```jldoctest example; filter = r"1\.0\d*" => "1.0"
julia> using ScaleInvariantAnalysis

julia> condscale(H)
1.0
```

(You may have some roundoff error in the last few digits.) This version of the
condition number matches our actual experience using `H`. In contrast,

```jldoctest example; filter = r"(19999\.0\d*|19998\.9\d+)" => "19999.0"
julia> condscale([1 0.9999; 0.9999 1])
19999.0
```

remains poorly-conditioned under all scale-transformations of the matrix.

## Index of available tools

```@index
```

## Reference documentation

```@autodocs
Modules = [ScaleInvariantAnalysis]
```
