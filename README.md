# ScaleInvariantAnalysis

<!--- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://HolyLab.github.io/ScaleInvariantAnalysis.jl/stable/) --->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://HolyLab.github.io/ScaleInvariantAnalysis.jl/dev/)
[![Build Status](https://github.com/HolyLab/ScaleInvariantAnalysis.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/HolyLab/ScaleInvariantAnalysis.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/HolyLab/ScaleInvariantAnalysis.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/HolyLab/ScaleInvariantAnalysis.jl)

This package provides tools for numerical analysis under conditions where the
underlying mathematics is scale-invariant. We work on computers with finite
precision, so operations like matrix-multiplication and matrix-division are
expected to have some error. However, naive estimates of the error, based on
quantities like `norm(x)`, may not be scale-invariant.

For example, if `H` is a diagonal nonnegative (Hessian) matrix (i.e., a rank-2
covariant tensor), with a change-of-scale in the variables all such `H` are
equivalent to the identity matrix. Therefore we might think that its [condition
number](https://en.wikipedia.org/wiki/Condition_number) should in fact be 1.
This package provides tool to calculate condition number, as well as several
other quantities, under conditions of scale-invariance.

See the
[documentation](https://HolyLab.github.io/ScaleInvariantAnalysis.jl/dev/) for
more information.
