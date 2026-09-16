# Package com.eignex.koblas.dense

Dense BLAS contracts and the implementations behind them.

The surface splits at the level, not at the backend. [DenseVectorKernels] is Level 1: arithmetic over one run
of a `DoubleArray`, taken as a buffer, an origin, a length and a step. It stays portable Kotlin because a
foreign call over a single vector costs more than the arithmetic it carries, and because it has to keep
working on a host with no library installed.

[DenseBlas] is Level 2 and 3, and every operation on it is one whole call to the selected vendor BLAS.
Koblas contributes shape and aliasing validation and nothing else: the arithmetic, including how zero
multipliers, infinities and accumulation order are treated, is the library's. There is no portable fallback
there, so an accelerator-dependent call on a host without a library raises
[com.eignex.koblas.vendor.MissingVendorException] rather than quietly computing the same answer far slower
under the same name.

The platform default is exposed through [com.eignex.koblas.koblas], selected once and immutable. Tests and
benchmarks reach an exact Level 1 selection through [com.eignex.koblas.BuiltinEngines], and
[com.eignex.koblas.KoblasEngine.explain] names the component a [DenseOperation] of a given length and spacing
actually reaches, so an attribution cannot claim a kernel the call did not run.

Operands are the public containers. [com.eignex.koblas.DenseMatrix] is contiguous column-major, and
[com.eignex.koblas.DenseVector] covers both dense spacings, adjacent entries and a fixed step, because a
vendor takes either as a pointer and an increment. Structure and transposition travel beside an operand as
flags rather than in its type: [MatrixStructure] says which triangle is stored and whether the diagonal is
implied, and a transpose is a boolean the call passes through.

`gemmt` is the triangular-result general product Netlib calls `GEMMTR`; OpenBLAS and oneMKL expose the common
`cblas_dgemmt` spelling. It uses ordinary full column-major operands and a selected full-storage destination
triangle, not conventional compact BLAS packed storage. A vendor that does not export it composes the result
from `gemm` plus a triangle copy, and the route of the call says which of the two ran.
