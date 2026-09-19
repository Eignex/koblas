# Package com.eignex.koblas.dense

Dense BLAS contracts and the implementations behind them.

The surface splits at the level, not at the backend. [DenseVectorKernels] is Level 1: arithmetic over one run
of a `DoubleArray`, taken as a buffer, an origin, a length and a step. It stays portable Kotlin because a
foreign call over a single vector costs more than the arithmetic it carries, and because it has to keep
working on a host with no library installed.

[DenseBlas] is Level 2 and 3. Built-in engines implement it in common Kotlin, with validation, zero-multiplier
no-read behavior and alias staging before mutation. This portable scalar implementation is the availability
floor; later stages add architecture-selected JVM panels and tiles behind the same shared operations.

Installed host bindings expose the same operation family separately. Their arithmetic and exceptional behavior
belong to the resolved library, and their route reports its binary and entry point. A requested engine name is
never used as proof that such a binding ran.

The platform default is exposed through [com.eignex.koblas.koblas], selected once and immutable. Tests and
benchmarks reach exact portable and JVM SIMD compositions through [com.eignex.koblas.BuiltinEngines], and
[com.eignex.koblas.KoblasEngine.explain] names the component a [DenseOperation] of a given length and spacing
actually reaches, so an attribution cannot claim a kernel the call did not run.

Operands are the public containers. [com.eignex.koblas.DenseMatrix] is contiguous column-major, and
[com.eignex.koblas.DenseVector] covers both dense spacings, adjacent entries and a fixed step, because a
vendor takes either as a pointer and an increment. Structure and transposition travel beside an operand as
flags rather than in its type: [MatrixStructure] says which triangle is stored and whether the diagonal is
implied, and a transpose is a boolean the call passes through.

`gemmt` is the triangular-result general product Netlib calls `GEMMTR`. It uses ordinary full column-major
operands and a selected full-storage destination triangle, not conventional compact BLAS packed storage.
Explicit vendor bindings report whether they called `cblas_dgemmt` or composed the operation.
