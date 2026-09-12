# Package com.eignex.koblas.dense

Dense BLAS contracts and built-in implementations.

[DenseVectorKernels], [DensePanelKernels], and [PackedKernels] define separate contiguous-vector,
matrix-panel, and padded-tile responsibilities. [DenseBlas] contains dense matrix algorithms bound to one immutable
composition. The platform default is exposed through [com.eignex.koblas.koblas]; tests and benchmarks can
construct independent exact engines through [com.eignex.koblas.BuiltinEngines].

`gemmt` is the triangular-result general product Netlib calls `GEMMTR`; OpenBLAS and oneMKL expose the common
`cblas_dgemmt` spelling. It uses ordinary full column-major operands and a selected full-storage destination
triangle, not conventional compact BLAS packed storage.

[com.eignex.koblas.StridedMatrixView] and [com.eignex.koblas.StridedVectorView] are live zero-copy views.
Operations preserve offsets, increments, and leading dimensions. Disjoint views may share storage, while an
actual destination/input overlap is rejected where the BLAS contract does not permit aliasing.

[MatrixWindow] and [VectorWindow] validate backing-buffer bounds with explicit strides. Matrix windows can
represent general, symmetric, and triangular input, including implicit unit diagonals. Their transpose is a
zero-copy window. [PackedMatrix] retains its buffer and a [PackedMatrixLayout] with an explicit role, group,
physical strides, padding, and version. The layout does not depend on the engine that later consumes it.

[ScalarLayoutKernels] packs, unpacks, and transposes these windows. [ScalarBlockKernels] is the independently
callable product reference for direct, one-packed, both-packed, and retained operands. Alpha and beta apply to
the logical values already represented by the operands. Prefer unscaled retained operands for reuse across
changing alpha. [ProductEvaluation] distinguishes dot-then-scale arithmetic from ordered column updates;
packing alone does not authorize reassociation. [BlockOutput] limits writes to a selected triangle, including
blocks whose row and column origins differ. Aliased block calls stage output before committing it.

These logical block contracts are the migration target for the existing fixed-tile [PackedPanels] callers.
GEMM migrates in PR 04 of the repository implementation plan; structured and triangular consumers migrate
in PRs 07–09. Internal panel packing can prepare strided windows without materializing a whole contiguous matrix.

The default engine resolves immutable performance rules once, with separate scalar-to-C and SIMD-to-C
crossovers. [com.eignex.koblas.KoblasEngine.explain] describes the actual component selected for a
[DenseOperation] and length. Exact C engines bypass these rules while retaining capability checks and semantic
no-work exits. Packed geometry compatibility remains required when composing legacy tile implementations.
