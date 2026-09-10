# Package com.eignex.koblas.dense

Dense BLAS contracts and built-in implementations.

[DenseVectorKernels], [DensePanelKernels], and [PackedKernels] define separate contiguous-vector,
matrix-panel, and padded-tile responsibilities. [Blas] contains dense matrix algorithms bound to one immutable
composition. The platform default is exposed through [com.eignex.koblas.koblas]; tests and benchmarks can
construct independent exact engines through [com.eignex.koblas.BuiltinEngines].

`gemmt` is the triangular-result general product Netlib calls `GEMMTR`; OpenBLAS and oneMKL expose the common
`cblas_dgemmt` spelling. It uses ordinary full column-major operands and a selected full-storage destination
triangle, not conventional compact BLAS packed storage.

[com.eignex.koblas.StridedMatrixView] and [com.eignex.koblas.StridedVectorView] are live zero-copy views.
Operations preserve offsets, increments, and leading dimensions. Disjoint views may share storage, while an
actual destination/input overlap is rejected where the BLAS contract does not permit aliasing.
