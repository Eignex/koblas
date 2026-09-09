# Package com.eignex.koblas.dense

Dense BLAS contracts and built-in implementations.

[Kernels] contains level-1 and packed-panel leaves. [Blas] contains dense matrix algorithms bound to one
immutable [Kernels] instance. The platform default is exposed through [com.eignex.koblas.koblas]; tests and
benchmarks can construct independent exact engines through [com.eignex.koblas.BuiltinKernels].

[com.eignex.koblas.StridedMatrixView] and [com.eignex.koblas.StridedVectorView] are live zero-copy views.
Operations preserve offsets, increments, and leading dimensions. Disjoint views may share storage, while an
actual destination/input overlap is rejected where the BLAS contract does not permit aliasing.
