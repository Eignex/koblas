# Package com.eignex.koblas

The koblas linear-algebra API — a well-defined subset of double-precision BLAS/LAPACK (see the README's
"BLAS coverage" table for the routine-by-routine mapping and deliberate deviations):

- Containers: [MatrixView] / [DenseMatrix] and [VectorView] / [DenseVector] / [SparseVector] (all
  `@Serializable`).
- Free-function arithmetic over the views: [dot], [axpy], [scale], [norm2], [asum], [iamax], [copy],
  [swap], [ger], [gemv], [transpose], [forEachStored]; the triangular solves [trsv] / [trsm];
  and the SPD suite [cholesky], [solveSpd],
  [invertSpd].
- The swappable [LinearAlgebra] backend for the heavier dense ops (`gemv` / `gemm` in full BLAS
  alpha/beta/transpose forms, `syrk`, LU [factor][LinearAlgebra.factor] / [solve][LinearAlgebra.solve]
  with [determinant]) with the portable [ReferenceLinearAlgebra], the resolved default [koblas], the
  [LuDecomposition] result, and the ergonomic [lu] / [LuDecomposition.solve][solve] / [matMul] entry
  points. Backends are offered through [registerBlas] / [registerLapack], or forced with
  [installLinearAlgebra].
- The [VectorKernels] half, for the level-1 kernels that sit below that seam: offered through
  [registerVectorKernels], forced with [installVectorKernels].
- Sparse linear algebra: the CSC [SparseMatrix], the [SparseLu] Markowitz factorization with FTRAN/BTRAN
  solves, and the [EtaBasis] product-form-of-the-inverse for rank-1 basis updates.
- The platform backend id [mathBackend].
