# Package com.eignex.koblas

The koblas linear-algebra API:

- Containers: [MatrixView] / [DenseMatrix] and [VectorView] / [DenseVector] / [SparseVector] (all
  `@Serializable`).
- Free-function arithmetic over the views: [dot], [axpy], [scale], [addOuter], [matVec], [forEachStored],
  and the SPD suite [cholesky], [choleskyDowndateInPlace], [solveSpd], [invertSpd].
- The swappable [LinearAlgebra] backend for the heavier dense ops (`gemv`, `gemm`, LU
  [factor][LinearAlgebra.factor] / [solve][LinearAlgebra.solve]) with the portable
  [ReferenceLinearAlgebra], the [platformLinearAlgebra] native-backend seam, the resolved default
  [koblas], the [LuDecomposition] result, and the ergonomic [lu] / [LuDecomposition.solve][solve] /
  [matMul] entry points.
- Sparse linear algebra: the CSC [SparseMatrix], the [SparseLu] Markowitz factorization with FTRAN/BTRAN
  solves, and the [EtaBasis] product-form-of-the-inverse for rank-1 basis updates.
- The platform backend id [mathBackend].
