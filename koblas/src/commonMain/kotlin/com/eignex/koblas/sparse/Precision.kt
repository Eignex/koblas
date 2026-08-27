package com.eignex.koblas.sparse

/* Current double-precision convenience names; [com.eignex.koblas.Precision] explains the element prefix. */

/** Double-precision [F64SparseKernels], the half an unqualified `SparseKernels` means. */
public typealias SparseKernels = F64SparseKernels

/** Double-precision [F64SparseBlas], the sparse matrix half an unqualified `SparseBlas` means. */
public typealias SparseBlas = F64SparseBlas

/** Double-precision [F64SparseDecompositions], the sparse factorization half an unqualified
 *  `SparseLapack` means, as [com.eignex.koblas.dense.Lapack] is the dense one. */
public typealias SparseLapack = F64SparseDecompositions

/** Double-precision [F64SparseLinearAlgebra], the pair of sparse halves an unqualified `SparseLinearAlgebra` means. */
public typealias SparseLinearAlgebra = F64SparseLinearAlgebra

/** Double-precision [F64SparseFactorization], the reusable factorization an unqualified `SparseFactorization` means. */
public typealias SparseFactorization = F64SparseFactorization

/** Double-precision [F64SingularSparseFactorization], what an unqualified `SingularSparseFactorization` means. */
public typealias SingularSparseFactorization = F64SingularSparseFactorization

/** Double-precision [F64SparseLuFactorization]. */
public typealias SparseLuFactorization = F64SparseLuFactorization

/** Double-precision [F64SparseCholeskyFactorization]. */
public typealias SparseCholeskyFactorization = F64SparseCholeskyFactorization

/** Double-precision [F64SparseLdlFactorization]. */
public typealias SparseLdlFactorization = F64SparseLdlFactorization

/** Double-precision [F64SparseQrFactorization]. */
public typealias SparseQrFactorization = F64SparseQrFactorization
