package com.eignex.koblas.sparse

import com.eignex.koblas.sparse.factorization.cholesky.F64SparseUpLookingCholesky
import com.eignex.koblas.sparse.factorization.lu.F64SparseMarkowitzLu

/*
 * The double-precision names of this package under the aliases they had before the element type reached
 * them. [com.eignex.koblas.Precision] says why they carry it.
 */

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

/** Double-precision [F64SparseMarkowitzLu], the sparse LU an unqualified `SparseLu` means. */
public typealias SparseLu = F64SparseMarkowitzLu

/** Double-precision [F64SparseUpLookingCholesky], the sparse Cholesky an unqualified `SparseCholesky`
 *  means. */
public typealias SparseCholesky = F64SparseUpLookingCholesky
