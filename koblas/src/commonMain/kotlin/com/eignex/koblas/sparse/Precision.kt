package com.eignex.koblas.sparse

import com.eignex.koblas.sparse.factorization.lu.F64SparseLuFactorization

/*
 * The double-precision names of this package under the aliases they had before the element type reached
 * them. [com.eignex.koblas.Precision] says why they carry it.
 */

/** Double-precision [F64SparseVectorKernels], the half an unqualified `SparseVectorKernels` means. */
public typealias SparseVectorKernels = F64SparseVectorKernels

/** Double-precision [F64SparseBlas], the sparse matrix half an unqualified `SparseBlas` means. */
public typealias SparseBlas = F64SparseBlas

/** Double-precision [F64SparseLu], the sparse LU backend half. */
public typealias SparseLuBackend = F64SparseLu

/** Double-precision [F64SparseLinearAlgebra], the pair of sparse halves an unqualified `SparseLinearAlgebra` means. */
public typealias SparseLinearAlgebra = F64SparseLinearAlgebra

/** Double-precision [F64SparseFactorization], the reusable factorization an unqualified `SparseFactorization` means. */
public typealias SparseFactorization = F64SparseFactorization

/** Double-precision [F64SingularSparseFactorization], what an unqualified `SingularSparseFactorization` means. */
public typealias SingularSparseFactorization = F64SingularSparseFactorization

/** Double-precision [F64SparseLuFactorization], the sparse LU an unqualified `SparseLu` means. */
public typealias SparseLu = F64SparseLuFactorization
