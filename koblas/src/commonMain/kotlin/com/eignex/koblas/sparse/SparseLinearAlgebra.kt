package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.koblas

/**
 * Both sparse matrix seams at once, the counterpart of [com.eignex.koblas.dense.LinearAlgebra].
 *
 * Implement this when a backend provides both, which is the usual case for a host library — SuiteSparse
 * ships the solver and the products together, unlike the CBLAS-without-LAPACKE split that made the dense
 * halves worth ranking separately. The halves are still registered independently, because costing nothing
 * to allow is better than discovering later that something needed it.
 */
interface SparseLinearAlgebra :
    SparseBlas,
    SparseLapack

/**
 * Factorize this sparse matrix with the active backend ([koblas]) — the sparse counterpart of
 * `DenseMatrix.lu()`, carrying the same name for the same operation on the other storage.
 */
fun SparseMatrix.lu(equilibrate: Boolean = false): SparseFactorization = koblas.factor(this, equilibrate)

/**
 * `this · x`, or `thisᵀ · x` when [transpose], with the active backend ([koblas]).
 *
 * An extension rather than a member of `SparseMatrix`, matching `DenseMatrix.matMul`: the containers are
 * storage, and which backend multiplies them is a separate concern that would otherwise make the root
 * package depend on this one.
 */
fun SparseMatrix.gemv(x: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.gemv(this, x, transpose)

/**
 * Solve `op(T) · x = b` in place against this matrix's [lower] or upper triangle, with the active backend
 * ([koblas]) — the sparse counterpart of the dense `trsv` free function. See [SparseBlas.trsv].
 */
fun SparseMatrix.trsv(x: DoubleArray, lower: Boolean, transpose: Boolean = false) =
    koblas.trsv(this, x, lower, transpose)
