package com.eignex.koblas.sparse

import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.factorization.lu.*

/** The sparse matrix halves, with the active sparse-vector kernels used by their surrounding operations. */
public interface F64SparseLinearAlgebra :
    F64SparseBlas,
    F64SparseLu {
    /** The sparse vector kernels used by operations around these matrix halves. */
    public val sparseVectorKernels: F64SparseVectorKernels get() = koblas.sparseVectorKernels
}

/** Factorize this sparse matrix with the active backend ([koblas]), the counterpart of `F64DenseMatrix.lu`. */
public fun F64SparseMatrix.lu(equilibrate: Boolean = false, dropTolerance: Double = NO_DROP): F64SparseFactorization =
    koblas.factor(this, equilibrate, dropTolerance)

/** `this · x`, or `thisᵀ · x` when [transpose], with the active backend ([koblas]). */
public fun F64SparseMatrix.gemv(x: DoubleArray, transpose: Boolean = false): DoubleArray = koblas.gemv(
    this,
    x,
    transpose,
)

/**
 * Solve `op(T) · x = b` in place against this matrix's [lower] or upper triangle, with the active backend
 * ([koblas]). See [F64SparseBlas.trsv].
 */
public fun F64SparseMatrix.trsv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
): Unit = koblas.trsv(this, x, lower, transpose, unitDiag)
