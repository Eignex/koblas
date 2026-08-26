package com.eignex.koblas.sparse

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.koblas

/** Factorize this sparse matrix with the active backend ([koblas]), the counterpart of `F64DenseMatrix.lu`. */
public fun F64SparseMatrix.lu(): F64SparseFactorization = koblas.factor(this)

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

/** `this · B` for a dense [b], with the active backend ([koblas]). See [F64SparseBlas.gemm]. */
public fun F64SparseMatrix.gemm(b: F64DenseMatrix): F64DenseMatrix = koblas.sparseBlas.gemm(this, b)

/**
 * Solve `op(T) · X = B` in place against this matrix's [lower] or upper triangle, for every column of [b] at
 * once, with the active backend ([koblas]). See [F64SparseBlas.trsm].
 */
@Suppress("LongParameterList") // the BLAS dtrsm signature
public fun F64SparseMatrix.trsm(
    b: F64DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
): Unit = koblas.sparseBlas.trsm(this, b, lower, transpose, unitDiag, right, alpha)
