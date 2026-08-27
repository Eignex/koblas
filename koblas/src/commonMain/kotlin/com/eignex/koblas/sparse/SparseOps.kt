package com.eignex.koblas.sparse

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.koblas

/** Factorize this sparse matrix with the active backend ([koblas]), the counterpart of `F64DenseMatrix.lu`. */
public fun F64SparseMatrix.lu(): F64SparseLuFactorization = koblas.factor(this)

/**
 * Cholesky-factorize this symmetric positive-definite matrix with the active backend ([koblas]), reading
 * only its lower triangle. See [F64SparseDecompositions.cholesky].
 */
public fun F64SparseMatrix.cholesky(): F64SparseCholeskyFactorization = koblas.cholesky(this)

/**
 * Factorize this symmetric matrix into `L·D·Lᵀ` with the active backend ([koblas]), reading only its lower
 * triangle. See [F64SparseDecompositions.ldl], which says what it does and does not promise.
 */
public fun F64SparseMatrix.ldl(): F64SparseLdlFactorization = koblas.ldl(this)

/**
 * QR-factorize this tall or square matrix with the active backend ([koblas]), for the least-squares solve
 * `min ‖A·x − b‖₂`. See [F64SparseDecompositions.qr].
 */
public fun F64SparseMatrix.qr(): F64SparseQrFactorization = koblas.qr(this)

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

/** `this · B` for a sparse [b], into a fresh sparse matrix, with the active backend ([koblas]). */
public fun F64SparseMatrix.gemm(b: F64SparseMatrix): F64SparseMatrix = koblas.sparseBlas.gemm(this, b)

/** Prepares an immutable snapshot of this matrix for repeated products with the active backend. */
public fun F64SparseMatrix.prepare(): F64PreparedSparseMatrix = koblas.sparseBlas.prepare(this)

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
