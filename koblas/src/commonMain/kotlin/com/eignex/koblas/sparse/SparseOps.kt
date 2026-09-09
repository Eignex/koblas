package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas

/** Factorize this sparse matrix with the active backend ([koblas]), the counterpart of `DenseMatrix.lu`. */
public fun SparseMatrix.lu(): SparseLuFactorization = koblas.factor(this)

/**
 * Cholesky-factorize this symmetric positive-definite matrix with the active backend ([koblas]), reading
 * only its lower triangle. See [SparseLapack.cholesky].
 */
public fun SparseMatrix.cholesky(): SparseCholeskyFactorization = koblas.cholesky(this)

/**
 * Factorize this symmetric matrix into `L·D·Lᵀ` with the active backend ([koblas]), reading only its lower
 * triangle. See [SparseLapack.quasiDefiniteLdl], which says what it does and does not promise.
 */
public fun SparseMatrix.quasiDefiniteLdl(): QuasiDefiniteLdlFactorization = koblas.quasiDefiniteLdl(this)

/**
 * QR-factorize this tall or square matrix with the active backend ([koblas]), for the least-squares solve
 * `min ‖A·x − b‖₂`. See [SparseLapack.qr].
 */
public fun SparseMatrix.qr(): SparseQrFactorization = koblas.qr(this)

/**
 * Solve `op(T) · x = b` in place against this matrix's [lower] or upper triangle, with the active backend
 * ([koblas]). See [SparseBlas.trsv].
 */
public fun SparseMatrix.trsv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
): Unit = koblas.trsv(this, x, lower, transpose, unitDiag)

/** Multiply `x = op(T) · x` in place against this matrix's selected triangle with the active backend ([koblas]). */
public fun SparseMatrix.trmv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
): Unit = koblas.trmv(this, x, lower, transpose, unitDiag)

/** Prepares an immutable snapshot of this matrix for repeated products with the active backend. */
public fun SparseMatrix.prepare(): PreparedSparseMatrix = koblas.sparseBlas.prepare(this)

/**
 * Solve `op(T) · X = B` in place against this matrix's [lower] or upper triangle, for every column of [b] at
 * once, with the active backend ([koblas]). See [SparseBlas.trsm].
 */
@Suppress("LongParameterList") // the BLAS dtrsm signature
public fun SparseMatrix.trsm(
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    workspace: Workspace? = null,
): Unit = koblas.sparseBlas.trsm(this, b, lower, transpose, unitDiag, right, alpha, workspace)

/** `B = alpha · op(T) · B`, or `B = alpha · B · op(T)` when [right], in place with the active backend ([koblas]). */
@Suppress("LongParameterList") // the BLAS dtrmm signature
public fun SparseMatrix.trmm(
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
): Unit = koblas.trmm(this, b, lower, transpose, unitDiag, right, alpha)
