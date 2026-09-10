package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas

/** Solve `op(T) · x = b` in place against this matrix's selected triangle. */
public fun SparseMatrix.trsv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
): Unit = koblas.trsv(this, x, lower, transpose, unitDiag)

/** Multiply `x = op(T) · x` in place against this matrix's selected triangle. */
public fun SparseMatrix.trmv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
): Unit = koblas.trmv(this, x, lower, transpose, unitDiag)

/** Prepares an immutable snapshot of this matrix for repeated products. */
public fun SparseMatrix.prepare(): PreparedSparseMatrix = koblas.prepare(this)

/** Symmetric product with [x] into a fresh dense vector, reading only this matrix's selected triangle. */
public fun SparseMatrix.symv(x: DoubleArray, lower: Boolean = true): DoubleArray =
    DoubleArray(rows).also { koblas.symv(1.0, this, x, 0.0, it, lower) }

/** Symmetric `y = alpha · A · x + beta · y`, reading only this matrix's selected triangle. */
@Suppress("LongParameterList")
public fun SparseMatrix.symv(
    alpha: Double,
    x: DoubleArray,
    beta: Double,
    y: DoubleArray,
    lower: Boolean = true,
): Unit = koblas.symv(alpha, this, x, beta, y, lower)

/** Symmetric sparse-dense product into [c], with this selected-triangle matrix on either side. */
@Suppress("LongParameterList")
public fun SparseMatrix.symm(
    alpha: Double,
    b: DenseMatrix,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean = true,
    right: Boolean = false,
    workspace: Workspace? = null,
): Unit = koblas.symm(alpha, this, b, beta, c, lower, right, workspace)

/** Fresh selected CSC triangle of `op(A) · op(A)ᵀ`. */
public fun SparseMatrix.syrk(transpose: Boolean = false, lower: Boolean = true): SparseMatrix =
    koblas.syrk(this, transpose, lower)

/** Dense selected-triangle sparse rank-k product into [c]. */
@Suppress("LongParameterList")
public fun SparseMatrix.syrk(
    alpha: Double,
    transpose: Boolean,
    beta: Double,
    c: DenseMatrix,
    lower: Boolean = true,
    workspace: Workspace? = null,
): Unit = koblas.syrk(alpha, this, transpose, beta, c, lower, workspace)

/** Fresh CSC `alpha · op(A) + B`, retaining the structural union. */
public fun SparseMatrix.addScaled(alpha: Double, transpose: Boolean, b: SparseMatrix): SparseMatrix =
    koblas.addScaled(alpha, this, transpose, b)

/** Solve `op(T) · X = B` in place for every column of [b]. */
@Suppress("LongParameterList")
public fun SparseMatrix.trsm(
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    workspace: Workspace? = null,
): Unit = koblas.trsm(this, b, lower, transpose, unitDiag, right, alpha, workspace)

/** Multiply [b] in place by this triangular matrix. */
@Suppress("LongParameterList")
public fun SparseMatrix.trmm(
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
): Unit = koblas.trmm(this, b, lower, transpose, unitDiag, right, alpha)
