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
public fun SparseMatrix.prepare(): PreparedSparseMatrix = koblas.sparseBlas.prepare(this)

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
): Unit = koblas.sparseBlas.trsm(this, b, lower, transpose, unitDiag, right, alpha, workspace)

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
