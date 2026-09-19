@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, T, X
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.DenseBlas
import com.eignex.koblas.sparse.SparseBlas
import kotlin.jvm.JvmOverloads

/**
 * Solve `op(T) · x = b` in place (BLAS `dtrsv`) for dense or sparse storage; see [DenseBlas.trsv] and
 * [SparseBlas.trsv]. Reads only the triangle [lower] selects, and does not check the diagonal, so a singular
 * triangle yields infinities or NaNs.
 *
 * The two storages differ in one place the contract names: a missing sparse diagonal entry is the sparse
 * representation of zero and is divided by as such, so it behaves as an explicitly stored zero would.
 */
@JvmOverloads
public fun MatrixStorage.trsv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
): Unit = when (this) {
    is DenseMatrix -> koblas.trsv(this, x, lower, transpose, unitDiag)
    is SparseMatrix -> koblas.trsv(this, x, lower, transpose, unitDiag)
}

/** `B = alpha · op(T)⁻¹ · B`, or `B = alpha · B · op(T)⁻¹` when [right] (BLAS `dtrsm`); see
 *  [DenseBlas.trsm] and [SparseBlas.trsm]. Reads only the triangle [lower] selects, and a singular triangle
 *  yields infinities or NaNs. */
@Suppress("LongParameterList") // the BLAS dtrsm signature plus the workspace
@JvmOverloads
public fun MatrixStorage.trsm(
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    workspace: MatrixWorkspace? = null,
): Unit = when (this) {
    is DenseMatrix -> koblas.trsm(this, b, lower, transpose, unitDiag, right, alpha)
    is SparseMatrix -> koblas.trsm(this, b, lower, transpose, unitDiag, right, alpha, workspace)
}

/** Multiply `x = op(T) · x` in place (BLAS `dtrmv`) for dense or sparse storage. */
@JvmOverloads
public fun MatrixStorage.trmv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
): Unit = when (this) {
    is DenseMatrix -> koblas.trmv(this, x, lower, transpose, unitDiag)
    is SparseMatrix -> koblas.trmv(this, x, lower, transpose, unitDiag)
}

/** `B = alpha · op(T) · B`, or `B = alpha · B · op(T)` when [right] (BLAS `dtrmm`). */
@Suppress("LongParameterList") // the BLAS dtrmm signature plus the workspace
@JvmOverloads
public fun MatrixStorage.trmm(
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    workspace: MatrixWorkspace? = null,
): Unit = when (this) {
    is DenseMatrix -> koblas.trmm(this, b, lower, transpose, unitDiag, right, alpha)
    is SparseMatrix -> koblas.trmm(this, b, lower, transpose, unitDiag, right, alpha, workspace)
}
