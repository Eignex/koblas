@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, T, X
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.DenseBlas
import com.eignex.koblas.sparse.SparseBlas
import kotlin.jvm.JvmOverloads

/**
 * Solve `op(T) · x = b` in place (BLAS `dtrsv`) for dense or sparse storage; see [DenseBlas.trsv] and
 * [SparseBlas.trsv]. Reads only the triangle [lower] selects. No singularity check is performed, and singular
 * inputs have no guaranteed numerical result.
 *
 * The two storages differ in one place the contract names: a missing sparse diagonal entry is the sparse
 * representation of zero.
 */
@Suppress("LongParameterList") // the BLAS dtrsv signature plus the workspace
@JvmOverloads
public fun Matrix.trsv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    workspace: Workspace? = null,
): Unit = koblas.trsv(this, x, lower, transpose, unitDiag, workspace)

/** `B = alpha · op(T)⁻¹ · B`, or `B = alpha · B · op(T)⁻¹` when [right] (BLAS `dtrsm`); see
 *  [DenseBlas.trsm] and [SparseBlas.trsm]. Reads only the triangle [lower] selects; singular
 *  inputs have no guaranteed numerical result. */
@Suppress("LongParameterList") // the BLAS dtrsm signature plus the workspace
@JvmOverloads
public fun Matrix.trsm(
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    workspace: Workspace? = null,
): Unit = koblas.trsm(this, b, lower, transpose, unitDiag, right, alpha, workspace)

/** Multiply `x = op(T) · x` in place (BLAS `dtrmv`) for dense or sparse storage. */
@Suppress("LongParameterList") // the BLAS dtrmv signature plus the workspace
@JvmOverloads
public fun Matrix.trmv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    workspace: Workspace? = null,
): Unit = koblas.trmv(this, x, lower, transpose, unitDiag, workspace)

/** `B = alpha · op(T) · B`, or `B = alpha · B · op(T)` when [right] (BLAS `dtrmm`). */
@Suppress("LongParameterList") // the BLAS dtrmm signature plus the workspace
@JvmOverloads
public fun Matrix.trmm(
    b: DenseMatrixRegion,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    workspace: Workspace? = null,
): Unit = koblas.trmm(this, b, lower, transpose, unitDiag, right, alpha, workspace)

/** Reads a custom triangle once, excluding the unselected half and an implicit diagonal. */
internal inline fun <T> triangleOperand(
    workspace: Workspace?,
    a: Matrix,
    lower: Boolean,
    unitDiag: Boolean,
    block: (DenseMatrixRegion) -> T,
): T {
    if (a is DenseMatrixRegion) return block(a)
    return workspace.borrow(DenseMatrix.entryCount(a.rows, a.cols)) { values ->
        for (j in 0 until a.cols) {
            for (i in 0 until a.rows) {
                if (if (lower) i >= j else i <= j) values[i + j * a.rows] = if (unitDiag && i == j) 1.0 else a[i, j]
            }
        }
        block(DenseMatrix.wrap(a.rows, a.cols, values))
    }
}
