@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, T, X

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.koblas

/** Solve `op(T) · x = b` in place (BLAS `dtrsv`); see [Blas.trsv]. Reads only the triangle [lower]
 *  selects, and does not check the diagonal, so a singular triangle yields infinities or NaNs. */
public fun DenseMatrix.trsv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
): Unit = koblas.trsv(this, x, lower, transpose, unitDiag)

/** `B = alpha · op(T)⁻¹ · B`, or `B = alpha · B · op(T)⁻¹` when [right] (BLAS `dtrsm`); see
 *  [Blas.trsm]. Reads only the triangle [lower] selects, and a singular triangle yields infinities
 *  or NaNs. */
@Suppress("LongParameterList") // the BLAS dtrsm signature
public fun DenseMatrix.trsm(
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    workspace: Workspace? = null,
): Unit = koblas.trsm(this, b, lower, transpose, unitDiag, right, alpha, workspace)

/** Multiply `x = op(T) · x` in place (BLAS `dtrmv`); see [Blas.trmv]. Reads only the triangle
 *  [lower] selects. */
public fun DenseMatrix.trmv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
): Unit = koblas.trmv(this, x, lower, transpose, unitDiag)

/** `B = alpha · op(T) · B`, or `B = alpha · B · op(T)` when [right] (BLAS `dtrmm`); see
 *  [Blas.trmm]. Reads only the triangle [lower] selects. */
@Suppress("LongParameterList") // the BLAS dtrmm signature
public fun DenseMatrix.trmm(
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
    workspace: Workspace? = null,
): Unit = koblas.trmm(this, b, lower, transpose, unitDiag, right, alpha, workspace)
