@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, T, X

package com.eignex.koblas.dense

import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.koblas

/** Solve `op(T) · x = b` in place (BLAS `dtrsv`); see [F64LinearAlgebra.trsv]. Reads only the triangle [lower]
 *  selects, and does not check the diagonal, so a singular triangle yields infinities or NaNs. */
public fun F64DenseMatrix.trsv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
): Unit = koblas.trsv(this, x, lower, transpose, unitDiag)

/** `B = alpha · op(T)⁻¹ · B`, or `B = alpha · B · op(T)⁻¹` when [right] (BLAS `dtrsm`); see
 *  [F64LinearAlgebra.trsm]. Reads only the triangle [lower] selects, and a singular triangle yields infinities
 *  or NaNs. */
@Suppress("LongParameterList") // the BLAS dtrsm signature
public fun F64DenseMatrix.trsm(
    b: F64DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
): Unit = koblas.trsm(this, b, lower, transpose, unitDiag, right, alpha)

/** Multiply `x = op(T) · x` in place (BLAS `dtrmv`); see [F64LinearAlgebra.trmv]. Reads only the triangle
 *  [lower] selects. */
public fun F64DenseMatrix.trmv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
): Unit = koblas.trmv(this, x, lower, transpose, unitDiag)

/** `B = alpha · op(T) · B`, or `B = alpha · B · op(T)` when [right] (BLAS `dtrmm`); see
 *  [F64LinearAlgebra.trmm]. Reads only the triangle [lower] selects. */
@Suppress("LongParameterList") // the BLAS dtrmm signature
public fun F64DenseMatrix.trmm(
    b: F64DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
): Unit = koblas.trmm(this, b, lower, transpose, unitDiag, right, alpha)

/** Invert the [lower] or upper triangle of [a] (LAPACK `dtrtri`); see [F64LinearAlgebra.trtri]. */
public fun F64DenseMatrix.trtri(lower: Boolean, unitDiag: Boolean = false): F64DenseMatrix = koblas.trtri(
    this,
    lower,
    unitDiag,
)
