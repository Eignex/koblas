@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, T, X
@file:kotlin.jvm.JvmName("Koblas")
@file:kotlin.jvm.JvmMultifileClass

package com.eignex.koblas

import com.eignex.koblas.dense.DenseBlas
import kotlin.jvm.JvmOverloads

/**
 * Solve `op(T) · x = b` in place (BLAS `dtrsv`); see [DenseBlas.trsv]. Reads only the triangle [lower]
 * selects, and does not check the diagonal, so a singular triangle yields infinities or NaNs.
 *
 * Dense storage only: a sparse triangular solve is a solver workflow rather than a numerical leaf, and belongs
 * to the consumer that owns its factors.
 */
@JvmOverloads
public fun DenseMatrix.trsv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
): Unit = koblas.trsv(this, x, lower, transpose, unitDiag)

/** `B = alpha · op(T)⁻¹ · B`, or `B = alpha · B · op(T)⁻¹` when [right] (BLAS `dtrsm`); see
 *  [DenseBlas.trsm]. Reads only the triangle [lower] selects, and a singular triangle yields infinities
 *  or NaNs. */
@Suppress("LongParameterList") // the BLAS dtrsm signature
@JvmOverloads
public fun DenseMatrix.trsm(
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
): Unit = koblas.trsm(this, b, lower, transpose, unitDiag, right, alpha)

/** Multiply `x = op(T) · x` in place (BLAS `dtrmv`). */
@JvmOverloads
public fun DenseMatrix.trmv(
    x: DoubleArray,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
): Unit = koblas.trmv(this, x, lower, transpose, unitDiag)

/** `B = alpha · op(T) · B`, or `B = alpha · B · op(T)` when [right] (BLAS `dtrmm`). */
@Suppress("LongParameterList") // the BLAS dtrmm signature
@JvmOverloads
public fun DenseMatrix.trmm(
    b: DenseMatrix,
    lower: Boolean,
    transpose: Boolean = false,
    unitDiag: Boolean = false,
    right: Boolean = false,
    alpha: Double = 1.0,
): Unit = koblas.trmm(this, b, lower, transpose, unitDiag, right, alpha)
