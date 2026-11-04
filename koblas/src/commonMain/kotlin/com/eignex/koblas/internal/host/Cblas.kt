package com.eignex.koblas.internal.host

import com.eignex.koblas.*
import com.eignex.koblas.MatrixStructure
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.*
import kotlin.math.abs

/**
 * The CBLAS enumeration values and the rules for mapping Koblas operands onto them.
 *
 * These constants are part of the CBLAS ABI rather than of any one vendor, so JVM and Native share this file
 * and neither restates them.
 *
 * Matrices use column-major addressing with their own leading dimensions. A borrowed region retains its
 * parent column spacing, and transposition is a flag.
 */
internal object Cblas {
    /** `CblasColMajor`. */
    const val COL_MAJOR: Int = 102

    /** `CblasNoTrans`. */
    const val NO_TRANS: Int = 111

    /** `CblasTrans`. */
    const val TRANS: Int = 112

    /** `CblasUpper`. */
    const val UPPER: Int = 121

    /** `CblasLower`. */
    const val LOWER: Int = 122

    /** `CblasNonUnit`. */
    const val NON_UNIT: Int = 131

    /** `CblasUnit`. */
    const val UNIT: Int = 132

    /** `CblasLeft`. */
    const val LEFT: Int = 141

    /** `CblasRight`. */
    const val RIGHT: Int = 142
}

/** The transpose flag for an operand a call reads transposed. */
internal fun transposeFor(transpose: Boolean): Int = if (transpose) Cblas.TRANS else Cblas.NO_TRANS

/** The triangle [structure] declares stored. */
internal fun uploFor(structure: MatrixStructure): Int = when (structure) {
    MatrixStructure.SymmetricLower, MatrixStructure.TriangularLower, MatrixStructure.UnitLower -> Cblas.LOWER
    MatrixStructure.SymmetricUpper, MatrixStructure.TriangularUpper, MatrixStructure.UnitUpper -> Cblas.UPPER
    MatrixStructure.General -> error("general matrix has no stored triangle")
}

/** Whether [structure] carries an implicit unit diagonal that the vendor must not read. */
internal fun diagFor(structure: MatrixStructure): Int = if (structure.isUnitDiagonal) Cblas.UNIT else Cblas.NON_UNIT

/** The side flag for a two-sided Level 3 call. */
internal fun sideFor(rightSide: Boolean): Int = if (rightSide) Cblas.RIGHT else Cblas.LEFT

/** A singleton never advances, and its legal stride may have an unrepresentable positive magnitude. */
internal fun positiveIncrement(stride: Int, size: Int): Int = if (size == 1) 1 else abs(stride)

/**
 * Lowest storage index passed to BLAS. Negative increments retain their sign but start at the
 * low address, allowing BLAS to reconstruct the logical first entry at the high end.
 */
internal fun baseIndex(vector: DenseVector): Int = baseIndex(vector.offset, vector.stride, vector.size)

/** The same rule over a raw run, which is what the Level 1 entry points hand BLAS. */
internal fun baseIndex(offset: Int, stride: Int, size: Int): Int =
    if (size <= 1 || stride >= 0) offset else offset + (size - 1) * stride
