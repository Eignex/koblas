package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixStructure

/**
 * The CBLAS enumeration values and the rules for mapping Koblas operands onto them.
 *
 * These constants are part of the CBLAS ABI rather than of any one vendor, so JVM and Native share this file
 * and neither restates them.
 *
 * Every matrix Koblas hands over is contiguous column-major with a leading dimension equal to its row count,
 * so the layout is always column-major and a transpose is a flag rather than a different addressing. There is
 * no case here where an operand has to be repacked to be described.
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
internal fun diagFor(structure: MatrixStructure): Int = when (structure) {
    MatrixStructure.UnitLower, MatrixStructure.UnitUpper -> Cblas.UNIT
    else -> Cblas.NON_UNIT
}

/** The side flag for a two-sided Level 3 call. */
internal fun sideFor(rightSide: Boolean): Int = if (rightSide) Cblas.RIGHT else Cblas.LEFT
