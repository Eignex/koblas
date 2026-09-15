package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.dense.MatrixWindow

/**
 * The CBLAS enumeration values and the rules for mapping a [MatrixWindow] onto them.
 *
 * These constants are part of the CBLAS ABI rather than of any one vendor, so JVM and Native share this file
 * and neither restates them.
 *
 * The mapping turns on one fact. A column-major block with leading dimension `lda`, read under the row-major
 * layout with the same `lda`, is exactly its own transpose: `M(i, j)` sits at `base + i + j · lda`, and the
 * row-major reading of that address is entry `(j, i)`. So a window whose addressing disagrees with the layout
 * the call settled on does not need a copy; it needs a transpose flag, and a symmetric or triangular one needs
 * its triangle flag flipped with it. That is why an offset, transposed panel still reaches the vendor as a
 * leading dimension instead of as a staged buffer.
 */
internal object Cblas {
    /** `CblasRowMajor`. */
    const val ROW_MAJOR: Int = 101

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

/**
 * The layout a call runs under, chosen from the operand that has no transpose flag to express a disagreement.
 *
 * That is the output matrix for a Level 3 call and the only matrix for a Level 2 one. A staged operand is
 * packed column-major, so it settles on that.
 */
internal fun layoutFor(addressing: Addressing): Int =
    if (addressing == Addressing.RowMajor) Cblas.ROW_MAJOR else Cblas.COL_MAJOR

/**
 * The layout an operand presents to the call. A staged operand is packed column-major, so it presents that
 * however its source window was addressed.
 */
internal fun presentedLayout(addressing: Addressing): Int =
    if (addressing == Addressing.Staged) Cblas.COL_MAJOR else layoutFor(addressing)

/** The transpose flag that reconciles an operand's [addressing] with the call's [layout]. */
internal fun transposeFor(addressing: Addressing, layout: Int): Int =
    if (presentedLayout(addressing) == layout) Cblas.NO_TRANS else Cblas.TRANS

/**
 * The triangle flag for [window] under [layout].
 *
 * A stored triangle is named in the window's own addressing, so reading the window under the opposite layout
 * swaps which triangle the vendor will find. A staged operand is packed column-major with its structure
 * already applied, so it is described as it was packed.
 */
internal fun uploFor(window: MatrixWindow, addressing: Addressing, layout: Int): Int {
    val lower = when (window.structure) {
        MatrixStructure.SymmetricLower, MatrixStructure.TriangularLower, MatrixStructure.UnitLower -> true
        MatrixStructure.SymmetricUpper, MatrixStructure.TriangularUpper, MatrixStructure.UnitUpper -> false
        MatrixStructure.General -> error("general matrix has no stored triangle")
    }
    val flipped = presentedLayout(addressing) != layout
    return if (lower != flipped) Cblas.LOWER else Cblas.UPPER
}

/** Whether [window] carries an implicit unit diagonal that the vendor must not read. */
internal fun diagFor(window: MatrixWindow): Int = when (window.structure) {
    MatrixStructure.UnitLower, MatrixStructure.UnitUpper -> Cblas.UNIT
    else -> Cblas.NON_UNIT
}

/** The side flag for a two-sided Level 3 call. */
internal fun sideFor(rightSide: Boolean): Int = if (rightSide) Cblas.RIGHT else Cblas.LEFT
