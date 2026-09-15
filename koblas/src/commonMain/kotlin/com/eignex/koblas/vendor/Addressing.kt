package com.eignex.koblas.vendor

import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.dense.MatrixWindow
import com.eignex.koblas.dense.VectorWindow

/**
 * How one matrix operand reaches BLAS.
 *
 * This is the single decision behind both what runs and what the route says ran. Execution switches on it and
 * [VendorBlas.routeOf] describes it; neither computes it a second way.
 */
internal enum class Addressing {
    /** `A(i, j)` is at `offset + i + j · lda`, which is what a column-major BLAS call wants. */
    ColumnMajor,

    /** `A(i, j)` is at `offset + i · lda + j`, passed with the row-major CBLAS layout flag. */
    RowMajor,

    /** Neither stride is one, so the window is copied into a packed column-major buffer for the call. */
    Staged,
}

/**
 * The addressing for [window], and the leading dimension that goes with it.
 *
 * BLAS addresses a matrix with one stride and one unit step, so a window is expressible exactly when one of
 * its strides is one and the other is a positive leading dimension large enough to hold a line. Everything
 * else is a staged copy: a negative column stride, two strides above one, or a structured block whose diagonal
 * has moved off the parent's, which no `uplo` flag can express.
 */
internal fun addressingOf(window: MatrixWindow): Addressing {
    if (window.structure != MatrixStructure.General && window.diagonalOffset != 0) return Addressing.Staged
    val rows = window.rows
    val columns = window.columns
    if (window.rowStride == 1 && window.columnStride >= maxOf(1, rows)) return Addressing.ColumnMajor
    if (window.columnStride == 1 && window.rowStride >= maxOf(1, columns)) return Addressing.RowMajor
    return Addressing.Staged
}

/** The leading dimension for a window already known not to be [Addressing.Staged]. */
internal fun leadingDimension(window: MatrixWindow, addressing: Addressing): Int = when (addressing) {
    Addressing.ColumnMajor -> window.columnStride
    Addressing.RowMajor -> window.rowStride
    Addressing.Staged -> maxOf(1, window.rows)
}

/**
 * The lowest storage index the window touches, which is the pointer BLAS is handed.
 *
 * For a positive stride that is the window's own offset. For a negative one it is the far end, because BLAS
 * walks a negatively stepped vector from the lowest address upward and treats the last element it reaches as
 * the logical first.
 */
internal fun baseIndex(window: VectorWindow): Int =
    if (window.stride >= 0) window.offset else window.offset + (window.size - 1) * window.stride

/**
 * The number of storage entries a directly addressed window spans, from its offset to its far corner.
 *
 * Both strides are positive for a window that is not [Addressing.Staged], so the span runs forward from the
 * offset and includes whatever padding the leading dimension leaves between lines. That padding is carried
 * across a transfer unchanged, which is what lets the same span be written back without disturbing storage the
 * operation never touched.
 */
internal fun reachableSpan(window: MatrixWindow): Int {
    if (window.rows == 0 || window.columns == 0) return 0
    return (window.rows - 1) * window.rowStride + (window.columns - 1) * window.columnStride + 1
}

/** Copies [window] into [destination] as a packed column-major block, applying its declared structure. */
internal fun stageInto(window: MatrixWindow, destination: DoubleArray) {
    var target = 0
    for (column in 0 until window.columns) {
        for (row in 0 until window.rows) {
            destination[target++] = window.value(row, column)
        }
    }
}

/** Copies a packed column-major block back into the general [window] it was staged from. */
internal fun unstageFrom(source: DoubleArray, window: MatrixWindow) {
    var origin = 0
    for (column in 0 until window.columns) {
        for (row in 0 until window.rows) {
            window.data[window.index(row, column)] = source[origin++]
        }
    }
}

/** Whether the window's entries can be written, which a structure with implicit entries forbids. */
internal fun writable(window: MatrixWindow): Boolean = when (window.structure) {
    MatrixStructure.General, MatrixStructure.SymmetricLower, MatrixStructure.SymmetricUpper,
    MatrixStructure.TriangularLower, MatrixStructure.TriangularUpper,
    -> true

    MatrixStructure.UnitLower, MatrixStructure.UnitUpper -> false
}
