@file:Suppress("LongParameterList") // a packing call carries its source window, its extents and its group

package com.eignex.koblas.internal.dense

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*

/**
 * Copies the `rows` by `depth` window of `op(A)` starting at ([rowStart], [depthStart]) into left groups.
 *
 * [lda] is the stored leading dimension of the source, before [transpose] is considered, so an untransposed
 * operand's rows are adjacent and a transposed one's are [lda] apart. [sourceOffset] is where the operand's
 * own logical origin sits in [source], which is how a strip of a larger matrix is packed without being
 * copied out of it first; a caller works it out from the transpose it is passing. A group the window does
 * not fill is padded with positive zero, which is what lets a tile accumulate over it and store nothing
 * there.
 */
internal fun packLeftPanel(
    source: DoubleArray,
    sourceOffset: Int,
    lda: Int,
    transpose: Boolean,
    rowStart: Int,
    depthStart: Int,
    rows: Int,
    depth: Int,
    destination: DoubleArray,
    destinationOffset: Int,
    group: Int,
) {
    // An empty window has no entries and no padding, and walking its groups anyway would step a counter
    // through an extent large enough to wrap it.
    if (rows <= 0 || depth <= 0) return
    var target = destinationOffset
    var row = 0
    while (row < rows) {
        val present = if (group < rows - row) group else rows - row
        var step = 0
        while (step < depth) {
            var lane = 0
            if (transpose) {
                val base = sourceOffset + depthStart + step + (rowStart + row) * lda
                while (lane < present) {
                    destination[target + lane] = source[base + lane * lda]
                    lane++
                }
            } else {
                val base = sourceOffset + rowStart + row + (depthStart + step) * lda
                while (lane < present) {
                    destination[target + lane] = source[base + lane]
                    lane++
                }
            }
            if (present < group) destination.fill(0.0, target + present, target + group)
            target += group
            step++
        }
        row += group
    }
}

/** The right-operand counterpart of [packLeftPanel], grouping columns of `op(B)` instead of rows. */
internal fun packRightPanel(
    source: DoubleArray,
    sourceOffset: Int,
    ldb: Int,
    transpose: Boolean,
    depthStart: Int,
    columnStart: Int,
    depth: Int,
    columns: Int,
    destination: DoubleArray,
    destinationOffset: Int,
    group: Int,
) {
    if (columns <= 0 || depth <= 0) return
    var target = destinationOffset
    var column = 0
    while (column < columns) {
        val present = if (group < columns - column) group else columns - column
        var step = 0
        while (step < depth) {
            var lane = 0
            if (transpose) {
                val base = sourceOffset + columnStart + column + (depthStart + step) * ldb
                while (lane < present) {
                    destination[target + lane] = source[base + lane]
                    lane++
                }
            } else {
                val base = sourceOffset + depthStart + step + (columnStart + column) * ldb
                while (lane < present) {
                    destination[target + lane] = source[base + lane * ldb]
                    lane++
                }
            }
            if (present < group) destination.fill(0.0, target + present, target + group)
            target += group
            step++
        }
        column += group
    }
}
