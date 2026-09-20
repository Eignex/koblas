@file:Suppress("LongParameterList") // a packing call carries its source window, its extents and its group

package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.requireShape

/** Which side of a product a packed operand was prepared for. */
public enum class PackedRole {
    /** The left operand, whose rows are grouped and whose columns are the shared dimension. */
    Left,

    /** The right operand, whose columns are grouped and whose rows are the shared dimension. */
    Right,
}

/**
 * The physical shape of a packed product operand, and the whole of what a consumer has to agree with.
 *
 * A packed operand is stored in groups of [group] adjacent entries of the grouped dimension: rows for a
 * [PackedRole.Left] operand and columns for a [PackedRole.Right] one. Inside a group the steps of the shared
 * dimension follow one another, so a group is [depth] blocks of [group] entries and the group that holds
 * lane `l` starts [groupStride] entries after the one before it. A final group that the extent does not fill
 * is padded with positive zero so that every group has the same shape. What that padding promises is that
 * the lanes it occupies are never stored, not that they contribute nothing: a padded zero against an infinite
 * entry of the other operand evaluates to a NaN, which stays in a lane the destination does not have.
 *
 * These four numbers are the description. There is no format version, alignment promise or vector length
 * here: a heap `DoubleArray` gives natural alignment and nothing stronger, and the one fact a consumer needs
 * beyond the extents is [group]. That has to match the one tile dimension this operand is grouped along,
 * which is [DenseProductKernels.tileRows] for a left panel and [DenseProductKernels.tileColumns] for a right
 * one, and nothing about the other dimension: a left panel grouped by eight is readable by an eight by four
 * tile and by an eight by two one alike, and a right panel grouped by four by an eight by four tile and a
 * four by four one. [requireUsableBy] refuses the rest, before anything is written.
 *
 * An operand with no rows or no columns has no entries and needs no storage. Its strides are zero rather
 * than a product of extents it never uses, so that a matrix with a large empty dimension describes itself
 * instead of overflowing an intermediate nobody reads.
 *
 * @property role which operand of a product this was packed for.
 * @property rows logical rows of the matrix this represents, after any transpose the packing applied.
 * @property columns logical columns of the matrix this represents, after any transpose the packing applied.
 * @property group adjacent entries of the grouped dimension in one depth step.
 */
public class PackedLayout(
    public val role: PackedRole,
    public val rows: Int,
    public val columns: Int,
    public val group: Int,
) {
    /** Steps of the shared dimension this layout holds, which is the columns of a left operand. */
    public val depth: Int get() = if (role == PackedRole.Left) columns else rows

    /** Entries of the grouped dimension, which is the rows of a left operand. */
    public val extent: Int get() = if (role == PackedRole.Left) rows else columns

    /** Distance between one group and the next, which is [depth] blocks of [group] entries. */
    public val groupStride: Int

    /** Entries in the complete physical extent, padding included. */
    public val storageSize: Int

    init {
        require(rows >= 0 && columns >= 0) { "negative packed extent" }
        require(group > 0) { "packed group must be positive" }
        if (rows == 0 || columns == 0) {
            groupStride = 0
            storageSize = 0
        } else {
            groupStride = checkedPackedSize(depth.toLong() * group)
            storageSize = checkedPackedSize((extent.toLong() + group - 1) / group * groupStride.toLong())
        }
    }

    /**
     * Where logical entry ([row], [column]) sits, relative to the start of the panel.
     *
     * Public because this formula is the layout: a caller writing its own packer, or filling a panel to hand
     * to a [DenseProductKernels] block directly, needs to know where an entry goes, and a description whose
     * one operative detail has to be inferred is not a description.
     */
    public fun index(row: Int, column: Int): Int {
        val lane = if (role == PackedRole.Left) row else column
        val step = if (role == PackedRole.Left) column else row
        return lane / group * groupStride + step * group + lane % group
    }

    override fun toString(): String =
        "PackedLayout(${role.name.lowercase()} ${rows}x$columns group=$group size=$storageSize)"

    /**
     * Refuses a consumer this panel was not packed for, before it can write anything.
     *
     * Both halves matter. A left panel in the right operand's position has its shared dimension on the wrong
     * axis and would read transposed nonsense, and a panel grouped along one width read by a tile grouped
     * along another would take a neighbouring row's value for its own. Neither is detectable from the
     * numbers afterwards. [tile] is the one dimension this operand is grouped along, so a consumer whose
     * other dimension differs is compatible and is not refused here.
     */
    internal fun requireUsableBy(expected: PackedRole, tile: Int) {
        requireShape(
            role == expected,
        ) { "packed operand is a ${role.name.lowercase()} panel, not a ${expected.name.lowercase()} one" }
        requireShape(
            group == tile,
        ) { "packed operand is grouped by $group, which this kernel's tile of $tile cannot read" }
    }
}

/**
 * A matrix already copied into the grouped layout a [DenseProductKernels] tile reads.
 *
 * Retained packing exists so a caller multiplying the same operand many times pays the copy once. The object
 * owns the array it was packed into and hands it to nobody: mutating the matrix it came from afterwards does
 * not change it, and neither does anything else. It holds no workspace loan, so it stays valid after the call
 * that produced it returns, and being immutable it is safe to read from several products at once.
 *
 * What it can be used with is [layout]'s business and is checked before a product writes anything. It is not
 * a [com.eignex.koblas.Matrix]: the logical entries are readable through [get] for a test or a diagnostic,
 * but a packed operand is a physical layout prepared for one kernel rather than a storage to compute over.
 */
public class PackedMatrix internal constructor(
    internal val values: DoubleArray,
    /** The physical layout, which is what a consumer is checked against. */
    public val layout: PackedLayout,
) {
    /** Logical rows of the matrix this represents. */
    public val rows: Int get() = layout.rows

    /** Logical columns of the matrix this represents. */
    public val columns: Int get() = layout.columns

    /** The logical entry at ([row], [column]), read back out of the packed storage. */
    public operator fun get(row: Int, column: Int): Double {
        if (row !in 0 until rows || column !in 0 until columns) {
            throw IndexOutOfBoundsException("packed index ($row, $column) outside ${rows}x$columns")
        }
        return values[layout.index(row, column)]
    }

    override fun toString(): String =
        "PackedMatrix(${layout.role.name.lowercase()} ${rows}x$columns group=${layout.group})"
}

/** Packs `op(A)` for the left of a product, in groups of [group] rows. */
internal fun packedLeft(a: DenseMatrix, transpose: Boolean, group: Int): PackedMatrix {
    val rows = if (transpose) a.cols else a.rows
    val depth = if (transpose) a.rows else a.cols
    val layout = PackedLayout(PackedRole.Left, rows, depth, group)
    val values = DoubleArray(layout.storageSize)
    packLeftPanel(a.values, 0, a.rows, transpose, 0, 0, rows, depth, values, 0, group)
    return PackedMatrix(values, layout)
}

/** Packs `op(B)` for the right of a product, in groups of [group] columns. */
internal fun packedRight(b: DenseMatrix, transpose: Boolean, group: Int): PackedMatrix {
    val depth = if (transpose) b.cols else b.rows
    val columns = if (transpose) b.rows else b.cols
    val layout = PackedLayout(PackedRole.Right, depth, columns, group)
    val values = DoubleArray(layout.storageSize)
    packRightPanel(b.values, 0, b.rows, transpose, 0, 0, depth, columns, values, 0, group)
    return PackedMatrix(values, layout)
}

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

/** A packed extent that does not fit an array, reported before anything is allocated or written. */
internal fun checkedPackedSize(size: Long): Int {
    if (size > Int.MAX_VALUE.toLong()) {
        throw DimensionMismatch(
            "packed operand needs $size entries, which no array holds",
        )
    }
    return size.toInt()
}
