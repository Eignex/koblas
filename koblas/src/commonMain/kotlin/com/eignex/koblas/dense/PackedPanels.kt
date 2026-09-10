package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.ExperimentalKoblasApi
import com.eignex.koblas.Workspace

/**
 * Packing helpers for retaining matrix panels in the format consumed by koblas's built-in level-3 kernels.
 *
 * A left panel contains row tiles in order. Within each tile, every shared-dimension step contributes
 * [tileRows] contiguous values. A right panel contains column tiles in order, with [tileColumns] contiguous
 * values for every shared-dimension step. Short edge groups are filled with zeroes. These formats are not
 * compact column-major matrices: use [writeLeft] or [writeRight] to copy their valid entries back.
 *
 * All buffers belong to the caller and may be retained or reused across calls. Packing and writeback allocate
 * nothing when the matrix backing and packed buffer differ. When they are the same array, `workspace` lends a
 * staging copy so overlapping input and output are safe; omitting it allocates that staging copy.
 *
 * The normalized packed triangular solve is `X * T = B`: `T` is a single right-format square panel with order
 * at most [tileColumns], while `B` and the overwritten `X` are a single left-format panel with at most [tileRows]
 * rows and the same depth as `T`. This is also the column-major tile written by [PackedKernels.gemmTile], so a
 * solved panel can be retained for later packed updates without conversion.
 */
@ExperimentalKoblasApi
public object PackedPanels {
    /** Number of contiguous values in each shared-dimension step of a left panel on this platform. */
    public val tileRows: Int get() = platformPackedKernels.gemmTileRows

    /** Number of contiguous values in each shared-dimension step of a right panel on this platform. */
    public val tileColumns: Int get() = platformPackedKernels.gemmTileCols

    /** Exact number of doubles needed for a left panel representing a [rows] by [depth] logical matrix. */
    public fun leftSize(rows: Int, depth: Int): Int = packedLeftSize(rows, depth, tileRows)

    /** Exact number of doubles needed for a right panel representing a [depth] by [columns] logical matrix. */
    public fun rightSize(depth: Int, columns: Int): Int = packedRightSize(depth, columns, tileColumns)

    /**
     * Packs a [rows] by [depth] window of `op(source)` as a left panel, multiplying valid entries by [alpha].
     * [sourceRow] and [sourceColumn] address the logical matrix after [transpose]. Padding is always positive
     * zero, including when [alpha] is NaN or infinite. [destinationOffset] starts the caller-owned output window.
     */
    @Suppress("LongParameterList") // source and destination windows plus the operation applied while copying
    public fun packLeft(
        source: DenseMatrix,
        destination: DoubleArray,
        rows: Int,
        depth: Int,
        sourceRow: Int = 0,
        sourceColumn: Int = 0,
        transpose: Boolean = false,
        alpha: Double = 1.0,
        destinationOffset: Int = 0,
        workspace: Workspace? = null,
    ) {
        packLeftPanel(
            source, destination, rows, depth, sourceRow, sourceColumn, transpose, alpha,
            destinationOffset, workspace, PackedPanelStructure.General,
        )
    }

    /**
     * Packs a symmetric window as a left panel, reading only the triangle selected by [lower] and mirroring it.
     * The source must be square. [alpha] scales valid entries; padding remains positive zero.
     */
    @Suppress("LongParameterList") // source and destination windows plus the structure applied while copying
    public fun packSymmetricLeft(
        source: DenseMatrix,
        destination: DoubleArray,
        rows: Int,
        depth: Int,
        lower: Boolean,
        sourceRow: Int = 0,
        sourceColumn: Int = 0,
        alpha: Double = 1.0,
        destinationOffset: Int = 0,
        workspace: Workspace? = null,
    ) {
        packLeftPanel(
            source, destination, rows, depth, sourceRow, sourceColumn, false, alpha,
            destinationOffset, workspace, PackedPanelStructure.Symmetric, lower,
        )
    }

    /**
     * Packs a triangular window of `op(source)` as a left panel. Entries outside the stored triangle become
     * positive zero. With [unitDiagonal], diagonal entries are materialized as one without reading the source,
     * then scaled by [alpha]. [lower] identifies the source triangle before [transpose].
     */
    @Suppress("LongParameterList") // source and destination windows plus the structure applied while copying
    public fun packTriangularLeft(
        source: DenseMatrix,
        destination: DoubleArray,
        rows: Int,
        depth: Int,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiagonal: Boolean = false,
        sourceRow: Int = 0,
        sourceColumn: Int = 0,
        alpha: Double = 1.0,
        destinationOffset: Int = 0,
        workspace: Workspace? = null,
    ) {
        packLeftPanel(
            source, destination, rows, depth, sourceRow, sourceColumn, transpose, alpha,
            destinationOffset, workspace, PackedPanelStructure.Triangular, lower, unitDiagonal,
        )
    }

    /**
     * Packs a [depth] by [columns] window of `op(source)` as a right panel. [sourceRow] and [sourceColumn]
     * address the logical matrix after [transpose]. Valid entries are copied without scaling and edge padding
     * is positive zero.
     */
    @Suppress("LongParameterList") // source and destination windows plus the operation applied while copying
    public fun packRight(
        source: DenseMatrix,
        destination: DoubleArray,
        depth: Int,
        columns: Int,
        sourceRow: Int = 0,
        sourceColumn: Int = 0,
        transpose: Boolean = false,
        destinationOffset: Int = 0,
        workspace: Workspace? = null,
    ) {
        packRightPanel(
            source, destination, depth, columns, sourceRow, sourceColumn, transpose,
            destinationOffset, workspace, PackedPanelStructure.General,
        )
    }

    /**
     * Packs a symmetric window as a right panel, reading only the triangle selected by [lower] and mirroring it.
     * The source must be square. Valid entries are not scaled and padding is positive zero.
     */
    @Suppress("LongParameterList") // source and destination windows plus the structure applied while copying
    public fun packSymmetricRight(
        source: DenseMatrix,
        destination: DoubleArray,
        depth: Int,
        columns: Int,
        lower: Boolean,
        sourceRow: Int = 0,
        sourceColumn: Int = 0,
        destinationOffset: Int = 0,
        workspace: Workspace? = null,
    ) {
        packRightPanel(
            source, destination, depth, columns, sourceRow, sourceColumn, false,
            destinationOffset, workspace, PackedPanelStructure.Symmetric, lower,
        )
    }

    /**
     * Packs a triangular window of `op(source)` as a right panel. Entries outside the stored triangle become
     * positive zero. With [unitDiagonal], diagonal entries are materialized as one without reading the source.
     * [lower] identifies the source triangle before [transpose]; diagonals are never inverted or scaled.
     */
    @Suppress("LongParameterList") // source and destination windows plus the structure applied while copying
    public fun packTriangularRight(
        source: DenseMatrix,
        destination: DoubleArray,
        depth: Int,
        columns: Int,
        lower: Boolean,
        transpose: Boolean = false,
        unitDiagonal: Boolean = false,
        sourceRow: Int = 0,
        sourceColumn: Int = 0,
        destinationOffset: Int = 0,
        workspace: Workspace? = null,
    ) {
        packRightPanel(
            source, destination, depth, columns, sourceRow, sourceColumn, transpose,
            destinationOffset, workspace, PackedPanelStructure.Triangular, lower, unitDiagonal,
        )
    }

    /**
     * Overwrites a destination window with the valid entries of a left panel. With [transpose], packed entry
     * `(i, p)` is written to destination entry `(p, i)`. Padding is ignored. Same-array overlap is safe through
     * `workspace` staging.
     */
    @Suppress("LongParameterList") // source and destination windows plus the operation applied while copying
    public fun writeLeft(
        source: DoubleArray,
        destination: DenseMatrix,
        rows: Int,
        depth: Int,
        sourceOffset: Int = 0,
        destinationRow: Int = 0,
        destinationColumn: Int = 0,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ) {
        writeLeftPanel(
            source, destination, rows, depth, sourceOffset, destinationRow, destinationColumn,
            transpose, workspace,
        )
    }

    /**
     * Overwrites a destination window with the valid entries of a right panel. With [transpose], packed entry
     * `(p, j)` is written to destination entry `(j, p)`. Padding is ignored. Same-array overlap is safe through
     * `workspace` staging.
     */
    @Suppress("LongParameterList") // source and destination windows plus the operation applied while copying
    public fun writeRight(
        source: DoubleArray,
        destination: DenseMatrix,
        depth: Int,
        columns: Int,
        sourceOffset: Int = 0,
        destinationRow: Int = 0,
        destinationColumn: Int = 0,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ) {
        writeRightPanel(
            source, destination, depth, columns, sourceOffset, destinationRow, destinationColumn,
            transpose, workspace,
        )
    }

    /**
     * Restores positive-zero edge padding in a left panel after a kernel that may use the physical tile as
     * scratch. Valid entries are unchanged. This allocates nothing and is safe for every caller-owned panel.
     */
    public fun clearLeftPadding(panel: DoubleArray, rows: Int, depth: Int, panelOffset: Int = 0) {
        clearLeftPanelPadding(panel, rows, depth, panelOffset, tileRows)
    }

    /**
     * Restores positive-zero edge padding in a right panel after a kernel that may use the physical tile as
     * scratch. Valid entries are unchanged. This allocates nothing and is safe for every caller-owned panel.
     */
    public fun clearRightPadding(panel: DoubleArray, depth: Int, columns: Int, panelOffset: Int = 0) {
        clearRightPanelPadding(panel, depth, columns, panelOffset, tileColumns)
    }

    /**
     * Solves the normalized packed system X * T = B in place.
     *
     * [triangle] is one right-format square panel containing the effective, non-transposed triangle.
     * [rightHandSide] is one left-format [rows] by [order] panel and is overwritten by X. Thus [rows] must
     * not exceed [tileRows] and [order] must not exceed [tileColumns]. [unitDiagonal] never reads the stored
     * diagonal, division occurs directly at each pivot, and exact-zero off-diagonal coefficients are skipped.
     *
     * The two arrays may be the same. Such aliasing is staged through [workspace], or through a temporary
     * allocation when no workspace is supplied.
     */
    @Suppress("LongParameterList") // packed source and destination windows plus triangular flags
    public fun trsm(
        triangle: DoubleArray,
        rightHandSide: DoubleArray,
        rows: Int,
        order: Int,
        lower: Boolean,
        unitDiagonal: Boolean = false,
        triangleOffset: Int = 0,
        rightHandSideOffset: Int = 0,
        workspace: Workspace? = null,
    ) {
        requireSolveShape(rows, order)
        requireArrayWindow(triangle, triangleOffset, rightSize(order, order), "packed triangle")
        requireArrayWindow(rightHandSide, rightHandSideOffset, leftSize(rows, order), "packed right-hand side")
        if (rows == 0 || order == 0) return
        withStableSource(triangle, rightHandSide, workspace) { stableTriangle ->
            platformPackedKernels.trsmTile(
                rows,
                order,
                stableTriangle,
                triangleOffset,
                lower,
                unitDiagonal,
                rightHandSide,
                rightHandSideOffset,
            )
        }
    }

    /**
     * Subtracts packedLeft * packedRight from [rightHandSide], then solves the normalized packed system
     * X * T = B without an intermediate panel write on implementations with a beneficial fused leaf.
     *
     * The product inputs use the ordinary left/right layouts with shared dimension [depth]. The triangle
     * and overwritten right-hand side have the same layouts and bounds as [trsm]. Scaling belongs in
     * [packLeft]'s alpha, so callers can reverse or otherwise scale the update without another kernel parameter.
     *
     * Any read-only input may share its backing array with [rightHandSide]. Aliased inputs are staged through
     * [workspace], or through temporary allocations when no workspace is supplied.
     */
    @Suppress("LongParameterList") // three packed source windows, one destination window and triangular flags
    public fun gemmTrsm(
        packedLeft: DoubleArray,
        packedRight: DoubleArray,
        triangle: DoubleArray,
        rightHandSide: DoubleArray,
        rows: Int,
        order: Int,
        depth: Int,
        lower: Boolean,
        unitDiagonal: Boolean = false,
        leftOffset: Int = 0,
        rightOffset: Int = 0,
        triangleOffset: Int = 0,
        rightHandSideOffset: Int = 0,
        workspace: Workspace? = null,
    ) {
        requireSolveShape(rows, order)
        require(depth >= 0) { "packed update depth must be non-negative, got $depth" }
        requireArrayWindow(packedLeft, leftOffset, leftSize(rows, depth), "packed left update")
        requireArrayWindow(packedRight, rightOffset, rightSize(depth, order), "packed right update")
        requireArrayWindow(triangle, triangleOffset, rightSize(order, order), "packed triangle")
        requireArrayWindow(rightHandSide, rightHandSideOffset, leftSize(rows, order), "packed right-hand side")
        if (rows == 0 || order == 0) return
        withStableSource(packedLeft, rightHandSide, workspace) { stableLeft ->
            withStableSource(packedRight, rightHandSide, workspace) { stableRight ->
                withStableSource(triangle, rightHandSide, workspace) { stableTriangle ->
                    platformPackedKernels.gemmTrsmTile(
                        depth,
                        rows,
                        order,
                        stableLeft,
                        leftOffset,
                        stableRight,
                        rightOffset,
                        stableTriangle,
                        triangleOffset,
                        lower,
                        unitDiagonal,
                        rightHandSide,
                        rightHandSideOffset,
                    )
                }
            }
        }
    }

    private fun requireSolveShape(rows: Int, order: Int) {
        require(rows in 0..tileRows) { "packed solve rows $rows exceed tileRows $tileRows" }
        require(order in 0..tileColumns) { "packed solve order $order exceeds tileColumns $tileColumns" }
    }
}
