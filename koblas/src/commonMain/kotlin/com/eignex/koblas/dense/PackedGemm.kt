package com.eignex.koblas.dense

import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import kotlin.math.min

/*
 * The packed matrix product: both operands are copied into panels laid out in the order the kernel reads
 * them, and a tile of C is accumulated in [PackedKernels.gemmTile].
 *
 * This is the whole of the matrix product, for every shape and every target. What it replaced was built
 * out of AXPY calls, so a block of C was read and written once for every step of the shared dimension.
 * Here C is touched once per tile per depth block instead, and both operand streams are contiguous, which
 * is what lets a vector unit run at its arithmetic rate rather than at the rate cache can feed it.
 * Measured against the old update on Level3Benchmark.gemm at order 1024, this is 3.3 times as fast and
 * within a tenth of a single-threaded OpenBLAS.
 *
 * The blocking and the packing are here rather than behind the seam because they are the same everywhere.
 * The tile is the one piece that varies by target, and it varies there rather than here.
 */

/**
 * Adds `alpha * op(A) * op(B)` into [c], where op is a transpose when the matching flag is set.
 *
 * [a] is `m x k` before its transpose flag, [b] is `k x n` before its, and [c] is `m x n` with rows
 * contiguous. Beta is the caller's business and has already been applied.
 *
 * [symmetricA] and [symmetricB] describe an operand held as one triangle of a symmetric matrix: true for a
 * lower triangle, false for an upper one, null for an ordinary matrix. Packing then mirrors across the
 * diagonal as it copies, which is the whole of what a symmetric product needs. Reading the triangle here
 * rather than materialising the full matrix first means the copy the packing already performs does the
 * mirroring for free, and the kernel never learns that anything was symmetric.
 */
@Suppress("LongParameterList") // both operands, both transpose flags, three dimensions and the scratch
internal fun packedGemm(
    kernels: PackedKernels,
    alpha: Double,
    a: DoubleArray,
    lda: Int,
    transposeA: Boolean,
    b: DoubleArray,
    ldb: Int,
    transposeB: Boolean,
    c: DoubleArray,
    m: Int,
    n: Int,
    k: Int,
    workspace: Workspace?,
    symmetricA: Boolean? = null,
    symmetricB: Boolean? = null,
): Unit = packedProduct(
    kernels, alpha, a, lda, transposeA, b, ldb, transposeB, c, m, n, k, workspace,
    symmetricA, symmetricB, triangle = null,
)

/**
 * Adds `alpha * op(A) * op(B)` into one triangle of [c], using the same packed panels as [packedGemm].
 * Tiles outside the triangle are skipped, interior tiles write C directly, and diagonal tiles pass through
 * scratch so only their selected entries are copied back.
 */
@Suppress("LongParameterList") // both operands, both transpose flags, three dimensions and the scratch
internal fun packedTriangularGemm(
    kernels: PackedKernels,
    alpha: Double,
    a: DoubleArray,
    lda: Int,
    transposeA: Boolean,
    b: DoubleArray,
    ldb: Int,
    transposeB: Boolean,
    c: DoubleArray,
    order: Int,
    k: Int,
    lower: Boolean,
    workspace: Workspace?,
): Unit = packedProduct(
    kernels, alpha, a, lda, transposeA, b, ldb, transposeB, c, order, order, k, workspace,
    symmetricA = null, symmetricB = null, triangle = lower,
)

/** Shared blocking for rectangular and triangular packed products. */
@Suppress("LongParameterList") // both operands, both transpose flags, three dimensions and the scratch
private fun packedProduct(
    kernels: PackedKernels,
    alpha: Double,
    a: DoubleArray,
    lda: Int,
    transposeA: Boolean,
    b: DoubleArray,
    ldb: Int,
    transposeB: Boolean,
    c: DoubleArray,
    m: Int,
    n: Int,
    k: Int,
    workspace: Workspace?,
    symmetricA: Boolean?,
    symmetricB: Boolean?,
    triangle: Boolean?,
) {
    val rows = kernels.gemmTileRows
    val cols = kernels.gemmTileCols
    val blockRows = DenseTuning.packedBlockRows
    val blockColumns = DenseTuning.packedBlockColumns
    // Blocked to the problem, not to the constants: a product smaller than a block would otherwise borrow
    // and clear scratch for rows and columns it does not have, which on a small product costs more than the
    // arithmetic. Rounded up to whole tiles, because a short edge is packed with its padding.
    val mc = min(blockRows - blockRows % rows, roundUp(m, rows))
    val nc = min(blockColumns - blockColumns % cols, roundUp(n, cols))
    val kc = min(DenseTuning.packedBlockDepth, k)
    workspace.borrow(mc * kc) { packedA ->
        workspace.borrow(kc * nc) { packedB ->
            workspace.borrow(rows * cols) { tile ->
                var columnBlock = 0
                while (columnBlock < n) {
                    val columns = min(nc, n - columnBlock)
                    var depthBlock = 0
                    while (depthBlock < k) {
                        val depth = min(kc, k - depthBlock)
                        packRightLayout(
                            b,
                            ldb,
                            packedB,
                            0,
                            depth,
                            columns,
                            depthBlock,
                            columnBlock,
                            transposeB,
                            if (symmetricB == null) PackedPanelStructure.General else PackedPanelStructure.Symmetric,
                            symmetricB ?: false,
                            unitDiagonal = false,
                            cols,
                        )
                        var rowBlock = 0
                        while (rowBlock < m) {
                            val rowCount = min(mc, m - rowBlock)
                            val outside = triangle != null && if (triangle) {
                                rowBlock + rowCount - 1 < columnBlock
                            } else {
                                rowBlock > columnBlock + columns - 1
                            }
                            if (!outside) {
                                packLeftLayout(
                                    a,
                                    lda,
                                    packedA,
                                    0,
                                    rowCount,
                                    depth,
                                    rowBlock,
                                    depthBlock,
                                    transposeA,
                                    alpha,
                                    if (symmetricA == null) {
                                        PackedPanelStructure.General
                                    } else {
                                        PackedPanelStructure.Symmetric
                                    },
                                    symmetricA ?: false,
                                    unitDiagonal = false,
                                    rows,
                                )
                                macroKernel(
                                    kernels, packedA, packedB, c, m,
                                    rowBlock, rowCount, columnBlock, columns, depth, tile, triangle,
                                )
                            }
                            rowBlock += rowCount
                        }
                        depthBlock += depth
                    }
                    columnBlock += columns
                }
            }
        }
    }
}

/** The next multiple of [step] at or above [value]. */
private fun roundUp(value: Int, step: Int): Int = (value + step - 1) / step * step

/**
 * Walks the packed panels tile by tile. A full selected tile is accumulated straight into C; a short edge
 * or a tile crossing [triangle]'s diagonal goes through [tile] first, because the kernel writes its whole
 * shape and the caller must discard either padding or entries from the unselected triangle.
 */
@Suppress("LongParameterList") // both panels, the destination with its window, and the scratch tile
private fun macroKernel(
    kernels: PackedKernels,
    packedA: DoubleArray,
    packedB: DoubleArray,
    c: DoubleArray,
    ldc: Int,
    rowBlock: Int,
    rowCount: Int,
    columnBlock: Int,
    columns: Int,
    depth: Int,
    tile: DoubleArray,
    triangle: Boolean?,
) {
    val rows = kernels.gemmTileRows
    val cols = kernels.gemmTileCols
    var column = 0
    while (column < columns) {
        val presentColumns = min(cols, columns - column)
        val bPanel = (column / cols) * depth * cols
        var row = 0
        while (row < rowCount) {
            val presentRows = min(rows, rowCount - row)
            val aPanel = (row / rows) * depth * rows
            val targetRow = rowBlock + row
            val targetColumn = columnBlock + column
            val target = targetRow + targetColumn * ldc
            accumulatePackedProductTile(
                kernels, depth, packedA, aPanel, packedB, bPanel, c, target, ldc,
                presentRows, presentColumns, targetRow, targetColumn, triangle, tile,
            )
            row += presentRows
        }
        column += presentColumns
    }
}
