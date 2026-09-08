package com.eignex.koblas.dense

import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import kotlin.math.min

/*
 * The packed matrix product: both operands are copied into panels laid out in the order the kernel reads
 * them, and a tile of C is accumulated in [F64Kernels.gemmTile].
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
 * Side of the square tile the portable kernel holds, in both rows and columns.
 *
 * Not in [DenseTuning] with the block sizes: the blocks are numbers this file compares against, while this
 * is the shape [portableGemmTile] is written out to, sixteen named accumulators. Changing it means writing
 * a different kernel, so it is a property of that code rather than a knob on it.
 */
internal const val PORTABLE_TILE: Int = 4

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
    kernels: F64Kernels,
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
    kernels: F64Kernels,
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
    kernels: F64Kernels,
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
                        packB(b, ldb, transposeB, symmetricB, packedB, cols, depthBlock, depth, columnBlock, columns)
                        var rowBlock = 0
                        while (rowBlock < m) {
                            val rowCount = min(mc, m - rowBlock)
                            packA(
                                a, lda, transposeA, symmetricA, alpha, packedA,
                                rows, rowBlock, rowCount, depthBlock, depth,
                            )
                            macroKernel(
                                kernels, packedA, packedB, c, m,
                                rowBlock, rowCount, columnBlock, columns, depth, tile, triangle,
                            )
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
 * Copies a row block of op(A) into panels of [rows] rows, each holding one step of the shared dimension
 * contiguously, and scales it by [alpha] on the way. Scaling here rather than in the tile costs one
 * multiply per packed element instead of one per multiply-add, and the panel is read many times.
 */
@Suppress("LongParameterList") // the source with its layout, the destination panel, and both windows
private fun packA(
    a: DoubleArray,
    lda: Int,
    transposeA: Boolean,
    symmetric: Boolean?,
    alpha: Double,
    packed: DoubleArray,
    rows: Int,
    rowBlock: Int,
    rowCount: Int,
    depthBlock: Int,
    depth: Int,
) {
    var target = 0
    var row = 0
    while (row < rowCount) {
        val present = min(rows, rowCount - row)
        for (step in 0 until depth) {
            val i = rowBlock + row
            val p = depthBlock + step
            if (symmetric != null) {
                // A symmetric operand is its own transpose, so the flag says nothing here; what matters is
                // which side of the diagonal holds the value.
                for (r in 0 until present) {
                    val stored = if (symmetric) i + r >= p else i + r <= p
                    packed[target + r] =
                        alpha * if (stored) a[i + r + p * lda] else a[p + (i + r) * lda]
                }
            } else if (transposeA) {
                for (r in 0 until present) packed[target + r] = alpha * a[p + (i + r) * lda]
            } else {
                val source = i + p * lda
                for (r in 0 until present) packed[target + r] = alpha * a[source + r]
            }
            for (r in present until rows) packed[target + r] = 0.0
            target += rows
        }
        row += rows
    }
}

/** Copies a column block of op(B) into panels of [cols] columns, one step of the shared dimension per row. */
@Suppress("LongParameterList") // the source with its layout, the destination panel, and both windows
private fun packB(
    b: DoubleArray,
    ldb: Int,
    transposeB: Boolean,
    symmetric: Boolean?,
    packed: DoubleArray,
    cols: Int,
    depthBlock: Int,
    depth: Int,
    columnBlock: Int,
    columns: Int,
) {
    var target = 0
    var column = 0
    while (column < columns) {
        val present = min(cols, columns - column)
        for (step in 0 until depth) {
            val p = depthBlock + step
            val j = columnBlock + column
            if (symmetric != null) {
                for (q in 0 until present) {
                    val stored = if (symmetric) p >= j + q else p <= j + q
                    packed[target + q] = if (stored) b[p + (j + q) * ldb] else b[j + q + p * ldb]
                }
            } else if (transposeB) {
                val source = j + p * ldb
                for (q in 0 until present) packed[target + q] = b[source + q]
            } else {
                for (q in 0 until present) packed[target + q] = b[p + (j + q) * ldb]
            }
            for (q in present until cols) packed[target + q] = 0.0
            target += cols
        }
        column += cols
    }
}

/**
 * Walks the packed panels tile by tile. A full selected tile is accumulated straight into C; a short edge
 * or a tile crossing [triangle]'s diagonal goes through [tile] first, because the kernel writes its whole
 * shape and the caller must discard either padding or entries from the unselected triangle.
 */
@Suppress("LongParameterList") // both panels, the destination with its window, and the scratch tile
private fun macroKernel(
    kernels: F64Kernels,
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
            val lastRow = targetRow + presentRows - 1
            val lastColumn = targetColumn + presentColumns - 1
            val outside = triangle != null && if (triangle) lastRow < targetColumn else targetRow > lastColumn
            if (outside) {
                row += presentRows
                continue
            }
            val inside = triangle == null || if (triangle) targetRow >= lastColumn else lastRow <= targetColumn
            val target = targetRow + targetColumn * ldc
            if (inside && presentRows == rows && presentColumns == cols) {
                kernels.gemmTile(depth, packedA, aPanel, packedB, bPanel, c, target, ldc)
            } else {
                tile.fill(0.0, 0, rows * cols)
                kernels.gemmTile(depth, packedA, aPanel, packedB, bPanel, tile, 0, rows)
                for (q in 0 until presentColumns) {
                    val source = q * rows
                    val destination = target + q * ldc
                    for (r in 0 until presentRows) {
                        val selected = triangle == null || if (triangle) {
                            targetRow + r >= targetColumn + q
                        } else {
                            targetRow + r <= targetColumn + q
                        }
                        if (selected) c[destination + r] += tile[source + r]
                    }
                }
            }
            row += presentRows
        }
        column += presentColumns
    }
}

/**
 * The portable tile: four rows by four columns held in sixteen scalar accumulators.
 *
 * This is what [F64Kernels.gemmTile] runs where a target has nothing better. The accumulators are separate
 * locals rather than an array so that a compiler with registers to spare can keep them there, which is the
 * whole reason the tile exists; an array would put them back in memory and leave the product bounded by
 * cache traffic again.
 */
@Suppress("LongParameterList") // two packed panels, a destination tile, and the shared depth
internal fun portableGemmTile(
    depth: Int,
    packedA: DoubleArray,
    aOff: Int,
    packedB: DoubleArray,
    bOff: Int,
    c: DoubleArray,
    cOff: Int,
    ldc: Int,
) {
    var c00 = 0.0
    var c10 = 0.0
    var c20 = 0.0
    var c30 = 0.0
    var c01 = 0.0
    var c11 = 0.0
    var c21 = 0.0
    var c31 = 0.0
    var c02 = 0.0
    var c12 = 0.0
    var c22 = 0.0
    var c32 = 0.0
    var c03 = 0.0
    var c13 = 0.0
    var c23 = 0.0
    var c33 = 0.0
    var ap = aOff
    var bp = bOff
    for (p in 0 until depth) {
        val a0 = packedA[ap]
        val a1 = packedA[ap + 1]
        val a2 = packedA[ap + 2]
        val a3 = packedA[ap + 3]
        var coefficient = packedB[bp]
        c00 += a0 * coefficient
        c10 += a1 * coefficient
        c20 += a2 * coefficient
        c30 += a3 * coefficient
        coefficient = packedB[bp + 1]
        c01 += a0 * coefficient
        c11 += a1 * coefficient
        c21 += a2 * coefficient
        c31 += a3 * coefficient
        coefficient = packedB[bp + 2]
        c02 += a0 * coefficient
        c12 += a1 * coefficient
        c22 += a2 * coefficient
        c32 += a3 * coefficient
        coefficient = packedB[bp + 3]
        c03 += a0 * coefficient
        c13 += a1 * coefficient
        c23 += a2 * coefficient
        c33 += a3 * coefficient
        ap += PORTABLE_TILE
        bp += PORTABLE_TILE
    }
    c[cOff] += c00
    c[cOff + 1] += c10
    c[cOff + 2] += c20
    c[cOff + 3] += c30
    c[cOff + ldc] += c01
    c[cOff + ldc + 1] += c11
    c[cOff + ldc + 2] += c21
    c[cOff + ldc + 3] += c31
    c[cOff + 2 * ldc] += c02
    c[cOff + 2 * ldc + 1] += c12
    c[cOff + 2 * ldc + 2] += c22
    c[cOff + 2 * ldc + 3] += c32
    c[cOff + 3 * ldc] += c03
    c[cOff + 3 * ldc + 1] += c13
    c[cOff + 3 * ldc + 2] += c23
    c[cOff + 3 * ldc + 3] += c33
}
