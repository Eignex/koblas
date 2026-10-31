@file:Suppress(
    "VariableNaming",
    "FunctionParameterNaming",
    "TooManyFunctions",
    "LongParameterList",
    "MaxLineLength",
)

package com.eignex.koblas.internal.dense

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.operands.regionVectorOverlap
import com.eignex.koblas.internal.operands.sameRegion
import com.eignex.koblas.internal.operands.scaleRegion
import com.eignex.koblas.internal.operands.staged
import com.eignex.koblas.internal.operands.stagedRegion
import com.eignex.koblas.internal.operands.stagedVectors
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*

/**
 * Common Kotlin dense BLAS with validation, triangle selection, alias staging and shared scheduling.
 *
 * [panels] owns Level 2 arithmetic and column grouping, [products] owns product tiles, and [triangles]
 * owns diagonal substitution. Schedules use backend recommendations without assuming a fixed width.
 *
 * Structured Level 3 calls share product scheduling: selected triangles skip or merge blocks, symmetric
 * operands expose diagonal blocks and stored strips, and triangular routines interleave substitutions
 * with product windows.
 */
internal class PortableDenseBlas(
    private val vectors: DenseVectorKernels,
    private val panels: DensePanelKernels,
    private val products: DenseProductKernels = PortableProductKernels,
    private val triangles: DenseTriangularKernels = PortableTriangularKernels,
) : DenseBlas {

    override fun gemv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        transpose: Boolean,
        workspace: Workspace?,
    ) {
        requireGemvOperands(a, transpose, x.size, y.size)
        if (y.isEmpty()) return
        if (alpha == 0.0 || (if (transpose) a.rows else a.cols) == 0) {
            applyBeta(vectors, y, 0, y.size, beta)
            return
        }
        if (a.values === y || x === y) {
            gemv(alpha, a, DenseVector.wrap(x), beta, DenseVector.wrap(y), transpose, workspace)
        } else {
            gemvCore(alpha, a, x, 0, 1, beta, y, 0, 1, transpose)
        }
    }

    override fun gemv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
        transpose: Boolean,
        workspace: Workspace?,
    ) {
        requireGemvOperands(a, transpose, x.size, y.size)
        if (y.size == 0) return
        val depth = if (transpose) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            applyBeta(vectors, y.values, y.offset, y.size, beta, y.stride)
            return
        }
        // Snapshot aliased operands before scaling the destination, and retain them for the call.
        stagedRegion(workspace, a, y) { stableA ->
            staged(workspace, x, x.overlaps(y)) { stableX ->
                gemvCore(alpha, stableA, stableX.values, stableX.offset, stableX.stride, beta, y.values, y.offset, y.stride, transpose)
            }
        }
    }

    private fun gemvCore(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        transpose: Boolean,
    ) {
        val rows = a.rows
        if (transpose) {
            forEachPanel(
                a.cols,
                panels.executionGroup(PanelWork.MultiDot, rows, a.cols, xStride == 1),
            ) { start, width ->
                panels.multiDot(
                    alpha, a.values, a.offset + start * a.leadingDimension, a.leadingDimension, x,
                    xOffset, xStride, rows, width, beta, y,
                    yOffset + start * yStride, yStride,
                )
            }
        } else {
            applyBeta(vectors, y, yOffset, rows, beta, yStride)
            forEachPanel(
                a.cols,
                panels.executionGroup(PanelWork.ColumnUpdate, rows, a.cols, yStride == 1),
            ) { start, width ->
                panels.columnUpdate(
                    alpha, a.values, a.offset + start * a.leadingDimension, a.leadingDimension, x,
                    xOffset + start * xStride, xStride, rows, width, y, yOffset, yStride,
                )
            }
        }
    }

    override fun transpose(a: DenseMatrixRegion): DenseMatrix {
        val result = DenseMatrix.zero(a.cols, a.rows)
        for (j in 0 until a.cols) {
            for (i in 0 until a.rows) {
                result.values[j + i * a.cols] =
                    a.values[a.offset + i + j * a.leadingDimension]
            }
        }
        return result
    }

    override fun gemm(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        workspace: Workspace?,
    ) {
        requireGemmOperands(a, transposeA, b, transposeB, c)
        // Return before borrowing operand-sized scratch for an empty destination.
        if (c.rows == 0 || c.cols == 0) return
        val depth = if (transposeA) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            scaleRegion(c, beta)
            return
        }
        stagedRegion(workspace, a, c) { sa ->
            stagedRegion(workspace, b, c, stable = sa, original = a) { sb ->
                gemmCore(alpha, sa, transposeA, sb, transposeB, beta, c, depth, workspace)
            }
        }
    }

    /**
     * The product itself, over operands already staged against an overlap with the destination.
     *
     * Which of the two paths runs is the extents' answer rather than the engine's.
     */
    private fun gemmCore(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        depth: Int,
        workspace: Workspace?,
    ) {
        productWindow(
            products, panels, alpha, a.values, a.offset, a.leadingDimension, transposeA, b.values, b.offset,
            b.leadingDimension, transposeB,
            beta, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth, OutputTriangle.Full, workspace,
        )
    }

    override fun gemmt(
        alpha: Double,
        a: DenseMatrixRegion,
        transposeA: Boolean,
        b: DenseMatrixRegion,
        transposeB: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireGemmtOperands(a, transposeA, b, transposeB, c)
        // An empty destination needs no staging, even when operands share an empty array.
        if (c.rows == 0) return
        val depth = if (transposeA) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            scaleRegion(c, beta, lower)
            return
        }
        stagedRegion(workspace, a, c) { sa ->
            stagedRegion(workspace, b, c, stable = sa, original = a) { sb ->
                productWindow(
                    products, panels, alpha, sa.values, sa.offset, sa.leadingDimension, transposeA, sb.values,
                    sb.offset, sb.leadingDimension, transposeB,
                    beta, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth, selectedTriangle(lower),
                    workspace,
                )
            }
        }
    }

    /**
     * `y = alpha · A · x + beta · y` over the stored triangle only, one pass over each column.
     *
     * A stored column of a symmetric matrix is two things at once: the part of a column below the diagonal,
     * and the part of the mirrored row to the left of it. [DensePanelKernels.coupledUpdateDot] does both from
     * one read, and the square corner where a group's columns overlap each other is the remainder that walk
     * cannot cover, which is scalar work here.
     */
    @Suppress("LongParameterList") // the BLAS dsymv signature plus the workspace
    override fun symv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        beta: Double,
        y: DoubleArray,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSymvOperands(a, x.size, y.size)
        if (y.isEmpty()) return
        if (alpha == 0.0 || (a.rows) == 0) {
            applyBeta(vectors, y, 0, y.size, beta)
            return
        }
        if (a.values === y || x === y) {
            symv(alpha, a, DenseVector.wrap(x), beta, DenseVector.wrap(y), lower, workspace)
        } else {
            symvCore(alpha, a, x, 0, 1, beta, y, 0, 1, lower)
        }
    }

    override fun symv(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DenseVector,
        beta: Double,
        y: DenseVector,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSymvOperands(a, x.size, y.size)
        val n = a.rows
        if (alpha == 0.0 || n == 0) {
            applyBeta(vectors, y.values, y.offset, y.size, beta, y.stride)
            return
        }
        // Snapshot aliased operands before beta overwrites values still needed by the product.
        stagedRegion(workspace, a, y, lower = lower) { sa ->
            staged(workspace, x, x.overlaps(y)) { xv ->
                symvCore(alpha, sa, xv.values, xv.offset, xv.stride, beta, y.values, y.offset, y.stride, lower)
            }
        }
    }

    @Suppress("LongParameterList") // the operands, their snapshots and the BLAS dsymv scalars
    private fun symvCore(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        lower: Boolean,
    ) {
        val n = a.rows
        applyBeta(vectors, y, yOffset, a.rows, beta, yStride)
        val group = panels.executionGroup(PanelWork.CoupledDotUpdate, n, n, xStride == 1 && yStride == 1)
        forEachTrianglePanel(n, group, lower, fromDiagonal = false) { start, width, window, rows ->
            symmetricCorner(alpha, a, x, xOffset, xStride, y, yOffset, yStride, start, width, lower)
            if (rows > 0) {
                panels.coupledUpdateDot(
                    alpha, a.values, a.offset + window + start * a.leadingDimension, a.leadingDimension, x,
                    xOffset + window * xStride, rows, width, y, yOffset + window * yStride, x,
                    xOffset + start * xStride, y, yOffset + start * yStride, xStride, yStride,
                    xStride, yStride,
                )
            }
        }
    }

    /**
     * The square block where a group's own columns and rows meet, which the shared window cannot reach.
     *
     * Every stored position in it contributes twice, once down its column and once along the mirrored row,
     * except the diagonal which is its own mirror. The arithmetic is the panel's, written out: a coefficient
     * scaled by alpha multiplies down, and the reduction is scaled by alpha on its way into the destination.
     */
    private fun symmetricCorner(
        alpha: Double,
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        start: Int,
        width: Int,
        lower: Boolean,
    ) {
        for (c in start until start + width) {
            val coefficient = alpha * x[xOffset + (c) * xStride]
            for (i in start until start + width) {
                if (if (lower) i < c else i > c) continue
                val value = a.values[a.offset + i + c * a.leadingDimension]
                y[yOffset + (i) * yStride] += coefficient * value
                if (i != c) y[yOffset + (c) * yStride] += alpha * (value * x[xOffset + (i) * xStride])
            }
        }
    }

    override fun symm(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        right: Boolean,
        workspace: Workspace?,
    ) {
        requireSymmOperands(a, b, c, right)
        if (c.rows == 0 || c.cols == 0) return
        if (alpha == 0.0) {
            scaleRegion(c, beta, vectors = vectors)
            return
        }
        stagedRegion(workspace, a, c, lower = if (sameRegion(a, b)) null else lower) { sa ->
            stagedRegion(workspace, b, c, stable = sa, original = a) { sb ->
                // Scale once because multiple symmetric windows accumulate into each destination entry.
                scaleRegion(c, beta, vectors = vectors)
                symmetricProduct(
                    products, panels, alpha, sa.values, a.rows, lower, sb.values, sb.leadingDimension, c.values,
                    c.leadingDimension,
                    c.rows, c.cols, right, workspace, sa.offset, sa.leadingDimension, sb.offset, c.offset,
                )
            }
        }
    }

    @Suppress("LongParameterList") // the BLAS dsyrk signature plus the workspace

    override fun syrk(
        alpha: Double,
        a: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSyrkOperands(a, transpose, c)
        if (c.rows == 0) return
        val depth = if (transpose) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            scaleRegion(c, beta, lower)
            return
        }
        stagedRegion(workspace, a, c) { sa ->
            productWindow(
                products, panels, alpha, sa.values, sa.offset, sa.leadingDimension, transpose,
                sa.values, sa.offset, sa.leadingDimension, !transpose,
                beta, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth,
                selectedTriangle(lower), workspace,
            )
        }
    }

    override fun syr2k(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSyr2kOperands(a, b, transpose, c)
        productTriangle(alpha, a, b, transpose, beta, c, lower, doubled = true, workspace = workspace)
    }

    /**
     * `alpha · op(A) · op(B)ᵀ + beta · C` in one triangle, and the same with the operands swapped when
     * [doubled].
     *
     * A rank-k update is a product whose right operand is the left one transposed, and a rank-2k update is
     * the sum of the two products its definition names. Composing rather than fusing is deliberate: both
     * halves are ordinary windows of shared product scheduling, so each is packed, blocked and tiled like
     * any other product, while a fused traversal would need a second implementation of all of that to beat
     * them together. The local evidence compares the two.
     *
     * `beta` is carried by the first of the two and the second accumulates, which is how it reaches every
     * selected entry exactly once.
     */
    private fun productTriangle(
        alpha: Double,
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        transpose: Boolean,
        beta: Double,
        c: DenseMatrixRegion,
        lower: Boolean,
        doubled: Boolean,
        workspace: Workspace?,
    ) {
        if (c.rows == 0) return
        val depth = if (transpose) a.rows else a.cols
        if (alpha == 0.0 || depth == 0) {
            scaleRegion(c, beta, lower)
            return
        }
        val selected = selectedTriangle(lower)
        stagedRegion(workspace, a, c) { sa ->
            stagedRegion(workspace, b, c, stable = sa, original = a) { sb ->
                productWindow(
                    products, panels, alpha, sa.values, sa.offset, sa.leadingDimension, transpose, sb.values,
                    sb.offset, sb.leadingDimension, !transpose,
                    beta, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth, selected, workspace,
                )
                if (doubled) {
                    productWindow(
                        products, panels, alpha, sb.values, sb.offset, sb.leadingDimension, transpose, sa.values,
                        sa.offset, sa.leadingDimension, !transpose,
                        1.0, c.values, c.offset, c.leadingDimension, c.rows, c.cols, depth, selected, workspace,
                    )
                }
            }
        }
    }

    override fun ger(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion, workspace: Workspace?) {
        requireGerOperands(x.size, y.size, a)
        if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
        stagedVectors(workspace, x, y, a) { sx, sy ->
            gerCore(
                alpha, sx.values, sx.offset, sx.stride, sy.values, sy.offset, sy.stride,
                a, a.offset, a.rows, a.cols,
            )
        }
    }

    private fun gerCore(
        alpha: Double,
        sx: DoubleArray,
        sxOffset: Int,
        sxStride: Int,
        sy: DoubleArray,
        syOffset: Int,
        syStride: Int,
        a: DenseMatrixRegion,
        aOffset: Int,
        rows: Int,
        columns: Int,
    ) {
        forEachPanel(
            columns,
            panels.executionGroup(PanelWork.RankUpdate, rows, columns, sxStride == 1),
        ) { start, width ->
            panels.rankUpdate(
                alpha, a.values, aOffset + start * a.leadingDimension, a.leadingDimension,
                sx, sxOffset, sxStride, rows, width,
                sy, syOffset + start * syStride, syStride,
            )
        }
    }

    override fun ger(alpha: Double, x: DoubleArray, y: DoubleArray, a: DenseMatrixRegion, workspace: Workspace?) {
        requireGerOperands(x.size, y.size, a)
        if (alpha == 0.0 || a.rows == 0 || a.cols == 0) return
        if (x === a.values || y === a.values) {
            ger(alpha, DenseVector.wrap(x), DenseVector.wrap(y), a, workspace)
        } else {
            gerCore(alpha, x, 0, 1, y, 0, 1, a, a.offset, a.rows, a.cols)
        }
    }

    override fun ger(
        alpha: Double,
        x: DoubleArray,
        y: DoubleArray,
        a: DenseMatrixRegion,
        rowOffset: Int,
        columnOffset: Int,
        rows: Int,
        columns: Int,
        xOffset: Int,
        yOffset: Int,
        workspace: Workspace?,
    ) {
        requireGerWindowOperands(x.size, y.size, a, rowOffset, columnOffset, rows, columns, xOffset, yOffset)
        if (alpha == 0.0 || rows == 0 || columns == 0) return
        if (x === a.values || y === a.values) {
            ger(
                alpha,
                StridedVector(x, xOffset, rows, 1),
                StridedVector(y, yOffset, columns, 1),
                a.view(rowOffset, columnOffset, rows, columns),
                workspace,
            )
        } else {
            gerCore(
                alpha, x, xOffset, 1, y, yOffset, 1, a,
                a.offset + rowOffset + columnOffset * a.leadingDimension, rows, columns,
            )
        }
    }

    /**
     * `A += alpha · x · xᵀ` over the stored triangle, as one rank update per group plus its corner.
     *
     * Every column of a group stores the rows below the last of them, which is the window they share; what
     * is left is the small triangle between the group's first and last column.
     */
    @Suppress("LongParameterList") // the BLAS dsyr signature plus the workspace
    override fun syr(alpha: Double, x: DenseVector, a: DenseMatrixRegion, lower: Boolean, workspace: Workspace?) {
        requireSyrOperands(a, x.size, "syr")
        val n = a.rows
        if (alpha == 0.0 || n == 0) return
        // Snapshot an aliased vector before updates overwrite coefficients needed by later columns.
        staged(workspace, x, regionVectorOverlap(a, x)) { sx ->
            val xv = sx.values
            val origin = sx.offset
            val step = sx.stride
            val group = panels.executionGroup(PanelWork.RankUpdate, n, n, sx.stride == 1)
            forEachTrianglePanel(n, group, lower, fromDiagonal = true) { start, width, window, rows ->
                // Only the diagonal corner has differing stored rows; the remaining window is shared by the group.
                val cornerFirst = if (lower) start else start + 1
                val cornerLast = if (lower) start + width - 1 else start + width
                for (c in start until start + width) {
                    val coefficient = alpha * xv[origin + c * step]
                    for (i in cornerFirst until cornerLast) {
                        if (if (lower) i < c else i > c) continue
                        a.values[a.offset + i + c * a.leadingDimension] += coefficient * xv[origin + i * step]
                    }
                }
                panels.rankUpdate(
                    alpha, a.values, a.offset + window + start * a.leadingDimension, a.leadingDimension, xv,
                    origin + window * step, step, rows, width,
                    xv, origin + start * step, step,
                )
            }
        }
    }

    /**
     * `A += alpha · (x · yᵀ + y · xᵀ)` over the stored triangle, as two rank updates per group plus its corner.
     *
     * Each entry takes its two contributions in two read-modify-writes rather than one, but the second finds
     * the group's window still in cache. Against an entry-at-a-time loop this is behind at the small orders,
     * where both stay well ahead of the measured vendors, and ahead from order 64, where the loop was behind
     * them.
     */
    @Suppress("LongParameterList") // the BLAS dsyr2 signature plus the workspace
    override fun syr2(
        alpha: Double,
        x: DenseVector,
        y: DenseVector,
        a: DenseMatrixRegion,
        lower: Boolean,
        workspace: Workspace?,
    ) {
        requireSyr2Operands(a, x.size, y.size, "syr2")
        val n = a.rows
        if (alpha == 0.0 || n == 0) return
        stagedVectors(workspace, x, y, a) { sx, sy ->
            syr2Core(alpha, sx, sy, a, lower)
        }
    }

    private fun syr2Core(alpha: Double, x: DenseVector, y: DenseVector, a: DenseMatrixRegion, lower: Boolean) {
        val n = a.rows
        val xv = x.values
        val yv = y.values
        val xOrigin = x.offset
        val xStep = x.stride
        val yOrigin = y.offset
        val yStep = y.stride
        val lda = a.leadingDimension
        val group = panels.executionGroup(PanelWork.RankUpdate, n, n, xStep == 1 && yStep == 1)
        forEachTrianglePanel(n, group, lower, fromDiagonal = true) { start, width, window, rows ->
            // Only the diagonal corner has differing stored rows; the remaining window is shared by the group.
            val cornerFirst = if (lower) start else start + 1
            val cornerLast = if (lower) start + width - 1 else start + width
            for (c in start until start + width) {
                val xc = xv[xOrigin + c * xStep]
                val yc = yv[yOrigin + c * yStep]
                for (i in cornerFirst until cornerLast) {
                    if (if (lower) i < c else i > c) continue
                    a.values[a.offset + i + c * lda] +=
                        alpha * (xv[xOrigin + i * xStep] * yc + yv[yOrigin + i * yStep] * xc)
                }
            }
            // The second update finds the group's window still in cache from the first.
            val panel = a.offset + window + start * lda
            panels.rankUpdate(
                alpha, a.values, panel, lda, xv, xOrigin + window * xStep, xStep, rows, width,
                yv, yOrigin + start * yStep, yStep,
            )
            panels.rankUpdate(
                alpha, a.values, panel, lda, yv, yOrigin + window * yStep, yStep, rows, width,
                xv, xOrigin + start * xStep, xStep,
            )
        }
    }

    /**
     * Solve `op(T) · x = b` in place, one stored column at a time.
     *
     * Untransposed, a finished entry is divided by its diagonal and then removed from every entry that
     * depends on it, which is one column update. Transposed, an entry is finished by reducing the column
     * against the entries already done, which is one multi-dot. Either way the dependency order is the
     * substitution's and cannot be grouped away; what a panel covers is the independent arithmetic within
     * one step.
     */
    @Suppress("LongParameterList") // the BLAS dtrsv signature plus the workspace

    override fun trsv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) {
        requireTriangularVectorOperands(a, x.size, "trsv")
        if (a.rows == 0) return
        if (a.values === x) {
            trsv(a, DenseVector.wrap(x), lower, transpose, unitDiag, workspace)
        } else {
            trsvCore(a, x, 0, 1, lower, transpose, unitDiag)
        }
    }

    override fun trsv(
        a: DenseMatrixRegion,
        x: DenseVector,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) {
        requireTriangularVectorOperands(a, x.size, "trsv")
        val n = a.rows
        if (n == 0) return
        stagedRegion(workspace, a, x, lower = lower, unitDiag = unitDiag) { sa ->
            trsvCore(sa, x.values, x.offset, x.stride, lower, transpose, unitDiag)
        }
    }

    private fun trsvCore(
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
    ) {
        val n = a.rows
        val lda = a.leadingDimension
        val ascending = lower != transpose
        forEachTriangularBlock(n, ascending) { start, end ->
            // The rest of the stored triangle beside this block: below it when lower, above it when upper.
            val outside = if (lower) end else 0
            val outsideRows = if (lower) n - end else start
            val block = a.offset + outside + start * lda
            if (transpose && outsideRows > 0) {
                panels.multiDot(
                    -1.0, a.values, block, lda, x, xOffset + outside * xStride, xStride, outsideRows, end - start,
                    1.0, x, xOffset + start * xStride, xStride,
                )
            }
            forEachTriangularColumn(start, end, lower, ascending) { j, window, rows ->
                if (transpose) {
                    if (rows > 0) {
                        panels.multiDot(
                            -1.0, a.values, a.offset + window + j * lda, lda, x, xOffset + window * xStride,
                            xStride, rows, 1, 1.0, x, xOffset + j * xStride, xStride,
                        )
                    }
                    if (!unitDiag) x[xOffset + j * xStride] /= a.values[a.offset + j + j * lda]
                } else {
                    if (!unitDiag) x[xOffset + j * xStride] /= a.values[a.offset + j + j * lda]
                    if (rows > 0) {
                        panels.columnUpdate(
                            -1.0, a.values, a.offset + window + j * lda, lda, x, xOffset + j * xStride, xStride,
                            rows, 1, x, xOffset + window * xStride, xStride,
                        )
                    }
                }
            }
            if (!transpose && outsideRows > 0) {
                panels.columnUpdate(
                    -1.0, a.values, block, lda, x, xOffset + start * xStride, xStride, outsideRows, end - start,
                    x, xOffset + outside * xStride, xStride,
                )
            }
        }
    }

    /**
     * `x = op(T) · x` in place, one stored column at a time and in the order that leaves no entry needed
     * after it has been overwritten.
     *
     * Untransposed, a column's entries multiply the entry on its diagonal into every row below it, so the
     * columns are walked away from the diagonal; transposed, an entry is the reduction of its own column
     * against entries not yet touched. Neither direction needs a copy of the input.
     */
    @Suppress("LongParameterList") // the BLAS dtrmv signature plus the workspace
    override fun trmv(
        a: DenseMatrixRegion,
        x: DoubleArray,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) {
        requireTriangularVectorOperands(a, x.size, "trmv")
        if (a.rows == 0) return
        if (a.values === x) {
            trmv(a, DenseVector.wrap(x), lower, transpose, unitDiag, workspace)
        } else {
            trmvCore(a, x, 0, 1, lower, transpose, unitDiag)
        }
    }

    override fun trmv(
        a: DenseMatrixRegion,
        x: DenseVector,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        workspace: Workspace?,
    ) {
        requireTriangularVectorOperands(a, x.size, "trmv")
        val n = a.rows
        stagedRegion(workspace, a, x, lower = lower, unitDiag = unitDiag) { sa ->
            trmvCore(sa, x.values, x.offset, x.stride, lower, transpose, unitDiag)
        }
    }

    private fun trmvCore(
        a: DenseMatrixRegion,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
    ) {
        val n = a.rows
        val lda = a.leadingDimension
        // Traverse in opposite directions for transpose variants to preserve values until their last use.
        val ascending = transpose == lower
        forEachTriangularBlock(n, ascending) { start, end ->
            // The rest of the stored triangle beside this block: below it when lower, above it when upper.
            val outside = if (lower) end else 0
            val outsideRows = if (lower) n - end else start
            val block = a.offset + outside + start * lda
            if (!transpose && outsideRows > 0) {
                panels.columnUpdate(
                    1.0, a.values, block, lda, x, xOffset + start * xStride, xStride, outsideRows, end - start,
                    x, xOffset + outside * xStride, xStride,
                )
            }
            forEachTriangularColumn(start, end, lower, ascending) { j, window, rows ->
                if (transpose) {
                    if (!unitDiag) x[xOffset + j * xStride] *= a.values[a.offset + j + j * lda]
                    if (rows > 0) {
                        panels.multiDot(
                            1.0, a.values, a.offset + window + j * lda, lda, x, xOffset + window * xStride,
                            xStride, rows, 1, 1.0, x, xOffset + j * xStride, xStride,
                        )
                    }
                } else {
                    if (rows > 0) {
                        panels.columnUpdate(
                            1.0, a.values, a.offset + window + j * lda, lda, x, xOffset + j * xStride, xStride,
                            rows, 1, x, xOffset + window * xStride, xStride,
                        )
                    }
                    if (!unitDiag) x[xOffset + j * xStride] *= a.values[a.offset + j + j * lda]
                }
            }
            if (transpose && outsideRows > 0) {
                panels.multiDot(
                    1.0, a.values, block, lda, x, xOffset + outside * xStride, xStride, outsideRows, end - start,
                    1.0, x, xOffset + start * xStride, xStride,
                )
            }
        }
    }

    override fun trsm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) = triangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = true, workspace = workspace)

    override fun trmm(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        workspace: Workspace?,
    ) = triangularMatrix(a, b, lower, transpose, unitDiag, right, alpha, solve = false, workspace = workspace)

    /**
     * Both triangular matrix routines, over a triangle staged against an overlap with the block it works on.
     *
     * The staging comes first and the scaling second. `alpha` is spent on the right-hand sides before the
     * substitution, and where the triangle shares their buffer that write would otherwise reach the
     * coefficients the substitution is about to read.
     */
    @Suppress("LongParameterList") // the BLAS dtrsm signature, which of the two it is, and the workspace
    private fun triangularMatrix(
        a: DenseMatrixRegion,
        b: DenseMatrixRegion,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
        alpha: Double,
        solve: Boolean,
        workspace: Workspace?,
    ) {
        val what = if (solve) "trsm" else "trmm"
        requireTriangularMatrixOperands(a, b, right, what)
        if (alpha == 0.0) {
            scaleRegion(b, 0.0)
            return
        }
        val sides = if (right) b.rows else b.cols
        val order = a.rows
        if (sides == 0 || order == 0) return
        stagedRegion(workspace, a, b, lower = lower, unitDiag = unitDiag) { sa ->
            triangularMatrix(
                triangles, products, panels, vectors, sa.values, order, b.values, b.leadingDimension, sides,
                lower, transpose, unitDiag, right, alpha, solve, workspace, sa.offset, sa.leadingDimension, b.offset,
            )
        }
    }
}

/** The part of a square destination a selected-triangle routine writes. */
internal fun selectedTriangle(lower: Boolean): OutputTriangle =
    if (lower) OutputTriangle.Lower else OutputTriangle.Upper
