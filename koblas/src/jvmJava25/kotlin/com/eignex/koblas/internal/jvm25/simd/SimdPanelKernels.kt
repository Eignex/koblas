@file:Suppress("LongParameterList") // a panel carries its window, its extents and its scaling as plain numbers

package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.implementationFor
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine

/**
 * Vector API panels with [ScalarEngine] fallbacks. Lanes run down rows independently of column grouping;
 * reductions may reorder additions. Short multi-dots and updates switch axes or use scalar bodies, while
 * resident updates retain destination vectors across columns. [implementationFor] reports the actual body.
 * [KoblasEngine] checks module availability before species initialization.
 */
internal object SimdPanelKernels : KernelProvider {
    override val id: String = "jvm-simd"

    override val implementations = setOf(EngineImplementation.Panels)

    private val SPECIES = SIMD_SPECIES

    private val LANE = SPECIES.length()

    /** Stored so scheduling decisions that compare it do not allocate. */
    override val name: String = "simd-panel($LANE lanes)"

    /**
     * Structural minimum of one whole lane block. Measurements did not establish a robust higher
     * crossover on the measured host; this does not promise a performance bound on other machines.
     */
    private val VECTOR_MINIMUM = MINIMUM_BLOCKS * LANE

    /** Whether a panel of this shape and spacing reaches the vector bodies rather than the portable ones. */
    private fun vectorizes(rows: Int, contiguous: Boolean): Boolean = contiguous && rows >= VECTOR_MINIMUM

    /**
     * A multi-dot is handed over whole and reuses one loaded vector across eight columns at a time; column
     * updates share destination traffic across eight. Coupled bodies take four columns per pass over the
     * destination. Rank-update bodies are two columns wide; wider groups would repeat those bodies and grow
     * a triangular caller's scalar corner. Group sizes reflect measurements on one host.
     */
    override fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): Int {
        val preferred = when {
            contiguous && groupsWholeRow(work, rows) -> columns

            work == PanelWork.ColumnUpdate && contiguous -> COLUMN_UPDATE_GROUP

            work == PanelWork.SparseRightHandSideReduction && !contiguous -> SPARSE_REDUCTION_GROUP

            work == PanelWork.SparseRightHandSides || work == PanelWork.SparseRightHandSideReduction ->
                if (contiguous) ADJACENT_SPARSE_BLOCKS * LANE else SPARSE_GROUP

            work == PanelWork.RankUpdate -> if (LANE <= 2) WIDE_GROUP else NARROW_GROUP

            else -> WIDE_GROUP
        }
        return if (columns <= 0) 1 else minOf(preferred, columns)
    }

    /**
     * Pack from two lane blocks of adjacent rows. At one block, copying the panel costs a data pass for
     * one vector operation per column and did not pay for itself on the measured host.
     */
    override fun prefersContiguous(work: PanelWork, rows: Int, columns: Int): Boolean =
        rows >= COPY_WORTH_BLOCKS * LANE && !vectorizes(rows, contiguous = false)

    override fun implementationFor(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): String =
        if (vectorizes(rows, contiguous) || (contiguous && reachesShortBody(work, rows))) {
            name
        } else {
            ScalarEngine.implementationFor(work, rows, columns, contiguous)
        }

    override fun multiDot(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ) {
        if (isShort(rows) && xStride == 1) {
            multiDotShort(alpha, a, aOffset, lda, x, xOffset, rows, columns, beta, y, yOffset, yStride)
            return
        }
        if (!vectorizes(rows, xStride == 1)) {
            ScalarEngine.multiDot(alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, beta, y, yOffset, yStride)
            return
        }
        var c = 0
        while (c + MULTI_DOT_GROUP <= columns) {
            dotEight(alpha, a, aOffset + c * lda, lda, x, xOffset, rows, beta, y, yOffset + c * yStride, yStride)
            c += MULTI_DOT_GROUP
        }
        while (c + WIDE_GROUP <= columns) {
            dotFour(alpha, a, aOffset + c * lda, lda, x, xOffset, rows, beta, y, yOffset + c * yStride, yStride)
            c += WIDE_GROUP
        }
        while (c + NARROW_GROUP <= columns) {
            dotTwo(alpha, a, aOffset + c * lda, lda, x, xOffset, rows, beta, y, yOffset + c * yStride, yStride)
            c += NARROW_GROUP
        }
        while (c < columns) {
            store(y, yOffset + c * yStride, alpha, SimdVectorOps.dot(a, aOffset + c * lda, x, xOffset, rows), beta)
            c++
        }
    }

    override fun columnUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ) {
        if (rows <= 0 || columns <= 0) return
        if (isShort(rows) && (rows == 1 || yStride == 1)) {
            columnUpdateShort(alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, y, yOffset)
            return
        }
        if (isResident(rows) && yStride == 1) {
            columnUpdateResident(alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, y, yOffset)
            return
        }
        if (!vectorizes(rows, yStride == 1)) {
            ScalarEngine.columnUpdate(alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, y, yOffset, yStride)
            return
        }
        var c = 0
        while (c + COLUMN_UPDATE_GROUP <= columns) {
            updateEight(
                a, aOffset + c * lda, lda, rows, y, yOffset,
                alpha * x[xOffset + (c + 0) * xStride],
                alpha * x[xOffset + (c + 1) * xStride],
                alpha * x[xOffset + (c + 2) * xStride],
                alpha * x[xOffset + (c + 3) * xStride],
                alpha * x[xOffset + (c + 4) * xStride],
                alpha * x[xOffset + (c + 5) * xStride],
                alpha * x[xOffset + (c + 6) * xStride],
                alpha * x[xOffset + (c + 7) * xStride],
            )
            c += COLUMN_UPDATE_GROUP
        }
        while (c + WIDE_GROUP <= columns) {
            updateFour(
                a, aOffset + c * lda, lda, rows, y, yOffset,
                alpha * x[xOffset + c * xStride],
                alpha * x[xOffset + (c + 1) * xStride],
                alpha * x[xOffset + (c + 2) * xStride],
                alpha * x[xOffset + (c + 3) * xStride],
            )
            c += WIDE_GROUP
        }
        while (c + NARROW_GROUP <= columns) {
            val at = aOffset + c * lda
            indexedUpdateTwo(
                a,
                at,
                at + lda,
                rows,
                y,
                yOffset,
                alpha * x[xOffset + c * xStride],
                alpha * x[xOffset + (c + 1) * xStride],
            )
            c += NARROW_GROUP
        }
        while (c < columns) {
            updateOne(a, aOffset + c * lda, rows, y, yOffset, alpha * x[xOffset + c * xStride])
            c++
        }
    }

    override fun coupledUpdateDot(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        columns: Int,
        y: DoubleArray,
        yOffset: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        sums: DoubleArray,
        sumOffset: Int,
        xStride: Int,
        yStride: Int,
        coefficientStride: Int,
        sumStride: Int,
    ) {
        if (rows <= 0 || columns <= 0) return
        if (!vectorizes(rows, contiguous = xStride == 1 && yStride == 1 && coefficientStride == 1 && sumStride == 1)) {
            ScalarEngine.coupledUpdateDot(
                alpha, a, aOffset, lda, x, xOffset, rows, columns, y, yOffset,
                coefficients, coefficientOffset, sums, sumOffset, xStride, yStride, coefficientStride, sumStride,
            )
            return
        }
        var c = 0
        while (c + WIDE_GROUP <= columns) {
            coupledFour(
                alpha, a, aOffset + c * lda, lda, x, xOffset, rows, y, yOffset,
                coefficients, coefficientOffset + c, sums, sumOffset + c,
            )
            c += WIDE_GROUP
        }
        while (c + NARROW_GROUP <= columns) {
            coupledTwo(
                alpha, a, aOffset + c * lda, lda, x, xOffset, rows, y, yOffset,
                coefficients, coefficientOffset + c, sums, sumOffset + c,
            )
            c += NARROW_GROUP
        }
        while (c < columns) {
            coupledOne(
                alpha, a, aOffset + c * lda, x, xOffset, rows, y, yOffset,
                coefficients[coefficientOffset + c], sums, sumOffset + c,
            )
            c++
        }
    }

    override fun rankUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        coefficientStride: Int,
    ) {
        if (rows <= 0 || columns <= 0) return
        if (!vectorizes(rows, xStride == 1)) {
            ScalarEngine.rankUpdate(
                alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns,
                coefficients, coefficientOffset, coefficientStride,
            )
            return
        }
        var c = 0
        if (LANE <= 2) {
            while (c + WIDE_GROUP <= columns) {
                rankFour(
                    a, aOffset + c * lda, lda, x, xOffset, rows,
                    alpha * coefficients[coefficientOffset + c * coefficientStride],
                    alpha * coefficients[coefficientOffset + (c + 1) * coefficientStride],
                    alpha * coefficients[coefficientOffset + (c + 2) * coefficientStride],
                    alpha * coefficients[coefficientOffset + (c + 3) * coefficientStride],
                )
                c += WIDE_GROUP
            }
        }
        while (c + NARROW_GROUP <= columns) {
            val at = aOffset + c * lda
            rankTwoAt(
                a,
                at,
                at + lda,
                x,
                xOffset,
                rows,
                alpha * coefficients[coefficientOffset + c * coefficientStride],
                alpha * coefficients[coefficientOffset + (c + 1) * coefficientStride],
            )
            c += NARROW_GROUP
        }
        while (c < columns) {
            updateOne(
                x,
                xOffset,
                rows,
                a,
                aOffset + c * lda,
                alpha * coefficients[coefficientOffset + c * coefficientStride],
            )
            c++
        }
    }

    override fun rankTwoUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        rows: Int,
        columns: Int,
        u: DoubleArray,
        uOffset: Int,
        uStride: Int,
        v: DoubleArray,
        vOffset: Int,
        vStride: Int,
    ) {
        if (rows <= 0 || columns <= 0) return
        if (!vectorizes(rows, xStride == 1 && yStride == 1)) {
            ScalarEngine.rankTwoUpdate(
                alpha, a, aOffset, lda, x, xOffset, xStride, y, yOffset, yStride, rows, columns,
                u, uOffset, uStride, v, vOffset, vStride,
            )
            return
        }
        var c = 0
        while (c + NARROW_GROUP <= columns) {
            symmetricTwo(
                a, aOffset + c * lda, lda, x, xOffset, y, yOffset, rows,
                alpha * u[uOffset + c * uStride], alpha * v[vOffset + c * vStride],
                alpha * u[uOffset + (c + 1) * uStride], alpha * v[vOffset + (c + 1) * vStride],
            )
            c += NARROW_GROUP
        }
        if (c < columns) {
            symmetricOne(
                a, aOffset + c * lda, x, xOffset, y, yOffset, rows,
                alpha * u[uOffset + c * uStride], alpha * v[vOffset + c * vStride],
            )
        }
    }

    /**
     * Lanes cover independent right-hand sides. Scalar indices select columns, so no Vector API indexed
     * load or store is needed.
     */
    override fun indexedColumnUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        y: DoubleArray,
        yOffset: Int,
    ) {
        if (rows <= 0 || columns <= 0) return
        if (!vectorizes(rows, rowStride == 1)) {
            ScalarEngine.indexedColumnUpdate(
                alpha, a, aOffset, rowStride, indexStride, indices, values, fromIndex, columns, rows, y, yOffset,
            )
            return
        }
        var c = 0
        while (c + NARROW_GROUP <= columns) {
            indexedUpdateTwo(
                a,
                aOffset + indices[fromIndex + c] * indexStride,
                aOffset + indices[fromIndex + c + 1] * indexStride,
                rows,
                y,
                yOffset,
                alpha * values[fromIndex + c],
                alpha * values[fromIndex + c + 1],
            )
            c += NARROW_GROUP
        }
        while (c < columns) {
            updateOne(
                a,
                aOffset + indices[fromIndex + c] * indexStride,
                rows,
                y,
                yOffset,
                alpha * values[fromIndex + c],
            )
            c++
        }
    }

    /** The indexed rank update in lanes of right-hand sides, the counterpart of [indexedColumnUpdate]. */
    override fun indexedRankUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        x: DoubleArray,
        xOffset: Int,
    ) {
        if (rows <= 0 || columns <= 0) return
        if (!vectorizes(rows, rowStride == 1)) {
            ScalarEngine.indexedRankUpdate(
                alpha, a, aOffset, rowStride, indexStride, indices, values, fromIndex, columns, rows, x, xOffset,
            )
            return
        }
        var c = 0
        while (c + NARROW_GROUP <= columns) {
            rankTwoAt(
                a,
                aOffset + indices[fromIndex + c] * indexStride,
                aOffset + indices[fromIndex + c + 1] * indexStride,
                x,
                xOffset,
                rows,
                alpha * values[fromIndex + c],
                alpha * values[fromIndex + c + 1],
            )
            c += NARROW_GROUP
        }
        while (c < columns) {
            val at = aOffset + indices[fromIndex + c] * indexStride
            updateOne(x, xOffset, rows, a, at, alpha * values[fromIndex + c])
            c++
        }
    }

    /**
     * Coupled indexed pass over lanes of right-hand sides.
     *
     * Scatter and reduction share an address and a broadcast coefficient. Loading the pivot column
     * in the loop avoids gathering it into a separate buffer for each column.
     */
    override fun indexedCoupledUpdate(
        alpha: Double,
        a: DoubleArray,
        b: DoubleArray,
        offset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        pivot: Int,
        sums: DoubleArray,
        sumOffset: Int,
        excluded: Int,
        destinationOffset: Int,
        destinationRowStride: Int,
        destinationIndexStride: Int,
    ) {
        if (rows <= 0 || columns <= 0) return
        if (!vectorizes(rows, rowStride == 1 && destinationRowStride == 1)) {
            ScalarEngine.indexedCoupledUpdate(
                alpha, a, b, offset, rowStride, indexStride, indices, values, fromIndex, columns, rows,
                pivot, sums, sumOffset, excluded, destinationOffset, destinationRowStride, destinationIndexStride,
            )
            return
        }
        for (c in 0 until columns) {
            val t = alpha * values[fromIndex + c]
            val at = destinationOffset + indices[fromIndex + c] * destinationIndexStride
            val sourceAt = offset + indices[fromIndex + c] * indexStride
            // Separate bodies avoid a branch between stores that prevents vector allocation elimination.
            if (c == excluded) {
                updateOne(b, pivot, rows, a, at, t)
            } else {
                scatterAndReduce(t, a, at, b, pivot, sums, sumOffset, rows, sourceAt)
            }
        }
    }

    /** A panel with fewer rows than one lane block, which a column cannot fill and the work runs along. */
    private fun isShort(rows: Int): Boolean = rows in 1 until VECTOR_MINIMUM

    /**
     * Panels handed over whole. A multi-dot groups its columns itself and reuses the shared vector across all of
     * them, so a call per group only adds calls; so does a column update short enough that a call per group
     * costs more than the arithmetic in it.
     */
    private fun groupsWholeRow(work: PanelWork, rows: Int): Boolean = work == PanelWork.MultiDot ||
        (work == PanelWork.ColumnUpdate && (isShort(rows) || isResident(rows)))

    /** The short panels this backend runs a body of its own for; a longer short column update is portable. */
    private fun reachesShortBody(work: PanelWork, rows: Int): Boolean =
        (work == PanelWork.MultiDot || work == PanelWork.ColumnUpdate) && isShort(rows)

    /** Rows a column update keeps in registers across every column instead of reloading per group. */
    private fun isResident(rows: Int): Boolean = rows >= LANE && rows < RESIDENT_BLOCKS * LANE

    /** Lane blocks a panel needs before its vector body can run, which is one whole vector. */
    private const val MINIMUM_BLOCKS = 1

    /**
     * Packing threshold in lane blocks, above [MINIMUM_BLOCKS] because a copy must pay for its data pass.
     */
    private const val COPY_WORTH_BLOCKS = 2

    /** Columns one pass of a multi-dot reduces against the shared vector. */
    private const val MULTI_DOT_GROUP = 8

    /** Columns grouped where one loaded vector serves all of them. */
    private const val COLUMN_UPDATE_GROUP = 8

    /** Lane blocks below which a column update keeps its destination in registers across the columns. */
    private const val RESIDENT_BLOCKS = 3

    private const val WIDE_GROUP = 4

    /** Columns grouped where the body itself is that wide. */
    private const val NARROW_GROUP = 2

    /**
     * Group strided right-hand sides to share one walk of the sparse column. These use the portable
     * body and its group width, independently of the species.
     */
    private const val SPARSE_GROUP = 8

    /**
     * A strided reduction uses one right-hand side so its scalar accumulator can stay in a register.
     */
    private const val SPARSE_REDUCTION_GROUP = 1

    /**
     * Group adjacent right-hand sides in whole lane blocks to use every lane and amortize the sparse
     * index walk. Eight blocks are the conservative choice from the measured near-parity pair; sparse
     * scheduling caps the group again to fit its staging buffer.
     */
    private const val ADJACENT_SPARSE_BLOCKS = 8
}
