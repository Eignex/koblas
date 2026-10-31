@file:Suppress("LongParameterList") // a panel carries its window, its extents and its scaling as plain numbers

package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.dense.DensePanelKernels
import com.eignex.koblas.internal.dense.PanelWork
import com.eignex.koblas.internal.dense.PortablePanelKernels
import com.eignex.koblas.sparse.*
import com.eignex.koblas.vendor.*
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/**
 * JVM Vector API panel arithmetic with [PortablePanelKernels] fallbacks.
 *
 * Explicit lanes allow floating-point reductions to reorder additions. Column and rank updates
 * keep a destination strip in registers while several columns accumulate into it. Column grouping
 * is independent of the lane count: lanes run down rows.
 *
 * A multi-dot or column update shorter than one lane block runs along its row instead, as a dot, an
 * elementwise update or a scalar pass over all its columns at once, and a column update of one or two lane
 * blocks keeps its destination in registers across every column. Other short panels and strided operands
 * requiring gathers use the portable bodies; [implementationFor] reports the body reached. Species
 * initialization requires the Vector API module, so [com.eignex.koblas.BuiltinEngines] checks availability
 * before using this backend.
 */
internal object SimdPanelKernels : DensePanelKernels {
    private val SPECIES = DoubleVector.SPECIES_PREFERRED
    private val LANE = if (simdAvailable) SPECIES.length() else 0

    /**
     * Cache the name so scheduling decisions that compare it do not allocate.
     */
    private val NAME: String = "simd-panel($LANE lanes)"

    override val name: String get() = NAME

    /**
     * Structural minimum of one whole lane block. Measurements did not establish a robust higher
     * crossover on the measured host; this does not promise a performance bound on other machines.
     */
    private val VECTOR_MINIMUM = MINIMUM_BLOCKS * LANE

    /** Column length from which a reduction carries two accumulators per column; shorter ones are reduce-bound. */
    private val PAIRED_MINIMUM = 16 * LANE

    /** Whether a panel of this shape and spacing reaches the vector bodies rather than the portable ones. */
    private fun vectorizes(rows: Int, contiguous: Boolean): Boolean =
        simdAvailable && contiguous && rows >= VECTOR_MINIMUM

    /**
     * A multi-dot is handed over whole and reuses one loaded vector across eight columns at a time; column
     * updates share destination traffic across eight, or twelve on long strips with wide vector registers.
     * Coupled bodies take four columns per pass over the destination. Rank-update bodies are two columns
     * wide; wider groups would repeat those bodies and grow a triangular caller's scalar corner. Group sizes
     * reflect measurements on one host.
     */
    override fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): Int {
        if (!simdAvailable) return PortablePanelKernels.executionGroup(work, rows, columns, contiguous)
        val preferred = when {
            contiguous && groupsWholeRow(work, rows) -> columns

            work == PanelWork.ColumnUpdate && contiguous ->
                if (prefersLargeColumnUpdate(rows)) LARGE_COLUMN_UPDATE_GROUP else COLUMN_UPDATE_GROUP

            work == PanelWork.SparseRightHandSideReduction && !contiguous -> SPARSE_REDUCTION_GROUP

            work == PanelWork.SparseRightHandSides || work == PanelWork.SparseRightHandSideReduction ->
                if (contiguous) ADJACENT_SPARSE_BLOCKS * LANE else SPARSE_GROUP

            work == PanelWork.RankUpdate -> NARROW_GROUP

            else -> WIDE_GROUP
        }
        return if (columns <= 0) 1 else minOf(preferred, columns)
    }

    private fun prefersLargeColumnUpdate(rows: Int): Boolean =
        SPECIES.vectorBitSize() >= 512 && rows >= LARGE_COLUMN_UPDATE_MINIMUM_ROWS

    /**
     * Pack from two lane blocks of adjacent rows. At one block, copying the panel costs a data pass for
     * one vector operation per column and did not pay for itself on the measured host.
     */
    override fun prefersContiguous(work: PanelWork, rows: Int, columns: Int): Boolean =
        simdAvailable && rows >= COPY_WORTH_BLOCKS * LANE && !vectorizes(rows, contiguous = false)

    override fun implementationFor(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): String =
        if (vectorizes(rows, contiguous) || (contiguous && reachesShortBody(work, rows))) {
            name
        } else {
            PortablePanelKernels.implementationFor(work, rows, columns, contiguous)
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
            PortablePanelKernels.multiDot(
                alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, beta, y, yOffset, yStride,
            )
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
            store(y, yOffset + c * yStride, alpha, dotOne(a, aOffset + c * lda, x, xOffset, rows), beta)
            c++
        }
    }

    /**
     * Eight columns per pass over the shared vector, which halves its reloads against four and gives every
     * column a chain of its own: eight chains cover the FMA latency without pairing.
     */
    private fun dotEight(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ) {
        var v0 = DoubleVector.zero(SPECIES)
        var v1 = DoubleVector.zero(SPECIES)
        var v2 = DoubleVector.zero(SPECIES)
        var v3 = DoubleVector.zero(SPECIES)
        var v4 = DoubleVector.zero(SPECIES)
        var v5 = DoubleVector.zero(SPECIES)
        var v6 = DoubleVector.zero(SPECIES)
        var v7 = DoubleVector.zero(SPECIES)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
            val at = aOffset + i
            v0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, at), xv, v0)
            v1 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, at + lda), xv, v1)
            v2 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, at + 2 * lda), xv, v2)
            v3 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, at + 3 * lda), xv, v3)
            v4 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, at + 4 * lda), xv, v4)
            v5 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, at + 5 * lda), xv, v5)
            v6 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, at + 6 * lda), xv, v6)
            v7 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, at + 7 * lda), xv, v7)
            i += LANE
        }
        var s0 = v0.reduceLanes(VectorOperators.ADD)
        var s1 = v1.reduceLanes(VectorOperators.ADD)
        var s2 = v2.reduceLanes(VectorOperators.ADD)
        var s3 = v3.reduceLanes(VectorOperators.ADD)
        var s4 = v4.reduceLanes(VectorOperators.ADD)
        var s5 = v5.reduceLanes(VectorOperators.ADD)
        var s6 = v6.reduceLanes(VectorOperators.ADD)
        var s7 = v7.reduceLanes(VectorOperators.ADD)
        while (i < rows) {
            val xi = x[xOffset + i]
            val at = aOffset + i
            s0 += a[at] * xi
            s1 += a[at + lda] * xi
            s2 += a[at + 2 * lda] * xi
            s3 += a[at + 3 * lda] * xi
            s4 += a[at + 4 * lda] * xi
            s5 += a[at + 5 * lda] * xi
            s6 += a[at + 6 * lda] * xi
            s7 += a[at + 7 * lda] * xi
            i++
        }
        store(y, yOffset, alpha, s0, beta)
        store(y, yOffset + yStride, alpha, s1, beta)
        store(y, yOffset + 2 * yStride, alpha, s2, beta)
        store(y, yOffset + 3 * yStride, alpha, s3, beta)
        store(y, yOffset + 4 * yStride, alpha, s4, beta)
        store(y, yOffset + 5 * yStride, alpha, s5, beta)
        store(y, yOffset + 6 * yStride, alpha, s6, beta)
        store(y, yOffset + 7 * yStride, alpha, s7, beta)
    }

    /**
     * Two accumulators per column: one FMA per column and lane block would wait out the instruction's
     * latency on a long column rather than run at the load rate.
     */
    private fun dotFour(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ) {
        var v0 = DoubleVector.zero(SPECIES)
        var v1 = DoubleVector.zero(SPECIES)
        var v2 = DoubleVector.zero(SPECIES)
        var v3 = DoubleVector.zero(SPECIES)
        var w0 = DoubleVector.zero(SPECIES)
        var w1 = DoubleVector.zero(SPECIES)
        var w2 = DoubleVector.zero(SPECIES)
        var w3 = DoubleVector.zero(SPECIES)
        val a1 = aOffset + lda
        val a2 = aOffset + 2 * lda
        val a3 = aOffset + 3 * lda
        var i = 0
        val pairs = if (rows >= PAIRED_MINIMUM) rows - rows % (2 * LANE) else 0
        while (i < pairs) {
            val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
            val xw = DoubleVector.fromArray(SPECIES, x, xOffset + i + LANE)
            v0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i), xv, v0)
            v1 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a1 + i), xv, v1)
            v2 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a2 + i), xv, v2)
            v3 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a3 + i), xv, v3)
            w0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i + LANE), xw, w0)
            w1 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a1 + i + LANE), xw, w1)
            w2 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a2 + i + LANE), xw, w2)
            w3 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a3 + i + LANE), xw, w3)
            i += 2 * LANE
        }
        while (i + LANE <= rows) {
            val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
            v0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i), xv, v0)
            v1 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a1 + i), xv, v1)
            v2 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a2 + i), xv, v2)
            v3 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a3 + i), xv, v3)
            i += LANE
        }
        var s0 = v0.add(w0).reduceLanes(VectorOperators.ADD)
        var s1 = v1.add(w1).reduceLanes(VectorOperators.ADD)
        var s2 = v2.add(w2).reduceLanes(VectorOperators.ADD)
        var s3 = v3.add(w3).reduceLanes(VectorOperators.ADD)
        while (i < rows) {
            val xi = x[xOffset + i]
            s0 += a[aOffset + i] * xi
            s1 += a[a1 + i] * xi
            s2 += a[a2 + i] * xi
            s3 += a[a3 + i] * xi
            i++
        }
        store(y, yOffset, alpha, s0, beta)
        store(y, yOffset + yStride, alpha, s1, beta)
        store(y, yOffset + 2 * yStride, alpha, s2, beta)
        store(y, yOffset + 3 * yStride, alpha, s3, beta)
    }

    private fun dotTwo(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ) {
        var v0 = DoubleVector.zero(SPECIES)
        var v1 = DoubleVector.zero(SPECIES)
        var w0 = DoubleVector.zero(SPECIES)
        var w1 = DoubleVector.zero(SPECIES)
        val a1 = aOffset + lda
        var i = 0
        val pairs = if (rows >= PAIRED_MINIMUM) rows - rows % (2 * LANE) else 0
        while (i < pairs) {
            val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
            val xw = DoubleVector.fromArray(SPECIES, x, xOffset + i + LANE)
            v0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i), xv, v0)
            v1 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a1 + i), xv, v1)
            w0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i + LANE), xw, w0)
            w1 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a1 + i + LANE), xw, w1)
            i += 2 * LANE
        }
        while (i + LANE <= rows) {
            val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
            v0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i), xv, v0)
            v1 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a1 + i), xv, v1)
            i += LANE
        }
        var s0 = v0.add(w0).reduceLanes(VectorOperators.ADD)
        var s1 = v1.add(w1).reduceLanes(VectorOperators.ADD)
        while (i < rows) {
            val xi = x[xOffset + i]
            s0 += a[aOffset + i] * xi
            s1 += a[a1 + i] * xi
            i++
        }
        store(y, yOffset, alpha, s0, beta)
        store(y, yOffset + yStride, alpha, s1, beta)
    }

    /** A single column is the vector kernel's dot, which keeps several accumulators on a long run. */
    private fun dotOne(a: DoubleArray, aOffset: Int, x: DoubleArray, xOffset: Int, rows: Int): Double =
        SimdOps.dot(a, aOffset, x, xOffset, rows)

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
            PortablePanelKernels.columnUpdate(
                alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, y, yOffset, yStride,
            )
            return
        }
        var c = 0
        if (prefersLargeColumnUpdate(rows)) {
            while (c + LARGE_COLUMN_UPDATE_GROUP <= columns) {
                updateTwelve(
                    a, aOffset + c * lda, lda, rows, y, yOffset,
                    alpha * x[xOffset + (c + 0) * xStride],
                    alpha * x[xOffset + (c + 1) * xStride],
                    alpha * x[xOffset + (c + 2) * xStride],
                    alpha * x[xOffset + (c + 3) * xStride],
                    alpha * x[xOffset + (c + 4) * xStride],
                    alpha * x[xOffset + (c + 5) * xStride],
                    alpha * x[xOffset + (c + 6) * xStride],
                    alpha * x[xOffset + (c + 7) * xStride],
                    alpha * x[xOffset + (c + 8) * xStride],
                    alpha * x[xOffset + (c + 9) * xStride],
                    alpha * x[xOffset + (c + 10) * xStride],
                    alpha * x[xOffset + (c + 11) * xStride],
                )
                c += LARGE_COLUMN_UPDATE_GROUP
            }
        }
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
            updateTwo(
                a,
                aOffset + c * lda,
                lda,
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

    /** A separate compilation keeps the long strip's vector graph out of the small-panel inlining budget. */
    private fun updateTwelve(
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        rows: Int,
        y: DoubleArray,
        yOffset: Int,
        t0: Double,
        t1: Double,
        t2: Double,
        t3: Double,
        t4: Double,
        t5: Double,
        t6: Double,
        t7: Double,
        t8: Double,
        t9: Double,
        t10: Double,
        t11: Double,
    ) {
        val c0 = DoubleVector.broadcast(SPECIES, t0)
        val c1 = DoubleVector.broadcast(SPECIES, t1)
        val c2 = DoubleVector.broadcast(SPECIES, t2)
        val c3 = DoubleVector.broadcast(SPECIES, t3)
        val c4 = DoubleVector.broadcast(SPECIES, t4)
        val c5 = DoubleVector.broadcast(SPECIES, t5)
        val c6 = DoubleVector.broadcast(SPECIES, t6)
        val c7 = DoubleVector.broadcast(SPECIES, t7)
        val c8 = DoubleVector.broadcast(SPECIES, t8)
        val c9 = DoubleVector.broadcast(SPECIES, t9)
        val c10 = DoubleVector.broadcast(SPECIES, t10)
        val c11 = DoubleVector.broadcast(SPECIES, t11)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            var acc = DoubleVector.fromArray(SPECIES, y, yOffset + i)
            acc = multiplyAdd(c0, DoubleVector.fromArray(SPECIES, a, aOffset + i + 0 * lda), acc)
            acc = multiplyAdd(c1, DoubleVector.fromArray(SPECIES, a, aOffset + i + 1 * lda), acc)
            acc = multiplyAdd(c2, DoubleVector.fromArray(SPECIES, a, aOffset + i + 2 * lda), acc)
            acc = multiplyAdd(c3, DoubleVector.fromArray(SPECIES, a, aOffset + i + 3 * lda), acc)
            acc = multiplyAdd(c4, DoubleVector.fromArray(SPECIES, a, aOffset + i + 4 * lda), acc)
            acc = multiplyAdd(c5, DoubleVector.fromArray(SPECIES, a, aOffset + i + 5 * lda), acc)
            acc = multiplyAdd(c6, DoubleVector.fromArray(SPECIES, a, aOffset + i + 6 * lda), acc)
            acc = multiplyAdd(c7, DoubleVector.fromArray(SPECIES, a, aOffset + i + 7 * lda), acc)
            acc = multiplyAdd(c8, DoubleVector.fromArray(SPECIES, a, aOffset + i + 8 * lda), acc)
            acc = multiplyAdd(c9, DoubleVector.fromArray(SPECIES, a, aOffset + i + 9 * lda), acc)
            acc = multiplyAdd(c10, DoubleVector.fromArray(SPECIES, a, aOffset + i + 10 * lda), acc)
            acc = multiplyAdd(c11, DoubleVector.fromArray(SPECIES, a, aOffset + i + 11 * lda), acc)
            acc.intoArray(y, yOffset + i)
            i += LANE
        }
        while (i < rows) {
            y[yOffset + i] += t0 * a[aOffset + i + 0 * lda] +
                t1 * a[aOffset + i + 1 * lda] +
                t2 * a[aOffset + i + 2 * lda] +
                t3 * a[aOffset + i + 3 * lda] +
                t4 * a[aOffset + i + 4 * lda] +
                t5 * a[aOffset + i + 5 * lda] +
                t6 * a[aOffset + i + 6 * lda] +
                t7 * a[aOffset + i + 7 * lda] +
                t8 * a[aOffset + i + 8 * lda] +
                t9 * a[aOffset + i + 9 * lda] +
                t10 * a[aOffset + i + 10 * lda] +
                t11 * a[aOffset + i + 11 * lda]
            i++
        }
    }

    /** Small panel calls otherwise pay another JVM frame for every eight columns. */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun updateEight(
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        rows: Int,
        y: DoubleArray,
        yOffset: Int,
        t0: Double,
        t1: Double,
        t2: Double,
        t3: Double,
        t4: Double,
        t5: Double,
        t6: Double,
        t7: Double,
    ) {
        val c0 = DoubleVector.broadcast(SPECIES, t0)
        val c1 = DoubleVector.broadcast(SPECIES, t1)
        val c2 = DoubleVector.broadcast(SPECIES, t2)
        val c3 = DoubleVector.broadcast(SPECIES, t3)
        val c4 = DoubleVector.broadcast(SPECIES, t4)
        val c5 = DoubleVector.broadcast(SPECIES, t5)
        val c6 = DoubleVector.broadcast(SPECIES, t6)
        val c7 = DoubleVector.broadcast(SPECIES, t7)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            var acc = DoubleVector.fromArray(SPECIES, y, yOffset + i)
            acc = multiplyAdd(c0, DoubleVector.fromArray(SPECIES, a, aOffset + i + 0 * lda), acc)
            acc = multiplyAdd(c1, DoubleVector.fromArray(SPECIES, a, aOffset + i + 1 * lda), acc)
            acc = multiplyAdd(c2, DoubleVector.fromArray(SPECIES, a, aOffset + i + 2 * lda), acc)
            acc = multiplyAdd(c3, DoubleVector.fromArray(SPECIES, a, aOffset + i + 3 * lda), acc)
            acc = multiplyAdd(c4, DoubleVector.fromArray(SPECIES, a, aOffset + i + 4 * lda), acc)
            acc = multiplyAdd(c5, DoubleVector.fromArray(SPECIES, a, aOffset + i + 5 * lda), acc)
            acc = multiplyAdd(c6, DoubleVector.fromArray(SPECIES, a, aOffset + i + 6 * lda), acc)
            acc = multiplyAdd(c7, DoubleVector.fromArray(SPECIES, a, aOffset + i + 7 * lda), acc)
            acc.intoArray(y, yOffset + i)
            i += LANE
        }
        while (i < rows) {
            y[yOffset + i] += t0 * a[aOffset + i + 0 * lda] +
                t1 * a[aOffset + i + 1 * lda] +
                t2 * a[aOffset + i + 2 * lda] +
                t3 * a[aOffset + i + 3 * lda] +
                t4 * a[aOffset + i + 4 * lda] +
                t5 * a[aOffset + i + 5 * lda] +
                t6 * a[aOffset + i + 6 * lda] +
                t7 * a[aOffset + i + 7 * lda]
            i++
        }
    }

    private fun updateFour(
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        rows: Int,
        y: DoubleArray,
        yOffset: Int,
        t0: Double,
        t1: Double,
        t2: Double,
        t3: Double,
    ) {
        val c0 = DoubleVector.broadcast(SPECIES, t0)
        val c1 = DoubleVector.broadcast(SPECIES, t1)
        val c2 = DoubleVector.broadcast(SPECIES, t2)
        val c3 = DoubleVector.broadcast(SPECIES, t3)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            var acc = DoubleVector.fromArray(SPECIES, y, yOffset + i)
            acc = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i), c0, acc)
            acc = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i + lda), c1, acc)
            acc = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i + 2 * lda), c2, acc)
            acc = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i + 3 * lda), c3, acc)
            acc.intoArray(y, yOffset + i)
            i += LANE
        }
        while (i < rows) {
            y[yOffset + i] += t0 * a[aOffset + i] + t1 * a[aOffset + i + lda] +
                t2 * a[aOffset + i + 2 * lda] + t3 * a[aOffset + i + 3 * lda]
            i++
        }
    }

    private fun updateTwo(
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        rows: Int,
        y: DoubleArray,
        yOffset: Int,
        t0: Double,
        t1: Double,
    ) {
        val c0 = DoubleVector.broadcast(SPECIES, t0)
        val c1 = DoubleVector.broadcast(SPECIES, t1)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            var acc = DoubleVector.fromArray(SPECIES, y, yOffset + i)
            acc = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i), c0, acc)
            acc = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i + lda), c1, acc)
            acc.intoArray(y, yOffset + i)
            i += LANE
        }
        while (i < rows) {
            y[yOffset + i] += t0 * a[aOffset + i] + t1 * a[aOffset + i + lda]
            i++
        }
    }

    private fun updateOne(a: DoubleArray, aOffset: Int, rows: Int, y: DoubleArray, yOffset: Int, t: Double) {
        val coefficient = DoubleVector.broadcast(SPECIES, t)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aOffset + i),
                coefficient,
                DoubleVector.fromArray(SPECIES, y, yOffset + i),
            ).intoArray(y, yOffset + i)
            i += LANE
        }
        while (i < rows) {
            y[yOffset + i] += t * a[aOffset + i]
            i++
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
            PortablePanelKernels.coupledUpdateDot(
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

    /** Four columns share each pass over the destination strip, which two columns spend twice as often. */
    private fun coupledFour(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        y: DoubleArray,
        yOffset: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        sums: DoubleArray,
        sumOffset: Int,
    ) {
        val t0 = alpha * coefficients[coefficientOffset]
        val t1 = alpha * coefficients[coefficientOffset + 1]
        val t2 = alpha * coefficients[coefficientOffset + 2]
        val t3 = alpha * coefficients[coefficientOffset + 3]
        val c0 = DoubleVector.broadcast(SPECIES, t0)
        val c1 = DoubleVector.broadcast(SPECIES, t1)
        val c2 = DoubleVector.broadcast(SPECIES, t2)
        val c3 = DoubleVector.broadcast(SPECIES, t3)
        var d0 = DoubleVector.zero(SPECIES)
        var d1 = DoubleVector.zero(SPECIES)
        var d2 = DoubleVector.zero(SPECIES)
        var d3 = DoubleVector.zero(SPECIES)
        val a1 = aOffset + lda
        val a2 = aOffset + 2 * lda
        val a3 = aOffset + 3 * lda
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            val v0 = DoubleVector.fromArray(SPECIES, a, aOffset + i)
            val v1 = DoubleVector.fromArray(SPECIES, a, a1 + i)
            val v2 = DoubleVector.fromArray(SPECIES, a, a2 + i)
            val v3 = DoubleVector.fromArray(SPECIES, a, a3 + i)
            val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
            d0 = multiplyAdd(v0, xv, d0)
            d1 = multiplyAdd(v1, xv, d1)
            d2 = multiplyAdd(v2, xv, d2)
            d3 = multiplyAdd(v3, xv, d3)
            val low = multiplyAdd(v1, c1, multiplyAdd(v0, c0, DoubleVector.fromArray(SPECIES, y, yOffset + i)))
            multiplyAdd(v3, c3, multiplyAdd(v2, c2, low)).intoArray(y, yOffset + i)
            i += LANE
        }
        var s0 = d0.reduceLanes(VectorOperators.ADD)
        var s1 = d1.reduceLanes(VectorOperators.ADD)
        var s2 = d2.reduceLanes(VectorOperators.ADD)
        var s3 = d3.reduceLanes(VectorOperators.ADD)
        while (i < rows) {
            val v0 = a[aOffset + i]
            val v1 = a[a1 + i]
            val v2 = a[a2 + i]
            val v3 = a[a3 + i]
            y[yOffset + i] += t0 * v0 + t1 * v1 + t2 * v2 + t3 * v3
            val xi = x[xOffset + i]
            s0 += v0 * xi
            s1 += v1 * xi
            s2 += v2 * xi
            s3 += v3 * xi
            i++
        }
        sums[sumOffset] += alpha * s0
        sums[sumOffset + 1] += alpha * s1
        sums[sumOffset + 2] += alpha * s2
        sums[sumOffset + 3] += alpha * s3
    }

    private fun coupledTwo(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        y: DoubleArray,
        yOffset: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        sums: DoubleArray,
        sumOffset: Int,
    ) {
        val t0 = alpha * coefficients[coefficientOffset]
        val t1 = alpha * coefficients[coefficientOffset + 1]
        val c0 = DoubleVector.broadcast(SPECIES, t0)
        val c1 = DoubleVector.broadcast(SPECIES, t1)
        var d0 = DoubleVector.zero(SPECIES)
        var d1 = DoubleVector.zero(SPECIES)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            val a0 = DoubleVector.fromArray(SPECIES, a, aOffset + i)
            val a1 = DoubleVector.fromArray(SPECIES, a, aOffset + i + lda)
            var acc = DoubleVector.fromArray(SPECIES, y, yOffset + i)
            acc = multiplyAdd(a0, c0, acc)
            acc = multiplyAdd(a1, c1, acc)
            acc.intoArray(y, yOffset + i)
            val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
            d0 = multiplyAdd(a0, xv, d0)
            d1 = multiplyAdd(a1, xv, d1)
            i += LANE
        }
        var s0 = d0.reduceLanes(VectorOperators.ADD)
        var s1 = d1.reduceLanes(VectorOperators.ADD)
        while (i < rows) {
            val v0 = a[aOffset + i]
            val v1 = a[aOffset + i + lda]
            y[yOffset + i] += t0 * v0 + t1 * v1
            val xi = x[xOffset + i]
            s0 += v0 * xi
            s1 += v1 * xi
            i++
        }
        sums[sumOffset] += alpha * s0
        sums[sumOffset + 1] += alpha * s1
    }

    private fun coupledOne(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        y: DoubleArray,
        yOffset: Int,
        coefficient: Double,
        sums: DoubleArray,
        sumOffset: Int,
    ) {
        val t = alpha * coefficient
        val scale = DoubleVector.broadcast(SPECIES, t)
        var d = DoubleVector.zero(SPECIES)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            val av = DoubleVector.fromArray(SPECIES, a, aOffset + i)
            multiplyAdd(av, scale, DoubleVector.fromArray(SPECIES, y, yOffset + i)).intoArray(y, yOffset + i)
            d = multiplyAdd(av, DoubleVector.fromArray(SPECIES, x, xOffset + i), d)
            i += LANE
        }
        var sum = d.reduceLanes(VectorOperators.ADD)
        while (i < rows) {
            val v = a[aOffset + i]
            y[yOffset + i] += t * v
            sum += v * x[xOffset + i]
            i++
        }
        sums[sumOffset] += alpha * sum
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
            PortablePanelKernels.rankUpdate(
                alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns,
                coefficients, coefficientOffset, coefficientStride,
            )
            return
        }
        var c = 0
        while (c + NARROW_GROUP <= columns) {
            rankTwo(
                a,
                aOffset + c * lda,
                lda,
                x,
                xOffset,
                rows,
                alpha * coefficients[coefficientOffset + c * coefficientStride],
                alpha * coefficients[coefficientOffset + (c + 1) * coefficientStride],
            )
            c += NARROW_GROUP
        }
        while (c < columns) {
            rankOne(
                a,
                aOffset + c * lda,
                x,
                xOffset,
                rows,
                alpha * coefficients[coefficientOffset + c * coefficientStride],
            )
            c++
        }
    }

    private fun rankTwo(
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        t0: Double,
        t1: Double,
    ) {
        val c0 = DoubleVector.broadcast(SPECIES, t0)
        val c1 = DoubleVector.broadcast(SPECIES, t1)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
            multiplyAdd(xv, c0, DoubleVector.fromArray(SPECIES, a, aOffset + i)).intoArray(a, aOffset + i)
            multiplyAdd(xv, c1, DoubleVector.fromArray(SPECIES, a, aOffset + i + lda))
                .intoArray(a, aOffset + i + lda)
            i += LANE
        }
        while (i < rows) {
            val v = x[xOffset + i]
            a[aOffset + i] += t0 * v
            a[aOffset + i + lda] += t1 * v
            i++
        }
    }

    private fun rankOne(a: DoubleArray, aOffset: Int, x: DoubleArray, xOffset: Int, rows: Int, t: Double) {
        val coefficient = DoubleVector.broadcast(SPECIES, t)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            multiplyAdd(
                DoubleVector.fromArray(SPECIES, x, xOffset + i),
                coefficient,
                DoubleVector.fromArray(SPECIES, a, aOffset + i),
            ).intoArray(a, aOffset + i)
            i += LANE
        }
        while (i < rows) {
            a[aOffset + i] += t * x[xOffset + i]
            i++
        }
    }

    /**
     * The indexed column update in lanes of right-hand sides.
     *
     * The vectorised axis is the group of right-hand sides, which is the one with nothing to carry between
     * its entries. The indices select where in the panel each column sits and are never themselves a vector
     * operation, so no indexed load or store is involved and a host without one is unaffected.
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
            PortablePanelKernels.indexedColumnUpdate(
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

    private fun indexedUpdateTwo(
        a: DoubleArray,
        at0: Int,
        at1: Int,
        rows: Int,
        y: DoubleArray,
        yOffset: Int,
        t0: Double,
        t1: Double,
    ) {
        val c0 = DoubleVector.broadcast(SPECIES, t0)
        val c1 = DoubleVector.broadcast(SPECIES, t1)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            var acc = DoubleVector.fromArray(SPECIES, y, yOffset + i)
            acc = multiplyAdd(DoubleVector.fromArray(SPECIES, a, at0 + i), c0, acc)
            acc = multiplyAdd(DoubleVector.fromArray(SPECIES, a, at1 + i), c1, acc)
            acc.intoArray(y, yOffset + i)
            i += LANE
        }
        while (i < rows) {
            y[yOffset + i] += t0 * a[at0 + i] + t1 * a[at1 + i]
            i++
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
            PortablePanelKernels.indexedRankUpdate(
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
            rankOne(a, aOffset + indices[fromIndex + c] * indexStride, x, xOffset, rows, alpha * values[fromIndex + c])
            c++
        }
    }

    private fun rankTwoAt(
        a: DoubleArray,
        at0: Int,
        at1: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        t0: Double,
        t1: Double,
    ) {
        val c0 = DoubleVector.broadcast(SPECIES, t0)
        val c1 = DoubleVector.broadcast(SPECIES, t1)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
            multiplyAdd(xv, c0, DoubleVector.fromArray(SPECIES, a, at0 + i)).intoArray(a, at0 + i)
            multiplyAdd(xv, c1, DoubleVector.fromArray(SPECIES, a, at1 + i)).intoArray(a, at1 + i)
            i += LANE
        }
        while (i < rows) {
            val v = x[xOffset + i]
            a[at0 + i] += t0 * v
            a[at1 + i] += t1 * v
            i++
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
            PortablePanelKernels.indexedCoupledUpdate(
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
                scatterOnly(t, a, at, b, pivot, rows)
            } else {
                scatterAndReduce(t, a, at, b, pivot, sums, sumOffset, rows, sourceAt)
            }
        }
    }

    /** One position of the coupled pass that scatters and is not reduced, which is a symmetric diagonal. */
    private fun scatterOnly(t: Double, a: DoubleArray, at: Int, b: DoubleArray, pivot: Int, rows: Int) {
        val coefficient = DoubleVector.broadcast(SPECIES, t)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            multiplyAdd(
                DoubleVector.fromArray(SPECIES, b, pivot + i),
                coefficient,
                DoubleVector.fromArray(SPECIES, a, at + i),
            ).intoArray(a, at + i)
            i += LANE
        }
        while (i < rows) {
            a[at + i] += t * b[pivot + i]
            i++
        }
    }

    /** One position of the coupled pass that does both halves, which is every position but the diagonal. */
    @Suppress("LongParameterList") // the multiplier, the two windows it lands in, and the one it reads
    private fun scatterAndReduce(
        t: Double,
        a: DoubleArray,
        at: Int,
        b: DoubleArray,
        pivot: Int,
        sums: DoubleArray,
        sumOffset: Int,
        rows: Int,
        sourceAt: Int,
    ) {
        val coefficient = DoubleVector.broadcast(SPECIES, t)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            multiplyAdd(
                DoubleVector.fromArray(SPECIES, b, pivot + i),
                coefficient,
                DoubleVector.fromArray(SPECIES, a, at + i),
            ).intoArray(a, at + i)
            multiplyAdd(
                DoubleVector.fromArray(SPECIES, b, sourceAt + i),
                coefficient,
                DoubleVector.fromArray(SPECIES, sums, sumOffset + i),
            ).intoArray(sums, sumOffset + i)
            i += LANE
        }
        while (i < rows) {
            a[at + i] += t * b[pivot + i]
            sums[sumOffset + i] += t * b[sourceAt + i]
            i++
        }
    }

    /** `y = alpha · sum + beta · y`, where a zero beta overwrites without reading the destination. */
    /** A panel with fewer rows than one lane block, which a column cannot fill and the work runs along. */
    private fun isShort(rows: Int): Boolean = simdAvailable && rows in 1 until VECTOR_MINIMUM

    /**
     * Panels handed over whole. A multi-dot groups its columns itself and reuses the shared vector across all of
     * them, so a call per group only adds calls; so does a column update short enough that a call per group
     * costs more than the arithmetic in it.
     */
    private fun groupsWholeRow(work: PanelWork, rows: Int): Boolean = work == PanelWork.MultiDot ||
        (work == PanelWork.ColumnUpdate && (isShort(rows) || isResident(rows)))

    /** The short panels this backend runs a body of its own for; a longer short column update is portable. */
    private fun reachesShortBody(work: PanelWork, rows: Int): Boolean =
        (work == PanelWork.MultiDot && isShort(rows)) ||
            (work == PanelWork.ColumnUpdate && rows == 1 && simdAvailable)

    /**
     * A short [multiDot]. A single row is an elementwise update along it; otherwise each column is a scalar
     * dot over its few rows. Masked lane blocks would read only the window, but their masks are heap objects
     * on the measured runtime.
     */
    private fun multiDotShort(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        columns: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ) {
        if (rows == 1) {
            scaledRow(alpha * x[xOffset], a, aOffset, lda, columns, beta, y, yOffset, yStride)
            return
        }
        for (c in 0 until columns) {
            val at = aOffset + c * lda
            var sum = 0.0
            for (i in 0 until rows) sum += a[at + i] * x[xOffset + i]
            store(y, yOffset + c * yStride, alpha, sum, beta)
        }
    }

    /** `y(c) = t · a(0, c) + beta · y(c)` along one stored row. */
    private fun scaledRow(
        t: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        columns: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ) {
        var c = 0
        if (lda == 1 && yStride == 1) {
            val tv = DoubleVector.broadcast(SPECIES, t)
            val bound = SPECIES.loopBound(columns)
            if (beta == 0.0) {
                while (c < bound) {
                    DoubleVector.fromArray(SPECIES, a, aOffset + c).mul(tv).intoArray(y, yOffset + c)
                    c += LANE
                }
            } else {
                val bv = DoubleVector.broadcast(SPECIES, beta)
                while (c < bound) {
                    multiplyAdd(
                        DoubleVector.fromArray(SPECIES, a, aOffset + c),
                        tv,
                        DoubleVector.fromArray(SPECIES, y, yOffset + c).mul(bv),
                    ).intoArray(y, yOffset + c)
                    c += LANE
                }
            }
        }
        while (c < columns) {
            store(y, yOffset + c * yStride, 1.0, t * a[aOffset + c * lda], beta)
            c++
        }
    }

    /**
     * A short [columnUpdate]. A single row is one dot along it; otherwise the portable body runs over every
     * column in one call, which the whole-width [executionGroup] is for.
     */
    private fun columnUpdateShort(
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
    ) {
        if (rows == 1) {
            val sum = if (lda == 1 && xStride == 1) {
                SimdOps.dot(a, aOffset, x, xOffset, columns)
            } else {
                stridedDot(a, aOffset, lda, x, xOffset, xStride, columns)
            }
            y[yOffset] += alpha * sum
            return
        }
        PortablePanelKernels.columnUpdate(alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, y, yOffset, 1)
    }

    /** Rows a column update keeps in registers across every column instead of reloading per group. */
    private fun isResident(rows: Int): Boolean = simdAvailable && rows >= LANE && rows < RESIDENT_BLOCKS * LANE

    /**
     * A column update over one or two lane blocks of rows, with the destination held in registers across
     * every column. On a panel this short each group of columns is a single vector iteration, and a group
     * reloading what the previous one stored makes the whole call one dependency chain through memory; here
     * two chains per block run across the columns and the destination is written once.
     */
    private fun columnUpdateResident(
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
    ) {
        val two = rows >= 2 * LANE
        var even0 = DoubleVector.zero(SPECIES)
        var odd0 = DoubleVector.zero(SPECIES)
        var even1 = DoubleVector.zero(SPECIES)
        var odd1 = DoubleVector.zero(SPECIES)
        var c = 0
        if (two) {
            while (c + 2 <= columns) {
                val t0 = DoubleVector.broadcast(SPECIES, x[xOffset + c * xStride])
                val t1 = DoubleVector.broadcast(SPECIES, x[xOffset + (c + 1) * xStride])
                val at = aOffset + c * lda
                even0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, at), t0, even0)
                even1 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, at + LANE), t0, even1)
                odd0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, at + lda), t1, odd0)
                odd1 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, at + lda + LANE), t1, odd1)
                c += 2
            }
            if (c < columns) {
                val t0 = DoubleVector.broadcast(SPECIES, x[xOffset + c * xStride])
                even0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + c * lda), t0, even0)
                even1 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + c * lda + LANE), t0, even1)
            }
        } else {
            while (c + 2 <= columns) {
                val at = aOffset + c * lda
                even0 = multiplyAdd(
                    DoubleVector.fromArray(SPECIES, a, at),
                    DoubleVector.broadcast(SPECIES, x[xOffset + c * xStride]),
                    even0,
                )
                odd0 = multiplyAdd(
                    DoubleVector.fromArray(SPECIES, a, at + lda),
                    DoubleVector.broadcast(SPECIES, x[xOffset + (c + 1) * xStride]),
                    odd0,
                )
                c += 2
            }
            if (c < columns) {
                even0 = multiplyAdd(
                    DoubleVector.fromArray(SPECIES, a, aOffset + c * lda),
                    DoubleVector.broadcast(SPECIES, x[xOffset + c * xStride]),
                    even0,
                )
            }
        }
        val scale = DoubleVector.broadcast(SPECIES, alpha)
        multiplyAdd(even0.add(odd0), scale, DoubleVector.fromArray(SPECIES, y, yOffset)).intoArray(y, yOffset)
        var i = LANE
        if (two) {
            multiplyAdd(even1.add(odd1), scale, DoubleVector.fromArray(SPECIES, y, yOffset + LANE))
                .intoArray(y, yOffset + LANE)
            i = 2 * LANE
        }
        while (i < rows) {
            y[yOffset + i] += alpha * stridedDot(a, aOffset + i, lda, x, xOffset, xStride, columns)
            i++
        }
    }

    /** A dot whose operands are both strided, with four sums so the additions do not wait on each other. */
    private fun stridedDot(
        a: DoubleArray,
        aOffset: Int,
        aStride: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        length: Int,
    ): Double {
        var s0 = 0.0
        var s1 = 0.0
        var s2 = 0.0
        var s3 = 0.0
        var c = 0
        while (c + 4 <= length) {
            s0 += a[aOffset + c * aStride] * x[xOffset + c * xStride]
            s1 += a[aOffset + (c + 1) * aStride] * x[xOffset + (c + 1) * xStride]
            s2 += a[aOffset + (c + 2) * aStride] * x[xOffset + (c + 2) * xStride]
            s3 += a[aOffset + (c + 3) * aStride] * x[xOffset + (c + 3) * xStride]
            c += 4
        }
        while (c < length) {
            s0 += a[aOffset + c * aStride] * x[xOffset + c * xStride]
            c++
        }
        return (s0 + s1) + (s2 + s3)
    }

    private fun store(y: DoubleArray, at: Int, alpha: Double, sum: Double, beta: Double) {
        y[at] = if (beta == 0.0) alpha * sum else alpha * sum + beta * y[at]
    }

    /** Lane blocks a panel needs before its vector body can run, which is one whole vector. */
    private const val MINIMUM_BLOCKS = 1

    /**
     * Packing threshold in lane blocks, above [MINIMUM_BLOCKS] because a copy must pay for its data pass.
     */
    private const val COPY_WORTH_BLOCKS = 2

    /** Columns grouped where one loaded vector serves all of them. */
    /** Columns one pass of a multi-dot reduces against the shared vector. */
    private const val MULTI_DOT_GROUP = 8
    private const val COLUMN_UPDATE_GROUP = 8


    /** Lane blocks below which a column update keeps its destination in registers across the columns. */
    private const val RESIDENT_BLOCKS = 3

    private const val LARGE_COLUMN_UPDATE_GROUP = 12
    private const val LARGE_COLUMN_UPDATE_MINIMUM_ROWS = 4_096

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
