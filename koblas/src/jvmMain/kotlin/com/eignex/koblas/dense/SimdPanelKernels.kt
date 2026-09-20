@file:Suppress("LongParameterList") // a panel carries its window, its extents and its scaling as plain numbers

package com.eignex.koblas.dense

import com.eignex.koblas.internal.numeric.hardwareFusedMultiplyAdd
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/**
 * The JVM Vector API panel arithmetic, with the portable bodies underneath it.
 *
 * A multi-dot and a coupled pass are reductions, and splitting a floating point sum across lanes changes the
 * answer, which is why the portable loops are written in the order they are and why these are written in
 * lanes instead. A column update and a rank update are elementwise, where the same reordering is not a
 * question; what these add is that the destination strip stays in registers while several columns accumulate
 * into it. Whether either is faster than what the compiler makes of the portable loop is a measurement rather
 * than a property of the source, and the stage evidence is where the comparison is recorded.
 *
 * These are not what an ordinary call runs yet. [com.eignex.koblas.BuiltinEngines.simd] is the arm that holds
 * them and is where they are measured; the platform default keeps the portable panels until a crossover has
 * been established, which is a decision with its own evidence rather than a consequence of writing them.
 *
 * The grouping below is this backend's own, and it is not a lane count: the panels group columns, and a
 * column's lanes run down its rows.
 *
 * A panel shorter than one lane block and a strided operand the Vector API would have to gather both fall to
 * [PortablePanelKernels], and [implementationFor] says which of the two a given panel reaches rather than
 * letting the selection's name imply the vector one.
 *
 * Resolving the species is what initializing this costs, so a runtime without the module must not reach it at
 * all: [com.eignex.koblas.BuiltinEngines] offers no engine holding this backend there, and the portable one
 * is what every call then schedules against.
 */
internal object SimdPanelKernels : DensePanelKernels {
    private val SPECIES = DoubleVector.SPECIES_PREFERRED
    private val LANE = if (simdAvailable) SPECIES.length() else 0

    /**
     * Built once. A name is constant for the life of this object, and rebuilding it per access would make
     * every question about which body a shape reaches allocate, including the ones a scheduling decision
     * asks on the way into a call.
     */
    private val NAME: String = "simd-panel($LANE lanes)"

    override val name: String get() = NAME

    /**
     * Rows a panel needs before its vector body is reached, as whole lane blocks of this machine.
     *
     * One block is the structural minimum: below it there is no whole vector to load and the body cannot
     * run at all. No measured crossover sits above it. The calibration timed whole Level 2 operations at
     * orders of twelve and below, where a vector body has one or two blocks to work with, at the preferred
     * width and at two lanes and with the fused multiply-add on and off; none of those was behind the
     * portable panels on the measured host. That is one host, so the number here stays the structural
     * minimum rather than becoming a tuned constant that another machine would have to undo.
     */
    private val VECTOR_MINIMUM = MINIMUM_BLOCKS * LANE

    /** Whether a panel of this shape and spacing reaches the vector bodies rather than the portable ones. */
    private fun vectorizes(rows: Int, contiguous: Boolean): Boolean =
        simdAvailable && contiguous && rows >= VECTOR_MINIMUM

    /**
     * Four columns for the two panels whose columns share one loaded vector, two for the two that do not.
     *
     * A multi-dot holds one accumulator per column and a column update holds one destination strip, and
     * either way every column of a group is served by the same loaded vector, so a wider group pays for more
     * arithmetic with each load; four measured ahead of two at every extent tried. The coupled and
     * rank-update bodies below are two columns wide, so asking for four would only mean two calls fused into
     * one, and for a triangular caller a wider group also grows the scalar corner with the square of it.
     *
     * The comparison is one shared machine's at four lanes and is kept with the stage evidence rather than
     * quoted here. It says which choice was ahead there; a lane count and a cache are not the same on the
     * next machine, which is why the number is not in this file.
     */
    override fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): Int {
        if (!simdAvailable) return PortablePanelKernels.executionGroup(work, rows, columns, contiguous)
        val preferred = when {
            work == PanelWork.SparseRightHandSideReduction && !contiguous -> SPARSE_REDUCTION_GROUP

            work == PanelWork.SparseRightHandSides || work == PanelWork.SparseRightHandSideReduction ->
                if (contiguous) ADJACENT_SPARSE_BLOCKS * LANE else SPARSE_GROUP

            work == PanelWork.CoupledDotUpdate || work == PanelWork.RankUpdate -> NARROW_GROUP

            else -> WIDE_GROUP
        }
        return if (columns <= 0) 1 else minOf(preferred, columns)
    }

    /**
     * From two lane blocks of adjacent rows upward, which is wider than where the vector bodies start.
     *
     * One lane block is one vector operation per column, and a caller that copied a panel that narrow would
     * pay a pass over its data for a single instruction's worth of arithmetic. A measured crossover of this
     * backend's own rather than a structural minimum like the one [implementationFor] answers with: the
     * copy was timed at both widths and the narrow one did not pay for itself. The measurement and its
     * limits are in the stage evidence, since a figure quoted here would be one machine's.
     */
    override fun prefersContiguous(work: PanelWork, rows: Int, columns: Int): Boolean =
        simdAvailable && rows >= COPY_WORTH_BLOCKS * LANE && !vectorizes(rows, contiguous = false)

    override fun implementationFor(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): String =
        if (vectorizes(rows, contiguous)) {
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
        if (!vectorizes(rows, xStride == 1)) {
            PortablePanelKernels.multiDot(
                alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, beta, y, yOffset, yStride,
            )
            return
        }
        var c = 0
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
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
            v0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i), xv, v0)
            v1 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i + lda), xv, v1)
            v2 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i + 2 * lda), xv, v2)
            v3 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i + 3 * lda), xv, v3)
            i += LANE
        }
        var s0 = v0.reduceLanes(VectorOperators.ADD)
        var s1 = v1.reduceLanes(VectorOperators.ADD)
        var s2 = v2.reduceLanes(VectorOperators.ADD)
        var s3 = v3.reduceLanes(VectorOperators.ADD)
        while (i < rows) {
            val xi = x[xOffset + i]
            s0 += a[aOffset + i] * xi
            s1 += a[aOffset + i + lda] * xi
            s2 += a[aOffset + i + 2 * lda] * xi
            s3 += a[aOffset + i + 3 * lda] * xi
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
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
            v0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i), xv, v0)
            v1 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i + lda), xv, v1)
            i += LANE
        }
        var s0 = v0.reduceLanes(VectorOperators.ADD)
        var s1 = v1.reduceLanes(VectorOperators.ADD)
        while (i < rows) {
            val xi = x[xOffset + i]
            s0 += a[aOffset + i] * xi
            s1 += a[aOffset + i + lda] * xi
            i++
        }
        store(y, yOffset, alpha, s0, beta)
        store(y, yOffset + yStride, alpha, s1, beta)
    }

    private fun dotOne(a: DoubleArray, aOffset: Int, x: DoubleArray, xOffset: Int, rows: Int): Double {
        var v = DoubleVector.zero(SPECIES)
        var i = 0
        val bound = SPECIES.loopBound(rows)
        while (i < bound) {
            v = multiplyAdd(
                DoubleVector.fromArray(SPECIES, a, aOffset + i),
                DoubleVector.fromArray(SPECIES, x, xOffset + i),
                v,
            )
            i += LANE
        }
        var sum = v.reduceLanes(VectorOperators.ADD)
        while (i < rows) {
            sum += a[aOffset + i] * x[xOffset + i]
            i++
        }
        return sum
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
        if (!vectorizes(rows, yStride == 1)) {
            PortablePanelKernels.columnUpdate(
                alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, y, yOffset, yStride,
            )
            return
        }
        var c = 0
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
    ) {
        if (rows <= 0 || columns <= 0) return
        if (!vectorizes(rows, contiguous = true)) {
            PortablePanelKernels.coupledUpdateDot(
                alpha, a, aOffset, lda, x, xOffset, rows, columns, y, yOffset,
                coefficients, coefficientOffset, sums, sumOffset,
            )
            return
        }
        var c = 0
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
     * The coupled indexed pass in lanes of right-hand sides.
     *
     * One address serves both halves: the position a stored coefficient scatters into is the position the
     * reduction reads back, so the walk computes it once and one broadcast coefficient drives both
     * multiply-adds. The scattered half reads the pivot column where it stands, which is a load the loop
     * repeats per position rather than a gather the caller pays for per column.
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
    ) {
        if (rows <= 0 || columns <= 0) return
        if (!vectorizes(rows, rowStride == 1)) {
            PortablePanelKernels.indexedCoupledUpdate(
                alpha, a, b, offset, rowStride, indexStride, indices, values, fromIndex, columns, rows,
                pivot, sums, sumOffset, excluded,
            )
            return
        }
        for (c in 0 until columns) {
            val t = alpha * values[fromIndex + c]
            val at = offset + indices[fromIndex + c] * indexStride
            // The exclusion is decided per position and never inside the lanes. A branch between the two
            // stores leaves the loop with a merge the virtual machine will not eliminate the vector boxes
            // across, and a body that allocates one object per lane block runs several times slower than the
            // portable loop it is there to beat.
            if (c == excluded) {
                scatterOnly(t, a, at, b, pivot, rows)
            } else {
                scatterAndReduce(t, a, at, b, pivot, sums, sumOffset, rows)
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
                DoubleVector.fromArray(SPECIES, b, at + i),
                coefficient,
                DoubleVector.fromArray(SPECIES, sums, sumOffset + i),
            ).intoArray(sums, sumOffset + i)
            i += LANE
        }
        while (i < rows) {
            a[at + i] += t * b[pivot + i]
            sums[sumOffset + i] += t * b[at + i]
            i++
        }
    }

    /** `y = alpha · sum + beta · y`, where a zero beta overwrites without reading the destination. */
    private fun store(y: DoubleArray, at: Int, alpha: Double, sum: Double, beta: Double) {
        y[at] = if (beta == 0.0) alpha * sum else alpha * sum + beta * y[at]
    }

    /**
     * One multiply-add, fused where [hardwareFusedMultiplyAdd] found the instruction and two operations
     * where it did not.
     *
     * Inline because a helper returning a [DoubleVector] that the JIT declines to inline makes the vector a
     * heap object, and the loop then allocates once per iteration. Kotlin's warning does not account for
     * avoiding that allocation. Inlining also folds the test to a constant, so one of the two forms reaches
     * the loop and neither a branch nor the other form survives in it.
     */
    @Suppress("NOTHING_TO_INLINE")
    private inline fun multiplyAdd(x: DoubleVector, y: DoubleVector, accumulator: DoubleVector): DoubleVector =
        if (hardwareFusedMultiplyAdd) x.fma(y, accumulator) else x.mul(y).add(accumulator)

    /** Lane blocks a panel needs before its vector body can run, which is one whole vector. */
    private const val MINIMUM_BLOCKS = 1

    /**
     * Lane blocks of adjacent rows a panel needs before a copy into adjacent storage pays for itself.
     *
     * Wider than [MINIMUM_BLOCKS], which is where the body starts running rather than where it is worth a
     * pass over the data to reach. The stage evidence is where the comparison at each width is recorded.
     */
    private const val COPY_WORTH_BLOCKS = 2

    /** Columns grouped where one loaded vector serves all of them. */
    private const val WIDE_GROUP = 4

    /** Columns grouped where the body itself is that wide. */
    private const val NARROW_GROUP = 2

    /**
     * Dense right-hand sides a sparse column walk serves at once, where they are a leading dimension apart.
     *
     * Not a vector width: a group this wide is what keeps one walk of a column's indices and values serving
     * several right-hand sides, and the arithmetic over a strided group is the portable body whatever the
     * species is. The same width the portable backend answers with, and measured the same way, since the
     * body that runs over a strided group is that backend's.
     */
    private const val SPARSE_GROUP = 8

    /**
     * Right-hand sides a reduction over a strided block serves at once, which is one for the reason the
     * portable backend gives: below a vector its accumulator wants a register rather than an array, and a
     * strided block reaches no vector body here either.
     */
    private const val SPARSE_REDUCTION_GROUP = 1

    /**
     * Lane blocks of adjacent right-hand sides this backend asks for, which is a different question.
     *
     * Where they are adjacent the arithmetic over a group is whole vectors, so the group wants to be lane
     * blocks rather than a count that has nothing to do with the species: a group of four on a machine with
     * eight lanes would leave half of every vector idle. Several blocks rather than one, because the walk of
     * a sparse column's indices is paid once for the whole group and a wider group spreads it further; the
     * ceiling is what stays resident while that column is walked, and the sparse scheduling caps this again
     * for its own staging buffer.
     *
     * Eight, from a sweep of the widths either side of it: the narrow ones lose across the grid, and the
     * two widest are close enough that either is defensible, with the narrower of them ahead on the denser
     * supports. Eight is the conservative end of that pair. The count is this backend's own, and the stage
     * evidence holds the comparison and the spread it was chosen from.
     */
    private const val ADJACENT_SPARSE_BLOCKS = 8
}
