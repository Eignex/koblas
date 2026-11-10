@file:Suppress("LongParameterList") // Panels address independent array windows.

package com.eignex.koblas.internal.jvm25.simd

import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

private val SPECIES = SIMD_SPECIES
private val LANE = SPECIES.length()

/**
 * Eight columns per pass over the shared vector, which halves its reloads against four and gives every
 * column a chain of its own: eight chains cover the FMA latency without pairing.
 */
internal fun dotEight(
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
 * Four columns per pass over the shared vector, one chain each. [lda] and [yStride] may be negative,
 * which is how a triangular group walks its lines from the far end.
 */
internal fun dotFour(
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
    val a1 = aOffset + lda
    val a2 = aOffset + 2 * lda
    val a3 = aOffset + 3 * lda
    var i = 0
    while (i + LANE <= rows) {
        val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
        v0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i), xv, v0)
        v1 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a1 + i), xv, v1)
        v2 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a2 + i), xv, v2)
        v3 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a3 + i), xv, v3)
        i += LANE
    }
    // A run shorter than a lane block, common in a triangular group, skips four empty reductions.
    val vectored = i > 0
    var s0 = if (vectored) v0.reduceLanes(VectorOperators.ADD) else 0.0
    var s1 = if (vectored) v1.reduceLanes(VectorOperators.ADD) else 0.0
    var s2 = if (vectored) v2.reduceLanes(VectorOperators.ADD) else 0.0
    var s3 = if (vectored) v3.reduceLanes(VectorOperators.ADD) else 0.0
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

internal fun dotTwo(
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
    val a1 = aOffset + lda
    var i = 0
    while (i + LANE <= rows) {
        val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
        v0 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, aOffset + i), xv, v0)
        v1 = multiplyAdd(DoubleVector.fromArray(SPECIES, a, a1 + i), xv, v1)
        i += LANE
    }
    var s0 = v0.reduceLanes(VectorOperators.ADD)
    var s1 = v1.reduceLanes(VectorOperators.ADD)
    while (i < rows) {
        val xi = x[xOffset + i]
        s0 += a[aOffset + i] * xi
        s1 += a[a1 + i] * xi
        i++
    }
    store(y, yOffset, alpha, s0, beta)
    store(y, yOffset + yStride, alpha, s1, beta)
}

/**
 * Short [SimdPanelKernels.multiDot] panels run along one row or reduce a few rows with shared x registers.
 * Masked lane blocks would stay inside the window, but their masks allocate on the measured runtime.
 */
internal fun multiDotShort(
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
    // The first rows' x stays in registers across every column, and a column's products sum as a tree so
    // neighbouring columns overlap; reloading x and chaining the sum per column made a wide panel of a
    // few rows several times slower than the vendors.
    val x0 = x[xOffset]
    val x1 = x[xOffset + 1]
    if (rows >= 4) {
        val x2 = x[xOffset + 2]
        val x3 = x[xOffset + 3]
        for (c in 0 until columns) {
            val at = aOffset + c * lda
            var sum = (a[at] * x0 + a[at + 1] * x1) + (a[at + 2] * x2 + a[at + 3] * x3)
            for (i in 4 until rows) sum += a[at + i] * x[xOffset + i]
            store(y, yOffset + c * yStride, alpha, sum, beta)
        }
        return
    }
    for (c in 0 until columns) {
        val at = aOffset + c * lda
        var sum = a[at] * x0 + a[at + 1] * x1
        if (rows == 3) sum += a[at + 2] * x[xOffset + 2]
        store(y, yOffset + c * yStride, alpha, sum, beta)
    }
}

/** `y(c) = t · a(0, c) + beta · y(c)` along one stored row. */
internal fun scaledRow(
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

/** `y = alpha · sum + beta · y`, where a zero beta overwrites without reading the destination. */
internal fun store(y: DoubleArray, at: Int, alpha: Double, sum: Double, beta: Double) {
    y[at] = if (beta == 0.0) alpha * sum else alpha * sum + beta * y[at]
}
