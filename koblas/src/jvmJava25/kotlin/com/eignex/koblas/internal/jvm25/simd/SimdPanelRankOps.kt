@file:Suppress("LongParameterList") // Panels address independent array windows.

package com.eignex.koblas.internal.jvm25.simd

import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

private val SPECIES = SIMD_SPECIES
private val LANE = SPECIES.length()

/** Four columns share each pass over the destination strip, which two columns spend twice as often. */
internal fun coupledFour(
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

internal fun coupledTwo(
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

internal fun coupledOne(
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

/** Two columns of a rank-two update, sharing each lane block of both source windows. */
internal fun symmetricTwo(
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    x: DoubleArray,
    xOffset: Int,
    y: DoubleArray,
    yOffset: Int,
    rows: Int,
    s0: Double,
    t0: Double,
    s1: Double,
    t1: Double,
) {
    val xs0 = DoubleVector.broadcast(SPECIES, s0)
    val yt0 = DoubleVector.broadcast(SPECIES, t0)
    val xs1 = DoubleVector.broadcast(SPECIES, s1)
    val yt1 = DoubleVector.broadcast(SPECIES, t1)
    var i = 0
    val bound = SPECIES.loopBound(rows)
    while (i < bound) {
        val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
        val yv = DoubleVector.fromArray(SPECIES, y, yOffset + i)
        multiplyAdd(yv, yt0, multiplyAdd(xv, xs0, DoubleVector.fromArray(SPECIES, a, aOffset + i)))
            .intoArray(a, aOffset + i)
        multiplyAdd(yv, yt1, multiplyAdd(xv, xs1, DoubleVector.fromArray(SPECIES, a, aOffset + i + lda)))
            .intoArray(a, aOffset + i + lda)
        i += LANE
    }
    while (i < rows) {
        val xi = x[xOffset + i]
        val yi = y[yOffset + i]
        a[aOffset + i] += s0 * xi + t0 * yi
        a[aOffset + i + lda] += s1 * xi + t1 * yi
        i++
    }
}

internal fun symmetricOne(
    a: DoubleArray,
    aOffset: Int,
    x: DoubleArray,
    xOffset: Int,
    y: DoubleArray,
    yOffset: Int,
    rows: Int,
    s: Double,
    t: Double,
) {
    val xs = DoubleVector.broadcast(SPECIES, s)
    val yt = DoubleVector.broadcast(SPECIES, t)
    var i = 0
    val bound = SPECIES.loopBound(rows)
    while (i < bound) {
        val partial = multiplyAdd(
            DoubleVector.fromArray(SPECIES, x, xOffset + i),
            xs,
            DoubleVector.fromArray(SPECIES, a, aOffset + i),
        )
        multiplyAdd(DoubleVector.fromArray(SPECIES, y, yOffset + i), yt, partial).intoArray(a, aOffset + i)
        i += LANE
    }
    while (i < rows) {
        a[aOffset + i] += s * x[xOffset + i] + t * y[yOffset + i]
        i++
    }
}

/** Four columns of a rank update sharing each loaded lane block of the source, for narrow vector units. */
internal fun rankFour(
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    x: DoubleArray,
    xOffset: Int,
    rows: Int,
    t0: Double,
    t1: Double,
    t2: Double,
    t3: Double,
) {
    val c0 = DoubleVector.broadcast(SPECIES, t0)
    val c1 = DoubleVector.broadcast(SPECIES, t1)
    val c2 = DoubleVector.broadcast(SPECIES, t2)
    val c3 = DoubleVector.broadcast(SPECIES, t3)
    val a1 = aOffset + lda
    val a2 = aOffset + 2 * lda
    val a3 = aOffset + 3 * lda
    var i = 0
    val bound = SPECIES.loopBound(rows)
    while (i < bound) {
        val xv = DoubleVector.fromArray(SPECIES, x, xOffset + i)
        multiplyAdd(xv, c0, DoubleVector.fromArray(SPECIES, a, aOffset + i)).intoArray(a, aOffset + i)
        multiplyAdd(xv, c1, DoubleVector.fromArray(SPECIES, a, a1 + i)).intoArray(a, a1 + i)
        multiplyAdd(xv, c2, DoubleVector.fromArray(SPECIES, a, a2 + i)).intoArray(a, a2 + i)
        multiplyAdd(xv, c3, DoubleVector.fromArray(SPECIES, a, a3 + i)).intoArray(a, a3 + i)
        i += LANE
    }
    while (i < rows) {
        val v = x[xOffset + i]
        a[aOffset + i] += t0 * v
        a[a1 + i] += t1 * v
        a[a2 + i] += t2 * v
        a[a3 + i] += t3 * v
        i++
    }
}

/** One position of the coupled pass that does both halves, which is every position but the diagonal. */
@Suppress("LongParameterList") // the multiplier, the two windows it lands in, and the one it reads
internal fun scatterAndReduce(
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
