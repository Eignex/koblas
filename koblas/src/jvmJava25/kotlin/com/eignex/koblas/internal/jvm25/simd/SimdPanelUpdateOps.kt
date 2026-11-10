@file:Suppress("LongParameterList") // Panels address independent array windows.

package com.eignex.koblas.internal.jvm25.simd

import jdk.incubator.vector.DoubleVector

private val SPECIES = SIMD_SPECIES
private val LANE = SPECIES.length()

/** Small panel calls otherwise pay another JVM frame for every eight columns. */
@Suppress("NOTHING_TO_INLINE")
internal inline fun updateEight(
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

/** `y += t0 · a0 + … + t3 · a3` over four lines [lda] apart; [lda] may be negative, as [dotFour]'s may. */
internal fun updateFour(
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

/** `y += t · a` over one contiguous run, shared by every single-column update and scatter. */
internal fun updateOne(a: DoubleArray, aOffset: Int, rows: Int, y: DoubleArray, yOffset: Int, t: Double) {
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

internal fun indexedUpdateTwo(
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

internal fun rankTwoAt(
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
 * Short [SimdPanelKernels.columnUpdate] panels arrive whole from [SimdPanelKernels.executionGroup].
 * One row runs as a dot; more rows stay in scalar registers across every column, four or two at a time.
 */
internal fun columnUpdateShort(
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
            SimdVectorOps.dot(a, aOffset, x, xOffset, columns)
        } else {
            stridedDot(a, aOffset, lda, x, xOffset, xStride, columns)
        }
        y[yOffset] += alpha * sum
        return
    }
    var r = 0
    while (r + 4 <= rows) {
        shortQuad(alpha, a, aOffset + r, lda, x, xOffset, xStride, columns, y, yOffset + r)
        r += 4
    }
    if (r + 2 <= rows) {
        shortPair(alpha, a, aOffset + r, lda, x, xOffset, xStride, columns, y, yOffset + r)
        r += 2
    }
    if (r < rows) y[yOffset + r] += alpha * stridedDot(a, aOffset + r, lda, x, xOffset, xStride, columns)
}

/**
 * Four rows of a short column update in scalar registers across every column, two chains per row over
 * the even and the odd columns. Written back per column pair, the destination was a chain through
 * memory that left a wide panel of a few rows several times behind the vendors.
 */
internal fun shortQuad(
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    columns: Int,
    y: DoubleArray,
    yOffset: Int,
) {
    var e0 = 0.0
    var e1 = 0.0
    var e2 = 0.0
    var e3 = 0.0
    var o0 = 0.0
    var o1 = 0.0
    var o2 = 0.0
    var o3 = 0.0
    var c = 0
    while (c + 2 <= columns) {
        val t0 = x[xOffset + c * xStride]
        val t1 = x[xOffset + (c + 1) * xStride]
        val at = aOffset + c * lda
        val bt = at + lda
        e0 += a[at] * t0
        e1 += a[at + 1] * t0
        e2 += a[at + 2] * t0
        e3 += a[at + 3] * t0
        o0 += a[bt] * t1
        o1 += a[bt + 1] * t1
        o2 += a[bt + 2] * t1
        o3 += a[bt + 3] * t1
        c += 2
    }
    if (c < columns) {
        val t0 = x[xOffset + c * xStride]
        val at = aOffset + c * lda
        e0 += a[at] * t0
        e1 += a[at + 1] * t0
        e2 += a[at + 2] * t0
        e3 += a[at + 3] * t0
    }
    y[yOffset] += alpha * (e0 + o0)
    y[yOffset + 1] += alpha * (e1 + o1)
    y[yOffset + 2] += alpha * (e2 + o2)
    y[yOffset + 3] += alpha * (e3 + o3)
}

/** The two-row counterpart of [shortQuad]. */
internal fun shortPair(
    alpha: Double,
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    columns: Int,
    y: DoubleArray,
    yOffset: Int,
) {
    var e0 = 0.0
    var e1 = 0.0
    var o0 = 0.0
    var o1 = 0.0
    var c = 0
    while (c + 2 <= columns) {
        val t0 = x[xOffset + c * xStride]
        val t1 = x[xOffset + (c + 1) * xStride]
        val at = aOffset + c * lda
        e0 += a[at] * t0
        e1 += a[at + 1] * t0
        o0 += a[at + lda] * t1
        o1 += a[at + lda + 1] * t1
        c += 2
    }
    if (c < columns) {
        val t0 = x[xOffset + c * xStride]
        e0 += a[aOffset + c * lda] * t0
        e1 += a[aOffset + c * lda + 1] * t0
    }
    y[yOffset] += alpha * (e0 + o0)
    y[yOffset + 1] += alpha * (e1 + o1)
}

/**
 * A column update over one or two lane blocks of rows, with the destination held in registers across
 * every column. On a panel this short each group of columns is a single vector iteration, and a group
 * reloading what the previous one stored makes the whole call one dependency chain through memory; here
 * two chains per block run across the columns and the destination is written once.
 */
internal fun columnUpdateResident(
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
internal fun stridedDot(
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
