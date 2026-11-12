// Primitive windows carry offsets and extents; fold their leaf arithmetic into backend entries.
@file:Suppress("LongParameterList", "TooManyFunctions", "NOTHING_TO_INLINE")

package com.eignex.koblas.internal.kernels.scalar.dense

internal inline fun scalarMultiDot(
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
    var c = 0
    while (c + PANEL_WIDE_GROUP <= columns) {
        val to = yOffset + c * yStride
        panelDotFour(a, aOffset + c * lda, lda, x, xOffset, xStride, rows, alpha, beta, y, to, yStride)
        c += PANEL_WIDE_GROUP
    }
    while (c < columns) {
        val sum = panelDotOne(a, aOffset + c * lda, x, xOffset, xStride, rows)
        panelStore(y, yOffset + c * yStride, alpha, sum, beta)
        c++
    }
}

/** One walk of the shared vector feeds four independent sums. */
private inline fun panelDotFour(
    a: DoubleArray,
    aOffset: Int,
    lda: Int,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    rows: Int,
    alpha: Double,
    beta: Double,
    y: DoubleArray,
    yOffset: Int,
    yStride: Int,
) {
    var s0 = 0.0
    var s1 = 0.0
    var s2 = 0.0
    var s3 = 0.0
    var at = xOffset
    for (i in 0 until rows) {
        val v = x[at]
        s0 += a[aOffset + i] * v
        s1 += a[aOffset + i + lda] * v
        s2 += a[aOffset + i + 2 * lda] * v
        s3 += a[aOffset + i + 3 * lda] * v
        at += xStride
    }
    panelStore(y, yOffset, alpha, s0, beta)
    panelStore(y, yOffset + yStride, alpha, s1, beta)
    panelStore(y, yOffset + 2 * yStride, alpha, s2, beta)
    panelStore(y, yOffset + 3 * yStride, alpha, s3, beta)
}

private inline fun panelDotOne(
    a: DoubleArray,
    aOffset: Int,
    x: DoubleArray,
    xOffset: Int,
    xStride: Int,
    rows: Int,
): Double {
    var sum = 0.0
    var at = xOffset
    for (i in 0 until rows) {
        sum += a[aOffset + i] * x[at]
        at += xStride
    }
    return sum
}

/** `y = alpha · sum + beta · y`, where a zero beta overwrites without reading the destination. */
private inline fun panelStore(y: DoubleArray, at: Int, alpha: Double, sum: Double, beta: Double) {
    y[at] = if (beta == 0.0) alpha * sum else alpha * sum + beta * y[at]
}

internal inline fun scalarColumnUpdate(
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
    var c = 0
    while (c + PANEL_WIDE_GROUP <= columns) {
        val t0 = alpha * x[xOffset + c * xStride]
        val t1 = alpha * x[xOffset + (c + 1) * xStride]
        val t2 = alpha * x[xOffset + (c + 2) * xStride]
        val t3 = alpha * x[xOffset + (c + 3) * xStride]
        val at = aOffset + c * lda
        var to = yOffset
        for (i in 0 until rows) {
            y[to] += t0 * a[at + i] + t1 * a[at + i + lda] + t2 * a[at + i + 2 * lda] + t3 * a[at + i + 3 * lda]
            to += yStride
        }
        c += PANEL_WIDE_GROUP
    }
    while (c < columns) {
        val t = alpha * x[xOffset + c * xStride]
        val at = aOffset + c * lda
        var to = yOffset
        for (i in 0 until rows) {
            y[to] += t * a[at + i]
            to += yStride
        }
        c++
    }
}

internal inline fun scalarCoupledUpdateDot(
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
    var c = 0
    while (c + PANEL_NARROW_GROUP <= columns) {
        val t0 = alpha * coefficients[coefficientOffset + c * coefficientStride]
        val t1 = alpha * coefficients[coefficientOffset + (c + 1) * coefficientStride]
        val at = aOffset + c * lda
        var s0 = 0.0
        var s1 = 0.0
        for (i in 0 until rows) {
            val v0 = a[at + i]
            val v1 = a[at + i + lda]
            y[yOffset + i * yStride] += t0 * v0 + t1 * v1
            val xi = x[xOffset + i * xStride]
            s0 += v0 * xi
            s1 += v1 * xi
        }
        sums[sumOffset + c * sumStride] += alpha * s0
        sums[sumOffset + (c + 1) * sumStride] += alpha * s1
        c += PANEL_NARROW_GROUP
    }
    while (c < columns) {
        val t = alpha * coefficients[coefficientOffset + c * coefficientStride]
        val at = aOffset + c * lda
        var sum = 0.0
        for (i in 0 until rows) {
            val v = a[at + i]
            y[yOffset + i * yStride] += t * v
            sum += v * x[xOffset + i * xStride]
        }
        sums[sumOffset + c * sumStride] += alpha * sum
        c++
    }
}

internal inline fun scalarRankUpdate(
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
    var c = 0
    while (c + PANEL_NARROW_GROUP <= columns) {
        val t0 = alpha * coefficients[coefficientOffset + c * coefficientStride]
        val t1 = alpha * coefficients[coefficientOffset + (c + 1) * coefficientStride]
        val at = aOffset + c * lda
        var from = xOffset
        for (i in 0 until rows) {
            val v = x[from]
            a[at + i] += t0 * v
            a[at + i + lda] += t1 * v
            from += xStride
        }
        c += PANEL_NARROW_GROUP
    }
    while (c < columns) {
        val t = alpha * coefficients[coefficientOffset + c * coefficientStride]
        val at = aOffset + c * lda
        var from = xOffset
        for (i in 0 until rows) {
            a[at + i] += t * x[from]
            from += xStride
        }
        c++
    }
}

internal inline fun scalarRankTwoUpdate(
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
    for (c in 0 until columns) {
        val s = alpha * u[uOffset + c * uStride]
        val t = alpha * v[vOffset + c * vStride]
        val at = aOffset + c * lda
        var fromX = xOffset
        var fromY = yOffset
        for (i in 0 until rows) {
            a[at + i] += s * x[fromX] + t * y[fromY]
            fromX += xStride
            fromY += yStride
        }
    }
}

/** Columns sharing each loaded vector entry. */
internal const val PANEL_WIDE_GROUP = 4

/** Columns held by each coupled or rank-update body. */
internal const val PANEL_NARROW_GROUP = 2
