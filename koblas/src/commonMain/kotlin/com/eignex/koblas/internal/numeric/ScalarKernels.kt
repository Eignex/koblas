package com.eignex.koblas.internal.numeric

/** `a · b` over [len] elements from each offset. */
// These reductions keep one accumulator each, deliberately. Four independent chains, which is what the
// bundled C kernels use above KOBLAS_UNROLL_MIN and what makes them 2 to 3 times faster there, were measured
// here both ways: Kotlin/Native runs them about 1.22 times faster at 32 and 128 elements, and HotSpot runs
// them 1.4 to 1.55 times slower, because it optimises the simple counted loop better than a hand-unrolled
// one. This file is compiled by both, so taking the native win would mean regressing the JVM or splitting
// these helpers per platform, and the only paths that would gain are an explicitly installed scalar provider
// and the crossScalar targets. Measure both backends again before revisiting.

internal fun scalarDot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
    var s = 0.0
    for (i in 0 until len) s += a[aOff + i] * b[bOff + i]
    return s
}

/** Exchange the two runs over [len] elements. */
internal fun scalarSwap(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int) {
    for (i in 0 until len) {
        val t = a[aOff + i]
        a[aOff + i] = b[bOff + i]
        b[bOff + i] = t
    }
}

/** `sum (a - b)^2` over [len] elements, without materialising the difference. */
internal fun scalarSsqd(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
    var s = 0.0
    for (i in 0 until len) {
        val d = a[aOff + i] - b[bOff + i]
        s += d * d
    }
    return s
}

/** `y += alpha * x` over [len] elements. A zero [alpha] leaves y alone rather than adding zeroes. */
internal fun scalarAxpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
    if (alpha == 0.0) return
    scalarAxpyArithmetic(y, yOff, alpha, x, xOff, len)
}

/** AXPY arithmetic without the standalone BLAS routine's zero-scalar quick return. */
internal fun scalarAxpyArithmetic(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
    for (i in 0 until len) y[yOff + i] += alpha * x[xOff + i]
}

/** `v *= alpha` over [len] elements. A unit [alpha] leaves v alone. */
internal fun scalarScale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
    if (alpha == 1.0) return
    for (i in 0 until len) v[vOff + i] *= alpha
}

/**
 * Four dots against one shared operand, as four accumulators over a single pass, so `b` is read once for
 * all four columns rather than once per column.
 */
@Suppress("LongParameterList") // four column offsets plus the shared operand
internal fun scalarDot4(
    a: DoubleArray,
    aOff: Int,
    stride: Int,
    b: DoubleArray,
    bOff: Int,
    len: Int,
    out: DoubleArray,
    outOff: Int,
) {
    val o1 = aOff + stride
    val o2 = aOff + 2 * stride
    val o3 = aOff + 3 * stride
    var r0 = 0.0
    var r1 = 0.0
    var r2 = 0.0
    var r3 = 0.0
    for (i in 0 until len) {
        val bi = b[bOff + i]
        r0 += a[aOff + i] * bi
        r1 += a[o1 + i] * bi
        r2 += a[o2 + i] * bi
        r3 += a[o3 + i] * bi
    }
    out[outOff] = r0
    out[outOff + 1] = r1
    out[outOff + 2] = r2
    out[outOff + 3] = r3
}
