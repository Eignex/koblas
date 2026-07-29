package com.eignex.koblas

/**
 * Scalar fallback for the dense primitives. The JVM target overrides these with a
 * SIMD-backed implementation using `jdk.incubator.vector`; non-JVM targets
 * (native, JS, Wasm) use the loops below - competitive with hand-written code
 * once the platform's compiler vectorises the inner loop, but without the
 * guaranteed lane-width win that the Vector API delivers on JVM.
 */

internal actual fun denseDot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
    var s = 0.0
    for (i in 0 until len) s += a[aOff + i] * b[bOff + i]
    return s
}

internal actual fun denseAxpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
    if (alpha == 0.0) return
    for (i in 0 until len) y[yOff + i] += alpha * x[xOff + i]
}

internal actual fun denseScale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
    if (alpha == 1.0) return
    for (i in 0 until len) v[vOff + i] *= alpha
}

// Four accumulators over one pass: the shared `b` element is read once per column and the four chains
// are independent, which is the same structural win the JVM kernel gets explicitly and which the
// platform compilers can vectorize here.
@Suppress("LongParameterList") // four row offsets plus the shared operand
internal actual fun denseDot4(
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
