package com.eignex.koblas

/**
 * Scalar implementations of the dense primitives, used directly by the JS, Wasm, iOS and Windows targets
 * and as the short-run path on Linux and macOS. The JVM target
 * overrides them with `jdk.incubator.vector` kernels.
 *
 * These loops are written to be vectorizable, but do not assume they are vectorized. Measured on
 * linuxX64 in a release binary against the JVM's SIMD kernels on the same machine, they run 4x to 7x
 * slower: `dot` at length 256 takes 165ns versus 34ns, `axpy` 178ns versus 31ns, and the gap holds
 * down to length 16. So on Kotlin/Native the host BLAS wins level 2 outright (2x to 15x, see
 * `CblasLinearAlgebra`) — the opposite of the JVM, where these operations stay on the SIMD kernels.
 * Narrowing that gap needs real vector kernels for native, not a better BLAS: the sparse traversals
 * these primitives do not cover are unaffected by either.
 */

internal fun scalarDot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
    var s = 0.0
    for (i in 0 until len) s += a[aOff + i] * b[bOff + i]
    return s
}

internal fun scalarAxpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
    if (alpha == 0.0) return
    for (i in 0 until len) y[yOff + i] += alpha * x[xOff + i]
}

internal fun scalarScale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
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
