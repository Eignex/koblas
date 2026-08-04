package com.eignex.koblas.dense

import com.eignex.koblas.dot
import jdk.incubator.vector.DoubleVector
import jdk.incubator.vector.VectorOperators

/**
 * JVM dense primitives. Routes to a SIMD path (`jdk.incubator.vector`) when the
 * incubator module is available at runtime, scalar fallback otherwise.
 *
 * Detection runs once at class load via `Class.forName`. The SIMD code lives in a
 * separate private object ([Simd]) so its static initializer (which references
 * `DoubleVector`) only runs when the probe succeeds. A JVM started without
 * `--add-modules=jdk.incubator.vector` loads this file cleanly and takes the
 * scalar path.
 *
 * Build wiring: koblas compiles with `-Xadd-modules=jdk.incubator.vector` to
 * make the SIMD code resolve at compile time. Tests pass the same flag at runtime
 * to exercise the SIMD path; consumers who pass it get SIMD, consumers who don't
 * get scalar. No extra config required for correctness.
 */

internal val simdAvailable: Boolean = try {
    Class.forName("jdk.incubator.vector.DoubleVector")
    true
} catch (_: Throwable) {
    false
}

/**
 * Lane width of the vector path, or 0 when it is unavailable. Runs below one lane execute no vector
 * body at all — `loopBound` is zero, so the work falls to the scalar tail after paying the vector
 * prologue (a horizontal reduce for [denseDot], a broadcast for the others). Short runs therefore
 * route straight to the scalar kernels; triangular solves and small factorizations issue many of
 * them.
 */
private val simdLanes: Int = if (simdAvailable) Simd.lanes() else 0

internal actual fun platformDot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double =
    if (simdAvailable && len >= simdLanes) Simd.dot(a, aOff, b, bOff, len) else scalarDot(a, aOff, b, bOff, len)

internal actual fun platformAxpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
    if (simdAvailable && len >= simdLanes) {
        Simd.axpy(y, yOff, alpha, x, xOff, len)
    } else {
        scalarAxpy(y, yOff, alpha, x, xOff, len)
    }
}

internal actual fun platformScale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
    if (simdAvailable && len >= simdLanes) {
        Simd.scale(v, vOff, alpha, len)
    } else {
        scalarScale(v, vOff, alpha, len)
    }
}

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
    if (simdAvailable && len >= simdLanes) {
        Simd.dot4(a, aOff, stride, b, bOff, len, out, outOff)
    } else {
        for (r in 0 until 4) out[outOff + r] = scalarDot(a, aOff + r * stride, b, bOff, len)
    }
}

private fun scalarDot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
    var s = 0.0
    for (i in 0 until len) s += a[aOff + i] * b[bOff + i]
    return s
}

private fun scalarAxpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
    if (alpha == 0.0) return
    for (i in 0 until len) y[yOff + i] += alpha * x[xOff + i]
}

private fun scalarScale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
    if (alpha == 1.0) return
    for (i in 0 until len) v[vOff + i] *= alpha
}

internal object Simd {
    private val SPECIES = DoubleVector.SPECIES_PREFERRED
    private val LANE = SPECIES.length()

    fun lanes(): Int = LANE

    /**
     * One accumulator, deliberately: multiple independent accumulators would break the FMA latency
     * chain, but measured against this loop they cost 20% at `len` 4-8 (the method grows past what the
     * JIT will inline, and triangular solves issue mostly short runs) and showed no gain at 256 beyond
     * run-to-run drift. [dot4] is where the chain gets broken, on the shapes that can amortize it.
     */
    fun dot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
        var i = 0
        val bound = SPECIES.loopBound(len)
        var sum = DoubleVector.zero(SPECIES)
        while (i < bound) {
            val va = DoubleVector.fromArray(SPECIES, a, aOff + i)
            val vb = DoubleVector.fromArray(SPECIES, b, bOff + i)
            sum = va.fma(vb, sum)
            i += LANE
        }
        var s = sum.reduceLanes(VectorOperators.ADD)
        while (i < len) {
            s += a[aOff + i] * b[bOff + i]
            i++
        }
        return s
    }

    /**
     * Four rows against one shared vector. Each `b` segment is loaded once and fused into four
     * independent accumulators, so load traffic drops from two vectors per FMA to five per four and
     * the accumulator chains are independent by construction — the reason [LinearAlgebra.gemv] wants
     * this over four separate [dot] calls, which would also pay four horizontal reductions.
     */
    @Suppress("LongParameterList") // four row offsets plus the shared operand
    fun dot4(
        a: DoubleArray,
        aOff: Int,
        stride: Int,
        b: DoubleArray,
        bOff: Int,
        len: Int,
        out: DoubleArray,
        outOff: Int,
    ) {
        var s0 = DoubleVector.zero(SPECIES)
        var s1 = DoubleVector.zero(SPECIES)
        var s2 = DoubleVector.zero(SPECIES)
        var s3 = DoubleVector.zero(SPECIES)
        val o1 = aOff + stride
        val o2 = aOff + 2 * stride
        val o3 = aOff + 3 * stride
        var i = 0
        val bound = SPECIES.loopBound(len)
        while (i < bound) {
            val vb = DoubleVector.fromArray(SPECIES, b, bOff + i)
            s0 = DoubleVector.fromArray(SPECIES, a, aOff + i).fma(vb, s0)
            s1 = DoubleVector.fromArray(SPECIES, a, o1 + i).fma(vb, s1)
            s2 = DoubleVector.fromArray(SPECIES, a, o2 + i).fma(vb, s2)
            s3 = DoubleVector.fromArray(SPECIES, a, o3 + i).fma(vb, s3)
            i += LANE
        }
        var r0 = s0.reduceLanes(VectorOperators.ADD)
        var r1 = s1.reduceLanes(VectorOperators.ADD)
        var r2 = s2.reduceLanes(VectorOperators.ADD)
        var r3 = s3.reduceLanes(VectorOperators.ADD)
        while (i < len) {
            val bi = b[bOff + i]
            r0 += a[aOff + i] * bi
            r1 += a[o1 + i] * bi
            r2 += a[o2 + i] * bi
            r3 += a[o3 + i] * bi
            i++
        }
        out[outOff] = r0
        out[outOff + 1] = r1
        out[outOff + 2] = r2
        out[outOff + 3] = r3
    }

    fun axpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
        if (alpha == 0.0) return
        val alphaVec = DoubleVector.broadcast(SPECIES, alpha)
        var i = 0
        val bound = SPECIES.loopBound(len)
        while (i < bound) {
            val vx = DoubleVector.fromArray(SPECIES, x, xOff + i)
            val vy = DoubleVector.fromArray(SPECIES, y, yOff + i)
            // y_new = alpha * x + y  ->  vx.fma(alphaVec, vy) computes vx * alphaVec + vy.
            vx.fma(alphaVec, vy).intoArray(y, yOff + i)
            i += LANE
        }
        while (i < len) {
            y[yOff + i] += alpha * x[xOff + i]
            i++
        }
    }

    fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
        if (alpha == 1.0) return
        val alphaVec = DoubleVector.broadcast(SPECIES, alpha)
        var i = 0
        val bound = SPECIES.loopBound(len)
        while (i < bound) {
            val vv = DoubleVector.fromArray(SPECIES, v, vOff + i)
            vv.mul(alphaVec).intoArray(v, vOff + i)
            i += LANE
        }
        while (i < len) {
            v[vOff + i] *= alpha
            i++
        }
    }
}
