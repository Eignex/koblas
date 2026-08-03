package com.eignex.koblas

// Dense-vector primitives - `internal` building blocks that higher-level ops (`dot`, `axpy`, `gemv`,
// `cholesky`, the eta updates) call on contiguous `DoubleArray` runs.
//
// Each has two layers, and they are separate on purpose:
//
//   - a compile-time leaf (`platformDot` and friends), specialized per target because whether a vector
//     unit is reachable is a compile-time fact. The JVM's is `jdk.incubator.vector` SIMD; every other
//     target gets a scalar loop from `nonJvmMain`.
//   - a runtime route to a registered [Level1] backend, for runs long enough that a foreign call pays
//     for itself. That threshold is [DispatchThresholds.level1].
//
// The routing lives here, once, rather than being duplicated into the source sets that can reach a host
// BLAS - which is what it used to be, at the cost of two extra source sets and a `denseDot` that meant
// something structurally different per target.
//
// Ordering matters. The [activeLevel1] field is tested first and the length second, so a target where
// nothing is ever registered pays one always-null field read and no threshold resolution at all. Reading
// the threshold first would be worse: it lives behind a `lazy`, so it would cost an initialization check
// on a path where the whole kernel is a few nanoseconds.
//
// All primitives take `DoubleArray` plus an `offset` and `length`, so the same call site works for whole
// vectors, matrix columns, or sub-slices.

/** `Sum a[aOff..aOff+len-1] * b[bOff..bOff+len-1]`. */
internal fun denseDot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double {
    val l1 = activeLevel1
    if (l1 != null && len >= dispatchThresholds.level1) return l1.dot(a, aOff, b, bOff, len)
    return platformDot(a, aOff, b, bOff, len)
}

/** `y[yOff..] = y[yOff..] + alpha * x[xOff..]`. */
internal fun denseAxpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int) {
    if (alpha == 0.0) return
    val l1 = activeLevel1
    if (l1 != null && len >= dispatchThresholds.level1) {
        l1.axpy(y, yOff, alpha, x, xOff, len)
        return
    }
    platformAxpy(y, yOff, alpha, x, xOff, len)
}

/** `v[vOff..vOff+len-1] = alpha * v[..]`. */
internal fun denseScale(v: DoubleArray, vOff: Int, alpha: Double, len: Int) {
    if (alpha == 1.0) return
    val l1 = activeLevel1
    if (l1 != null && len >= dispatchThresholds.level1) {
        l1.scale(v, vOff, alpha, len)
        return
    }
    platformScale(v, vOff, alpha, len)
}

/** The compile-time [denseDot] kernel for this target, with no routing. */
internal expect fun platformDot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double

/** The compile-time [denseAxpy] kernel for this target, with no routing. */
internal expect fun platformAxpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)

/** The compile-time [denseScale] kernel for this target, with no routing. */
internal expect fun platformScale(v: DoubleArray, vOff: Int, alpha: Double, len: Int)

/**
 * Four dots against a shared right operand: `out[outOff + r] = Sum a[aOff + r*stride + i] * b[bOff + i]`
 * for `r` in `0..3`. Four columns of a column-major matrix against one vector, which is the shape
 * [LinearAlgebra.gemv] and the `Aᵀ·B` [LinearAlgebra.gemm] branch need. A vectorized implementation
 * loads each `b` segment once for all four columns and keeps four independent accumulators, so it cuts
 * load traffic and breaks the accumulator dependency chain that limits a single [denseDot]; the
 * scalar fallback just runs four dots.
 *
 * No [Level1] counterpart, and no routing: BLAS has no batched-dot routine, so there is nothing to route
 * to. Callers wanting a host BLAS for this shape reach it through `gemv` or `gemm` on the [Blas] seam.
 */
@Suppress("LongParameterList") // four column offsets plus the shared operand
internal expect fun denseDot4(
    a: DoubleArray,
    aOff: Int,
    stride: Int,
    b: DoubleArray,
    bOff: Int,
    len: Int,
    out: DoubleArray,
    outOff: Int,
)
