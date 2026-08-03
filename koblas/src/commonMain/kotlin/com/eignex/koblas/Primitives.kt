package com.eignex.koblas

// Platform-dispatched dense-vector primitives - `internal` building blocks that
// higher-level ops (`dot`, `axpy`, `gemv`, `cholesky`) call on contiguous
// `DoubleArray` runs. JVM provides a SIMD implementation via the incubator
// `jdk.incubator.vector` API; every other target uses a scalar fallback in
// `nonJvmMain`.
//
// All primitives take `DoubleArray` plus an `offset` and `length` so the same
// call site works for whole vectors, matrix rows, or sub-slices.

/** `Sum a[aOff..aOff+len-1] * b[bOff..bOff+len-1]`. */
internal expect fun denseDot(a: DoubleArray, aOff: Int, b: DoubleArray, bOff: Int, len: Int): Double

/** `y[yOff..] = y[yOff..] + alpha * x[xOff..]`. */
internal expect fun denseAxpy(y: DoubleArray, yOff: Int, alpha: Double, x: DoubleArray, xOff: Int, len: Int)

/** `v[vOff..vOff+len-1] = alpha * v[..]`. */
internal expect fun denseScale(v: DoubleArray, vOff: Int, alpha: Double, len: Int)

/**
 * Four dots against a shared right operand: `out[outOff + r] = Sum a[aOff + r*stride + i] * b[bOff + i]`
 * for `r` in `0..3`. Four columns of a column-major matrix against one vector, which is the shape
 * [LinearAlgebra.gemv] and the `A·Bᵀ` [LinearAlgebra.gemm] branch need. A vectorized implementation
 * loads each `b` segment once for all four rows and keeps four independent accumulators, so it cuts
 * load traffic and breaks the accumulator dependency chain that limits a single [denseDot]; the
 * scalar fallback just runs four dots.
 */
@Suppress("LongParameterList") // four row offsets plus the shared operand
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
