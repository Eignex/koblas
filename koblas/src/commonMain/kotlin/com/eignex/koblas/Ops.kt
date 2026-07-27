@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter matrices L, M, etc.

package com.eignex.koblas

import kotlin.math.abs
import kotlin.math.sqrt

// Arithmetic over [VectorView] / [MatrixView] as free functions; the view types stay read-only.
//
// Iteration goes through [forEachStored], which dispatches dense to all-indices
// and sparse to stored entries. Densexdense paths delegate to [denseDot] /
// [denseAxpy] / [denseScale] in `Primitives.kt` (SIMD on JVM, scalar elsewhere).
//
// Naming: mutating functions take the destination first and return [Unit] (`scale`,
// `axpy`, `addOuter`); allocating functions return a fresh result (`matVec`,
// infix `dot`).

/**
 * Visit each stored entry of [this] as `(index, value)`. For [DenseVector] that's
 * every index in `0 until size`; for [SparseVector] that's the entries present in
 * the parallel index/value arrays (which may include numerical zeros).
 */
inline fun VectorView.forEachStored(block: (i: Int, v: Double) -> Unit) {
    when (this) {
        is DenseVector -> {
            val d = data
            for (i in 0 until d.size) block(i, d[i])
        }

        is SparseVector -> {
            val idx = indices
            val vals = values
            for (k in idx.indices) block(idx[k], vals[k])
        }
    }
}

/** `aT * b`. Densexdense routes through [denseDot] (SIMD on JVM); sparse paths
 *  iterate the cheaper operand's stored entries. */
infix fun VectorView.dot(other: VectorView): Double {
    require(size == other.size) { "size mismatch: $size vs ${other.size}" }
    if (this is DenseVector && other is DenseVector) {
        return denseDot(data, 0, other.data, 0, size)
    }
    // At least one sparse - iterate that side, gather from the other.
    return if (this is SparseVector || other !is SparseVector) {
        var s = 0.0
        this.forEachStored { i, v -> s += v * other[i] }
        s
    } else {
        var s = 0.0
        other.forEachStored { i, v -> s += v * this[i] }
        s
    }
}

/**
 * Euclidean norm `||v||₂` (BLAS `dnrm2`). Sparse vectors sum over stored entries only.
 *
 * Computed as `sqrt(sum of squares)` without netlib's overflow-guarding rescale: components must stay
 * within roughly `1e±150` for the squares not to overflow/underflow — ample for the intended workloads.
 */
fun norm2(v: VectorView): Double {
    var s = 0.0
    v.forEachStored { _, x -> s += x * x }
    return sqrt(s)
}

/** Sum of absolute values `Sum |v_i|` (BLAS `dasum`). Sparse vectors sum over stored entries only. */
fun asum(v: VectorView): Double {
    var s = 0.0
    v.forEachStored { _, x -> s += abs(x) }
    return s
}

/**
 * Index of the first entry with maximal `|v_i|` (BLAS `idamax`), or `-1` for a zero-length vector.
 * For [DenseVector] "first" is by index; for [SparseVector] ties resolve in storage order, and an
 * all-unstored (zero) vector returns index 0, matching the dense zero vector.
 */
fun iamax(v: VectorView): Int {
    if (v.size == 0) return -1
    var best = -1
    var bestAbs = -1.0
    v.forEachStored { i, x ->
        val a = abs(x)
        if (a > bestAbs) {
            bestAbs = a
            best = i
        }
    }
    return if (best == -1) 0 else best // no stored entries: the zero vector's max is its first element
}

/** `dst = src` (BLAS `dcopy`). Dense sources bulk-copy; sparse sources zero-fill then scatter. */
fun copy(src: VectorView, dst: DenseVector) {
    require(src.size == dst.size) { "size mismatch: ${src.size} vs ${dst.size}" }
    if (src is DenseVector) {
        src.data.copyInto(dst.data)
    } else {
        dst.data.fill(0.0)
        src.forEachStored { i, x -> dst.data[i] = x }
    }
}

/** Exchange the contents of [a] and [b] (BLAS `dswap`). */
fun swap(a: DenseVector, b: DenseVector) {
    require(a.size == b.size) { "size mismatch: ${a.size} vs ${b.size}" }
    val ad = a.data
    val bd = b.data
    for (i in ad.indices) {
        val t = ad[i]
        ad[i] = bd[i]
        bd[i] = t
    }
}

/** `y = y + alpha * x`. Dense `x` uses SIMD; sparse `x` walks stored entries. */
fun axpy(y: DenseVector, alpha: Double, x: VectorView) {
    require(y.size == x.size) { "size mismatch: ${y.size} vs ${x.size}" }
    if (alpha == 0.0) return
    if (x is DenseVector) {
        denseAxpy(y.data, 0, alpha, x.data, 0, y.size)
    } else {
        val yd = y.data
        x.forEachStored { i, v -> yd[i] += alpha * v }
    }
}

/** `v = alpha * v`. */
fun scale(v: DenseVector, alpha: Double) {
    if (alpha == 1.0) return
    denseScale(v.data, 0, alpha, v.size)
}

/**
 * `M = M + alpha * x * yT` (rank-1 update). Subtract by passing `alpha = -1.0`.
 *
 * Densexdense routes each row's update through [denseAxpy] (SIMD). Sparse paths
 * only visit the rows/cols where `x_i * y_j` could be non-zero.
 */
fun addOuter(M: DenseMatrix, alpha: Double, x: VectorView, y: VectorView) {
    require(M.rows == x.size && M.cols == y.size) {
        "addOuter shape mismatch: M is ${M.rows}x${M.cols}, x ${x.size}, y ${y.size}"
    }
    if (alpha == 0.0) return
    val md = M.data
    val cols = M.cols
    if (x is DenseVector && y is DenseVector) {
        val xd = x.data
        for (i in 0 until M.rows) {
            val xi = xd[i]
            if (xi != 0.0) denseAxpy(md, i * cols, alpha * xi, y.data, 0, cols)
        }
        return
    }
    // Mixed or sparse - fall back to per-stored-entry updates.
    x.forEachStored { i, xi ->
        if (xi != 0.0) {
            val row = i * cols
            val scaled = alpha * xi
            y.forEachStored { j, yj -> md[row + j] += scaled * yj }
        }
    }
}

/** Matrix-vector product `A * x` into a fresh dense result. */
fun matVec(A: MatrixView, x: VectorView): DenseVector {
    require(A.cols == x.size) { "matVec shape mismatch: A is ${A.rows}x${A.cols}, x size ${x.size}" }
    // Dense·dense routes through the backend's gemv (the single dense matrix-vector implementation).
    if (A is DenseMatrix && x is DenseVector) return DenseVector.wrap(koblas.gemv(A, x.data))
    val out = DenseVector(A.rows)
    val od = out.data
    if (A is DenseMatrix) {
        val ad = A.data
        val cols = A.cols
        for (i in 0 until A.rows) {
            val row = i * cols
            var s = 0.0
            x.forEachStored { j, v -> s += ad[row + j] * v }
            od[i] = s
        }
    } else {
        for (i in 0 until A.rows) {
            var s = 0.0
            x.forEachStored { j, v -> s += A[i, j] * v }
            od[i] = s
        }
    }
    return out
}
