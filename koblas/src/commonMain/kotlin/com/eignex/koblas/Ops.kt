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
// `axpy`, `ger`); allocating functions return a fresh result (`gemv`,
// infix `dot`).
//
// Naming follows the library-wide rule: BLAS routines keep their standard mnemonics, LAPACK routines get
// English names. Two deliberate exceptions live here. `norm2` and `asum` spell out what BLAS calls
// `nrm2` and `asum`, because `norm2` pairs with `norm1` (which is LAPACK `dlange`, not BLAS) and reading
// them side by side matters more than matching four characters exactly. And `iamax` keeps its mnemonic
// rather than becoming `indexOfMaxAbs`, since it is unambiguous to anyone who has met BLAS.

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
 * The fast path is a plain `sqrt(sum of squares)`; when that sum overflows or drowns in underflow
 * (components beyond roughly `1e±150`), a rescaled two-pass recovers the netlib-accurate result, so
 * any finite input yields the correct norm.
 */
fun norm2(v: VectorView): Double {
    // Dense operands route through the level-1 seam; a sparse one sums its stored entries, which no BLAS
    // routine can do, so it keeps the same rescaling logic inline.
    if (v is DenseVector) return denseNrm2(v.data, 0, v.size)
    var s = 0.0
    v.forEachStored { _, x -> s += x * x }
    if (s.isFinite() && s >= MIN_NORMAL) return sqrt(s)
    var amax = 0.0
    v.forEachStored { _, x ->
        val a = abs(x)
        if (a > amax) amax = a
    }
    // All-zero (0.0), NaN anywhere (NaN), or an infinite component (Inf) resolve through the raw sum.
    if (amax == 0.0 || amax.isInfinite()) return sqrt(s)
    var t = 0.0
    v.forEachStored { _, x ->
        val r = x / amax
        t += r * r
    }
    return amax * sqrt(t)
}

/** Smallest normal double; a squares-sum below this has lost precision to underflow. */
private const val MIN_NORMAL = 2.2250738585072014e-308

/** Sum of absolute values `Sum |v_i|` (BLAS `dasum`). Sparse vectors sum over stored entries only. */
fun asum(v: VectorView): Double {
    if (v is DenseVector) return denseAsum(v.data, 0, v.size)
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
 * Rank-one update `A = A + alpha · x · yᵀ` (BLAS `dger`). Subtract by passing `alpha = -1.0`.
 *
 * Two dense operands dispatch to the installed backend through [LinearAlgebra.ger]. A sparse or mixed
 * pair has no BLAS counterpart and stays here, visiting only the rows and columns where `x_i · y_j`
 * can be non-zero.
 */
fun ger(alpha: Double, x: VectorView, y: VectorView, a: DenseMatrix) {
    require(a.rows == x.size && a.cols == y.size) {
        "ger shape mismatch: A is ${a.rows}x${a.cols}, x ${x.size}, y ${y.size}"
    }
    if (alpha == 0.0) return
    if (x is DenseVector && y is DenseVector) {
        koblas.ger(alpha, x.data, y.data, a)
        return
    }
    // Mixed or sparse: no BLAS routine takes these, so walk the stored entries. Skipping a zero entry of
    // y skips a whole column of updates, which is the point of accepting a sparse operand at all. The
    // column is the outer loop because columns are the contiguous axis.
    val md = a.data
    val rows = a.rows
    y.forEachStored { j, yj ->
        if (yj != 0.0) {
            val col = j * rows
            val scaled = alpha * yj
            x.forEachStored { i, xi -> md[col + i] += scaled * xi }
        }
    }
}

/**
 * Matrix 1-norm: the maximum absolute column sum (LAPACK `dlange` with norm `1`). This is the `anorm`
 * input [LinearAlgebra.rcond] expects, computed on the matrix before factoring, so a solver that
 * estimates conditioning each refactorization calls both.
 *
 * Takes no workspace, unlike [LinearAlgebra.rcond], because it needs no scratch: a column is contiguous,
 * so each column sum completes before the next begins and one accumulator suffices. Row-major storage
 * forced a running total per column and therefore an `n`-wide array; that is what the workspace argument
 * used to be for.
 */
fun norm1(a: DenseMatrix): Double {
    val ad = a.data
    val rows = a.rows
    var m = 0.0
    for (j in 0 until a.cols) {
        val base = j * rows
        var s = 0.0
        for (i in 0 until rows) s += abs(ad[base + i])
        if (s > m) m = s
    }
    return m
}

/**
 * Fresh transposed matrix `Aᵀ`. Always materializes; for products against a transposed operand,
 * prefer the `transpose` flags on [LinearAlgebra.gemv] / [LinearAlgebra.gemm], which read the
 * original storage without copying.
 */
fun DenseMatrix.transpose(): DenseMatrix {
    val t = DenseMatrix(cols, rows)
    val td = t.data
    // Reads down a contiguous source column and writes across a stride in the destination.
    for (j in 0 until cols) {
        val base = j * rows
        for (i in 0 until rows) td[j + i * cols] = data[base + i]
    }
    return t
}

/**
 * Matrix-vector product `A · x` into a fresh dense result (BLAS `dgemv` with `alpha = 1`, `beta = 0`).
 *
 * The view-taking overload of [Blas.gemv], for the same reason [ger] has one: a [SparseVector] operand
 * has no BLAS counterpart, and walking only its stored entries is the point of passing one. Two dense
 * operands dispatch to the backend, so this is a shape adapter rather than a second implementation.
 *
 * No `transpose` flag: for a sparse matrix use [SparseMatrix.gemv], which takes one.
 */
fun gemv(A: MatrixView, x: VectorView): DenseVector {
    require(A.cols == x.size) { "gemv shape mismatch: A is ${A.rows}x${A.cols}, x size ${x.size}" }
    if (A is DenseMatrix && x is DenseVector) return DenseVector.wrap(koblas.gemv(A, x.data))
    val out = DenseVector(A.rows)
    val od = out.data
    if (A is DenseMatrix) {
        // One axpy per stored entry of x, down a contiguous column — so a sparse x touches only the
        // columns it has entries in, which is the reason this overload exists.
        val ad = A.data
        val rows = A.rows
        x.forEachStored { j, v ->
            if (v != 0.0) denseAxpy(od, 0, v, ad, j * rows, rows)
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
