@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.sqrt

/*
 * The portable Householder QR, its column-pivoted form and the solves over their factors, netlib dgeqrf,
 * dgeqp3, dormqr and dtrtrs. This is the semantic definition a native QR is validated against;
 * [F64ReferenceDecompositions] is the F64Decompositions surface over it.
 */

internal fun referenceQr(kernels: F64Kernels, a: F64DenseMatrix): F64QrDecomposition = referenceQrInto(
    kernels,
    a,
    F64QrDecomposition(a.rows, a.cols, DoubleArray(a.data.size), DoubleArray(minOf(a.rows, a.cols))),
)

/**
 * [referenceQr] into [out]'s existing buffers instead of fresh ones, discarding what [out] held. [out] may be
 * backed by [a]'s own buffer, which makes the copy below a no-op.
 */
internal fun referenceQrInto(kernels: F64Kernels, a: F64DenseMatrix, out: F64QrDecomposition): F64QrDecomposition {
    requireShape(out.m == a.rows && out.n == a.cols) {
        "qrInto: out is ${out.m}x${out.n}, expected ${a.rows}x${a.cols}"
    }
    val m = a.rows
    val n = a.cols
    val k = minOf(m, n)
    val buf = out.qr
    val tau = out.tau
    a.data.copyInto(buf)
    // Every tau entry below is written, so none of [out]'s previous reflectors survive.
    for (col in 0 until k) {
        tau[col] = householderColumn(kernels, buf, m, col)
        applyReflectorToTrailing(kernels, buf, m, n, col, tau[col])
    }
    return out
}

internal fun referenceQrPivoted(kernels: F64Kernels, a: F64DenseMatrix, tolerance: Double): F64PivotedQrDecomposition {
    requireRankTolerance(tolerance)
    val m = a.rows
    val n = a.cols
    val k = minOf(m, n)
    val buf = a.data.copyOf()
    val tau = DoubleArray(k)
    val pivots = IntArray(n) { it }
    val current = DoubleArray(n) { c -> kernels.nrm2(buf, c * m, m) }
    val computed = current.copyOf()
    for (col in 0 until k) {
        var best = col
        for (c in col + 1 until n) if (current[c] > current[best]) best = c
        if (best != col) swapColumns(buf, m, pivots, current, computed, col, best)
        tau[col] = householderColumn(kernels, buf, m, col)
        applyReflectorToTrailing(kernels, buf, m, n, col, tau[col])
        downdateNorms(kernels, buf, m, n, col, current, computed)
    }
    return F64PivotedQrDecomposition(
        F64QrDecomposition(m, n, buf, tau),
        pivots,
        rankOfPivotedR(buf, m, n, k, tolerance),
    )
}

/** Exchange columns [i] and [j] of [buf] and every per-column quantity that tracks them. */
@Suppress("LongParameterList") // one array per tracked quantity; bundling them would allocate per step
private fun swapColumns(
    buf: DoubleArray,
    m: Int,
    pivots: IntArray,
    current: DoubleArray,
    computed: DoubleArray,
    i: Int,
    j: Int,
) {
    for (r in 0 until m) {
        val t = buf[r + i * m]
        buf[r + i * m] = buf[r + j * m]
        buf[r + j * m] = t
    }
    pivots[i] = pivots[j].also { pivots[j] = pivots[i] }
    current[i] = current[j].also { current[j] = current[i] }
    computed[i] = computed[j].also { computed[j] = computed[i] }
}

/** Apply `H = I − tau·v·vᵀ` from column [col] to every column after it. */
@Suppress("LongParameterList") // the kernels on top of the reflector's own arguments
private fun applyReflectorToTrailing(
    kernels: F64Kernels,
    buf: DoubleArray,
    m: Int,
    n: Int,
    col: Int,
    tau: Double,
) {
    if (tau == 0.0) return
    val len = m - col - 1
    val vBase = col + 1 + col * m
    for (c in col + 1 until n) {
        val head = col + c * m
        val s = buf[head] + kernels.dot(buf, vBase, buf, head + 1, len)
        val f = -tau * s
        if (f != 0.0) {
            buf[head] += f
            kernels.axpy(buf, head + 1, f, buf, vBase, len)
        }
    }
}

/** Shrink each trailing column's norm by the component the reflector just removed, recomputing from
 *  the column when the downdate has drifted too far to trust. */
@Suppress("LongParameterList") // the kernels on top of one array per tracked quantity
private fun downdateNorms(
    kernels: F64Kernels,
    buf: DoubleArray,
    m: Int,
    n: Int,
    col: Int,
    current: DoubleArray,
    computed: DoubleArray,
) {
    for (c in col + 1 until n) {
        if (current[c] == 0.0) continue
        val ratio = abs(buf[col + c * m]) / current[c]
        val remaining = maxOf(0.0, (1.0 - ratio) * (1.0 + ratio))
        val drift = remaining * (current[c] / computed[c]).let { it * it }
        if (drift <= F64_NORM_RECOMPUTE_THRESHOLD) {
            val len = m - col - 1
            val base = col + 1 + c * m
            current[c] = if (len > 0) kernels.nrm2(buf, base, len) else 0.0
            computed[c] = current[c]
        } else {
            current[c] *= sqrt(remaining)
        }
    }
}

/** Build the Householder reflector for [col] in place (LAPACK `dlarfg`), returning `tau` and storing
 *  the scaled vector below the diagonal with the new `R` diagonal entry on it. */
private fun householderColumn(kernels: F64Kernels, buf: DoubleArray, m: Int, col: Int): Double {
    val base = col + col * m
    val len = m - col
    // dlarfg measures the tail alone and returns a zero reflector when it vanishes, leaving the
    // diagonal as it found it. Measuring the whole column instead would reflect a column already in
    // triangular form, which is valid but puts a sign in R that LAPACK does not, and the last column of
    // every square matrix takes that path.
    val alpha = buf[base]
    val tailNorm = kernels.nrm2(buf, base + 1, len - 1)
    if (tailNorm == 0.0) return 0.0
    // hypot rather than a second nrm2 over the whole column: the tail is already measured, and nrm2 is a
    // single-accumulator reduction running at add latency, so the second pass costs what the first did.
    val norm = hypot(alpha, tailNorm)
    val beta = if (alpha >= 0.0) -norm else norm
    // Division keeps a subnormal v0 from becoming an infinity.
    val v0 = alpha - beta
    for (i in base + 1 until base + len) buf[i] /= v0
    buf[base] = beta
    return (beta - alpha) / beta
}

internal fun referenceApplyQInto(
    kernels: F64Kernels,
    qr: F64QrDecomposition,
    y: DoubleArray,
    out: DoubleArray,
    transpose: Boolean,
): DoubleArray {
    val m = qr.m
    requireShape(y.size == m) { "applyQ: y length ${y.size} != $m" }
    requireShape(out.size == m) { "applyQ: out length ${out.size} != $m" }
    val x = out
    if (out !== y) y.copyInto(out)
    val k = qr.tau.size
    val buf = qr.qr
    val order = if (transpose) 0 until k else k - 1 downTo 0
    for (col in order) {
        val t = qr.tau[col]
        if (t == 0.0) continue
        val len = m - col - 1
        val vBase = col + 1 + col * m
        val s = x[col] + kernels.dot(buf, vBase, x, col + 1, len)
        val f = t * s
        x[col] -= f
        if (f != 0.0) kernels.axpy(x, col + 1, -f, buf, vBase, len)
    }
    return x
}

/** The pivoted least-squares solve into [out], which is returned. */
internal fun referencePivotedLeastSquaresInto(
    kernels: F64Kernels,
    qr: F64PivotedQrDecomposition,
    b: DoubleArray,
    out: DoubleArray,
    workspace: Workspace?,
): DoubleArray {
    requireShape(b.size == qr.m) { "solve: b length ${b.size} != ${qr.m}" }
    requireShape(out.size == qr.n) { "solve: out length ${out.size} != ${qr.n}" }
    val rank = qr.rank
    workspace.borrow(qr.m) { y ->
        referenceApplyQInto(kernels, qr.factorization, b, y, transpose = true)
        trsvCore(
            kernels,
            qr.factorization.qr,
            rank,
            y,
            lda = qr.m,
            lower = false,
            transpose = false,
            unitDiag = false,
        )
        out.fill(0.0)
        for (k in 0 until rank) out[qr.pivots[k]] = y[k]
    }
    return out
}

/** Least-squares solve into [out], which has length `n` and is returned. A [workspace] lends the
 *  length-`m` intermediate for `Qᵀb`. */
internal fun referenceLeastSquaresInto(
    kernels: F64Kernels,
    qr: F64QrDecomposition,
    b: DoubleArray,
    out: DoubleArray,
    workspace: Workspace?,
): DoubleArray {
    requireShape(qr.m >= qr.n) { "solve requires m >= n, got ${qr.m}x${qr.n}" }
    requireShape(b.size == qr.m) { "solve: b length ${b.size} != ${qr.m}" }
    requireShape(out.size == qr.n) { "solve: out length ${out.size} != ${qr.n}" }
    workspace.borrow(qr.m) { y ->
        referenceApplyQInto(kernels, qr, b, y, transpose = true)
        y.copyInto(out, 0, 0, qr.n)
    }
    trsvCore(kernels, qr.qr, qr.n, out, lda = qr.m, lower = false, transpose = false, unitDiag = false)
    return out
}

/** Minimum-norm solve into [out], which has length `m` (the wide system's column count) and is returned.
 *  A [workspace] lends the length-`n` intermediate. */
internal fun referenceMinimumNormInto(
    kernels: F64Kernels,
    qr: F64QrDecomposition,
    b: DoubleArray,
    out: DoubleArray,
    workspace: Workspace?,
): DoubleArray {
    requireShape(qr.m >= qr.n) { "solve with minimumNorm expects the QR of the transpose (tall), got ${qr.m}x${qr.n}" }
    requireShape(b.size == qr.n) { "solve with minimumNorm: b length ${b.size} != ${qr.n}" }
    requireShape(out.size == qr.m) { "solve with minimumNorm: out length ${out.size} != ${qr.m}" }
    workspace.borrow(qr.n) { w ->
        b.copyInto(w)
        trsvCore(kernels, qr.qr, qr.n, w, lda = qr.m, lower = false, transpose = true, unitDiag = false)
        w.copyInto(out)
    }
    out.fill(0.0, qr.n, out.size) // the leading n entries are the solve's, the tail is the wide part
    return referenceApplyQInto(kernels, qr, out, out, transpose = false)
}
