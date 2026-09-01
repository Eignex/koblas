@file:Suppress("VariableNaming", "FunctionParameterNaming") // math convention: single-letter A, B, L, U

package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import kotlin.math.abs

/*
 * The portable LU factorization, the solves over its factors and the Hager condition estimator built on
 * them, netlib dgetrf, dgetrs and dgecon. This is the semantic definition a native LU is validated against;
 * [F64ReferenceDecompositions] is the F64Decompositions surface over it.
 */

/** Doolittle LU with partial pivoting, writing into [out]'s buffers. */
internal fun referenceLuFactorInto(
    kernels: F64Kernels,
    a: F64DenseMatrix,
    out: F64LuDecomposition,
): F64LuDecomposition {
    val m = out.rows
    val n = out.cols
    val order = out.order
    val lu = out.lu
    a.data.copyInto(lu)
    val piv = out.piv
    for (i in 0 until m) piv[i] = i
    var failedAt = NOT_SINGULAR
    for (k in 0 until order) {
        var p = k
        var max = abs(lu[k + k * m])
        for (i in k + 1 until m) {
            val v = abs(lu[i + k * m])
            if (v > max) {
                max = v
                p = i
            }
        }
        if (max == 0.0) {
            if (failedAt == NOT_SINGULAR) failedAt = k
            continue
        }
        if (p != k) {
            for (j in 0 until n) {
                val t = lu[k + j * m]
                lu[k + j * m] = lu[p + j * m]
                lu[p + j * m] = t
            }
            val tp = piv[k]
            piv[k] = piv[p]
            piv[p] = tp
        }
        // Division keeps a subnormal pivot from becoming an infinity.
        val pivot = lu[k + k * m]
        val len = m - k - 1
        val colBase = k + 1 + k * m
        for (i in k + 1 until m) lu[i + k * m] = lu[i + k * m] / pivot
        for (j in k + 1 until n) {
            val ukj = lu[k + j * m]
            if (ukj != 0.0) kernels.axpy(lu, k + 1 + j * m, -ukj, lu, colBase, len)
        }
    }
    out.failedAt = failedAt
    return out
}

@Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
internal fun referenceLuSolveInto(
    kernels: F64Kernels,
    lu: F64LuDecomposition,
    b: DoubleArray,
    out: DoubleArray,
    transpose: Boolean = false,
    workspace: Workspace? = null,
): DoubleArray {
    requireLuSquare(lu, "solve")
    requireFactored(lu.failedAt, "solve")
    val n = lu.n
    requireShape(b.size == n) { "solve: b length ${b.size} != $n" }
    requireShape(out.size == n) { "solve: out length ${out.size} != $n" }
    val a = lu.lu
    val piv = lu.piv
    return if (transpose) {
        solveTranspose(kernels, n, a, piv, b, out, workspace)
    } else {
        solveNormal(kernels, n, a, piv, b, out, workspace)
    }
}

@Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
private fun solveNormal(
    kernels: F64Kernels,
    n: Int,
    a: DoubleArray,
    piv: IntArray,
    b: DoubleArray,
    out: DoubleArray,
    workspace: Workspace?,
): DoubleArray {
    // When out aliases b the permutation may read a slot already written, so it stages the gather.
    if (out === b) {
        workspace.borrow(n) { staged ->
            for (i in 0 until n) staged[i] = b[piv[i]]
            staged.copyInto(out)
        }
    } else {
        for (i in 0 until n) out[i] = b[piv[i]]
    }
    trsvCore(kernels, a, n, out, lower = true, transpose = false, unitDiag = true)
    trsvCore(kernels, a, n, out, lower = false, transpose = false, unitDiag = false)
    return out
}

@Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
private fun solveTranspose(
    kernels: F64Kernels,
    n: Int,
    a: DoubleArray,
    piv: IntArray,
    b: DoubleArray,
    out: DoubleArray,
    workspace: Workspace?,
): DoubleArray {
    workspace.borrow(n) { y ->
        b.copyInto(y)
        trsvCore(kernels, a, n, y, lower = false, transpose = true, unitDiag = false)
        trsvCore(kernels, a, n, y, lower = true, transpose = true, unitDiag = true)
        for (i in 0 until n) out[piv[i]] = y[i] // undo the row permutation
    }
    return out
}

/** Solve `A · X = B`, or `Aᵀ · X = B` when [transpose], into [out], which is returned. [out] may be [b],
 *  and a [workspace] lends the transposed direction's `n·nrhs` staging block. */
@Suppress("LongParameterList") // destination and scratch on top of the solve's own arguments
internal fun referenceLuSolveInto(
    kernels: F64Kernels,
    lu: F64LuDecomposition,
    b: F64DenseMatrix,
    out: F64DenseMatrix,
    transpose: Boolean,
    workspace: Workspace?,
): F64DenseMatrix {
    requireLuSquare(lu, "solve")
    requireFactored(lu.failedAt, "solve")
    val n = lu.n
    val nrhs = b.cols
    requireSolveShapes(n, b, out)
    if (nrhs == 0) return out
    val f = lu.lu
    return if (transpose) {
        workspace.borrow(n * nrhs) { y ->
            b.data.copyInto(y)
            trsmCore(kernels, f, n, y, nrhs, lower = false, transpose = true, unitDiag = false)
            trsmCore(kernels, f, n, y, nrhs, lower = true, transpose = true, unitDiag = true)
            permuteRows(y, out.data, n, nrhs, lu.piv, gather = false)
        }
        out
    } else {
        // The gather cannot read B in place once out shares its buffer, so that case stages. Two
        // matrices can be distinct objects over one array, so the test is on the storage.
        if (out.data === b.data) {
            workspace.borrow(n * nrhs) { staged ->
                permuteRows(b.data, staged, n, nrhs, lu.piv, gather = true)
                staged.copyInto(out.data)
            }
        } else {
            permuteRows(b.data, out.data, n, nrhs, lu.piv, gather = true)
        }
        trsmCore(kernels, f, n, out.data, nrhs, lower = true, transpose = false, unitDiag = true)
        trsmCore(kernels, f, n, out.data, nrhs, lower = false, transpose = false, unitDiag = false)
        out
    }
}

/** The estimator body, over caller-supplied vectors so [rcond] can pool them. */
@Suppress("LongParameterList") // the kernels on top of four scratch vectors the caller owns
internal fun hagerEstimate(
    kernels: F64Kernels,
    lu: F64LuDecomposition,
    anorm: Double,
    n: Int,
    x: DoubleArray,
    y: DoubleArray,
    signs: DoubleArray,
    probe: DoubleArray,
    workspace: Workspace?,
): Double {
    for (i in 0 until n) x[i] = 1.0 / n
    var estimate = 0.0
    var lastPivot = -1
    var sweep = 0
    while (sweep < RCOND_MAX_SWEEPS) {
        referenceLuSolveInto(kernels, lu, x, y, workspace = workspace)
        var e = 0.0
        for (i in 0 until n) e += abs(y[i])
        // A solve that stops producing numbers means the factor cannot be inverted in this arithmetic, so
        // the matrix is singular for any use the estimate has. Carrying on instead lets a NaN through every
        // comparison below, since none of them hold against it, and the search then settles on a small
        // estimate and reports the matrix as well conditioned.
        if (!e.isFinite()) return 0.0
        if (e > estimate) estimate = e
        for (i in 0 until n) signs[i] = if (y[i] >= 0.0) 1.0 else -1.0
        val z = referenceLuSolveInto(kernels, lu, signs, y, transpose = true, workspace = workspace)
        var j = 0
        for (i in 1 until n) if (abs(z[i]) > abs(z[j])) j = i
        var zx = 0.0
        for (i in 0 until n) zx += z[i] * x[i]
        if (j == lastPivot || abs(z[j]) <= zx) break
        x.fill(0.0)
        x[j] = 1.0
        lastPivot = j
        sweep++
    }
    // The alternating-sign safeguard from LAPACK's dlacn2.
    for (i in 0 until n) {
        probe[i] = (if (i % 2 == 0) 1.0 else -1.0) * (1.0 + i.toDouble() / maxOf(1, n - 1))
    }
    val alt = referenceLuSolveInto(kernels, lu, probe, y, workspace = workspace)
    var e = 0.0
    for (i in 0 until n) e += abs(alt[i])
    if (!e.isFinite()) return 0.0
    e = 2.0 * e / (3.0 * n)
    if (e > estimate) estimate = e
    val denominator = estimate * anorm
    return if (denominator.isFinite() && denominator > 0.0) 1.0 / denominator else 0.0
}
