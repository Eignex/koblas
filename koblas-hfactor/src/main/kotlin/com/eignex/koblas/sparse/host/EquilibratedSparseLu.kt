package com.eignex.koblas.sparse.host

import com.eignex.koblas.Workspace
import com.eignex.koblas.sparse.SparseFactorization
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.log2
import kotlin.math.pow

/** Power-of-two row factors near `1/max|row|`, so scaling is exact in binary floating point. */
internal fun f64EquilibrationScale(rows: Int, rowIdx: IntArray, values: DoubleArray): DoubleArray {
    val largest = DoubleArray(rows)
    for (k in values.indices) {
        val magnitude = abs(values[k])
        if (magnitude > largest[rowIdx[k]]) largest[rowIdx[k]] = magnitude
    }
    return DoubleArray(rows) { row ->
        val magnitude = largest[row]
        if (magnitude <= 0.0) 1.0 else 2.0.pow(-floor(log2(magnitude)).toInt()).takeIf { it.isFinite() } ?: 1.0
    }
}

internal fun f64ScaledValues(rowIdx: IntArray, values: DoubleArray, scale: DoubleArray): DoubleArray =
    DoubleArray(values.size) { values[it] * scale[rowIdx[it]] }

internal fun applyEquilibration(x: DoubleArray, scale: DoubleArray) {
    for (i in x.indices) x[i] *= scale[i]
}

/**
 * A native factorization of `E·A` presented as one of `A`.
 *
 * The library was handed row-scaled values, so a forward solve scales its right-hand side going in and a
 * transposed one scales its result coming out, by the reasoning [applyEquilibration] records. Everything
 * else is the library's own answer about the factors it holds.
 *
 * Written out rather than delegated with `by`: [SparseFactorization] gives its dense solves and its
 * `solve` overloads default bodies that call back into [solveInto], and a delegating class would send those
 * to the wrapped factorization, which would answer them without undoing the scaling.
 */
internal class EquilibratedSparseLu(private val inner: SparseFactorization, private val scale: DoubleArray) :
    SparseFactorization {
    override val n: Int get() = inner.n
    override val failedAt: Int get() = inner.failedAt
    override val nnz: Int get() = inner.nnz
    override val rcond: Double get() = inner.rcond

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        // The destination carries the scaled right-hand side in, which it may since the solve overwrites it
        // and reads each position before writing it where the two alias.
        if (!transpose) {
            for (i in b.indices) out[i] = b[i] * scale[i]
            inner.solveInto(out, out, transpose = false, workspace = workspace)
            return out
        }
        inner.solveInto(b, out, transpose = true, workspace = workspace)
        applyEquilibration(out, scale)
        return out
    }

    override fun close(): Unit = inner.close()
}
