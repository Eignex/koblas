package com.eignex.koblas.sparse.host.spqr

import com.eignex.koblas.AllocationCapability
import com.eignex.koblas.AllocationGuarantee
import com.eignex.koblas.ScratchKind
import com.eignex.koblas.ScratchRequirement
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.requireShape
import com.eignex.koblas.requireSolveShapes
import com.eignex.koblas.sparse.F64SparseQrFactorization
import com.eignex.koblas.sparse.sparseSnapshotOf

internal class SpqrFactorData(
    val rank: Int,
    val r: F64SparseMatrix,
    val columnOrder: IntArray,
    val householder: F64SparseMatrix,
    val rowPermutation: IntArray,
    val tau: DoubleArray,
)

/** A SPQR factorization retained in its coherent sparse Householder form. */
public class SpqrQrFactorization internal constructor(
    private val factors: SpqrFactorData,
    override val m: Int,
    override val n: Int,
) : F64SparseQrFactorization {
    private var closed = false
    private val solveCapability = AllocationCapability(
        AllocationGuarantee.NO_MANAGED,
        listOf(ScratchRequirement(ScratchKind.F64, m)),
    )

    override val rank: Int get() = checked { factors.rank }

    override val nnz: Int get() = checked { factors.r.nnz + factors.householder.nnz }

    override val r: F64SparseMatrix get() = checked { sparseSnapshotOf(factors.r) }

    override val columnOrder: IntArray get() = checked { factors.columnOrder.copyOf() }

    override fun solveAllocation(): AllocationCapability = solveCapability

    override fun applyQInto(y: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        checkOpen()
        requireShape(y.size == m) { "applyQ: y size ${y.size}, expected $m" }
        requireShape(out.size == m) { "applyQ: out size ${out.size}, expected $m" }
        workspace.borrow(m) { x ->
            if (transpose) {
                x.fill(0.0, 0, m)
                for (i in 0 until m) x[factors.rowPermutation[i]] = y[i]
                for (k in factors.tau.indices) applyReflection(k, x)
                x.copyInto(out, endIndex = m)
            } else {
                y.copyInto(x, endIndex = m)
                for (k in factors.tau.indices.reversed()) applyReflection(k, x)
                for (i in 0 until m) out[i] = x[factors.rowPermutation[i]]
            }
        }
        return out
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, workspace: Workspace?): DoubleArray {
        checkOpen()
        requireSolveShapes(m, n, b, out)
        if (rankDeficient) {
            throw SingularMatrix(rank, "solve: the QR factorization has rank $rank of $n columns")
        }
        workspace.borrow(m) { x ->
            x.fill(0.0, 0, m)
            for (i in 0 until m) x[factors.rowPermutation[i]] = b[i]
            for (k in factors.tau.indices) applyReflection(k, x)
            for (k in n - 1 downTo 0) {
                val last = factors.r.colPtr[k + 1] - 1
                check(last >= factors.r.colPtr[k] && factors.r.rowIdx[last] == k) {
                    "SPQR returned a full-rank R without diagonal $k"
                }
                val xk = x[k] / factors.r.values[last]
                x[k] = xk
                if (xk != 0.0) {
                    for (p in factors.r.colPtr[k] until last) {
                        x[factors.r.rowIdx[p]] -= factors.r.values[p] * xk
                    }
                }
            }
            for (k in 0 until n) out[factors.columnOrder[k]] = x[k]
        }
        return out
    }

    override fun close() {
        closed = true
    }

    private fun applyReflection(k: Int, x: DoubleArray) {
        var product = 0.0
        factors.householder.forEachInColumn(k) { row, value -> product += value * x[row] }
        val scale = factors.tau[k] * product
        if (scale == 0.0) return
        factors.householder.forEachInColumn(k) { row, value -> x[row] -= value * scale }
    }

    private fun checkOpen() {
        check(!closed) { "SPQR factorization is closed" }
    }

    private inline fun <T> checked(body: () -> T): T {
        checkOpen()
        return body()
    }
}
