package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.Workspace
import com.eignex.koblas.requireFactored
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.F64SparseFactorization
import java.lang.ref.Cleaner
import java.lang.ref.Reference

private val cleaner: Cleaner = Cleaner.create()

/** A KLU factorization whose native symbolic and numeric objects are reclaimed when it becomes unreachable. */
public class KluFactorization internal constructor(override val failedAt: Int, private val factor: KluFactor) :
    F64SparseFactorization {
    init {
        val heldFactor = factor
        cleaner.register(this) {
            KluCalls.free(heldFactor)
            heldFactor.arena.close()
        }
    }

    override val n: Int get() = factor.n
    override val nnz: Int get() = factor.nnz
    override val rcond: Double get() = factor.rcond

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireFactored(failedAt, "solve")
        requireShape(b.size == n) { "solve: b size ${b.size}, expected $n" }
        requireShape(out.size == n) { "solve: out size ${out.size}, expected $n" }
        if (out !== b) b.copyInto(out)
        try {
            KluCalls.solve(factor, out, transpose)
        } finally {
            Reference.reachabilityFence(this)
        }
        return out
    }
}
