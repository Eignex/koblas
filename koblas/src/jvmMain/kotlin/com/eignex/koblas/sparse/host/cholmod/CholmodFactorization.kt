package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.Workspace
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.requireSolveShapes
import java.lang.ref.Reference
import kotlin.math.sqrt

/** A CHOLMOD factorization whose native factor is reclaimed when it becomes unreachable. */
public class CholmodFactorization internal constructor(
    private val factor: CholmodFactor,
    private val calls: CholmodCalls,
) : F64SparseFactorization {

    init {
        nativeCleaner.register(this, Release(calls, factor))
    }

    private class Release(private val calls: CholmodCalls, private val factor: CholmodFactor) : Runnable {
        override fun run(): Unit = calls.free(factor)
    }

    override val n: Int get() = factor.n

    /** Always [NOT_SINGULAR]: a non-positive pivot is raised at factorization rather than carried here. */
    override val failedAt: Int get() = NOT_SINGULAR

    override val nnz: Int get() = try {
        factor.nzmax
    } finally {
        Reference.reachabilityFence(this)
    }

    /**
     * CHOLMOD's own estimate, which is the ratio of the smallest factor diagonal to the largest, squared for
     * an `L·Lᵀ`. The square root brings it back to the ratio this seam documents, so the number means the
     * same thing whichever backend produced it.
     */
    override val rcond: Double get() = try {
        val estimate = calls.rcond(factor)
        if (factor.isLl) sqrt(estimate) else estimate
    } finally {
        Reference.reachabilityFence(this)
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireSolveShapes(n, b, out)
        if (out !== b) b.copyInto(out)
        try {
            check(calls.solve(factor, out)) { "cholmod_solve failed on a factorization it produced" }
        } finally {
            Reference.reachabilityFence(this)
        }
        return out
    }
}
