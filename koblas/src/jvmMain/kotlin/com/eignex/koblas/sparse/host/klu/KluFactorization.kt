package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.requireFactored
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.requireSolveShapes
import java.lang.ref.Reference

/** A KLU factorization whose native symbolic and numeric objects are reclaimed when it becomes unreachable. */
public class KluFactorization internal constructor(
    initialFailedAt: Int,
    private val factor: KluFactor,
    private val calls: KluCalls,
) : F64SparseFactorization {
    override var failedAt: Int = initialFailedAt
        private set
    init {
        nativeCleaner.register(this, Release(calls, factor))
    }

    private class Release(private val calls: KluCalls, private val factor: KluFactor) : Runnable {
        override fun run() {
            calls.free(factor)
            factor.arena.close()
        }
    }

    override val n: Int get() = factor.n

    /** Reads KLU's numeric object, so the fence holds this factorization past the read. */
    override val nnz: Int get() = try {
        factor.nnz
    } finally {
        Reference.reachabilityFence(this)
    }

    override val rcond: Double get() = factor.rcond

    internal fun refactor(a: F64SparseMatrix, equilibrate: Boolean): KluRefactorResult {
        requireShape(a.rows == n) { "refactor: matrix size ${a.rows}, expected $n" }
        if (!a.colPtr.contentEquals(factor.colPtr) || !a.rowIdx.contentEquals(factor.rowIdx)) {
            return KluRefactorResult.Incompatible
        }
        return try {
            if (calls.refactor(factor, a.colPtr, a.rowIdx, a.values, equilibrate)) {
                failedAt = NOT_SINGULAR
                KluRefactorResult.Success
            } else {
                failedAt = SINGULAR_POSITION_UNKNOWN
                KluRefactorResult.Singular
            }
        } finally {
            Reference.reachabilityFence(this)
        }
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireFactored(failedAt, "solve")
        requireSolveShapes(n, b, out)
        if (out !== b) b.copyInto(out)
        try {
            calls.solve(factor, out, transpose)
        } finally {
            Reference.reachabilityFence(this)
        }
        return out
    }
}

internal enum class KluRefactorResult { Success, Incompatible, Singular }
