package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.internal.host.NativeResourceLifecycle
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorizationReport
import com.eignex.koblas.sparse.basicReport
import com.eignex.koblas.sparse.requireBlockSolveShapes
import com.eignex.koblas.sparse.requireSolveShapes
import java.lang.ref.Reference
import kotlin.math.sqrt

/** A CHOLMOD factorization with deterministic close and cleaner fallback for its native factor. */
public class CholmodFactorization internal constructor(
    private val factor: CholmodFactor,
    private val calls: CholmodCalls,
    /** The column a zero pivot stopped an `L·D·Lᵀ` at, or [NOT_SINGULAR]; a Cholesky raises instead. */
    override val failedAt: Int = NOT_SINGULAR,
) : F64SparseFactorization {
    private class Release(
        private val calls: CholmodCalls,
        private val factor: CholmodFactor,
        private val solveWorkspace: CholmodSolveWorkspace,
    ) {
        fun release() {
            try {
                calls.free(factor)
            } finally {
                solveWorkspace.close()
            }
        }
    }

    private val solveWorkspace = CholmodSolveWorkspace(factor.n)
    private val lifecycle = NativeResourceLifecycle(
        "CHOLMOD factorization",
        Release(calls, factor, solveWorkspace)::release,
    )
    private val cleanable = nativeCleaner.register(this, lifecycle)

    override val n: Int = factor.n

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability = noManagedAllocation

    override fun report(): F64SparseFactorizationReport = basicReport("cholmod").copy(
        details = mapOf("blockSolve" to "native"),
    )

    override val nnz: Int get() = lifecycle.withResource {
        if (singular) {
            0
        } else {
            try {
                factor.nzmax
            } finally {
                Reference.reachabilityFence(this)
            }
        }
    }

    /**
     * CHOLMOD's own estimate, which is the ratio of the smallest factor diagonal to the largest, squared for
     * an `L·Lᵀ`. The square root brings it back to the ratio this seam documents, so the number means the
     * same thing whichever backend produced it.
     */
    override val rcond: Double get() = lifecycle.withResource {
        if (singular) {
            0.0
        } else {
            try {
                val estimate = calls.rcond(factor)
                if (factor.isLl) sqrt(estimate) else estimate
            } finally {
                Reference.reachabilityFence(this)
            }
        }
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        lifecycle.withResource {
            if (singular) throw singularFailure(failedAt, "solve")
            requireSolveShapes(n, b, out)
            if (out !== b) b.copyInto(out)
            try {
                check(calls.solve(factor, out, solveWorkspace)) {
                    "cholmod_solve failed on a factorization it produced"
                }
            } finally {
                Reference.reachabilityFence(this)
            }
            out
        }

    override fun solveInto(
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        transpose: Boolean,
        workspace: Workspace?,
    ): F64DenseMatrix = lifecycle.withResource {
        if (singular) throw singularFailure(failedAt, "solve")
        requireBlockSolveShapes(n, b, out)
        if (b.cols == 0) return@withResource out
        if (out.data !== b.data) b.data.copyInto(out.data)
        try {
            CholmodSolveWorkspace(n, b.cols).use { blockWorkspace ->
                check(calls.solve(factor, out.data, blockWorkspace)) {
                    "cholmod_solve failed on a factorization it produced"
                }
            }
        } finally {
            Reference.reachabilityFence(this)
        }
        out
    }

    override fun close(): Unit = cleanable.clean()
}
