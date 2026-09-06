package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.internal.host.NativeOwnership
import com.eignex.koblas.requireSolveShapes

/** A CHOLMOD factorization with deterministic close and cleaner fallback for its native factor. */
internal class CholmodFactorization(
    private val factor: CholmodFactor,
    private val calls: CholmodCalls,
    /** The column a zero pivot stopped an `L·D·Lᵀ` at, or [NOT_SINGULAR]; a Cholesky raises instead. */
    override val failedAt: Int = NOT_SINGULAR,
) : CholmodFactorizationBase() {
    private class Release(
        private val calls: CholmodCalls,
        private val factor: CholmodFactor,
        private val solveWorkspace: CholmodSolveWorkspace,
    ) {
        fun release() {
            solveWorkspace.use { solveWorkspace ->
                calls.free(factor)
            }
        }
    }

    override val isLl: Boolean = factor.isLl

    private val solveWorkspace = CholmodSolveWorkspace(factor.n)

    override val ownership =
        NativeOwnership(this, "CHOLMOD factorization", Release(calls, factor, solveWorkspace)::release)

    override val n: Int = factor.n

    override fun extractFactors(): CholmodFactors? = calls.extractFactor(factor, asLl = isLl)

    override fun readNzmax(): Int = factor.nzmax

    override fun readRcond(): Double = calls.rcond(factor)

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        ownership.anchoring {
            if (singular) throw singularFailure(failedAt, "solve")
            requireSolveShapes(n, n, b, out)
            if (out !== b) b.copyInto(out)
            check(calls.solve(factor, out, solveWorkspace)) {
                "cholmod_solve failed on a factorization it produced"
            }

            out
        }

    override fun solveInto(
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        transpose: Boolean,
        workspace: Workspace?,
    ): F64DenseMatrix = ownership.anchoring {
        if (singular) throw singularFailure(failedAt, "solve")
        requireSolveShapes(n, n, b, out)
        if (b.cols == 0) return@anchoring out
        if (out.data !== b.data) b.data.copyInto(out.data)
        CholmodSolveWorkspace(n, b.cols).use { blockWorkspace ->
            check(calls.solve(factor, out.data, blockWorkspace)) {
                "cholmod_solve failed on a factorization it produced"
            }
        }

        out
    }
}
