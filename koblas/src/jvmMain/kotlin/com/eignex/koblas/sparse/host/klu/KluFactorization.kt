package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.NativeOwnership
import com.eignex.koblas.requireSolveShapes
import com.eignex.koblas.sparse.F64SparseLuFactorization
import com.eignex.koblas.sparse.FactorsNotExposed

/** A KLU factorization with deterministic close and cleaner fallback for its symbolic and numeric objects. */
public class KluFactorization internal constructor(
    initialFailedAt: Int,
    private val factor: KluFactor,
    private val calls: KluCalls,
) : F64SparseLuFactorization {
    override var failedAt: Int = initialFailedAt
        private set

    private class Release(private val calls: KluCalls, private val factor: KluFactor) {
        fun release() {
            try {
                calls.free(factor)
            } finally {
                factor.arena.close()
            }
        }
    }

    private val ownership = NativeOwnership(this, "KLU factorization", Release(calls, factor)::release)

    override val n: Int get() = factor.n

    /**
     * The factors, extracted on the first read. KLU copies them out of its numeric object, so a caller who
     * only solves or refactorizes never pays for them.
     */
    private var extracted: Lazy<KluFactors> = freshExtraction()

    private fun freshExtraction(): Lazy<KluFactors> = lazy {
        calls.extract(factor) ?: throw FactorsNotExposed("native factors")
    }

    private val factors: KluFactors
        get() = ownership.anchoring {
            requireFactored(failedAt, "factors")
            extracted.value
        }

    override val l: F64SparseMatrix get() = factors.lower

    override val u: F64SparseMatrix get() = factors.upper

    override val offDiagonal: F64SparseMatrix get() = factors.offDiagonal

    override val rowOrder: IntArray get() = factors.rowOrder.copyOf()

    override val columnOrder: IntArray get() = factors.columnOrder.copyOf()

    override val rowScaling: DoubleArray get() = factors.rowScaling.copyOf()

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability =
        noSizeDependentManagedAllocation

    /** Reads KLU's numeric object, so the fence holds this factorization past the read. */
    override val nnz: Int get() = ownership.anchoring {
        factor.nnz
    }

    override val rcond: Double get() = ownership.anchoring { factor.rcond }

    internal fun refactor(a: F64SparseMatrix, equilibrate: Boolean): KluRefactorResult = ownership.anchoring {
        // A different order is reported rather than raised, the same as a different pattern of the same
        // order: refactor already answers a foreign `previous` by factoring afresh, so a caller holding a
        // reusable analysis for the wrong matrix gets one answer whichever way it fails to match.
        if (a.rows != n || !a.colPtr.contentEquals(factor.colPtr) || !a.rowIdx.contentEquals(factor.rowIdx)) {
            return@anchoring KluRefactorResult.Incompatible
        }
        val succeeded = calls.refactor(factor, a.colPtr, a.rowIdx, a.values, equilibrate)
        extracted = freshExtraction()
        if (succeeded) {
            failedAt = NOT_SINGULAR
            KluRefactorResult.Success
        } else {
            failedAt = SINGULAR_POSITION_UNKNOWN
            KluRefactorResult.Singular
        }
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        ownership.anchoring {
            requireFactored(failedAt, "solve")
            requireSolveShapes(n, n, b, out)
            if (out !== b) b.copyInto(out)
            calls.solve(factor, out, transpose)

            out
        }

    override fun solveInto(
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        transpose: Boolean,
        workspace: Workspace?,
    ): F64DenseMatrix = ownership.anchoring {
        requireFactored(failedAt, "solve")
        requireSolveShapes(n, n, b, out)
        if (b.cols == 0) return@anchoring out
        if (out.data !== b.data) b.data.copyInto(out.data)
        calls.solve(factor, out.data, transpose, b.cols)

        out
    }

    override fun close(): Unit = ownership.close()
}
