package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.NativeResourceLifecycle
import com.eignex.koblas.internal.host.keepingReachable
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.requireSolveShapes
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.FactorsNotExposed
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
            solveWorkspace.use { solveWorkspace ->
                calls.free(factor)
            }
        }
    }

    /** Which factorization this holds, read once: converting a copy needs the same answer. */
    private val isLl: Boolean = factor.isLl

    private val solveWorkspace = CholmodSolveWorkspace(factor.n)
    private val lifecycle = NativeResourceLifecycle(
        "CHOLMOD factorization",
        Release(calls, factor, solveWorkspace)::release,
    )
    private val cleanable = nativeCleaner.register(this, lifecycle)

    override val n: Int = factor.n

    /**
     * `L` and the ordering, converted on the first read. CHOLMOD copies the factor to convert it, so a
     * caller who only solves never pays for this.
     */
    private val extracted: CholmodFactors by lazy {
        calls.extractFactor(factor, asLl = isLl) ?: throw FactorsNotExposed("native factors")
    }

    private val factors: CholmodFactors get() = lifecycle.withResource { extracted }

    /** The lower triangular factor, as [CholmodFactors.lower] documents it for each kind. */
    internal val lowerFactor: F64SparseMatrix
        get() {
            requireCholmodFactors("l")
            return factors.lower(isLl)
        }

    /** The diagonal factor of an `L·D·Lᵀ`, which CHOLMOD stores as the diagonal of `L`. */
    internal val diagonalFactor: DoubleArray
        get() {
            requireCholmodFactors("d")
            return factors.diagonal()
        }

    /** The fill-reducing ordering CHOLMOD chose. */
    internal val ordering: IntArray
        get() {
            requireCholmodFactors("order")
            return factors.permutation.copyOf()
        }

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability = noManagedAllocation

    override val nnz: Int get() = lifecycle.withResource {
        if (singular) {
            0
        } else {
            keepingReachable(this) {
                factor.nzmax
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
            keepingReachable(this) {
                val estimate = calls.rcond(factor)
                if (factor.isLl) sqrt(estimate) else estimate
            }
        }
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        lifecycle.withResource {
            if (singular) throw singularFailure(failedAt, "solve")
            requireSolveShapes(n, n, b, out)
            if (out !== b) b.copyInto(out)
            keepingReachable(this) {
                check(calls.solve(factor, out, solveWorkspace)) {
                    "cholmod_solve failed on a factorization it produced"
                }
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
        requireSolveShapes(n, n, b, out)
        if (b.cols == 0) return@withResource out
        if (out.data !== b.data) b.data.copyInto(out.data)
        keepingReachable(this) {
            CholmodSolveWorkspace(n, b.cols).use { blockWorkspace ->
                check(calls.solve(factor, out.data, blockWorkspace)) {
                    "cholmod_solve failed on a factorization it produced"
                }
            }
        }
        out
    }

    override fun close(): Unit = cleanable.clean()
}
