package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.NativeResourceLifecycle
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.sparse.F64SparseLuFactorization
import com.eignex.koblas.sparse.F64SparseFactorizationReport
import com.eignex.koblas.sparse.basicReport
import com.eignex.koblas.sparse.requireBlockSolveShapes
import com.eignex.koblas.sparse.requireSolveShapes
import java.lang.ref.Reference

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

    private val lifecycle = NativeResourceLifecycle("KLU factorization", Release(calls, factor)::release)
    private val cleanable = nativeCleaner.register(this, lifecycle)

    override val n: Int get() = factor.n

    /**
     * The factors, extracted on the first read. KLU copies them out of its numeric object, so a caller who
     * only solves or refactorizes never pays for them.
     */
    private val extracted: KluFactors by lazy {
        checkNotNull(calls.extract(factor)) {
            "this libklu does not expose klu_extract, so its factors cannot be read"
        }
    }

    private val factors: KluFactors get() = lifecycle.withResource { extracted }

    override val l: F64SparseMatrix get() = factors.lower

    override val u: F64SparseMatrix get() = factors.upper

    override val offDiagonal: F64SparseMatrix get() = factors.offDiagonal

    override val rowOrder: IntArray get() = factors.rowOrder.copyOf()

    override val columnOrder: IntArray get() = factors.columnOrder.copyOf()

    override val rowScaling: DoubleArray get() = factors.rowScaling.copyOf()

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability =
        noSizeDependentManagedAllocation

    /** Reads KLU's numeric object, so the fence holds this factorization past the read. */
    override val nnz: Int get() = lifecycle.withResource {
        try {
            factor.nnz
        } finally {
            Reference.reachabilityFence(this)
        }
    }

    override val rcond: Double get() = lifecycle.withResource { factor.rcond }

    internal fun refactor(a: F64SparseMatrix, equilibrate: Boolean): KluRefactorResult = lifecycle.withResource {
        requireShape(a.rows == n) { "refactor: matrix size ${a.rows}, expected $n" }
        if (!a.colPtr.contentEquals(factor.colPtr) || !a.rowIdx.contentEquals(factor.rowIdx)) {
            return@withResource KluRefactorResult.Incompatible
        }
        try {
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

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        lifecycle.withResource {
            requireFactored(failedAt, "solve")
            requireSolveShapes(n, b, out)
            if (out !== b) b.copyInto(out)
            try {
                calls.solve(factor, out, transpose)
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
        requireFactored(failedAt, "solve")
        requireBlockSolveShapes(n, b, out)
        if (b.cols == 0) return@withResource out
        if (out.data !== b.data) b.data.copyInto(out.data)
        try {
            calls.solve(factor, out.data, transpose, b.cols)
        } finally {
            Reference.reachabilityFence(this)
        }
        out
    }

    override fun report(): F64SparseFactorizationReport = basicReport("klu").copy(
        fillRatio = if (factor.rowIdx.isEmpty()) 0.0 else nnz.toDouble() / factor.rowIdx.size,
        details = mapOf("blockSolve" to "native"),
    )

    override fun close(): Unit = cleanable.clean()
}
