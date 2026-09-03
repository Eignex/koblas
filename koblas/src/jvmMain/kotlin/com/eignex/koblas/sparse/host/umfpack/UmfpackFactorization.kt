package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.NativeResourceLifecycle
import com.eignex.koblas.internal.host.keepingReachable
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.sparse.F64SparseLuFactorization
import com.eignex.koblas.sparse.FactorsNotExposed
import com.eignex.koblas.sparse.requireSolveShapes
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE

/**
 * Holds [matrix] alive because umfpack_di_solve takes Ap, Ai and Ax alongside the factors, and releases the
 * factors, which are UMFPACK's own allocation, on close or through [nativeCleaner] as a fallback.
 */
public class UmfpackFactorization internal constructor(
    private val matrix: F64SparseMatrix,
    override val failedAt: Int,
    arena: Arena,
    private val numericHolder: MemorySegment,
    private val calls: UmfpackCalls,
    private val control: MemorySegment?,
) : F64SparseLuFactorization {
    /**
     * `L`, `U`, the permutations and the scaling, extracted on the first read of any of them. UMFPACK copies
     * both factors out of its numeric object, so a caller who only solves never pays for them.
     */
    private val extracted: UmfpackFactors by lazy {
        calls.extractFactors(numericHolder.get(ADDRESS, 0L), matrix.rows) ?: throw FactorsNotExposed("native factors")
    }

    private val factors: UmfpackFactors get() = lifecycle.withResource { extracted }

    override val l: F64SparseMatrix get() = factors.lower

    override val u: F64SparseMatrix get() = factors.upper

    override val rowOrder: IntArray get() = factors.rowOrder.copyOf()

    override val columnOrder: IntArray get() = factors.columnOrder.copyOf()

    override val rowScaling: DoubleArray get() = factors.rowScaling.copyOf()

    private val emptyOffDiagonal: F64SparseMatrix by lazy {
        F64SparseMatrix.wrap(n, n, IntArray(n + 1), IntArray(0), DoubleArray(0))
    }

    override val offDiagonal: F64SparseMatrix get() = lifecycle.withResource { emptyOffDiagonal }

    private class Release(
        private val calls: UmfpackCalls,
        private val arena: Arena,
        private val numericHolder: MemorySegment,
    ) {
        fun release() {
            arena.use {
                calls.freeNumeric(numericHolder)
            }
        }
    }

    private val lifecycle = NativeResourceLifecycle(
        "UMFPACK factorization",
        Release(calls, arena, numericHolder)::release,
    )
    private val cleanable = nativeCleaner.register(this, lifecycle)

    override val n: Int get() = matrix.rows

    override val nnz: Int get() = lifecycle.withResource { lnz + unz }

    /** Fill in `L` and `U`, read out of UMFPACK's Info at factorization time. */
    internal var lnz: Int = 0

    internal var unz: Int = 0

    override var rcond: Double = 0.0
        get() = lifecycle.withResource { field }
        internal set

    /** Reused because factors are externally serialized; see the public sparse-factor lifecycle contract. */
    private val solveInfo = arena.allocate(JAVA_DOUBLE, INFO.toLong())
    private val aliasedSolveAllocation = AllocationCapability(
        AllocationGuarantee.NO_SIZE_DEPENDENT_MANAGED,
        listOf(ScratchRequirement(ScratchKind.F64, n)),
    )

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability =
        if (aliasing) aliasedSolveAllocation else noSizeDependentManagedAllocation

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        lifecycle.withResource {
            requireFactored(failedAt, "solve")
            requireSolveShapes(n, b, out)
            // UMFPACK writes X and reads B, so aliasing them would have it read its own partial output. The
            // separate right-hand side comes from the workspace when there is one to lend it.
            if (out !== b) return@withResource solveDistinct(b, out, transpose)
            workspace.borrow(n) { rhs ->
                b.copyInto(rhs)
                solveDistinct(rhs, out, transpose)
            }
        }

    /** The call itself, over a right-hand side that does not alias [out]. */
    private fun solveDistinct(b: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray {
        // The factors are reached as a raw address, so nothing the call holds keeps this factorization
        // reachable and the cleaner is free to run mid-call. The fence is what holds it to the return.
        keepingReachable(this) {
            val status = calls.solve(
                if (transpose) SYS_AT else SYS_A,
                MemorySegment.ofArray(matrix.colPtr),
                MemorySegment.ofArray(matrix.rowIdx),
                MemorySegment.ofArray(matrix.values),
                MemorySegment.ofArray(out),
                MemorySegment.ofArray(b),
                numericHolder.get(ADDRESS, 0),
                control,
                solveInfo,
            )
            check(status == OK) { "umfpack_di_solve failed with status $status" }
        }
        return out
    }

    override fun close(): Unit = cleanable.clean()

    internal companion object {
        /** Reads the fill counts out of a completed factorization's Info array. */
        fun fillOf(info: MemorySegment): Pair<Int, Int> = info.getAtIndex(JAVA_DOUBLE, INFO_LNZ.toLong()).toInt() to
            info.getAtIndex(JAVA_DOUBLE, INFO_UNZ.toLong()).toInt()

        fun rcondOf(info: MemorySegment): Double = info.getAtIndex(JAVA_DOUBLE, INFO_RCOND.toLong())
    }
}
