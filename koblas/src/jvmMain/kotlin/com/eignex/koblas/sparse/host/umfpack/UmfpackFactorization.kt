package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.requireSolveShapes
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.ref.Reference

/**
 * Holds [matrix] alive because umfpack_di_solve takes Ap, Ai and Ax alongside the factors, and releases the
 * factors, which are UMFPACK's own allocation, through [nativeCleaner] once this object is unreachable.
 */
public class UmfpackFactorization internal constructor(
    private val matrix: F64SparseMatrix,
    override val failedAt: Int,
    private val arena: Arena,
    private val numericHolder: MemorySegment,
    private val calls: UmfpackCalls,
    private val control: MemorySegment?,
) : F64SparseFactorization {

    init {
        nativeCleaner.register(this, Release(calls, arena, numericHolder))
    }

    private class Release(
        private val calls: UmfpackCalls,
        private val arena: Arena,
        private val numericHolder: MemorySegment,
    ) : Runnable {
        override fun run() {
            calls.freeNumeric(numericHolder)
            arena.close()
        }
    }

    override val n: Int get() = matrix.rows

    override val nnz: Int get() = lnz + unz

    /** Fill in `L` and `U`, read out of UMFPACK's Info at factorization time. */
    internal var lnz: Int = 0

    internal var unz: Int = 0

    override var rcond: Double = 0.0
        internal set

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireFactored(failedAt, "solve")
        requireSolveShapes(n, b, out)
        // UMFPACK writes X and reads B, so aliasing them would have it read its own partial output. The
        // separate right-hand side comes from the workspace when there is one to lend it.
        if (out !== b) return solveDistinct(b, out, transpose)
        return workspace.borrow(n) { rhs ->
            b.copyInto(rhs)
            solveDistinct(rhs, out, transpose)
        }
    }

    /** The call itself, over a right-hand side that does not alias [out]. */
    private fun solveDistinct(b: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray {
        // The factors are reached as a raw address, so nothing the call holds keeps this factorization
        // reachable and the cleaner is free to run mid-call. The fence is what holds it to the return.
        try {
            Arena.ofConfined().use { scratch ->
                val info = scratch.allocate(JAVA_DOUBLE, INFO.toLong())
                val status = calls.solve(
                    if (transpose) SYS_AT else SYS_A,
                    MemorySegment.ofArray(matrix.colPtr),
                    MemorySegment.ofArray(matrix.rowIdx),
                    MemorySegment.ofArray(matrix.values),
                    MemorySegment.ofArray(out),
                    MemorySegment.ofArray(b),
                    numericHolder.get(ADDRESS, 0),
                    control,
                    info,
                )
                check(status == OK) { "umfpack_di_solve failed with status $status" }
            }
        } finally {
            Reference.reachabilityFence(this)
        }
        return out
    }

    internal companion object {
        /** Reads the fill counts out of a completed factorization's Info array. */
        fun fillOf(info: MemorySegment): Pair<Int, Int> = info.getAtIndex(JAVA_DOUBLE, INFO_LNZ.toLong()).toInt() to
            info.getAtIndex(JAVA_DOUBLE, INFO_UNZ.toLong()).toInt()

        fun rcondOf(info: MemorySegment): Double = info.getAtIndex(JAVA_DOUBLE, INFO_RCOND.toLong())
    }
}
