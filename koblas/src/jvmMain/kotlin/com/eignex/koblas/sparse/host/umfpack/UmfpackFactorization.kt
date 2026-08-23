package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.Workspace
import com.eignex.koblas.core.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.requireFactored
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.F64SparseFactorization
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.ref.Cleaner
import java.lang.ref.Reference

/** Frees the native factors of unreachable factorizations. One for the whole process. */
private val cleaner: Cleaner = Cleaner.create()

/**
 * Holds [matrix] alive because umfpack_di_solve takes Ap, Ai and Ax alongside the factors, and releases the
 * factors, which are UMFPACK's own allocation, through a [Cleaner] once this object is unreachable.
 */
public class UmfpackFactorization internal constructor(
    private val matrix: F64SparseMatrix,
    override val failedAt: Int,
    private val handles: Handles,
    private val calls: UmfpackCalls,
) : F64SparseFactorization {

    /**
     * The arena and the `void **Numeric` holder, split out so the cleanup lambda captures them and not the
     * factorization, which it would otherwise keep reachable forever.
     */
    internal class Handles(val arena: Arena, val numericHolder: MemorySegment)

    init {
        val arena = handles.arena
        val holder = handles.numericHolder
        cleaner.register(this) {
            calls.freeNumeric(holder)
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
        requireShape(b.size == n) { "solve: b size ${b.size}, expected $n" }
        requireShape(out.size == n) { "solve: out size ${out.size}, expected $n" }
        // UMFPACK writes X and reads B, so aliasing them would have it read its own partial output.
        val rhs = if (out === b) b.copyOf() else b
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
                    MemorySegment.ofArray(rhs),
                    handles.numericHolder.get(ADDRESS, 0),
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
