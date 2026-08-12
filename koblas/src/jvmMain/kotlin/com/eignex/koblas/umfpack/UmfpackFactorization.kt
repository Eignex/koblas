package com.eignex.koblas.umfpack

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.SparseFactorization
import java.lang.foreign.Arena
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.ref.Cleaner
import kotlin.math.pow

/** Index of UMFPACK_LNZ in Info, the nonzeros in `L` with the diagonal included. */
private const val INFO_LNZ = 43

/** Index of UMFPACK_UNZ in Info, the nonzeros in `U` with the diagonal included. */
private const val INFO_UNZ = 44

/** Frees the native factors of unreachable factorizations. One for the whole process. */
private val cleaner: Cleaner = Cleaner.create()

/**
 * Holds [matrix] alive because umfpack_di_solve takes Ap, Ai and Ax alongside the factors, and releases the
 * factors, which are UMFPACK's own allocation, through a [Cleaner] once this object is unreachable.
 */
public class UmfpackFactorization internal constructor(
    private val matrix: SparseMatrix,
    override val failedAt: Int,
    private val handles: Handles,
) : SparseFactorization {

    /**
     * The arena and the `void **Numeric` holder, split out so the cleanup lambda captures them and not the
     * factorization, which it would otherwise keep reachable forever.
     */
    internal class Handles(val arena: Arena, val numericHolder: MemorySegment)

    init {
        val arena = handles.arena
        val holder = handles.numericHolder
        cleaner.register(this) {
            UmfpackCalls.freeNumeric(holder)
            arena.close()
        }
    }

    override val n: Int get() = matrix.rows

    override val nnz: Int get() = lnz + unz

    /** Fill in `L` and `U`, read out of UMFPACK's Info at factorization time. */
    internal var lnz: Int = 0

    internal var unz: Int = 0

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        check(!singular) {
            "cannot solve against a singular factorization: umfpack reported the matrix singular. " +
                "Check `singular` before solving; repair the matrix and factor again."
        }
        requireShape(b.size == n) { "solve: b size ${b.size}, expected $n" }
        requireShape(out.size == n) { "solve: out size ${out.size}, expected $n" }
        // UMFPACK writes X and reads B, so aliasing them would have it read its own partial output.
        val rhs = if (out === b) b.copyOf() else b
        Arena.ofConfined().use { scratch ->
            val info = scratch.allocate(JAVA_DOUBLE, UmfpackCalls.INFO.toLong())
            val status = UmfpackCalls.solve(
                if (transpose) UmfpackCalls.SYS_AT else UmfpackCalls.SYS_A,
                MemorySegment.ofArray(matrix.colPtr),
                MemorySegment.ofArray(matrix.rowIdx),
                MemorySegment.ofArray(matrix.values),
                MemorySegment.ofArray(out),
                MemorySegment.ofArray(rhs),
                handles.numericHolder.get(ADDRESS, 0),
                info,
            )
            check(status == UmfpackCalls.OK) { "umfpack_di_solve failed with status $status" }
        }
        return out
    }

    /**
     * UMFPACK reports a mantissa and a base-10 exponent, since the product of n pivots overflows a double
     * early. Recombining them here can still overflow to infinity.
     */
    override fun determinant(): Double {
        if (singular) return 0.0
        Arena.ofConfined().use { scratch ->
            val mx = scratch.allocate(JAVA_DOUBLE)
            val ex = scratch.allocate(JAVA_DOUBLE)
            val info = scratch.allocate(JAVA_DOUBLE, UmfpackCalls.INFO.toLong())
            val status = UmfpackCalls.determinant(
                mx,
                ex,
                handles.numericHolder.get(ADDRESS, 0),
                info,
            )
            check(status == UmfpackCalls.OK) { "umfpack_di_get_determinant failed with status $status" }
            return mx.get(JAVA_DOUBLE, 0) * 10.0.pow(ex.get(JAVA_DOUBLE, 0))
        }
    }

    internal companion object {
        /** Reads the fill counts out of a completed factorization's Info array. */
        fun fillOf(info: MemorySegment): Pair<Int, Int> =
            info.get(JAVA_DOUBLE, INFO_LNZ * 8L).toInt() to info.get(JAVA_DOUBLE, INFO_UNZ * 8L).toInt()
    }
}
