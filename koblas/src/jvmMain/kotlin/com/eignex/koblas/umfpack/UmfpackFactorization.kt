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

/** `Info[UMFPACK_LNZ]`: nonzeros in `L`, diagonal included. */
private const val INFO_LNZ = 43

/** `Info[UMFPACK_UNZ]`: nonzeros in `U`, diagonal included. */
private const val INFO_UNZ = 44

/** Frees the native factors of unreachable factorizations; one for the whole process. */
private val cleaner: Cleaner = Cleaner.create()

/**
 * UMFPACK's numeric factors, behind koblas's [SparseFactorization].
 *
 * This class is the payoff from making that a interface rather than a class. UMFPACK hands back a
 * `void *Numeric` and will not describe what is inside it, so there is no way to express these factors in
 * the shared packed format the dense side uses — a seam demanding a concrete `SparseLu` could never have
 * admitted a host solver at all. Here the opaque handle simply *is* the implementation's state.
 *
 * Holds its [matrix] alive on purpose: `umfpack_di_solve` takes `Ap`, `Ai` and `Ax` alongside the factors,
 * so dropping the matrix would leave the factorization unable to solve. The argument is required whether or
 * not iterative refinement is enabled — koblas disables it, see `UmfpackCalls.solveControl` — because the
 * signature has no way to say "no matrix".
 *
 * **Native memory.** The factors are UMFPACK's own allocation, so they outlive the JVM's heap accounting and
 * have to be released explicitly. A [Cleaner] does it when this object becomes unreachable, which keeps
 * `SparseFactorization` free of a `close()` that koblas's own portable factorization would have no use for.
 * The trade is that release is not deterministic: a program churning through large factorizations in a tight
 * loop holds native memory until GC notices. If that ever matters, the answer is an explicit `close` on this
 * class rather than on the interface.
 */
class UmfpackFactorization internal constructor(
    private val matrix: SparseMatrix,
    override val failedAt: Int,
    private val handles: Handles,
) : SparseFactorization {

    /**
     * The arena and the `void **Numeric` holder, split into their own object so the [Cleaner] action can
     * capture *them* and not the factorization.
     *
     * That is the whole reason this class exists: a cleanup lambda that closed over `UmfpackFactorization`
     * would keep it reachable forever and the factors would never be freed.
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

    /** Fill in `L` and `U`, read out of UMFPACK's `Info` at factorization time. */
    internal var lnz: Int = 0

    internal var unz: Int = 0

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        check(!singular) {
            "cannot solve against a singular factorization: umfpack reported the matrix singular. " +
                "Check `singular` before solving; repair the matrix and factor again."
        }
        requireShape(b.size == n) { "solve: b size ${b.size}, expected $n" }
        requireShape(out.size == n) { "solve: out size ${out.size}, expected $n" }
        // umfpack writes X and reads B, so aliasing them would have it read its own partial output.
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
     * `det(B)` from UMFPACK, which reports it as a mantissa and a base-10 exponent to survive the range a
     * sparse determinant can reach — the product of `n` pivots overflows a double long before `n` is large.
     *
     * Recombining them here can still overflow to infinity for a big matrix, which is the honest answer for
     * a value that does not fit; a caller needing more should read the pieces from UMFPACK directly.
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
        /** Reads the fill counts out of a completed factorization's `Info` array. */
        fun fillOf(info: MemorySegment): Pair<Int, Int> =
            info.get(JAVA_DOUBLE, INFO_LNZ * 8L).toInt() to info.get(JAVA_DOUBLE, INFO_UNZ * 8L).toInt()
    }
}
