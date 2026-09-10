@file:Suppress("UndocumentedPublicFunction", "UndocumentedPublicProperty")

package com.eignex.koblas.sparse.host.hfactor

import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.hfactor.internal.NativeOwnership
import com.eignex.koblas.requireHfactorIndex
import com.eignex.koblas.requireHfactorShape
import com.eignex.koblas.sparse.basis.BasisKernel
import com.eignex.koblas.sparse.basis.BasisRepair
import com.eignex.koblas.sparse.basis.BasisSnapshot
import com.eignex.koblas.sparse.basis.BasisSolveQuality
import com.eignex.koblas.sparse.basis.BasisUpdate
import com.eignex.koblas.sparse.basis.IndexedVector
import com.eignex.koblas.sparse.basis.RefactorizeReason
import com.eignex.koblas.sparse.basis.basisSolveQuality
import java.lang.foreign.MemorySegment

/**
 * A simplex basis held by HiGHS's HFactor: Markowitz factors, hypersparse solves that fall back to
 * conventional ones as the vectors fill, and Forrest-Tomlin updates.
 *
 * A solve crosses the seam without copying. [IndexedVector] stores what HFactor's own vector does, dense
 * values with the positions of the nonzeros beside them, so the call is handed those two arrays as they lie
 * and writes the result back over them.
 *
 * An update needs the entering column's forward solve and the pivotal row's transposed one, and needs them
 * as HFactor left them rather than as values alone. Both are what a dual simplex has just computed, so this
 * tracks which vector each of its solves last filled and reuses the native one where the caller hands the
 * same vector back. A caller solving in some other order is still correct; it pays the solve again.
 *
 * The tracking is by identity: a vector edited
 * between its solve and the update is read here as the solve left it, not as it now stands.
 */
@OptIn(UnsafeKoblasApi::class)
public class HfactorBasisSolver internal constructor(
    private val a: SparseMatrix,
    private val calls: HfactorCalls,
    private val handle: MemorySegment,
    private val rowScale: DoubleArray? = null,
) : AutoCloseable {
    private class Release(private val calls: HfactorCalls, private val handle: MemorySegment) {
        fun closeNative(): Unit = calls.free(handle)
    }

    private val ownership = NativeOwnership(this, "HFactor basis solver", Release(calls, handle)::closeNative)

    public val n: Int = a.rows

    private val columns = a.cols
    private val basicIndex = IntArray(n)
    private val pivotRange = DoubleArray(2)
    private var factorized = false

    /**
     * Which slots a repair filled with unit columns, or null while the basis is entirely columns of `A`.
     * [basicIndex] names nothing at those slots, so the residual check reads them from here instead.
     */
    private var unitRows: IntArray? = null
    private var lastFtran: IndexedVector? = null
    private var lastBtran: IndexedVector? = null

    public var singular: Boolean = true
        private set

    public val rcond: Double get() = ownership.anchoring {
        if (!factorized || singular) return@anchoring 0.0
        calls.pivotRange(handle, pivotRange)
        if (pivotRange[1] == 0.0) 0.0 else pivotRange[0] / pivotRange[1]
    }

    public val updateCount: Int get() = ownership.anchoring {
        if (factorized) calls.updateCount(handle) else 0
    }

    public val nnz: Int get() = ownership.anchoring {
        if (factorized) calls.fill(handle) else 0
    }

    public val refactorizeReason: RefactorizeReason? get() = ownership.anchoring {
        when (calls.refactorizeReason(handle)) {
            1 -> RefactorizeReason.FACTOR_ASKED
            2 -> RefactorizeReason.UPDATES_WORN
            else -> null
        }
    }

    public val kernel: BasisKernel? get() = ownership.anchoring {
        if (!factorized) return@anchoring null
        val sizes = IntArray(2)
        if (!calls.kernel(handle, sizes)) return@anchoring null
        BasisKernel(sizes[0], sizes[1])
    }

    public fun refactorize(basicIndex: IntArray): Boolean = ownership.anchoring {
        requireHfactorShape(basicIndex.size == n) { "refactorize: basicIndex size ${basicIndex.size} != $n" }
        for (t in 0 until n) {
            requireHfactorIndex(basicIndex[t] in 0 until columns) {
                "index ${basicIndex[t]} outside [0,$columns)"
            }
        }
        val deficiency = calls.build(handle, basicIndex)

        basicIndex.copyInto(this.basicIndex)
        unitRows = null
        forgetSolves()
        factorized = true
        /*
         * HFactor repairs a rank-deficient basis by substituting logicals for the dependent columns, which
         * is not a basis the caller asked for. It is reported as singular here, matching the portable
         * solver; [refactorizeRepairing] is where a caller asks to keep the repair instead.
         */
        singular = deficiency != 0
        !singular
    }

    /**
     * HFactor completes a rank-deficient factorization with unit pivots rather than abandoning it, so the
     * factors it leaves invert a basis of its own choosing. [refactorize] refuses that basis to match the
     * portable solver; this one keeps it and says what it is.
     *
     * The columns HFactor substitutes are numbered past the constraint matrix, since they are the slacks a
     * simplex would hold and not columns of `A`. They are translated to the seam's own reading here: the
     * slot names no column and carries the row its unit column stands for.
     */
    public fun refactorizeRepairing(basicIndex: IntArray): BasisRepair? = ownership.anchoring {
        requireHfactorShape(basicIndex.size == n) { "refactorize: basicIndex size ${basicIndex.size} != $n" }
        for (t in 0 until n) {
            requireHfactorIndex(basicIndex[t] in 0 until columns) {
                "index ${basicIndex[t]} outside [0,$columns)"
            }
        }
        val settled = IntArray(n)
        val deficiency = calls.buildRepairing(handle, basicIndex, settled)
        if (deficiency == null) {
            if (!refactorize(basicIndex)) return@anchoring null
            return@anchoring BasisRepair(basicIndex.copyOf(), IntArray(n) { -1 })
        }

        settled.copyInto(this.basicIndex)
        unitRows = if (deficiency == 0) null else rowsOfUnitColumns(settled)
        forgetSolves()
        factorized = true
        singular = false
        val rowsOf = unitRows ?: IntArray(n) { -1 }
        check(deficiency == 0 || rowsOf.any { it >= 0 }) { "HFactor reported a repair it did not make" }
        BasisRepair(IntArray(n) { if (rowsOf[it] >= 0) -1 else settled[it] }, rowsOf)
    }

    /**
     * The factors are of `E·A`, so a forward solve scales what goes in: `(E·B)` applied to `E·x` is `B` applied
     * to `x`, and the answer comes back in the caller's own numbers.
     */
    public fun ftran(x: IndexedVector, expectedDensity: Double = 1.0): Unit = ownership.anchoring {
        if (rowScale != null) scaleStored(x)
        solveNative(x, expectedDensity, transpose = false)
        lastFtran = x
    }

    /**
     * The transposed counterpart, which scales its result instead. HFactor's own vector keeps the answer in
     * the scaled space it factored, which is what [update] needs back from it, so only the caller's copy moves.
     */
    public fun btran(x: IndexedVector, expectedDensity: Double = 1.0): Unit = ownership.anchoring {
        solveNative(x, expectedDensity, transpose = true)
        if (rowScale != null) scaleStored(x)
        lastBtran = x
    }

    private fun scaleStored(x: IndexedVector) {
        val scale = rowScale ?: return
        for (k in 0 until x.count) {
            val row = x.indices[k]
            x.values[row] *= scale[row]
        }
    }

    public fun solveQuality(rhs: DoubleArray, solution: IndexedVector, transpose: Boolean = false): BasisSolveQuality =
        ownership.anchoring {
            checkSolvable()
            basisSolveQuality(a, basicIndex, unitRows, rhs, solution, transpose)
        }

    public fun update(
        pivotRow: Int,
        entering: Int,
        spike: IndexedVector,
        pivotEta: IndexedVector? = null,
    ): BasisUpdate = ownership.anchoring {
        requireHfactorIndex(pivotRow in 0 until n) { "index $pivotRow outside [0,$n)" }
        requireHfactorIndex(entering in 0 until columns) { "index $entering outside [0,$columns)" }
        requireHfactorShape(spike.size == n) { "update: spike size ${spike.size} != $n" }
        if (!factorized || singular) return@anchoring BasisUpdate.SINGULAR
        /*
         * Judged on the spike the caller passed rather than on the one HFactor may be about to recompute,
         * so an update is refused for the same inputs the portable solver refuses it for. The bridge checks
         * again on whatever it ends up with.
         */
        val pivot = spike[pivotRow]
        if (pivot == 0.0 || !pivot.isFinite()) return@anchoring BasisUpdate.SINGULAR
        val advice =
            calls.update(
                handle,
                pivotRow,
                entering,
                spike === lastFtran,
                pivotEta != null && pivotEta === lastBtran,
            )

        // The native update consumes its solve vectors, so neither remains reusable afterwards.
        forgetSolves()
        when (advice) {
            UPDATE_REFUSED -> BasisUpdate.SINGULAR

            else -> {
                basicIndex[pivotRow] = entering
                if (advice == UPDATE_REFACTORIZE) BasisUpdate.REFACTORIZE else BasisUpdate.APPLIED
            }
        }
    }

    private fun rowsOfUnitColumns(settled: IntArray): IntArray =
        IntArray(n) { if (settled[it] < columns) -1 else settled[it] - columns }

    private fun solveNative(x: IndexedVector, expectedDensity: Double, transpose: Boolean) {
        checkSolvable()
        requireHfactorShape(x.size == n) { "solve: x size ${x.size} != $n" }
        x.count = calls.solve(handle, x.count, x.indices, x.values, expectedDensity, transpose)
    }

    private fun checkSolvable() {
        if (!factorized) throw SingularMatrix(SINGULAR_POSITION_UNKNOWN, "solve: no basis has been factorized")
        if (singular) throw SingularMatrix(SINGULAR_POSITION_UNKNOWN, "solve: the basis is singular")
    }

    private fun forgetSolves() {
        lastFtran = null
        lastBtran = null
    }

    /**
     * A factorization HFactor set aside, with the two things its own representation does not carry: the
     * basis this binding reports and which of its slots a repair filled with unit columns.
     */
    private inner class NativeSnapshot(val pointer: MemorySegment, val basis: IntArray, val repair: IntArray?) :
        BasisSnapshot {
        private var released = false

        override fun close() {
            if (released) return
            released = true
            live.remove(this)
            calls.freeSnapshot(pointer)
        }
    }

    private val live = mutableSetOf<NativeSnapshot>()

    public fun snapshot(): BasisSnapshot? = ownership.anchoring {
        if (!factorized || singular) return@anchoring null
        val taken = calls.snapshot(handle) ?: return@anchoring null
        NativeSnapshot(taken, basicIndex.copyOf(), unitRows?.copyOf()).also { live.add(it) }
    }

    public fun restore(snapshot: BasisSnapshot): Boolean = ownership.anchoring {
        // Only snapshots still owned by this solver describe its native factorization.
        val native = snapshot as? NativeSnapshot ?: return@anchoring false
        if (native !in live) return@anchoring false
        if (!calls.restore(handle, native.pointer)) return@anchoring false
        native.basis.copyInto(basicIndex)
        unitRows = native.repair?.copyOf()
        forgetSolves()
        factorized = true
        singular = false
        true
    }

    override fun close() {
        for (snapshot in live.toList()) snapshot.close()
        ownership.close()
    }
}

private const val UPDATE_REFUSED = -1
private const val UPDATE_REFACTORIZE = 1
