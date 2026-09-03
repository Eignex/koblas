package com.eignex.koblas.sparse.host.hfactor

import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.NativeResourceLifecycle
import com.eignex.koblas.internal.host.keepingReachable
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.requireInBounds
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.basis.BasisUpdate
import com.eignex.koblas.sparse.basis.F64BasisSolveQuality
import com.eignex.koblas.sparse.basis.F64BasisSolver
import com.eignex.koblas.sparse.basis.F64IndexedVector
import com.eignex.koblas.sparse.basis.basisSolveQuality
import java.lang.foreign.MemorySegment

/**
 * A simplex basis held by HiGHS's HFactor: Markowitz factors, hypersparse solves that fall back to
 * conventional ones as the vectors fill, and Forrest-Tomlin updates.
 *
 * A solve crosses the seam without copying. [F64IndexedVector] stores what HFactor's own vector does, dense
 * values with the positions of the nonzeros beside them, so the call is handed those two arrays as they lie
 * and writes the result back over them.
 *
 * An update needs the entering column's forward solve and the pivotal row's transposed one, and needs them
 * as HFactor left them rather than as values alone. Both are what a dual simplex has just computed, so this
 * tracks which vector each of its solves last filled and reuses the native one where the caller hands the
 * same vector back. A caller solving in some other order is still correct; it pays the solve again.
 *
 * The tracking is by identity, which is what [F64BasisSolver.update] asks a caller for: a vector edited
 * between its solve and the update is read here as the solve left it, not as it now stands.
 */
@OptIn(UnsafeKoblasApi::class)
public class HfactorBasisSolver internal constructor(
    private val a: F64SparseMatrix,
    private val calls: HfactorCalls,
    private val handle: MemorySegment,
) : F64BasisSolver {
    private class Release(private val calls: HfactorCalls, private val handle: MemorySegment) {
        fun release(): Unit = calls.free(handle)
    }

    private val lifecycle = NativeResourceLifecycle("HFactor basis solver", Release(calls, handle)::release)
    private val cleanable = nativeCleaner.register(this, lifecycle)

    override val n: Int = a.rows

    private val columns = a.cols
    private val basicIndex = IntArray(n)
    private val pivotRange = DoubleArray(2)
    private var factorized = false
    private var lastFtran: F64IndexedVector? = null
    private var lastBtran: F64IndexedVector? = null

    override var singular: Boolean = true
        private set

    override val rcond: Double get() = lifecycle.withResource {
        if (!factorized || singular) return@withResource 0.0
        keepingReachable(this) {
            calls.pivotRange(handle, pivotRange)
            if (pivotRange[1] == 0.0) 0.0 else pivotRange[0] / pivotRange[1]
        }
    }

    override val updateCount: Int get() = lifecycle.withResource {
        keepingReachable(this) {
            if (factorized) calls.updateCount(handle) else 0
        }
    }

    override val nnz: Int get() = lifecycle.withResource {
        keepingReachable(this) {
            if (factorized) calls.fill(handle) else 0
        }
    }

    override fun refactorize(basicIndex: IntArray): Boolean = lifecycle.withResource {
        requireShape(basicIndex.size == n) { "refactorize: basicIndex size ${basicIndex.size} != $n" }
        for (t in 0 until n) requireInBounds(basicIndex[t], columns)
        val deficiency = keepingReachable(this) {
            calls.build(handle, basicIndex)
        }
        basicIndex.copyInto(this.basicIndex)
        forgetSolves()
        factorized = true
        /*
         * HFactor repairs a rank-deficient basis by substituting logicals for the dependent columns, which
         * is not a basis the caller asked for. It is reported as singular instead, matching the portable
         * solver, and the repair goes unused.
         */
        singular = deficiency != 0
        !singular
    }

    override fun ftran(x: F64IndexedVector, expectedDensity: Double): Unit = lifecycle.withResource {
        solveNative(x, expectedDensity, transpose = false)
        lastFtran = x
    }

    override fun btran(x: F64IndexedVector, expectedDensity: Double): Unit = lifecycle.withResource {
        solveNative(x, expectedDensity, transpose = true)
        lastBtran = x
    }

    override fun solveQuality(rhs: DoubleArray, solution: F64IndexedVector, transpose: Boolean): F64BasisSolveQuality =
        lifecycle.withResource {
            checkSolvable()
            basisSolveQuality(a, basicIndex, rhs, solution, transpose)
        }

    override fun update(
        pivotRow: Int,
        entering: Int,
        spike: F64IndexedVector,
        pivotEta: F64IndexedVector?,
    ): BasisUpdate = lifecycle.withResource {
        requireInBounds(pivotRow, n)
        requireInBounds(entering, columns)
        requireShape(spike.size == n) { "update: spike size ${spike.size} != $n" }
        if (!factorized || singular) return@withResource BasisUpdate.SINGULAR
        /*
         * Judged on the spike the caller passed rather than on the one HFactor may be about to recompute,
         * so an update is refused for the same inputs the portable solver refuses it for. The bridge checks
         * again on whatever it ends up with.
         */
        val pivot = spike[pivotRow]
        if (pivot == 0.0 || !pivot.isFinite()) return@withResource BasisUpdate.SINGULAR
        val advice = keepingReachable(this) {
            calls.update(handle, pivotRow, entering, spike === lastFtran, pivotEta != null && pivotEta === lastBtran)
        }
        // The update consumes both native vectors, so neither answers for a caller's vector afterwards.
        forgetSolves()
        when (advice) {
            HfactorUpdate.REFUSED -> BasisUpdate.SINGULAR

            else -> {
                basicIndex[pivotRow] = entering
                if (advice == HfactorUpdate.REFACTORIZE) BasisUpdate.REFACTORIZE else BasisUpdate.APPLIED
            }
        }
    }

    private fun solveNative(x: F64IndexedVector, expectedDensity: Double, transpose: Boolean) {
        checkSolvable()
        requireShape(x.size == n) { "solve: x size ${x.size} != $n" }
        x.count = keepingReachable(this) {
            calls.solve(handle, x.count, x.indices, x.values, expectedDensity, transpose)
        }
    }

    private fun checkSolvable() {
        if (!factorized) throw SingularMatrix(SINGULAR_POSITION_UNKNOWN, "solve: no basis has been factorized")
        if (singular) throw SingularMatrix(SINGULAR_POSITION_UNKNOWN, "solve: the basis is singular")
    }

    private fun forgetSolves() {
        lastFtran = null
        lastBtran = null
    }

    override fun close(): Unit = cleanable.clean()
}
