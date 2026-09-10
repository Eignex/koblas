package com.eignex.koblas.sparse.host.hfactor

import com.eignex.koblas.*
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.hfactor.internal.NativeOwnership
import com.eignex.koblas.requireHfactorSolveShapes
import com.eignex.koblas.sparse.SparseLuFactorization
import com.eignex.koblas.sparse.basis.IndexedVector
import com.eignex.koblas.sparse.host.factorNotExposed
import java.lang.foreign.MemorySegment

/**
 * A general square matrix factored by HFactor, which takes it as a basis of its own columns.
 *
 * HFactor solves through vectors it owns, so one factorization serves one thread, as a host basis
 * factorization does. Where a caller pivots rather than solving one matrix, [HfactorBasisSolver] is what
 * HFactor is for and this is the plainer surface beside it.
 */
@OptIn(UnsafeKoblasApi::class)
public class HfactorFactorization internal constructor(
    override val n: Int,
    private val calls: HfactorCalls,
    private val handle: MemorySegment,
) : SparseLuFactorization {
    private class Release(private val calls: HfactorCalls, private val handle: MemorySegment) {
        fun closeNative(): Unit = calls.free(handle)
    }

    private val ownership = NativeOwnership(this, "HFactor factorization", Release(calls, handle)::closeNative)

    private val carrier = IndexedVector(n)
    private val pivotRange = DoubleArray(2)

    override val failedAt: Int get() = NOT_SINGULAR

    override val l: SparseMatrix get() = factorNotExposed("l")

    override val u: SparseMatrix get() = factorNotExposed("u")

    override val rowOrder: IntArray get() = factorNotExposed("rowOrder")

    override val columnOrder: IntArray get() = factorNotExposed("columnOrder")

    override val rowScaling: DoubleArray get() = factorNotExposed("rowScaling")

    override val offDiagonal: SparseMatrix get() = factorNotExposed("offDiagonal")

    private fun factorNotExposed(factor: String): Nothing = ownership.factorNotExposed(factor)

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability =
        AllocationCapability(AllocationGuarantee.NO_SIZE_DEPENDENT_MANAGED)

    override val nnz: Int get() = ownership.anchoring {
        calls.fill(handle)
    }

    // Reading pivots copies the native factors, so callers should sample rather than poll this.
    override val rcond: Double get() = ownership.anchoring {
        calls.pivotRange(handle, pivotRange)
        if (pivotRange[1] == 0.0) 0.0 else pivotRange[0] / pivotRange[1]
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        ownership.anchoring {
            requireHfactorSolveShapes(n, n, b, out)
            carrier.scatter(b)
            carrier.count = calls.solve(handle, carrier.count, carrier.indices, carrier.values, DENSE, transpose)
            carrier.gather(out)
        }

    override fun close(): Unit = ownership.close()

    private companion object {
        const val DENSE = 1.0
    }
}
