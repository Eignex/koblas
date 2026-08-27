package com.eignex.koblas.sparse.host.hfactor

import com.eignex.koblas.*
import com.eignex.koblas.internal.host.NativeResourceLifecycle
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorizationReport
import com.eignex.koblas.sparse.basicReport
import com.eignex.koblas.sparse.basis.F64IndexedVector
import com.eignex.koblas.sparse.requireSolveShapes
import java.lang.foreign.MemorySegment
import java.lang.ref.Reference

/**
 * A general square matrix factored by HFactor, which takes it as a basis of its own columns.
 *
 * HFactor solves through vectors it owns, so one factorization serves one thread, as a host basis
 * factorization does. Where a caller pivots rather than solving one matrix, [HfactorBasisSolver] is what
 * HFactor is for and this is the plainer surface beside it.
 */
public class HfactorFactorization internal constructor(
    override val n: Int,
    private val calls: HfactorCalls,
    private val handle: MemorySegment,
) : F64SparseFactorization {
    private class Release(private val calls: HfactorCalls, private val handle: MemorySegment) {
        fun release(): Unit = calls.free(handle)
    }

    private val lifecycle = NativeResourceLifecycle("HFactor factorization", Release(calls, handle)::release)
    private val cleanable = nativeCleaner.register(this, lifecycle)

    private val carrier = F64IndexedVector(n)
    private val pivotRange = DoubleArray(2)

    /** Always [NOT_SINGULAR]: this exists only for a matrix HFactor factored at full rank. */
    override val failedAt: Int get() = NOT_SINGULAR

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability = noManagedAllocation

    override fun report(): F64SparseFactorizationReport = basicReport("hfactor")

    override val nnz: Int get() = lifecycle.withResource {
        try {
            calls.fill(handle)
        } finally {
            Reference.reachabilityFence(this)
        }
    }

    /** Reaching the pivots copies the whole factorization, so this is sampled rather than polled. */
    override val rcond: Double get() = lifecycle.withResource {
        try {
            calls.pivotRange(handle, pivotRange)
            if (pivotRange[1] == 0.0) 0.0 else pivotRange[0] / pivotRange[1]
        } finally {
            Reference.reachabilityFence(this)
        }
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        lifecycle.withResource {
            requireSolveShapes(n, b, out)
            carrier.scatter(b)
            carrier.count = try {
                calls.solve(handle, carrier.count, carrier.indices, carrier.values, DENSE, transpose)
            } finally {
                Reference.reachabilityFence(this)
            }
            carrier.gather(out)
        }

    override fun close(): Unit = cleanable.clean()

    private companion object {
        /** A dense right-hand side is what a general solve is handed, so the sweeps are chosen for one. */
        const val DENSE = 1.0
    }
}
