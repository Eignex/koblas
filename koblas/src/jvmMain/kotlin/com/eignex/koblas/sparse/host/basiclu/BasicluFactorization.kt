package com.eignex.koblas.sparse.host.basiclu

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.internal.host.NativeResourceLifecycle
import com.eignex.koblas.internal.host.keepingReachable
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.requireSolveShapes
import com.eignex.koblas.sparse.F64BasisFactorization
import com.eignex.koblas.sparse.F64SparseLuFactorization
import com.eignex.koblas.sparse.FactorsNotExposed
import com.eignex.koblas.sparse.host.applyF64Equilibration
import com.eignex.koblas.sparse.internal.replaceColumns
import com.eignex.koblas.sparse.internal.snapshot
import kotlin.math.abs

/** A host BASICLU factorization with deterministic close and cleaner fallback for its native object. */
public open class BasicluFactorization internal constructor(
    internal val target: BasicluObject,
    internal val calls: BasicluCalls,
    /** The equilibration the values were scaled by before factorization, or null when there was none. */
    private val rowScale: DoubleArray? = null,
) : F64SparseLuFactorization {
    private class Release(private val calls: BasicluCalls, private val target: BasicluObject) {
        fun release() {
            try {
                calls.free(target)
            } finally {
                target.arena.close()
            }
        }
    }

    private val lifecycle = NativeResourceLifecycle("BASICLU factorization", Release(calls, target)::release)
    private val cleanable = nativeCleaner.register(this, lifecycle)

    override val n: Int get() = target.n

    override val failedAt: Int get() = NOT_SINGULAR

    override val l: F64SparseMatrix get() = factorNotExposed("l")

    override val u: F64SparseMatrix get() = factorNotExposed("u")

    override val rowOrder: IntArray get() = factorNotExposed("rowOrder")

    override val columnOrder: IntArray get() = factorNotExposed("columnOrder")

    override val rowScaling: DoubleArray get() = factorNotExposed("rowScaling")

    override val offDiagonal: F64SparseMatrix get() = factorNotExposed("offDiagonal")

    private fun factorNotExposed(factor: String): Nothing = withNativeFactor { throw FactorsNotExposed(factor) }

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability =
        noSizeDependentManagedAllocation

    /** Reads BASICLU's own store, so the fence holds this factorization past the read. */
    override val nnz: Int get() = withNativeFactor {
        (calls.statistic(target, BasicluStore.LNZ) + calls.statistic(target, BasicluStore.UNZ)).toInt()
    }

    override val rcond: Double get() = withNativeFactor {
        val largest = abs(calls.statistic(target, BasicluStore.MAX_PIVOT))
        if (largest == 0.0) 0.0 else abs(calls.statistic(target, BasicluStore.MIN_PIVOT)) / largest
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        withNativeFactor {
            requireSolveShapes(n, n, b, out)
            // BASICLU solves through its own buffer, so the destination carries the right-hand side in.
            if (out !== b) b.copyInto(out)
            // The factors are of E·B, so a forward solve scales what goes in and a transposed one what comes out.
            if (rowScale != null && !transpose) applyF64Equilibration(out, rowScale)
            val status = calls.solve(target, out, transpose)
            check(status == BasicluStatus.OK) { "BASICLU solve failed with status $status" }
            if (rowScale != null && transpose) applyF64Equilibration(out, rowScale)
            out
        }

    internal fun <T> withNativeFactor(block: () -> T): T = lifecycle.withResource {
        keepingReachable(this) {
            block()
        }
    }

    override fun close(): Unit = cleanable.clean()
}

/** A host BASICLU basis whose factors follow a column replacement until BASICLU declines the update. */
public class BasicluBasisFactorization internal constructor(
    private val owner: BasicluSparseLu,
    initialBasis: F64SparseMatrix,
    target: BasicluObject,
    calls: BasicluCalls,
) : BasicluFactorization(target, calls),
    F64BasisFactorization {
    private var built: F64SparseMatrix = initialBasis
    private val pending = LinkedHashMap<Int, F64SparseVector>()

    /**
     * The basis these factors stand for, built from the replacements BASICLU took when it is asked for and
     * not before. Building it copies the structure of the whole matrix and validates it again, which is
     * work per pivot that a driver pivoting to optimality asks for once, if at all.
     */
    override val basis: F64SparseMatrix
        get() {
            if (pending.isNotEmpty()) {
                built = replaceColumns(built, pending)
                pending.clear()
            }
            return built
        }

    override fun replaceColumn(column: Int, entering: F64SparseVector): F64BasisFactorization {
        val status = withNativeFactor {
            calls.replaceColumn(target, column, entering.indices, entering.values)
        }
        // Kept as a copy, since the caller's vector stays live for it to write and the build comes later.
        pending[column] = entering.snapshot()
        if (status != BasicluStatus.OK) {
            val replacement = owner.factorBasis(basis)
            close()
            return replacement
        }
        return this
    }
}
