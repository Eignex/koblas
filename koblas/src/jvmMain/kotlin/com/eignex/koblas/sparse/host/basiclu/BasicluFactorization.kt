package com.eignex.koblas.sparse.host.basiclu

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.sparse.F64BasisFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.host.applyEquilibration
import com.eignex.koblas.sparse.internal.replaceColumns
import com.eignex.koblas.sparse.internal.snapshot
import com.eignex.koblas.sparse.requireSolveShapes
import java.lang.ref.Reference
import kotlin.math.abs

/** A host BASICLU factorization whose native object is reclaimed when it becomes unreachable. */
public open class BasicluFactorization internal constructor(
    internal val target: BasicluObject,
    internal val calls: BasicluCalls,
    /** The equilibration the values were scaled by before factorization, or null when there was none. */
    private val rowScale: DoubleArray? = null,
) : F64SparseFactorization {
    init {
        nativeCleaner.register(this, Release(calls, target))
    }

    private class Release(private val calls: BasicluCalls, private val target: BasicluObject) : Runnable {
        override fun run() {
            calls.free(target)
            target.arena.close()
        }
    }

    override val n: Int get() = target.n

    override val failedAt: Int get() = NOT_SINGULAR

    /** Reads BASICLU's own store, so the fence holds this factorization past the read. */
    override val nnz: Int get() = try {
        (calls.statistic(target, BasicluStore.LNZ) + calls.statistic(target, BasicluStore.UNZ)).toInt()
    } finally {
        Reference.reachabilityFence(this)
    }

    override val rcond: Double get() = try {
        val largest = abs(calls.statistic(target, BasicluStore.MAX_PIVOT))
        if (largest == 0.0) 0.0 else abs(calls.statistic(target, BasicluStore.MIN_PIVOT)) / largest
    } finally {
        Reference.reachabilityFence(this)
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireSolveShapes(n, b, out)
        // BASICLU solves through its own buffer, so the destination carries the right-hand side in.
        if (out !== b) b.copyInto(out)
        // The factors are of E·B, so a forward solve scales what goes in and a transposed one what comes out.
        if (rowScale != null && !transpose) applyEquilibration(out, rowScale)
        val status = try {
            calls.solve(target, out, transpose)
        } finally {
            Reference.reachabilityFence(this)
        }
        check(status == BasicluStatus.OK) { "BASICLU solve failed with status $status" }
        if (rowScale != null && transpose) applyEquilibration(out, rowScale)
        return out
    }
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
        val status = try {
            calls.replaceColumn(target, column, entering.indices, entering.values)
        } finally {
            Reference.reachabilityFence(this)
        }
        // Kept as a copy, since the caller's vector stays live for it to write and the build comes later.
        pending[column] = entering.snapshot()
        if (status != BasicluStatus.OK) return owner.factorBasis(basis)
        return this
    }
}
