package com.eignex.koblas.sparse.host.basiclu

import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.sparse.F64BasisFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.requireSolveShapes
import com.eignex.koblas.withColumn
import java.lang.ref.Reference
import kotlin.math.abs

/** A host BASICLU factorization whose native object is reclaimed when it becomes unreachable. */
public open class BasicluFactorization internal constructor(
    internal val target: BasicluObject,
    internal val calls: BasicluCalls,
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
        val status = try {
            calls.solve(target, out, transpose)
        } finally {
            Reference.reachabilityFence(this)
        }
        check(status == BasicluStatus.OK) { "BASICLU solve failed with status $status" }
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
    override var basis: F64SparseMatrix = initialBasis
        private set

    override fun replaceColumn(column: Int, entering: F64SparseVector): F64BasisFactorization {
        val next = basis.withColumn(column, entering)
        val status = try {
            calls.replaceColumn(target, column, entering.indices, entering.values)
        } finally {
            Reference.reachabilityFence(this)
        }
        if (status != BasicluStatus.OK) return owner.factorBasis(next)
        basis = next
        return this
    }
}
