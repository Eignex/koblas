package com.eignex.koblas.basiclu

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.host.F64SparseLuAdapter
import com.eignex.koblas.sparse.host.applyEquilibration
import com.eignex.koblas.sparse.host.equilibrationScale
import com.eignex.koblas.sparse.host.scaledValues
import java.lang.ref.Reference
import kotlin.math.abs

/** Sparse LU and Forrest-Tomlin basis updates backed by BASICLU. */
@OptIn(UnsafeKoblasApi::class)
public class BasicluSparseLu(
    /** Absolute path to the BASICLU bridge library, or the platform lookup chain when null. */
    public val libraryPath: String? = null,
    /** Smallest stored-entry count routed to the native factorization; null keeps the platform default. */
    public val factorizeMin: Int? = null,
) : F64SparseLuAdapter(factorizeMin) {
    private val calls = BasicluCalls(libraryPath)

    override val name: String get() = "basiclu"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 2
    override val nativeAvailable: Boolean get() = calls.available

    /**
     * BASICLU offers no row scaling of its own, so equilibration is applied to the values handed over and
     * undone in the solves, by the same power-of-two factors the portable factorization uses.
     */
    override fun factorNative(a: F64SparseMatrix, equilibrate: Boolean): F64SparseFactorization {
        val scale = if (equilibrate) equilibrationScale(a.rows, a.rowIdx, a.values) else null
        val values = if (scale == null) a.values else scaledValues(a.rowIdx, a.values, scale)
        return factorHandle(a, values)?.let { BasicluFactorization(a.rows, it, calls, scale) }
            ?: F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)
    }

    override val supportsBasisUpdates: Boolean get() = true

    override fun factorBasis(basis: F64SparseMatrix): F64BasisFactorization {
        require(basis.rows == basis.cols) { "factorBasis requires a square matrix; got ${basis.rows}x${basis.cols}" }
        return factorHandle(basis, basis.values)?.let { BasicluBasisFactorization(this, basis, it, calls) }
            ?: BasicluSingularBasisFactorization(this, basis)
    }

    /** [values] comes in separately because an equilibrated factorization scales it before handing it over. */
    private fun factorHandle(a: F64SparseMatrix, values: DoubleArray): BasicluHandle? = calls.factorize(
        a.rows,
        LongArray(a.cols) { a.colPtr[it].toLong() },
        LongArray(a.cols) { a.colPtr[it + 1].toLong() },
        LongArray(a.rowIdx.size) { a.rowIdx[it].toLong() },
        values,
    )
}

private open class BasicluFactorization(
    override val n: Int,
    protected val handle: BasicluHandle,
    protected val calls: BasicluCalls,
    private val rowScale: DoubleArray? = null,
) : F64SparseFactorization {
    init {
        nativeCleaner.register(this, Release(calls, handle))
    }

    private class Release(private val calls: BasicluCalls, private val handle: BasicluHandle) : Runnable {
        override fun run() {
            calls.free(handle)
        }
    }

    override val failedAt: Int get() = NOT_SINGULAR

    /** Reads BASICLU's own store, so the fence holds this factorization past the read. */
    override val nnz: Int get() = try {
        (calls.info(handle, BASICLU_LNZ) + calls.info(handle, BASICLU_UNZ)).toInt()
    } finally {
        Reference.reachabilityFence(this)
    }

    override val rcond: Double
        get() = try {
            val largest = abs(calls.info(handle, BASICLU_MAX_PIVOT))
            if (largest == 0.0) 0.0 else abs(calls.info(handle, BASICLU_MIN_PIVOT)) / largest
        } finally {
            Reference.reachabilityFence(this)
        }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireSolveShapes(n, b, out)
        // The factors are of E·B, so a forward solve scales what goes in, which cannot touch the caller's
        // right-hand side, and a transposed one scales what comes out.
        if (rowScale != null && !transpose) {
            return workspace.borrow(n) { rhs ->
                b.copyInto(rhs)
                applyEquilibration(rhs, rowScale)
                solve(rhs, out, transpose)
            }
        }
        // BASICLU reads the right-hand side and writes the destination, so an aliased pair needs a separate
        // buffer for one of them.
        val solved = if (out !== b) {
            solve(b, out, transpose)
        } else {
            workspace.borrow(n) { rhs ->
                b.copyInto(rhs)
                solve(rhs, out, transpose)
            }
        }
        if (rowScale != null) applyEquilibration(solved, rowScale)
        return solved
    }

    private fun solve(rhs: DoubleArray, out: DoubleArray, transpose: Boolean): DoubleArray {
        try {
            calls.solve(handle, rhs, out, transpose)
        } finally {
            Reference.reachabilityFence(this)
        }
        return out
    }
}

private class BasicluBasisFactorization(
    private val owner: BasicluSparseLu,
    override var basis: F64SparseMatrix,
    handle: BasicluHandle,
    calls: BasicluCalls,
) : BasicluFactorization(basis.rows, handle, calls),
    F64BasisFactorization {
    @OptIn(UnsafeKoblasApi::class)
    override fun replaceColumn(column: Int, entering: F64SparseVector): F64BasisFactorization {
        require(column in 0 until n) { "replaceColumn: column $column out of [0,$n)" }
        require(entering.size == n) { "replaceColumn: entering size ${entering.size}, expected $n" }
        val next = basis.withColumn(column, entering)
        val status = try {
            calls.replaceColumn(
                handle,
                LongArray(entering.indices.size) { entering.indices[it].toLong() },
                entering.values,
                column,
            )
        } finally {
            Reference.reachabilityFence(this)
        }
        return if (status == BASICLU_OK) {
            basis = next
            this
        } else {
            owner.factorBasis(next)
        }
    }
}

private class BasicluSingularBasisFactorization(
    private val owner: BasicluSparseLu,
    override val basis: F64SparseMatrix,
) : F64BasisFactorization {
    override val n: Int get() = basis.rows
    override val failedAt: Int get() = SINGULAR_POSITION_UNKNOWN
    override val nnz: Int get() = 0
    override val rcond: Double get() = 0.0

    override fun replaceColumn(column: Int, entering: F64SparseVector): F64BasisFactorization =
        owner.factorBasis(basis.withColumn(column, entering))

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        throw SingularMatrix(failedAt, "solve: the factorization is singular")
}
