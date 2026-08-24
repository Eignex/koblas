package com.eignex.koblas.basiclu

import com.eignex.koblas.HOST_BACKEND_PRIORITY
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.SINGULAR_POSITION_UNKNOWN
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.internal.host.nativeCleaner
import com.eignex.koblas.sparse.F64BasisFactorization
import com.eignex.koblas.sparse.F64SingularSparseFactorization
import com.eignex.koblas.sparse.F64SparseFactorization
import com.eignex.koblas.sparse.host.F64HostSparseLuAdapter
import com.eignex.koblas.sparse.requireSolveShapes
import java.lang.ref.Reference
import kotlin.math.abs

/** Sparse LU and Forrest-Tomlin basis updates backed by BASICLU. */
@OptIn(UnsafeKoblasApi::class)
public class BasicluSparseLu(
    /** Absolute path to the BASICLU bridge library, or the platform lookup chain when null. */
    public val libraryPath: String? = null,
    /** Smallest stored-entry count routed to the native factorization; null keeps the platform default. */
    public val factorizeMin: Int? = null,
) : F64HostSparseLuAdapter(factorizeMin) {
    private val calls = BasicluCalls(libraryPath)

    override val name: String get() = "basiclu"
    override val priority: Int get() = HOST_BACKEND_PRIORITY + 2
    override val nativeAvailable: Boolean get() = calls.available

    override fun factorNative(a: F64SparseMatrix, equilibrate: Boolean): F64SparseFactorization =
        factorHandle(a)?.let { BasicluFactorization(a.rows, it, calls) }
            ?: F64SingularSparseFactorization(a.rows, SINGULAR_POSITION_UNKNOWN)

    /** Factor a simplex [basis] for sparse column replacements. */
    override fun factorBasis(basis: F64SparseMatrix): F64BasisFactorization {
        require(basis.rows == basis.cols) { "factorBasis requires a square matrix; got ${basis.rows}x${basis.cols}" }
        return factorHandle(basis)?.let { BasicluBasisFactorization(this, basis, it, calls) }
            ?: BasicluSingularBasisFactorization(this, basis)
    }

    private fun factorHandle(a: F64SparseMatrix): BasicluHandle? = calls.factorize(
        a.rows,
        LongArray(a.cols) { a.colPtr[it].toLong() },
        LongArray(a.cols) { a.colPtr[it + 1].toLong() },
        LongArray(a.rowIdx.size) { a.rowIdx[it].toLong() },
        a.values,
    )
}

private open class BasicluFactorization(
    override val n: Int,
    protected val handle: BasicluHandle,
    protected val calls: BasicluCalls,
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
        return if (out === b && workspace != null) {
            workspace.borrow(n) { rhs ->
                b.copyInto(rhs)
                solve(rhs, out, transpose)
            }
        } else {
            solve(if (out === b) b.copyOf() else b, out, transpose)
        }
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

@OptIn(UnsafeKoblasApi::class)
private fun F64SparseMatrix.withColumn(column: Int, entering: F64SparseVector): F64SparseMatrix {
    val oldStart = colPtr[column]
    val oldEnd = colPtr[column + 1]
    val delta = entering.indices.size - (oldEnd - oldStart)
    val pointers = IntArray(cols + 1)
    for (j in 0..cols) pointers[j] = colPtr[j] + if (j <= column) 0 else delta
    val rows = IntArray(rowIdx.size + delta)
    val values = DoubleArray(this.values.size + delta)
    rowIdx.copyInto(rows, endIndex = oldStart)
    this.values.copyInto(values, endIndex = oldStart)
    entering.indices.copyInto(rows, oldStart)
    entering.values.copyInto(values, oldStart)
    rowIdx.copyInto(rows, oldStart + entering.indices.size, oldEnd)
    this.values.copyInto(values, oldStart + entering.indices.size, oldEnd)
    return F64SparseMatrix.wrap(this.rows, cols, pointers, rows, values)
}
