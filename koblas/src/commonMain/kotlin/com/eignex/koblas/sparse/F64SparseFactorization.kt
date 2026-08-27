package com.eignex.koblas.sparse

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.requireSquare

/**
 * A factorization held for reuse against further right-hand sides.
 *
 * Close a factorization when it is no longer needed. Portable implementations own no external resource and
 * use the default no-op [close]. Native implementations release their factors deterministically; closing
 * them is idempotent, and subsequent reads of [nnz] or [rcond] and calls to [solveInto] throw
 * [IllegalStateException]. [n], [failedAt], and [singular] remain available after close.
 */
public interface F64SparseFactorization : AutoCloseable {
    /** The dimension of the factored matrix. */
    public val n: Int

    /**
     * The pivot position that had no numerically acceptable candidate, or [NOT_SINGULAR] when the
     * factorization succeeded.
     */
    public val failedAt: Int

    /**
     * Whether the factorization failed for want of a numerically acceptable pivot. Solving against one
     * throws [com.eignex.koblas.SingularMatrix] rather than answering with infinities.
     */
    public val singular: Boolean get() = failedAt != NOT_SINGULAR

    /** Nonzeros in the factors, the fill. Zero for a singular factorization, which has none. */
    public val nnz: Int

    /**
     * A cheap pivot-quality estimate: `min(abs(U(k, k))) / max(abs(U(k, k)))`. A small value warns that
     * the factorization may be inaccurate; it is not a reciprocal condition-number estimate.
     */
    public val rcond: Double

    /**
     * Allocation behavior of [solveInto] for this factorization and argument shape. The default is
     * conservative for third-party implementations that have not declared a contract.
     */
    public fun solveAllocation(aliasing: Boolean = true, transpose: Boolean = false): AllocationCapability =
        unrestrictedAllocation

    /**
     * Solve `B x = b`, or `Bᵀ x = b` when [transpose], into [out], which is returned. [out] may be [b].
     * [solveAllocation] reports the precise managed scratch and intrinsic allocation guarantee; use the
     * strict overload below when the contract must be enforced before execution.
     */
    public fun solveInto(
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): DoubleArray

    /** Solve every column of [b] into a fresh dense result. */
    public fun solve(b: F64DenseMatrix, transpose: Boolean = false): F64DenseMatrix {
        requireShape(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
        return solveInto(b, F64DenseMatrix(n, b.cols), transpose)
    }

    /**
     * Solve every right-hand-side column of [b] into [out]. The default is alias-safe and calls the vector
     * solve once per column; providers with a block ABI override it with one foreign call.
     */
    public fun solveInto(
        b: F64DenseMatrix,
        out: F64DenseMatrix,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): F64DenseMatrix {
        requireBlockSolveShapes(n, b, out)
        if (b.cols == 0) return out
        workspace.borrow(n) { rhs ->
            workspace.borrow(n) { solved ->
                for (column in 0 until b.cols) {
                    b.data.copyInto(rhs, 0, column * n, (column + 1) * n)
                    solveInto(rhs, solved, transpose, workspace)
                    solved.copyInto(out.data, column * n, 0, n)
                }
            }
        }
        return out
    }

    /**
     * Strict form of [solveInto]. Rejects [allocationPolicy] before mutation when the declared capability or
     * currently idle [workspace] buffers cannot honor it.
     */
    public fun solveInto(
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
        allocationPolicy: AllocationPolicy,
    ): DoubleArray {
        requireSolveShapes(n, b, out)
        val capability = solveAllocation(b === out, transpose)
        if (!capability.supports(allocationPolicy, workspace)) {
            throw AllocationPolicyRejectedException(allocationPolicy, capability)
        }
        return solveInto(b, out, transpose, workspace)
    }

    /** Solve `B x = b`, or `Bᵀ x = b` when [transpose], into a fresh result. */
    public fun solve(b: DoubleArray, transpose: Boolean = false): DoubleArray = solveInto(b, DoubleArray(n), transpose)

    /** Common and provider-specific diagnostics sampled from this factorization. */
    public fun report(): F64SparseFactorizationReport = basicReport(provider = "unknown")

    /** Releases resources owned by this factorization. Portable implementations have nothing to release. */
    override fun close() {}
}

/**
 * A sparse factorization of a simplex basis that can follow a replacement of one basis column.
 *
 * [replaceColumn] returns the factorization of the resulting basis, which supersedes this factorization. An
 * implementation may update its factors directly or rebuild them when the replacement would be too dense or
 * numerically unsafe to update.
 */
public interface F64BasisFactorization : F64SparseFactorization {
    /** The square basis matrix represented by this factorization. */
    public val basis: F64SparseMatrix

    /**
     * Replace [column] of [basis] with [entering] and return the factorization of the resulting basis, which
     * supersedes this factorization.
     *
     * [column] must name a column of [basis], and [entering] must have length [n].
     */
    public fun replaceColumn(column: Int, entering: F64SparseVector): F64BasisFactorization
}

/** What [F64SparseDecompositions.factor] returns when no numerically acceptable pivot remains. */
public class F64SingularSparseFactorization(override val n: Int, override val failedAt: Int) : F64SparseFactorization {
    override val nnz: Int get() = 0

    override val rcond: Double get() = 0.0

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability =
        noManagedOrNativeAllocation

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        throw singularFailure(failedAt, "solve")
}

/** The shapes [F64SparseFactorization.solveInto] requires of its right-hand side and its destination. */
public fun requireSolveShapes(n: Int, b: DoubleArray, out: DoubleArray) {
    requireShape(b.size == n) { "solve: b size ${b.size}, expected $n" }
    requireShape(out.size == n) { "solve: out size ${out.size}, expected $n" }
}

/** The shapes required by a sparse block solve. */
public fun requireBlockSolveShapes(n: Int, b: F64DenseMatrix, out: F64DenseMatrix) {
    requireShape(b.rows == n) { "solve: B has ${b.rows} rows, expected $n" }
    requireShape(out.rows == n && out.cols == b.cols) {
        "solve: out is ${out.rows}x${out.cols}, expected ${n}x${b.cols}"
    }
}

/**
 * A basis factorization over any [F64SparseDecompositions], for a backend that cannot update its own factors. A
 * replacement refactorizes the basis it produces, so the factors stay exact at the cost of a factorization
 * per replacement.
 *
 * Public because it is what a caller wanting a basis factorization from a backend that does not offer one
 * builds it from. BASICLU offers one; nothing else koblas binds does.
 */
public class F64RefactoringBasisFactorization(
    private val lu: F64SparseDecompositions,
    override val basis: F64SparseMatrix,
    private val factors: F64SparseFactorization,
) : F64BasisFactorization {
    private var closed = false

    init {
        requireSquare(basis, "factorBasis")
    }

    override val n: Int get() = factors.n

    override val failedAt: Int get() = factors.failedAt

    override val nnz: Int get() {
        checkOpen()
        return factors.nnz
    }

    override val rcond: Double get() {
        checkOpen()
        return factors.rcond
    }

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability {
        checkOpen()
        return factors.solveAllocation(aliasing, transpose)
    }

    override fun report(): F64SparseFactorizationReport {
        checkOpen()
        return factors.report()
    }

    override fun replaceColumn(column: Int, entering: F64SparseVector): F64BasisFactorization {
        checkOpen()
        val next = basis.withColumn(column, entering)
        val replacement = F64RefactoringBasisFactorization(lu, next, lu.factor(next))
        close()
        return replacement
    }

    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        checkOpen()
        return factors.solveInto(b, out, transpose, workspace)
    }

    override fun close() {
        if (closed) return
        closed = true
        factors.close()
    }

    private fun checkOpen() {
        check(!closed) { "basis factorization is closed" }
    }
}
