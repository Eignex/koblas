package com.eignex.koblas.sparse

import com.eignex.koblas.AllocationCapability
import com.eignex.koblas.AllocationPolicy
import com.eignex.koblas.AllocationPolicyRejectedException
import com.eignex.koblas.Workspace
import com.eignex.koblas.borrow
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.requireShape
import com.eignex.koblas.unrestrictedAllocation

/**
 * A sparse QR factorization `A·P = Q·R` of a tall or square `A`, held for reuse.
 *
 * Square factors are [F64SparseFactorization]; this one is not. A QR is not unique, so [columnOrder] names
 * the ordering chosen and [r] is read against it. `Q` is an operator because it is `m×m` and dense in general.
 */
public interface F64SparseQrFactorization : AutoCloseable {
    /** Rows, and the length of a right-hand side. */
    public val m: Int

    /** Columns, and the length of a solution. */
    public val n: Int

    /** Columns with an acceptable diagonal in [r], by whatever tolerance the provider applies. */
    public val rank: Int

    /** Whether [rank] fell short of [n]. */
    public val rankDeficient: Boolean get() = rank < n

    /** Stored entries in the factors this implementation retains. */
    public val nnz: Int

    /** The upper triangular `n×n` factor, in this factorization's column ordering. */
    public val r: F64SparseMatrix

    /** The column of `A` that is column `k` here. */
    public val columnOrder: IntArray

    /** `y` through `Q`, or `Qᵀ` when [transpose], into [out]. Both have length [m]. */
    public fun applyQInto(
        y: DoubleArray,
        out: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): DoubleArray

    /** [applyQInto] into a fresh vector. */
    public fun applyQ(y: DoubleArray, transpose: Boolean = false): DoubleArray =
        applyQInto(y, DoubleArray(m), transpose)

    /** Allocation behavior of [solveInto]; conservative by default. */
    public fun solveAllocation(): AllocationCapability = unrestrictedAllocation

    /** `min ‖A·x − b‖₂` into [out]. A deficient `A` raises [com.eignex.koblas.SingularMatrix] here, not everywhere. */
    public fun solveInto(b: DoubleArray, out: DoubleArray, workspace: Workspace? = null): DoubleArray

    /** [solveInto] into a fresh vector. */
    public fun solve(b: DoubleArray, workspace: Workspace? = null): DoubleArray =
        solveInto(b, DoubleArray(n), workspace)

    /** [solveInto], rejecting [allocationPolicy] before mutation. */
    public fun solveInto(
        b: DoubleArray,
        out: DoubleArray,
        workspace: Workspace? = null,
        allocationPolicy: AllocationPolicy,
    ): DoubleArray {
        requireLeastSquaresShapes(m, n, b, out)
        val capability = solveAllocation()
        if (!capability.supports(allocationPolicy, workspace)) {
            throw AllocationPolicyRejectedException(allocationPolicy, capability)
        }
        return solveInto(b, out, workspace)
    }

    /** Solve every column of [b] into [out]. */
    public fun solveInto(b: F64DenseMatrix, out: F64DenseMatrix, workspace: Workspace? = null): F64DenseMatrix {
        requireLeastSquaresBlockShapes(m, n, b, out)
        if (b.cols == 0) return out
        workspace.borrow(m) { rhs ->
            workspace.borrow(n) { solved ->
                for (column in 0 until b.cols) {
                    b.data.copyInto(rhs, 0, column * m, (column + 1) * m)
                    solveInto(rhs, solved, workspace)
                    solved.copyInto(out.data, column * n, 0, n)
                }
            }
        }
        return out
    }

    /** Solve every column of [b] into a fresh dense result. */
    public fun solve(b: F64DenseMatrix, workspace: Workspace? = null): F64DenseMatrix {
        requireShape(b.rows == m) { "solve: B has ${b.rows} rows, expected $m" }
        return solveInto(b, F64DenseMatrix(n, b.cols), workspace)
    }

    /** Common and provider-specific diagnostics. */
    public fun report(): F64SparseFactorizationReport = F64SparseFactorizationReport(
        provider = "unknown",
        order = n,
        factorNonzeros = nnz,
        reciprocalPivotRange = null,
        columnPermutation = columnOrder.toList(),
        details = mapOf("rows" to m.toString(), "rank" to rank.toString()),
    )

    /** Releases resources owned by this factorization. */
    override fun close() {}
}

/** The shapes a least-squares solve requires. */
public fun requireLeastSquaresShapes(m: Int, n: Int, b: DoubleArray, out: DoubleArray) {
    requireShape(b.size == m) { "solve: b size ${b.size}, expected $m" }
    requireShape(out.size == n) { "solve: out size ${out.size}, expected $n" }
}

/** The shapes a blocked least-squares solve requires. */
public fun requireLeastSquaresBlockShapes(m: Int, n: Int, b: F64DenseMatrix, out: F64DenseMatrix) {
    requireShape(b.rows == m) { "solve: B has ${b.rows} rows, expected $m" }
    requireShape(out.rows == n && out.cols == b.cols) {
        "solve: out is ${out.rows}x${out.cols}, expected ${n}x${b.cols}"
    }
}
