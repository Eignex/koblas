@file:Suppress("UndocumentedPublicFunction", "UndocumentedPublicProperty")

package com.eignex.koblas.sparse

import com.eignex.koblas.*

/** A caller-owned HFactor general sparse LU factorization. */
public interface SparseFactorization : AutoCloseable {
    public val n: Int
    public val failedAt: Int
    public val singular: Boolean get() = failedAt != NOT_SINGULAR
    public val nnz: Int
    public val rcond: Double
    public fun solveAllocation(aliasing: Boolean = true, transpose: Boolean = false): AllocationCapability =
        unrestrictedAllocation
    public fun solveInto(
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): DoubleArray
    public fun solve(b: DenseMatrix, transpose: Boolean = false): DenseMatrix =
        solveInto(b, DenseMatrix(n, b.cols), transpose)
    public fun solveInto(
        b: DenseMatrix,
        out: DenseMatrix,
        transpose: Boolean = false,
        workspace: Workspace? = null,
    ): DenseMatrix {
        requireSolveShapes(n, n, b, out)
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
    public fun solveInto(
        b: DoubleArray,
        out: DoubleArray,
        transpose: Boolean = false,
        workspace: Workspace? = null,
        allocationPolicy: AllocationPolicy,
    ): DoubleArray {
        requireSolveShapes(n, n, b, out)
        val capability = solveAllocation(b === out, transpose)
        if (!capability.supports(allocationPolicy, workspace)) {
            throw AllocationPolicyRejectedException(allocationPolicy, capability)
        }
        return solveInto(b, out, transpose, workspace)
    }
    public fun solve(b: DoubleArray, transpose: Boolean = false): DoubleArray = solveInto(b, DoubleArray(n), transpose)
    override fun close() {}
}

/** Registry compatibility retained for HFactor removal sequencing. */
public interface BasisFactorization : SparseLuFactorization

/** A singular result from HFactor, which exposes no factors and cannot solve. */
public class SingularSparseFactorization(override val n: Int, override val failedAt: Int) : SparseLuFactorization {
    override val nnz: Int get() = 0
    override val rcond: Double get() = 0.0
    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability =
        noManagedOrNativeAllocation
    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray =
        throw singularFailure(failedAt, "solve")
}
