package com.eignex.koblas.sparse.factorization.cholesky

import com.eignex.koblas.AllocationCapability
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.NotPositiveDefinite
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.noManagedOrNativeAllocation
import com.eignex.koblas.requireSquare
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.F64SparseCholeskyFactorization
import com.eignex.koblas.sparse.F64SparseFactorizationReport
import com.eignex.koblas.sparse.factorization.columnPointers
import com.eignex.koblas.sparse.factorization.eliminationTree
import com.eignex.koblas.sparse.factorization.ereach
import com.eignex.koblas.sparse.internal.transposeCsc
import com.eignex.koblas.sparse.requireSolveShapes
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Sparse Cholesky factorization `A = L·Lᵀ` of a symmetric positive-definite matrix. The factor [l] is
 * lower triangular in CSC with the diagonal entry first in each column, which is where both solve sweeps
 * look for it.
 *
 * The ordering is the one the matrix arrives in. A fill-reducing permutation is what separates this from a
 * fast sparse Cholesky, and it is also what a caller cannot check: this is the semantic definition the
 * bindings are compared against, so it stays the arithmetic and nothing else.
 */
public class F64SparseUpLookingCholesky internal constructor(
    override val n: Int,
    private val colPtr: IntArray,
    private val rowIdx: IntArray,
    private val values: DoubleArray,
) : F64SparseCholeskyFactorization {

    override val l: F64SparseMatrix
        get() = F64SparseMatrix.wrap(n, n, colPtr.copyOf(), rowIdx.copyOf(), values.copyOf())

    /** The identity: this factorization reorders nothing, for the reason its own documentation gives. */
    override val order: IntArray get() = IntArray(n) { it }

    /** Always [NOT_SINGULAR]: a non-positive pivot is raised rather than recorded, so this only exists for a
     *  matrix that factored completely. */
    override val failedAt: Int get() = NOT_SINGULAR

    override val nnz: Int get() = values.size

    /** `min |L(k, k)| / max |L(k, k)|`, which for `A = L·Lᵀ` is what the seam documents over `U`. */
    override val rcond: Double
        get() {
            if (n == 0) return 1.0
            var minimum = Double.POSITIVE_INFINITY
            var maximum = 0.0
            for (k in 0 until n) {
                val magnitude = abs(values[colPtr[k]])
                minimum = minOf(minimum, magnitude)
                maximum = maxOf(maximum, magnitude)
            }
            return if (maximum == 0.0) 0.0 else minimum / maximum
        }

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability =
        noManagedOrNativeAllocation

    override fun report(): F64SparseFactorizationReport = F64SparseFactorizationReport(
        provider = F64ReferenceSparseLinearAlgebra.name,
        order = n,
        factorNonzeros = nnz,
        reciprocalPivotRange = rcond,
        ordering = "natural",
    )

    /**
     * Solve `A x = b` into [out], which is returned and may be [b]. [transpose] is accepted and ignored: `A`
     * is symmetric, so the transposed system is the same one. Allocates nothing, [workspace] or not, since
     * both sweeps run in the destination.
     */
    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        requireSolveShapes(n, b, out)
        if (out !== b) b.copyInto(out)
        // L y = b, forward over the columns.
        for (k in 0 until n) {
            val diagonal = colPtr[k]
            val yk = out[k] / values[diagonal]
            out[k] = yk
            if (yk != 0.0) {
                for (q in diagonal + 1 until colPtr[k + 1]) out[rowIdx[q]] -= values[q] * yk
            }
        }
        // Lᵀ x = y, back over the same columns, which are Lᵀ's rows.
        for (k in n - 1 downTo 0) {
            val diagonal = colPtr[k]
            var s = out[k]
            for (q in diagonal + 1 until colPtr[k + 1]) s -= values[q] * out[rowIdx[q]]
            out[k] = s / values[diagonal]
        }
        return out
    }

    /** Factories for the portable sparse Cholesky. */
    public companion object {
        /**
         * Factor the lower triangle of [a], which anything stored above the diagonal is ignored in favour of.
         *
         * @throws NotPositiveDefinite at the first column whose pivot is not positive.
         */
        public fun factorLower(a: F64SparseMatrix): F64SparseUpLookingCholesky {
            requireSquare(a, "cholesky")
            val n = a.rows
            // The up-looking sweep reads row k of A left of the diagonal, and CSC stores columns. Transposing
            // the lower triangle once turns each of those rows into a column, and costs one pass over A.
            val upper = transposeCsc(a)
            val parent = eliminationTree(n, upper)
            val colPtr = columnPointers(n, upper, parent)
            val rowIdx = IntArray(colPtr[n])
            val values = DoubleArray(colPtr[n])
            factorNumeric(n, upper, parent, colPtr, rowIdx, values)
            return F64SparseUpLookingCholesky(n, colPtr, rowIdx, values)
        }
    }
}

/**
 * The up-looking numeric factorization: row `k` of `L` is a sparse triangular solve against the rows already
 * built, and the diagonal is what the row leaves of `A(k, k)`.
 */
@Suppress("LongParameterList") // the shape, the tree and the three factor arrays being filled
private fun factorNumeric(
    n: Int,
    upper: F64SparseMatrix,
    parent: IntArray,
    colPtr: IntArray,
    rowIdx: IntArray,
    values: DoubleArray,
) {
    val x = DoubleArray(n)
    val stack = IntArray(n)
    val mark = IntArray(n) { -1 }
    val next = colPtr.copyOf() // next(i) is where column i's next entry goes, so it also ends its built part
    for (k in 0 until n) {
        val top = ereach(upper, k, parent, stack, mark)
        var d = 0.0
        upper.forEachInColumn(k) { i, v ->
            if (i < k) {
                x[i] = v
            } else if (i == k) {
                d = v
            }
        }
        for (t in top until n) {
            val i = stack[t]
            val diagonal = colPtr[i]
            val lki = x[i] / values[diagonal]
            x[i] = 0.0
            for (q in diagonal + 1 until next[i]) x[rowIdx[q]] -= values[q] * lki
            d -= lki * lki
            rowIdx[next[i]] = k
            values[next[i]] = lki
            next[i]++
        }
        if (d <= 0.0) {
            throw NotPositiveDefinite(k, d, "cholesky: pivot $d at column $k is not positive")
        }
        rowIdx[next[k]] = k
        values[next[k]] = sqrt(d)
        next[k]++
    }
}
