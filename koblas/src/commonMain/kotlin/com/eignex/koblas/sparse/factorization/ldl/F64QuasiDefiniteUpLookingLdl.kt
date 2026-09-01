package com.eignex.koblas.sparse.factorization.ldl

import com.eignex.koblas.AllocationCapability
import com.eignex.koblas.NOT_SINGULAR
import com.eignex.koblas.Workspace
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.noManagedOrNativeAllocation
import com.eignex.koblas.requireSquare
import com.eignex.koblas.singularFailure
import com.eignex.koblas.sparse.F64QuasiDefiniteLdlFactorization
import com.eignex.koblas.sparse.FactorizationInertia
import com.eignex.koblas.sparse.factorization.columnPointers
import com.eignex.koblas.sparse.factorization.eliminationTree
import com.eignex.koblas.sparse.factorization.ereach
import com.eignex.koblas.sparse.internal.transposeCsc
import com.eignex.koblas.sparse.requireSolveShapes
import kotlin.math.abs

/**
 * Sparse factorization `A = L·D·Lᵀ` of a symmetric matrix, up-looking over the elimination tree.
 *
 * [l] is unit lower triangular and its diagonal is not stored, since it is ones; [d] holds the diagonal
 * factor. Where the Cholesky takes a square root and so demands a positive pivot, this one does not, which
 * is the whole reason it exists: an indefinite matrix has an `L·D·Lᵀ` and has no `L·Lᵀ`.
 *
 * No numerical pivoting, so what it produces for an indefinite matrix is the exact factorization of the
 * matrix it was given and not necessarily a well conditioned one. It is for quasi-definite systems, whose
 * fill-reducing ordering must be retained.
 */
public class F64QuasiDefiniteUpLookingLdl internal constructor(
    override val n: Int,
    private val colPtr: IntArray,
    private val rowIdx: IntArray,
    private val values: DoubleArray,
    private val diagonal: DoubleArray,
    override val failedAt: Int,
) : F64QuasiDefiniteLdlFactorization {

    override val l: F64SparseMatrix
        get() {
            requireFactors("l")
            return F64SparseMatrix.wrap(
                n,
                n,
                colPtr.copyOf(),
                rowIdx.copyOf(),
                values.copyOf(),
            )
        }

    /** The diagonal factor `D`, which an indefinite matrix leaves entries of either sign in. */
    override val d: DoubleArray
        get() {
            requireFactors("d")
            return diagonal.copyOf()
        }

    override val inertia: FactorizationInertia
        get() {
            requireFactors("inertia")
            return FactorizationInertia(
                positive = diagonal.count { it > 0.0 },
                negative = diagonal.count { it < 0.0 },
                zero = diagonal.count { it == 0.0 },
            )
        }

    /** The identity: this factorization reorders nothing, for the reason its own documentation gives. */
    override val order: IntArray
        get() {
            requireFactors("order")
            return IntArray(n) { it }
        }

    private fun requireFactors(operation: String) {
        if (singular) throw singularFailure(failedAt, operation)
    }

    /** The stored entries of `L` plus the diagonal `D`, which together are what the factorization holds. */
    override val nnz: Int get() = if (singular) 0 else values.size + n

    /** `min |D(k, k)| / max |D(k, k)|`, which for `A = L·D·Lᵀ` is what the seam documents over `U`. */
    override val rcond: Double
        get() {
            if (singular) return 0.0
            if (n == 0) return 1.0
            var minimum = Double.POSITIVE_INFINITY
            var maximum = 0.0
            for (k in 0 until n) {
                val magnitude = abs(diagonal[k])
                minimum = minOf(minimum, magnitude)
                maximum = maxOf(maximum, magnitude)
            }
            return if (maximum == 0.0) 0.0 else minimum / maximum
        }

    override fun solveAllocation(aliasing: Boolean, transpose: Boolean): AllocationCapability =
        noManagedOrNativeAllocation

    /**
     * Solve `A x = b` into [out], which is returned and may be [b]. [transpose] is accepted and ignored: `A`
     * is symmetric, so the transposed system is the same one. Allocates nothing, [workspace] or not, since
     * all three sweeps run in the destination.
     */
    override fun solveInto(b: DoubleArray, out: DoubleArray, transpose: Boolean, workspace: Workspace?): DoubleArray {
        if (singular) throw singularFailure(failedAt, "solve")
        requireSolveShapes(n, b, out)
        if (out !== b) b.copyInto(out)
        // L y = b, forward, with the unit diagonal leaving each entry final as it is reached.
        for (k in 0 until n) {
            val yk = out[k]
            if (yk != 0.0) {
                for (q in colPtr[k] until colPtr[k + 1]) out[rowIdx[q]] -= values[q] * yk
            }
        }
        for (k in 0 until n) out[k] /= diagonal[k]
        // Lᵀ x = z, back over the same columns, which are Lᵀ's rows.
        for (k in n - 1 downTo 0) {
            var s = out[k]
            for (q in colPtr[k] until colPtr[k + 1]) s -= values[q] * out[rowIdx[q]]
            out[k] = s
        }
        return out
    }

    /** Factories for the portable sparse `L·D·Lᵀ`. */
    public companion object {
        /**
         * Factor the lower triangle of [a], which anything stored above the diagonal is ignored in favour of.
         *
         * A zero pivot comes back as a factorization reporting `singular` at that column, as the sparse LU
         * does. A negative one does not: it is what an indefinite matrix is expected to produce.
         */
        public fun factorLower(a: F64SparseMatrix): F64QuasiDefiniteUpLookingLdl {
            requireSquare(a, "quasiDefiniteLdl")
            val n = a.rows
            // The up-looking sweep reads row k of A left of the diagonal, and CSC stores columns.
            val upper = transposeCsc(a)
            val parent = eliminationTree(n, upper)
            val colPtr = columnPointers(n, upper, parent, storesDiagonal = false)
            val rowIdx = IntArray(colPtr[n])
            val values = DoubleArray(colPtr[n])
            val diagonal = DoubleArray(n)
            val failedAt = factorNumeric(n, upper, parent, colPtr, rowIdx, values, diagonal)
            return F64QuasiDefiniteUpLookingLdl(n, colPtr, rowIdx, values, diagonal, failedAt)
        }
    }
}

/**
 * The up-looking numeric factorization: row `k` of `L` is a sparse triangular solve against the rows already
 * built, and `D(k)` is what that row leaves of `A(k, k)`. Returns the column a zero pivot stopped it at, or
 * [NOT_SINGULAR].
 */
@Suppress("LongParameterList") // the shape, the tree, and the three factor arrays being filled beside D
private fun factorNumeric(
    n: Int,
    upper: F64SparseMatrix,
    parent: IntArray,
    colPtr: IntArray,
    rowIdx: IntArray,
    values: DoubleArray,
    diagonal: DoubleArray,
): Int {
    val y = DoubleArray(n)
    val stack = IntArray(n)
    val mark = IntArray(n) { -1 }
    val next = colPtr.copyOf() // next(i) is where column i's next entry goes, so it also ends its built part
    for (k in 0 until n) {
        val top = ereach(upper, k, parent, stack, mark)
        var dk = 0.0
        upper.forEachInColumn(k) { i, v ->
            if (i < k) {
                y[i] = v
            } else if (i == k) {
                dk = v
            }
        }
        for (t in top until n) {
            val i = stack[t]
            val yi = y[i]
            y[i] = 0.0
            for (q in colPtr[i] until next[i]) y[rowIdx[q]] -= values[q] * yi
            val lki = yi / diagonal[i]
            dk -= lki * yi
            rowIdx[next[i]] = k
            values[next[i]] = lki
            next[i]++
        }
        if (dk == 0.0) return k
        diagonal[k] = dk
    }
    return NOT_SINGULAR
}
