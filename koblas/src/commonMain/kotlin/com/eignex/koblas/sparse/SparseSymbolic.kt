package com.eignex.koblas.sparse

import com.eignex.koblas.F64SparseMatrix
import com.eignex.koblas.requireSquare

/**
 * Reads the upper triangle in column form, rows `i ≤ j` of column j, not the lower triangle the dense side
 * reads. Reorders by default, so the factorization is then of `P·A·Pᵀ`.
 *
 * @property n the dimension of the analysed matrix.
 * @property parent the elimination tree of the permuted matrix, so parent(j) is the parent of column j.
 * @property columnPointers L's column starts, length `n + 1`, so columnPointers(n) is the fill.
 * @property permutation the row and column of `A` eliminated at step `k`; the identity when natural.
 */
public class SparseSymbolic internal constructor(
    public val n: Int,
    public val parent: IntArray,
    public val columnPointers: IntArray,
    public val permutation: IntArray,
    private val analysedColumnStarts: IntArray,
    private val analysedRowIndices: IntArray,
) {
    /** Whether the analysis eliminates in the order given, in which case no permuting happens anywhere. */
    internal val isNatural: Boolean = permutation.indices.all { permutation[it] == it }

    /** The step at which each original index is eliminated, the inverse of [permutation]. */
    internal val inversePermutation: IntArray = inverseOf(permutation)

    /** Nonzeros in the strictly lower L, the fill this pattern will hold. The diagonal is D, not L. */
    public val nnz: Int get() = columnPointers[n]

    /**
     * Factorize [a] numerically against this analysis as `A = L·D·Lᵀ` with L unit lower triangular. [a] must
     * carry exactly the analysed pattern, structural zeros included, which is checked.
     */
    public fun factorLdl(a: F64SparseMatrix, policy: SparseLdlPolicy = SparseLdlPolicy.Strict): F64SparseFactorization =
        numericLdl(a, this, policy)

    /** Whether [a] carries exactly the analysed pattern. Compares the CSC index arrays, not the values. */
    internal fun describesPatternOf(a: F64SparseMatrix): Boolean = a.rows == n && a.cols == n &&
        a.colPtr.contentEquals(analysedColumnStarts) &&
        a.rowIdx.contentEquals(analysedRowIndices)

    /** Entry point for the symbolic analysis. */
    public companion object {
        /** Analyse [a]'s pattern, the elimination tree first and then the column counts of L. */
        public fun analyze(
            a: F64SparseMatrix,
            ordering: SparseOrdering = SparseOrdering.MinimumDegree,
        ): SparseSymbolic {
            requireSquare(a, "symbolic analysis")
            requireUpperTriangleStored(a)
            val n = a.rows
            val permutation = when (ordering) {
                SparseOrdering.Natural -> IntArray(n) { it }
                SparseOrdering.MinimumDegree -> minimumDegreeOrdering(a)
            }
            // The tree and the counts describe the matrix that is eliminated, the permuted one. Keyed on the
            // permutation being the identity rather than on the ordering asked for, which is the test
            // [SparseSymbolic.isNatural] makes and the one numericLdl reads: a fill-reducing ordering that
            // happens to come back as the identity has to leave both sides eliminating the same matrix.
            val natural = permutation.indices.all { permutation[it] == it }
            val analysed = if (natural) a else permutedUpperTriangle(a, inverseOf(permutation))
            val parent = eliminationTree(analysed, n)
            val counts = columnCounts(analysed, n, parent)
            val pointers = IntArray(n + 1)
            for (j in 0 until n) pointers[j + 1] = pointers[j] + counts[j]
            return SparseSymbolic(n, parent, pointers, permutation, a.colPtr, a.rowIdx)
        }

        /**
         * Rejects a lower-only matrix, which the analysis reading rows `i ≤ j` of column j would see as
         * diagonal and factor into something that solves nothing. Full and upper-only both pass.
         */
        private fun requireUpperTriangleStored(a: F64SparseMatrix) {
            var strictUpper = 0
            var strictLower = 0
            for (j in 0 until a.cols) {
                a.forEachInColumn(j) { i, _ ->
                    if (i < j) strictUpper++
                    if (i > j) strictLower++
                }
            }
            require(strictUpper > 0 || strictLower == 0) {
                "symbolic analysis reads the upper triangle in column form, and this matrix stores only its " +
                    "strict lower triangle ($strictLower entries). Pass the full symmetric matrix, or its " +
                    "upper triangle, or transpose it first."
            }
        }

        /**
         * The elimination tree of a symmetric pattern. Each column k walks upward from every row `i < k` it
         * holds, redirecting the ancestor pointers it passes to k.
         */
        private fun eliminationTree(a: F64SparseMatrix, n: Int): IntArray {
            val parent = IntArray(n) { -1 }
            val ancestor = IntArray(n) { -1 }
            for (k in 0 until n) {
                a.forEachInColumn(k) { row, _ ->
                    var i = row
                    while (i != -1 && i < k) {
                        val next = ancestor[i]
                        ancestor[i] = k
                        if (next == -1) parent[i] = k
                        i = next
                    }
                }
            }
            return parent
        }

        /**
         * How many entries each column of L will hold. For column k, every row `i < k` climbs to the first
         * already-visited ancestor and each node passed gains an entry in its own column.
         */
        private fun columnCounts(a: F64SparseMatrix, n: Int, parent: IntArray): IntArray {
            val counts = IntArray(n)
            val flag = IntArray(n) { -1 }
            for (k in 0 until n) {
                flag[k] = k
                a.forEachInColumn(k) { row, _ ->
                    if (row < k) {
                        var i = row
                        while (flag[i] != k) {
                            counts[i]++
                            flag[i] = k
                            i = parent[i]
                        }
                    }
                }
            }
            return counts
        }
    }
}

/** The inverse of the permutation [p], so `inverseOf(p)[p[k]] == k`. */
internal fun inverseOf(p: IntArray): IntArray = IntArray(p.size).also { for (k in p.indices) it[p[k]] = k }
