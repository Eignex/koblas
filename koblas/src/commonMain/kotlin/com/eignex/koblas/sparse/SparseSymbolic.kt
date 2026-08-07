package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix

/**
 * The symbolic analysis of a symmetric sparse matrix: its elimination tree and the nonzero pattern of `L`,
 * derived from the pattern of `A` alone.
 *
 * This is the phase [SparseLu] cannot have. Markowitz pivoting chooses its pivots by magnitude as it goes, so
 * the pattern of the factors is not knowable until the values are known — which is why the symbolic/numeric
 * split was closed as inapplicable there. A symmetric factorization pivots down the diagonal in a fixed
 * order, so the pattern falls out of the graph and nothing about it depends on a single value. Two things
 * follow, and they are the reason this class is public rather than an implementation detail:
 *
 * - **Refactorization is cheap.** A matrix whose values change but whose pattern does not — a Newton
 *   iteration, a KKT system re-linearized, a covariance updated in place — is refactorized by
 *   [factorLdl] against an analysis computed once. The analysis is the expensive part at high fill.
 * - **The fill is known before any arithmetic happens.** [nnz] answers "will this factorization fit" without
 *   computing it.
 *
 * The analysis is of the **upper triangle in column form**, which for a symmetric matrix stored in full is
 * simply what column `j` holds at rows `i ≤ j`. That is the orientation the up-looking factorization needs,
 * and it is the one place where the sparse side's convention differs from the dense side's read-the-lower-
 * triangle: transposing to match would cost a pass over the matrix to save a sentence of documentation.
 * A matrix carrying *only* its strict lower triangle is rejected rather than silently analysed as diagonal.
 *
 * The elimination tree and the up-looking pattern traversal are Davis's, from *Direct Methods for Sparse
 * Linear Systems*; [parent] and the column counts are computed exactly as `ldl_symbolic` computes them.
 *
 * @property n the dimension of the analysed matrix.
 * @property parent the elimination tree: `parent[j]` is the parent of column `j`, or `-1` at a root.
 * @property columnPointers `L`'s column starts, length `n + 1`, so `columnPointers[n]` is the fill.
 */
class SparseSymbolic internal constructor(
    val n: Int,
    val parent: IntArray,
    val columnPointers: IntArray,
    private val analysedColumnStarts: IntArray,
    private val analysedRowIndices: IntArray,
) {
    /** Nonzeros in the strictly lower `L`, the fill this pattern will hold. The diagonal is `D`, not `L`. */
    val nnz: Int get() = columnPointers[n]

    /**
     * Factorize [a] numerically against this analysis: `A = L·D·Lᵀ` with `L` unit lower triangular.
     *
     * [a] must have exactly the pattern this analysis was computed from — the same CSC index arrays, values
     * aside — and that is checked rather than assumed. A caller whose values have cancelled to zero should
     * keep the structural zero rather than rebuilding a sparser matrix, which is the cheaper thing to do
     * anyway.
     *
     * The alternative, accepting any subset, reads as more generous and is not worth it: the numeric phase
     * walks the elimination tree of the *analysed* pattern, so a matrix with an entry the analysis never saw
     * sends that walk off the end of the tree, and distinguishing "fewer entries" from "different entries"
     * costs the same pass this check does.
     */
    fun factorLdl(a: SparseMatrix, policy: SparseLdlPolicy = SparseLdlPolicy.Strict): SparseFactorization =
        numericLdl(a, this, policy)

    /**
     * Whether [a] carries exactly the pattern this analysis was computed from.
     *
     * Compares the CSC index arrays and nothing else, so it costs one pass over the pattern and is immune to
     * the values — which is the same standard the analysis itself was built to.
     */
    internal fun describesPatternOf(a: SparseMatrix): Boolean = a.rows == n && a.cols == n &&
        a.colPtr.contentEquals(analysedColumnStarts) &&
        a.rowIdx.contentEquals(analysedRowIndices)

    /** Analysis entry points. */
    companion object {
        /**
         * Analyse [a]'s pattern: the elimination tree, then the column counts of `L`.
         *
         * Costs one pass over the pattern for the tree and one up-looking traversal per column for the
         * counts, with no arithmetic on the values at all — they are not read.
         */
        fun analyze(a: SparseMatrix): SparseSymbolic {
            require(a.rows == a.cols) { "symbolic analysis requires a square matrix; got ${a.rows}x${a.cols}" }
            requireUpperTriangleStored(a)
            val n = a.rows
            val parent = eliminationTree(a, n)
            val counts = columnCounts(a, n, parent)
            val pointers = IntArray(n + 1)
            for (j in 0 until n) pointers[j + 1] = pointers[j] + counts[j]
            return SparseSymbolic(n, parent, pointers, a.colPtr, a.rowIdx)
        }

        /**
         * Rejects a matrix that carries only its strict lower triangle.
         *
         * The analysis reads rows `i ≤ j` of each column `j`, so a lower-only matrix looks diagonal to it and
         * would factor to something that solves nothing. A full or upper-only matrix both pass. The check is
         * over the pattern, so it costs one pass and cannot be fooled by values.
         */
        private fun requireUpperTriangleStored(a: SparseMatrix) {
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
         * The elimination tree of a symmetric pattern (Davis's `cs_etree`).
         *
         * Each column `k` walks the tree upward from every row `i < k` it holds, redirecting the ancestor
         * pointers it passes to `k`. The path compression is what keeps this near-linear: a node's ancestor
         * link is rewritten as it is traversed, so no path is walked twice.
         */
        private fun eliminationTree(a: SparseMatrix, n: Int): IntArray {
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
         * How many entries each column of `L` will hold, by walking the tree for each column of `A`.
         *
         * The same traversal the numeric phase performs, with the values left out: for column `k`, every row
         * `i < k` climbs to the first already-visited ancestor, and each node passed gains an entry in its
         * own column. `flag` marks per `k`, which is what makes each node counted once per column.
         */
        private fun columnCounts(a: SparseMatrix, n: Int, parent: IntArray): IntArray {
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
