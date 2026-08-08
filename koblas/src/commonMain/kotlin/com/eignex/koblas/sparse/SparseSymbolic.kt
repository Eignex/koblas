package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix

/**
 * The symbolic analysis of a symmetric sparse matrix: its elimination tree and the nonzero pattern of `L`,
 * derived from the pattern of `A` alone.
 *
 * Public rather than internal because it is reusable: a matrix whose values change while its pattern does not
 * refactorizes through [factorLdl] against one analysis, and [nnz] answers "will this fit" before any
 * arithmetic runs. [SparseLu] can offer neither, since Markowitz pivoting picks its pivots from the values.
 *
 * Reads the **upper triangle in column form** — rows `i ≤ j` of column `j` — which is what the up-looking
 * factorization needs and the one place the sparse side differs from the dense read-the-lower-triangle. A
 * matrix carrying only its strict lower triangle is rejected rather than analysed as diagonal.
 *
 * Reorders by default, since the order decides the fill; the factorization is of `P·A·Pᵀ` and solves permute
 * through [permutation] invisibly. Pass [SparseOrdering.Natural] to eliminate in the order given.
 *
 * Tree and column counts follow Davis's `ldl_symbolic`, from *Direct Methods for Sparse Linear Systems*.
 *
 * @property n the dimension of the analysed matrix.
 * @property parent the elimination tree of the *permuted* matrix: `parent[j]` is the parent of column `j`.
 * @property columnPointers `L`'s column starts, length `n + 1`, so `columnPointers[n]` is the fill.
 * @property permutation `permutation[k]` is the row and column of `A` eliminated at step `k`; the identity
 *  under [SparseOrdering.Natural]. Exposed so a caller can map back to their own numbering.
 */
class SparseSymbolic internal constructor(
    val n: Int,
    val parent: IntArray,
    val columnPointers: IntArray,
    val permutation: IntArray,
    private val analysedColumnStarts: IntArray,
    private val analysedRowIndices: IntArray,
) {
    /** Whether the analysis eliminates in the order given, in which case no permuting happens anywhere. */
    internal val isNatural: Boolean = permutation.indices.all { permutation[it] == it }

    /** `inversePermutation[original] = step`, derived rather than carried: one array, one use, cheap. */
    internal val inversePermutation: IntArray =
        IntArray(permutation.size).also { for (k in permutation.indices) it[permutation[k]] = k }

    /** Nonzeros in the strictly lower `L`, the fill this pattern will hold. The diagonal is `D`, not `L`. */
    val nnz: Int get() = columnPointers[n]

    /**
     * Factorize [a] numerically against this analysis: `A = L·D·Lᵀ` with `L` unit lower triangular.
     *
     * [a] must carry exactly the analysed pattern, which is checked rather than assumed — the numeric phase
     * walks the analysed elimination tree, so an unseen entry sends that walk off the end of it. A caller
     * whose values cancelled to zero should keep the structural zero rather than rebuild a sparser matrix.
     */
    fun factorLdl(a: SparseMatrix, policy: SparseLdlPolicy = SparseLdlPolicy.Strict): SparseFactorization =
        numericLdl(a, this, policy)

    /** Whether [a] carries exactly the analysed pattern. Compares the CSC index arrays, not the values. */
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
        fun analyze(a: SparseMatrix, ordering: SparseOrdering = SparseOrdering.MinimumDegree): SparseSymbolic {
            require(a.rows == a.cols) { "symbolic analysis requires a square matrix; got ${a.rows}x${a.cols}" }
            requireUpperTriangleStored(a)
            val n = a.rows
            val permutation = when (ordering) {
                SparseOrdering.Natural -> IntArray(n) { it }
                SparseOrdering.MinimumDegree -> minimumDegreeOrdering(a)
            }
            val inverse = IntArray(n)
            for (k in 0 until n) inverse[permutation[k]] = k
            // The tree and the counts describe the matrix that will actually be eliminated, which is the
            // permuted one. Only its pattern matters here, so the values it carries are the caller's.
            val analysed = if (ordering == SparseOrdering.Natural) a else permutedUpperTriangle(a, inverse)
            val parent = eliminationTree(analysed, n)
            val counts = columnCounts(analysed, n, parent)
            val pointers = IntArray(n + 1)
            for (j in 0 until n) pointers[j + 1] = pointers[j] + counts[j]
            return SparseSymbolic(n, parent, pointers, permutation, a.colPtr, a.rowIdx)
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
