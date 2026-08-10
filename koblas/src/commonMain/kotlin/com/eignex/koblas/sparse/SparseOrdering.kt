package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix

/**
 * Which permutation the symmetric analysis eliminates in.
 *
 * A symmetric factorization eliminates down the diagonal in the order it is given, so the order decides the
 * fill — the same matrix can factor with none or with a dense triangle depending only on how its rows and
 * columns are numbered. That makes the ordering part of the analysis rather than something a caller is left
 * to arrange, which is what CHOLMOD and CSparse both concluded.
 */
public enum class SparseOrdering {
    /**
     * Reorder to reduce fill, the default.
     *
     * The naive call should not be the slow one. A caller who has not thought about ordering gets a
     * factorization that does not fill needlessly, and one who has can ask for [Natural].
     */
    MinimumDegree,

    /**
     * Eliminate in the order given.
     *
     * For a matrix that already has a good ordering — a banded assembly, a KKT system built in blocks — where
     * rediscovering the identity permutation is work with nothing to show for it. Also the way to apply an
     * ordering koblas does not offer: permute the matrix, then analyse it with this.
     */
    Natural,
}

/**
 * A fill-reducing permutation by minimum degree over the quotient graph.
 *
 * At each step the variable with the fewest remaining neighbours is eliminated, which turns its neighbourhood
 * into a clique; representing that clique as an *element* rather than by adding its edges is what keeps the
 * graph from growing as the elimination proceeds, and absorbing elements into the one that supersedes them is
 * what keeps it shrinking. The heuristic is Markowitz's, restricted to the symmetric case where the row and
 * column counts are the same number.
 *
 * This is minimum degree with an exact external degree, not SuiteSparse's AMD. AMD's contribution is an
 * *approximate* degree that bounds the true one and costs less to maintain, which is what makes it near-linear
 * on large patterns; the orderings the two produce are comparable, and on the sizes koblas is built for the
 * exact degree is affordable and much harder to get subtly wrong. If analysis time ever dominates on a large
 * pattern, the approximate degree bound is the thing to add, and it slots in here without touching the seam.
 *
 * Takes the pattern of `A + Aᵀ` so it does not care which triangles are stored, and returns `perm` with
 * `perm[k]` the original index eliminated at step `k`.
 */
internal fun minimumDegreeOrdering(a: SparseMatrix): IntArray {
    val n = a.rows
    val adjacency = adjacencyOf(a, n)
    // Per variable: the elements it belongs to. A variable's neighbourhood is its remaining direct edges
    // together with the variables of every element it touches.
    val elements = Array(n) { mutableSetOf<Int>() }
    val alive = BooleanArray(n) { true }
    val elementMembers = arrayOfNulls<MutableSet<Int>>(n)
    val permutation = IntArray(n)
    val neighbourhood = mutableSetOf<Int>()

    for (step in 0 until n) {
        var pivot = -1
        var best = Int.MAX_VALUE
        for (v in 0 until n) {
            if (!alive[v]) continue
            val degree = degreeOf(v, adjacency, elements, elementMembers, alive, neighbourhood)
            if (degree < best) {
                best = degree
                pivot = v
            }
        }
        permutation[step] = pivot
        alive[pivot] = false
        // The pivot's neighbourhood becomes one element, superseding every element it touched.
        collectNeighbourhood(pivot, adjacency, elements, elementMembers, alive, neighbourhood)
        val members = neighbourhood.toMutableSet()
        elementMembers[pivot] = members
        for (absorbed in elements[pivot]) elementMembers[absorbed] = null
        for (v in members) {
            adjacency[v].remove(pivot)
            elements[v].removeAll(elements[pivot])
            elements[v].add(pivot)
            // Direct edges inside the element are now carried by it, so they need not be carried twice.
            adjacency[v].removeAll(members)
        }
        elements[pivot].clear()
    }
    return permutation
}

/**
 * The upper triangle of `P·A·Pᵀ`, which is the matrix a permuted factorization actually eliminates.
 *
 * Reads only `A`'s upper triangle — for a symmetric matrix stored in full that is half the entries, and for
 * one stored as its upper triangle it is all of them — and maps each entry to whichever side of the permuted
 * diagonal it lands on. Each unordered pair is therefore emitted exactly once, so nothing is doubled and
 * nothing is dropped.
 *
 * Materializing rather than reading `A` through the permutation in place: the numeric phase walks columns in
 * elimination order and needs each one sorted, which a permuted view cannot promise without sorting anyway.
 */
internal fun permutedUpperTriangle(a: SparseMatrix, inversePermutation: IntArray): SparseMatrix {
    val n = a.rows
    val columns = List(n) { ArrayList<Pair<Int, Double>>() }
    for (j in 0 until n) {
        a.forEachInColumn(j) { i, v ->
            if (i <= j) {
                val row = inversePermutation[i]
                val column = inversePermutation[j]
                if (row <= column) columns[column].add(row to v) else columns[row].add(column to v)
            }
        }
    }
    return SparseMatrix.ofColumns(n, n, columns)
}

/** The undirected adjacency of `A + Aᵀ`, without self-loops. */
private fun adjacencyOf(a: SparseMatrix, n: Int): Array<MutableSet<Int>> {
    val adjacency = Array(n) { mutableSetOf<Int>() }
    for (j in 0 until n) {
        a.forEachInColumn(j) { i, _ ->
            if (i != j) {
                adjacency[i].add(j)
                adjacency[j].add(i)
            }
        }
    }
    return adjacency
}

/** How many live variables [v] would connect to if it were eliminated now. */
@Suppress("LongParameterList") // the quotient graph is four parallel arrays; bundling them would allocate
private fun degreeOf(
    v: Int,
    adjacency: Array<MutableSet<Int>>,
    elements: Array<MutableSet<Int>>,
    elementMembers: Array<MutableSet<Int>?>,
    alive: BooleanArray,
    scratch: MutableSet<Int>,
): Int {
    collectNeighbourhood(v, adjacency, elements, elementMembers, alive, scratch)
    return scratch.size
}

/** [scratch] receives the live variables adjacent to [v], directly or through an element it belongs to. */
@Suppress("LongParameterList") // as [degreeOf]
private fun collectNeighbourhood(
    v: Int,
    adjacency: Array<MutableSet<Int>>,
    elements: Array<MutableSet<Int>>,
    elementMembers: Array<MutableSet<Int>?>,
    alive: BooleanArray,
    scratch: MutableSet<Int>,
) {
    scratch.clear()
    for (u in adjacency[v]) if (alive[u] && u != v) scratch.add(u)
    for (e in elements[v]) {
        val members = elementMembers[e] ?: continue
        for (u in members) if (alive[u] && u != v) scratch.add(u)
    }
}
