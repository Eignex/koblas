package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix

/** Which fill-reducing ordering a sparse factorization applies. */
public enum class SparseOrdering {
    /** Reorder to reduce fill, the default. */
    MinimumDegree,

    /**
     * Eliminate in the order given, for a matrix that already has a good one. Also the way to apply an
     * ordering koblas does not offer, by permuting the matrix and analysing it with this.
     */
    Natural,
}

/**
 * A fill-reducing permutation by exact minimum degree over the quotient graph. Takes the pattern of
 * `A + Aᵀ`, so it does not care which triangles are stored, and returns the index eliminated at each step.
 */
internal fun minimumDegreeOrdering(a: SparseMatrix): IntArray {
    val n = a.rows
    val adjacency = adjacencyOf(a, n)
    // A variable's neighbourhood is its remaining direct edges plus the variables of every element it is in.
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
 * The upper triangle of `P·A·Pᵀ`, the matrix a permuted factorization eliminates. Reads only A's upper
 * triangle and maps each entry to whichever side of the permuted diagonal it lands on, so none is doubled.
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
@Suppress("LongParameterList") // the quotient graph is four parallel arrays, and bundling them would allocate
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
@Suppress("LongParameterList") // as degreeOf
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
