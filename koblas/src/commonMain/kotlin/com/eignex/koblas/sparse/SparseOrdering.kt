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
 * A fill-reducing permutation by minimum degree over the quotient graph. Takes the pattern of `A + Aᵀ`, so
 * it does not care which triangles are stored, and returns the index eliminated at each step.
 */
internal fun minimumDegreeOrdering(a: SparseMatrix): IntArray {
    val n = a.rows
    if (n == 0) return IntArray(0)
    return QuotientGraph(a, n).eliminate()
}

/**
 * The graph a minimum-degree elimination walks: variables carrying direct edges, plus membership of the
 * elements earlier pivots left behind. Every structure is a primitive array, each step reads only the
 * pivot's own neighbourhood, and the degrees it invalidates are the only ones rescored.
 */
private class QuotientGraph(a: SparseMatrix, private val n: Int) {
    /** Direct variable-to-variable edges, [varAdjLen] long and shrinking as elements absorb them. */
    private val varAdj: Array<IntArray>
    private val varAdjLen = IntArray(n)

    /** The elements each variable belongs to, [varElemLen] long. */
    private val varElem = Array(n) { IntArray(INITIAL_ELEMENTS) }
    private val varElemLen = IntArray(n)

    /** Members of the element a pivot created, indexed by that pivot; null once absorbed. */
    private val elemVars = arrayOfNulls<IntArray>(n)

    private val alive = BooleanArray(n) { true }
    private val degree = IntArray(n)
    private val buckets = DegreeBuckets(n)

    /** The neighbourhood being gathered, with [inElement] stamped to [elementStamp] for membership. */
    private val element = IntArray(n)
    private var elementSize = 0
    private val inElement = IntArray(n) { -1 }
    private var elementStamp = 0

    /** Duplicate suppression for a degree count, independent of the [elementStamp] marks. */
    private val mark = IntArray(n) { -1 }
    private var markStamp = 0

    init {
        // Both directions of every off-diagonal entry, then a stamped pass to drop the duplicates.
        val counts = IntArray(n)
        for (j in 0 until n) {
            a.forEachInColumn(j) { i, _ ->
                if (i != j) {
                    counts[i]++
                    counts[j]++
                }
            }
        }
        varAdj = Array(n) { IntArray(counts[it]) }
        for (j in 0 until n) {
            a.forEachInColumn(j) { i, _ ->
                if (i != j) {
                    varAdj[i][varAdjLen[i]++] = j
                    varAdj[j][varAdjLen[j]++] = i
                }
            }
        }
        for (v in 0 until n) {
            markStamp++
            val adj = varAdj[v]
            var kept = 0
            for (t in 0 until varAdjLen[v]) {
                val u = adj[t]
                if (mark[u] != markStamp) {
                    mark[u] = markStamp
                    adj[kept++] = u
                }
            }
            varAdjLen[v] = kept
        }
    }

    fun eliminate(): IntArray {
        for (v in 0 until n) degree[v] = degreeOf(v)
        // Descending, since add() pushes onto the front: equal degrees come out lowest index first.
        for (v in n - 1 downTo 0) buckets.add(v, degree[v])
        val permutation = IntArray(n)
        for (step in 0 until n) {
            val pivot = buckets.removeSmallest()
            permutation[step] = pivot
            alive[pivot] = false
            gatherNeighbourhood(pivot)
            supersedeElements(pivot)
            // Nothing outside the pivot's neighbourhood can have changed degree.
            for (t in 0 until elementSize) {
                val v = element[t]
                val updated = degreeOf(v)
                buckets.move(v, degree[v], updated)
                degree[v] = updated
            }
        }
        return permutation
    }

    /** The live variables adjacent to [v], directly or through an element it belongs to. */
    private fun degreeOf(v: Int): Int {
        markStamp++
        mark[v] = markStamp // so v is never counted among its own neighbours
        var count = 0
        val adj = varAdj[v]
        for (t in 0 until varAdjLen[v]) {
            val u = adj[t]
            if (alive[u] && mark[u] != markStamp) {
                mark[u] = markStamp
                count++
            }
        }
        val elements = varElem[v]
        for (t in 0 until varElemLen[v]) {
            val members = elemVars[elements[t]] ?: continue
            for (u in members) {
                if (alive[u] && mark[u] != markStamp) {
                    mark[u] = markStamp
                    count++
                }
            }
        }
        return count
    }

    /** Collect [pivot]'s neighbourhood into [element], stamping each member's [inElement]. */
    private fun gatherNeighbourhood(pivot: Int) {
        elementStamp++
        elementSize = 0
        val adj = varAdj[pivot]
        for (t in 0 until varAdjLen[pivot]) {
            val u = adj[t]
            if (alive[u] && inElement[u] != elementStamp) {
                inElement[u] = elementStamp
                element[elementSize++] = u
            }
        }
        val elements = varElem[pivot]
        for (t in 0 until varElemLen[pivot]) {
            val members = elemVars[elements[t]] ?: continue
            for (u in members) {
                if (alive[u] && inElement[u] != elementStamp) {
                    inElement[u] = elementStamp
                    element[elementSize++] = u
                }
            }
        }
    }

    /**
     * Turn [pivot]'s neighbourhood into one element. It supersedes every element the pivot touched, and it
     * carries the direct edges inside it, so its members stop carrying those individually.
     */
    private fun supersedeElements(pivot: Int) {
        for (t in 0 until varElemLen[pivot]) elemVars[varElem[pivot][t]] = null
        elemVars[pivot] = element.copyOf(elementSize)
        for (t in 0 until elementSize) {
            val v = element[t]
            val adj = varAdj[v]
            var kept = 0
            for (s in 0 until varAdjLen[v]) {
                val u = adj[s]
                if (alive[u] && inElement[u] != elementStamp) adj[kept++] = u
            }
            varAdjLen[v] = kept
            val elements = varElem[v]
            var keptElements = 0
            for (s in 0 until varElemLen[v]) {
                val e = elements[s]
                if (elemVars[e] != null) elements[keptElements++] = e
            }
            varElemLen[v] = keptElements
            addElement(v, pivot)
        }
    }

    private fun addElement(v: Int, e: Int) {
        var elements = varElem[v]
        if (varElemLen[v] == elements.size) {
            elements = elements.copyOf(elements.size * 2)
            varElem[v] = elements
        }
        elements[varElemLen[v]++] = e
    }

    private companion object {
        const val INITIAL_ELEMENTS = 4
    }
}

/** Live variables bucketed by degree, so the smallest is found without scanning them all. */
private class DegreeBuckets(private val size: Int) {
    private val head = IntArray(size + 1) { -1 }
    private val next = IntArray(size) { -1 }
    private val previous = IntArray(size) { -1 }

    /** No occupied bucket is below this. Lowered by [add], raised as [removeSmallest] empties buckets. */
    private var lowest = 0

    fun add(item: Int, degree: Int) {
        val first = head[degree]
        next[item] = first
        previous[item] = -1
        if (first != -1) previous[first] = item
        head[degree] = item
        if (degree < lowest) lowest = degree
    }

    fun remove(item: Int, degree: Int) {
        val before = previous[item]
        val after = next[item]
        if (before == -1) head[degree] = after else next[before] = after
        if (after != -1) previous[after] = before
        previous[item] = -1
        next[item] = -1
    }

    fun move(item: Int, from: Int, to: Int) {
        if (from == to) return
        remove(item, from)
        add(item, to)
    }

    /** Remove and return the lowest-degree member; the caller guarantees one exists. */
    fun removeSmallest(): Int {
        while (lowest <= size && head[lowest] == -1) lowest++
        val item = head[lowest]
        remove(item, lowest)
        return item
    }
}

/**
 * The upper triangle of `P·A·Pᵀ`, the matrix a permuted factorization eliminates. Reads only A's upper
 * triangle and maps each entry to whichever side of the permuted diagonal it lands on, so none is doubled.
 */
internal fun permutedUpperTriangle(a: SparseMatrix, inversePermutation: IntArray): SparseMatrix {
    val n = a.rows
    var nnz = 0
    for (j in 0 until n) a.forEachInColumn(j) { i, _ -> if (i <= j) nnz++ }
    val rowIdx = IntArray(nnz)
    val colIdx = IntArray(nnz)
    val values = DoubleArray(nnz)
    var k = 0
    for (j in 0 until n) {
        a.forEachInColumn(j) { i, v ->
            if (i <= j) {
                val row = inversePermutation[i]
                val column = inversePermutation[j]
                rowIdx[k] = if (row <= column) row else column
                colIdx[k] = if (row <= column) column else row
                values[k] = v
                k++
            }
        }
    }
    return SparseMatrix.ofTriplets(n, n, rowIdx, colIdx, values)
}
