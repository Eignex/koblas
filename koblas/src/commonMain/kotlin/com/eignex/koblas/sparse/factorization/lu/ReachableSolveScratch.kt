package com.eignex.koblas.sparse.factorization.lu

/**
 * Scratch for a reachability-limited triangular solve, held by the caller so a solve allocates nothing.
 *
 * [mark] is stamped rather than cleared: a position belongs to the current traversal when its stamp matches,
 * so a solve that touches ten positions costs ten writes rather than a pass over all `m`. The stamp advances
 * per phase, and wrapping is handled by clearing when it would.
 */
internal class ReachableSolveScratch(m: Int) {
    val mark: IntArray = IntArray(m) { -1 }
    val stack: IntArray = IntArray(m)
    val next: IntArray = IntArray(m)
    val order: IntArray = IntArray(m)

    /** Pivot-space accumulator, left at zero between solves so a solve clears only what it touched. */
    val values: DoubleArray = DoubleArray(m)

    private val orderCopy: IntArray = IntArray(m)
    private var inverse: IntArray? = null
    private var inverseOf: IntArray? = null
    private var stamp: Int = 0

    /**
     * The inverse of [perm], built once per permutation. The forward solve needs original-row to
     * pivot-position, which the factorization stores the other way round.
     *
     * Keyed on the array itself rather than cached outright: a caller holds one scratch across the life of a
     * solver and refactorizes under it, and each factorization pivots to its own order, so an unconditional
     * cache would answer every later solve with the first factorization's permutation.
     */
    fun invPerm(perm: IntArray): IntArray {
        inverse?.let { if (inverseOf === perm) return it }
        val inv = IntArray(perm.size)
        for (k in perm.indices) inv[perm[k]] = k
        inverse = inv
        inverseOf = perm
        return inv
    }

    /**
     * A stable copy of the first [count] of [order], since the next traversal overwrites it and the sweep
     * that follows still needs the one before.
     */
    fun copyOrder(count: Int): IntArray {
        order.copyInto(orderCopy, 0, 0, count)
        return orderCopy
    }

    /** The stamp for a new traversal. */
    fun nextStamp(): Int {
        if (stamp == Int.MAX_VALUE) {
            mark.fill(-1)
            stamp = 0
        }
        return ++stamp
    }
}

/**
 * The pivot positions reachable from [sources] through [columns], in an order where every position comes
 * after everything it depends on.
 *
 * This is the Gilbert and Peierls reach: a triangular solve only has to visit the positions the right-hand
 * side can actually influence, which for a sparse right-hand side is a small fraction of `m`. Sweeping all
 * `m` positions instead is what makes an otherwise `O(1)`-sized solve cost `O(m)`.
 *
 * The traversal is iterative rather than recursive, since a chain through the factor can be as long as `m`.
 * Which direction the solve runs is already carried by the graph: L's columns point at later positions and
 * U's at earlier ones, so reverse postorder is the processing order for both.
 *
 * @return the number of positions written to [scratch].order.
 */
internal fun reachableOrder(
    columns: Array<IntArray>,
    sources: IntArray,
    sourceCount: Int,
    scratch: ReachableSolveScratch,
): Int {
    val mark = scratch.mark
    val stack = scratch.stack
    val next = scratch.next
    val order = scratch.order
    val stamp = scratch.nextStamp()
    var found = 0

    for (s in 0 until sourceCount) {
        val root = sources[s]
        if (mark[root] == stamp) continue
        mark[root] = stamp
        var top = 0
        stack[0] = root
        next[0] = 0
        while (top >= 0) {
            val node = stack[top]
            val edges = columns[node]
            var edge = next[top]
            var descended = false
            while (edge < edges.size) {
                val child = edges[edge]
                edge++
                if (mark[child] != stamp) {
                    mark[child] = stamp
                    next[top] = edge
                    top++
                    stack[top] = child
                    next[top] = 0
                    descended = true
                    break
                }
            }
            if (descended) continue
            next[top] = edge
            // Finished: everything this position depends on is already recorded behind it.
            order[found++] = node
            top--
        }
    }

    // Postorder finishes a position after everything it feeds, so reversing puts each one before the
    // positions its value has to reach.
    var lo = 0
    var hi = found - 1
    while (lo < hi) {
        val swap = order[lo]
        order[lo] = order[hi]
        order[hi] = swap
        lo++
        hi--
    }
    return found
}
