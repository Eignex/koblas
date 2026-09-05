package com.eignex.koblas.sparse.factorization

import com.eignex.koblas.core.F64SparseMatrix

/*
 * The symbolic half the symmetric factorizations share. `A = L·Lᵀ` and `A = L·D·Lᵀ` differ in what they
 * compute per column and agree on which columns those are, so the tree, the row patterns it gives and the
 * column counts they sum to are worked out once here.
 */

/**
 * The elimination tree of a symmetric matrix given its upper triangle, where `parent(i)` is the row of the
 * first subdiagonal entry of column `i` of `L` and `-1` for a root. Path compression through `ancestor`
 * keeps this near linear in the stored entries.
 */
internal fun eliminationTree(n: Int, upper: F64SparseMatrix): IntArray {
    val parent = IntArray(n) { -1 }
    val ancestor = IntArray(n) { -1 }
    for (k in 0 until n) {
        upper.forEachInColumn(k) { row, _ ->
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
 * The nonzero pattern of row [k] of `L`, written into [stack] from the returned index up to `n`, in an order
 * where a column comes before its ancestors. [mark] carries the stamp of the row already visited, so the
 * traversal never walks a subtree twice.
 */
internal fun ereach(upper: F64SparseMatrix, k: Int, parent: IntArray, stack: IntArray, mark: IntArray): Int {
    val n = stack.size
    var top = n
    mark[k] = k
    upper.forEachInColumn(k) { row, _ ->
        if (row <= k) {
            var length = 0
            var i = row
            while (i != -1 && mark[i] != k) {
                stack[length++] = i
                mark[i] = k
                i = parent[i]
            }
            // Reversed onto the top of the stack, so a column lands after every column it depends on.
            while (length > 0) stack[--top] = stack[--length]
        }
    }
    return top
}

/**
 * Column starts for `L`, from a symbolic pass that walks the same row patterns the numeric one will.
 *
 * [storesDiagonal] is what separates the two factorizations here: `L·Lᵀ` keeps the diagonal of `L` and puts
 * it first in each column, where `L·D·Lᵀ` holds a unit diagonal it does not store and a `D` of its own.
 */
internal fun columnPointers(
    n: Int,
    upper: F64SparseMatrix,
    parent: IntArray,
    storesDiagonal: Boolean = true,
): IntArray {
    val counts = IntArray(n)
    val stack = IntArray(n)
    val mark = IntArray(n) { -1 }
    for (k in 0 until n) {
        val top = ereach(upper, k, parent, stack, mark)
        for (t in top until n) counts[stack[t]]++
        if (storesDiagonal) counts[k]++
    }
    val colPtr = IntArray(n + 1)
    for (k in 0 until n) colPtr[k + 1] = colPtr[k] + counts[k]
    return colPtr
}

/**
 * The pattern-only half of an up-looking symmetric factorization, which depends on nothing but the structure
 * and so survives a change of values.
 *
 * @property n the order of the analyzed matrix.
 * @property parent the elimination tree.
 * @property colPtr the column bounds of `L`, so the numeric pass never grows its arrays.
 */
internal class UpLookingSymbolic(val n: Int, val parent: IntArray, val colPtr: IntArray)

/** [UpLookingSymbolic] of the matrix whose transposed lower triangle is [upper]. */
internal fun analyzeUpLooking(n: Int, upper: F64SparseMatrix, storesDiagonal: Boolean): UpLookingSymbolic {
    val parent = eliminationTree(n, upper)
    return UpLookingSymbolic(n, parent, columnPointers(n, upper, parent, storesDiagonal))
}
