package com.eignex.koblas.sparse.factorization.qr

import com.eignex.koblas.core.F64SparseMatrix

internal class SparseQrSymbolic(
    val parent: IntArray,
    val leftmost: IntArray,
    val rowPermutation: IntArray,
    val rows: Int,
    val householderNonzeros: Int,
)

internal fun columnEliminationTree(a: F64SparseMatrix): IntArray {
    val parent = IntArray(a.cols) { -1 }
    val ancestor = IntArray(a.cols) { -1 }
    val lastColumn = IntArray(a.rows) { -1 }
    for (k in 0 until a.cols) {
        a.forEachInColumn(k) { row, _ ->
            var i = lastColumn[row]
            while (i != -1 && i < k) {
                val next = ancestor[i]
                ancestor[i] = k
                if (next == -1) parent[i] = k
                i = next
            }
            lastColumn[row] = k
        }
    }
    return parent
}

internal fun analyzeQr(a: F64SparseMatrix): SparseQrSymbolic {
    val m = a.rows
    val n = a.cols
    val parent = columnEliminationTree(a)
    val leftmost = IntArray(m) { -1 }
    for (k in n - 1 downTo 0) a.forEachInColumn(k) { row, _ -> leftmost[row] = k }

    // Rows queued by the column they may first pivot in, one linked list per column.
    val next = IntArray(m)
    val head = IntArray(n) { -1 }
    val tail = IntArray(n) { -1 }
    val queued = IntArray(n)
    val permutation = IntArray(m + n) { -1 }
    for (i in m - 1 downTo 0) {
        val k = leftmost[i]
        if (k == -1) continue
        if (queued[k]++ == 0) tail[k] = i
        next[i] = head[k]
        head[k] = i
    }

    var householderNonzeros = 0
    var rows = m
    for (k in 0 until n) {
        var i = head[k]
        householderNonzeros++
        // No row left to pivot on, so column k takes a fictitious one.
        if (i < 0) i = rows++
        permutation[i] = k
        if (--queued[k] <= 0) continue
        householderNonzeros += queued[k]
        val ancestor = parent[k]
        if (ancestor != -1) {
            if (queued[ancestor] == 0) tail[ancestor] = tail[k]
            next[tail[k]] = head[ancestor]
            head[ancestor] = next[i]
            queued[ancestor] += queued[k]
        }
    }
    var free = n
    for (i in 0 until m) if (permutation[i] < 0) permutation[i] = free++
    return SparseQrSymbolic(parent, leftmost, permutation, rows, householderNonzeros)
}

internal fun countUpperNonzeros(a: F64SparseMatrix, symbolic: SparseQrSymbolic): Int {
    val n = a.cols
    val mark = IntArray(maxOf(symbolic.rows, n)) { -1 }
    var total = 0
    for (k in 0 until n) {
        mark[k] = k
        var reached = 0
        a.forEachInColumn(k) { row, _ ->
            var i = symbolic.leftmost[row]
            while (i != -1 && mark[i] != k) {
                mark[i] = k
                reached++
                i = symbolic.parent[i]
            }
        }
        // The path union, plus the diagonal the pass stops short of.
        total += reached + 1
    }
    return total
}
