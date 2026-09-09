package com.eignex.koblas.sparse.internal

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix

/**
 * `A · B` for two CSC operands, by Gustavson's method: one column of the result at a time, accumulated in a
 * dense scratch row indexed by the result's rows and read back through the positions it touched.
 *
 * The scratch is what makes this linear in the work rather than in the shape. A column of `B` selects the
 * columns of `A` that contribute to it, and each contribution scatters into the scratch; the positions
 * touched are collected as they are first written, so the column is read back without sweeping the rows that
 * stayed empty.
 *
 * Which positions those are is structural: an operand's stored zero selects and scatters like any other
 * entry, so the result stores what the two patterns meet at, as an entry the arithmetic cancels to zero is
 * also kept.
 */
internal fun multiplySparse(
    a: SparseMatrix,
    b: SparseMatrix,
    alpha: Double = 1.0,
    lower: Boolean? = null,
): SparseMatrix {
    val rows = a.rows
    val values = DoubleArray(rows)
    // The column each row was last touched in, so a first touch is told from a repeat without clearing.
    val touchedIn = IntArray(rows) { -1 }
    val touched = IntArray(rows)

    val colPtr = IntArray(b.cols + 1)
    var outIdx = IntArray(a.nnz + b.nnz)
    var outVal = DoubleArray(outIdx.size)
    var count = 0

    for (j in 0 until b.cols) {
        var used = 0
        for (bp in b.colPtr[j] until b.colPtr[j + 1]) {
            val l = b.rowIdx[bp]
            // A stored zero of B contributes its column of A as stored zeros rather than dropping it: the
            // pattern of the product is the pattern of the operands, whatever the arithmetic makes of it.
            for (ap in a.colPtr[l] until a.colPtr[l + 1]) {
                val i = a.rowIdx[ap]
                if (lower != null && if (lower) i < j else i > j) continue
                if (touchedIn[i] != j) {
                    touchedIn[i] = j
                    values[i] = if (alpha == 0.0) alpha else a.values[ap] * b.values[bp]
                    touched[used++] = i
                } else if (alpha != 0.0) {
                    values[i] += a.values[ap] * b.values[bp]
                }
            }
        }
        if (count + used > outIdx.size) {
            val grown = maxOf(outIdx.size * 2, count + used)
            outIdx = outIdx.copyOf(grown)
            outVal = outVal.copyOf(grown)
        }
        // Rows arrive in whatever order the contributing columns held them, and CSC wants them ascending.
        // Sorted where they were collected, since the scratch is already this column's and nothing else
        // reads it before the next column overwrites the same prefix.
        touched.sort(0, used)
        for (t in 0 until used) {
            val i = touched[t]
            outIdx[count] = i
            outVal[count] = if (alpha == 0.0) alpha else alpha * values[i]
            count++
        }
        colPtr[j + 1] = count
    }
    // Each column's rows were sorted where they were collected, and a scatter list holds each row once.
    return SparseMatrix.wrapTrusted(rows, b.cols, colPtr, outIdx.copyOf(count), outVal.copyOf(count))
}

/**
 * Adds `alpha · A · B` directly into dense [c]. Both sparse operands are already in the requested
 * orientation. [lower] limits writes to one triangle when non-null. Stored traversal deliberately avoids
 * products with implicit sparse zeros.
 */
internal fun multiplySparseInto(
    alpha: Double,
    a: SparseMatrix,
    b: SparseMatrix,
    c: DenseMatrix,
    lower: Boolean? = null,
) {
    if (alpha == 0.0) return
    val cd = c.data
    for (j in 0 until b.cols) {
        b.forEachInColumn(j) { p, bv ->
            a.forEachInColumn(p) { i, av ->
                if (lower == null || if (lower) i >= j else i <= j) {
                    cd[i + j * c.rows] += alpha * (av * bv)
                }
            }
        }
    }
}

/** Fresh selected triangle of `op(A) · op(A)ᵀ` without materializing a transpose or full product. */
internal fun symmetricRankProduct(a: SparseMatrix, transpose: Boolean, lower: Boolean): SparseMatrix {
    val order = if (transpose) a.cols else a.rows
    val sums = DoubleArray(order)
    val touchedAt = IntArray(order) { -1 }
    val touched = IntArray(order)
    val pointers = IntArray(order + 1)
    var rows = IntArray(maxOf(1, a.nnz))
    var values = DoubleArray(rows.size)
    var count = 0
    for (j in 0 until order) {
        var used = 0
        forEachRankContribution(a, transpose, j) { i, value ->
            if (if (lower) i >= j else i <= j) {
                if (touchedAt[i] != j) {
                    touchedAt[i] = j
                    sums[i] = value
                    touched[used++] = i
                } else {
                    sums[i] += value
                }
            }
        }
        if (count + used > rows.size) {
            val size = maxOf(rows.size * 2, count + used)
            rows = rows.copyOf(size)
            values = values.copyOf(size)
        }
        touched.sort(0, used)
        for (at in 0 until used) {
            val row = touched[at]
            rows[count] = row
            values[count] = sums[row]
            count++
        }
        pointers[j + 1] = count
    }
    return SparseMatrix.wrapTrusted(order, order, pointers, rows.copyOf(count), values.copyOf(count))
}

/** Adds the selected triangle of `alpha · op(A) · op(A)ᵀ` directly into dense [c]. */
internal fun symmetricRankInto(
    alpha: Double,
    a: SparseMatrix,
    transpose: Boolean,
    c: DenseMatrix,
    lower: Boolean,
    sums: DoubleArray,
    touchedAt: IntArray,
    touched: IntArray,
) {
    val order = c.rows
    touchedAt.fill(-1)
    for (j in 0 until order) {
        var used = 0
        forEachRankContribution(a, transpose, j) { i, value ->
            if (if (lower) i >= j else i <= j) {
                if (touchedAt[i] != j) {
                    touchedAt[i] = j
                    sums[i] = value
                    touched[used++] = i
                } else {
                    sums[i] += value
                }
            }
        }
        for (at in 0 until used) {
            val i = touched[at]
            c.data[i + j * order] += alpha * sums[i]
        }
    }
}

/** Emits one stored product contribution for output column [j]. */
private inline fun forEachRankContribution(
    a: SparseMatrix,
    transpose: Boolean,
    j: Int,
    action: (row: Int, value: Double) -> Unit,
) {
    if (!transpose) {
        for (p in 0 until a.cols) {
            val jp = storedPosition(a, j, p)
            if (jp < 0) continue
            val jv = a.values[jp]
            for (ip in a.colPtr[p] until a.colPtr[p + 1]) action(a.rowIdx[ip], a.values[ip] * jv)
        }
        return
    }
    for (jp in a.colPtr[j] until a.colPtr[j + 1]) {
        val p = a.rowIdx[jp]
        val jv = a.values[jp]
        for (i in 0 until a.cols) {
            val ip = storedPosition(a, p, i)
            if (ip >= 0) action(i, a.values[ip] * jv)
        }
    }
}

/** Stored-position lookup without turning an absent sparse entry into a numerical zero. */
private fun storedPosition(a: SparseMatrix, row: Int, column: Int): Int {
    var low = a.colPtr[column]
    var high = a.colPtr[column + 1] - 1
    while (low <= high) {
        val middle = (low + high) ushr 1
        when {
            a.rowIdx[middle] < row -> low = middle + 1
            a.rowIdx[middle] > row -> high = middle - 1
            else -> return middle
        }
    }
    return -1
}
