package com.eignex.koblas.sparse.internal

import com.eignex.koblas.core.F64SparseMatrix

/**
 * `A · B` for two CSC operands, by Gustavson's method: one column of the result at a time, accumulated in a
 * dense scratch row indexed by the result's rows and read back through the positions it touched.
 *
 * The scratch is what makes this linear in the work rather than in the shape. A column of `B` selects the
 * columns of `A` that contribute to it, and each contribution scatters into the scratch; the positions
 * touched are collected as they are first written, so the column is read back without sweeping the rows that
 * stayed empty.
 */
internal fun multiplySparse(a: F64SparseMatrix, b: F64SparseMatrix): F64SparseMatrix {
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
        b.forEachInColumn(j) { l, bv ->
            if (bv != 0.0) {
                a.forEachInColumn(l) { i, av ->
                    if (touchedIn[i] != j) {
                        touchedIn[i] = j
                        values[i] = av * bv
                        touched[used++] = i
                    } else {
                        values[i] += av * bv
                    }
                }
            }
        }
        if (count + used > outIdx.size) {
            val grown = maxOf(outIdx.size * 2, count + used)
            outIdx = outIdx.copyOf(grown)
            outVal = outVal.copyOf(grown)
        }
        // Rows arrive in whatever order the contributing columns held them, and CSC wants them ascending.
        val column = touched.copyOfRange(0, used)
        column.sort()
        for (i in column) {
            outIdx[count] = i
            outVal[count] = values[i]
            count++
        }
        colPtr[j + 1] = count
    }
    return F64SparseMatrix.wrap(rows, b.cols, colPtr, outIdx.copyOf(count), outVal.copyOf(count))
}
