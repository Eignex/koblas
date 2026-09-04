package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.singularFailure
import com.eignex.koblas.sparse.F64SparseFactorization

/**
 * `L` in CSC and the fill-reducing permutation, copied out of a converted factor.
 *
 * The diagonal is first in each column, which is where an `L·D·Lᵀ` caller finds `D`. Both bindings extract
 * this the same way once the native factor has been converted, so what each one carries is the extraction
 * and everything below reads the plain arrays.
 */
internal class CholmodFactors(
    val order: Int,
    val colPtr: IntArray,
    val rowIdx: IntArray,
    val values: DoubleArray,
    val permutation: IntArray,
) {
    /**
     * The lower triangular factor, in the form each kind's interface documents: an `L·Lᵀ` keeps its real
     * diagonal, and an `L·D·Lᵀ` drops the diagonal CHOLMOD stores `D` in, leaving it implicit.
     */
    fun lower(isLl: Boolean): F64SparseMatrix {
        if (isLl) {
            return F64SparseMatrix.wrap(order, order, colPtr.copyOf(), rowIdx.copyOf(), values.copyOf())
        }
        // An `L·D·Lᵀ` keeps `D` on the diagonal CHOLMOD stores, and this seam holds it separately with the
        // unit diagonal implicit, so that entry is dropped rather than replaced by a one.
        val trimmed = IntArray(order + 1)
        for (k in 0 until order) {
            trimmed[k + 1] = trimmed[k] + (colPtr[k + 1] - colPtr[k] - 1)
        }
        val rows = IntArray(trimmed[order])
        val entries = DoubleArray(trimmed[order])
        for (k in 0 until order) {
            var slot = trimmed[k]
            for (q in colPtr[k] + 1 until colPtr[k + 1]) {
                rows[slot] = rowIdx[q]
                entries[slot] = values[q]
                slot++
            }
        }
        return F64SparseMatrix.wrap(order, order, trimmed, rows, entries)
    }

    /** The diagonal factor of an `L·D·Lᵀ`, which CHOLMOD stores as the diagonal of `L`. */
    fun diagonal(): DoubleArray = DoubleArray(order) { values[colPtr[it]] }
}

/** The guard both bindings take before handing out factors: a singular factorization has none to give. */
internal fun F64SparseFactorization.requireCholmodFactors(operation: String) {
    if (singular) throw singularFailure(failedAt, operation)
}
