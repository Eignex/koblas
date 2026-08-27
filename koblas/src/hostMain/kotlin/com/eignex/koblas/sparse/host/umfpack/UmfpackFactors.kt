@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.umfpack

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.internal.transposeRaw
import kotlinx.cinterop.*

/** What UMFPACK's extraction hands back, in CSC with the scaling normalised to a multiplier. */
internal class UmfpackFactors(
    val lower: F64SparseMatrix,
    val upper: F64SparseMatrix,
    val rowOrder: IntArray,
    val columnOrder: IntArray,
    val rowScaling: DoubleArray,
)

/**
 * `L`, `U`, the two permutations and the row scaling, or null when this libumfpack lacks the extraction
 * symbols.
 *
 * `L` arrives in row form and is transposed to reach CSC. The scaling arrives as a multiplier or a divisor
 * depending on `do_recip`, and is normalised to a multiplier so `L·U = P·diag(rowScaling)·A·Q` holds without
 * asking which it was.
 */
internal fun extractUmfpackFactors(f: UmfpackFunctions, numeric: COpaquePointer?, order: Int): UmfpackFactors? {
    val lunz = f.getLunz ?: return null
    val getNumeric = f.getNumeric ?: return null
    return memScoped {
        val lNonzeros = alloc<IntVar>()
        val uNonzeros = alloc<IntVar>()
        val rows = alloc<IntVar>()
        val cols = alloc<IntVar>()
        val diagonals = alloc<IntVar>()
        if (lunz(lNonzeros.ptr, uNonzeros.ptr, rows.ptr, cols.ptr, diagonals.ptr, numeric) != 0) {
            return@memScoped null
        }
        val lCount = maxOf(lNonzeros.value, 1)
        val uCount = maxOf(uNonzeros.value, 1)
        val lPtr = allocArray<IntVar>(order + 1)
        val lIdx = allocArray<IntVar>(lCount)
        val lVal = allocArray<DoubleVar>(lCount)
        val uPtr = allocArray<IntVar>(order + 1)
        val uIdx = allocArray<IntVar>(uCount)
        val uVal = allocArray<DoubleVar>(uCount)
        val rowPerm = allocArray<IntVar>(order)
        val colPerm = allocArray<IntVar>(order)
        val reciprocal = alloc<IntVar>()
        val scaling = allocArray<DoubleVar>(order)
        val status = getNumeric(
            lPtr, lIdx, lVal, uPtr, uIdx, uVal, rowPerm, colPerm, null, reciprocal.ptr, scaling, numeric,
        )
        if (status != 0) return@memScoped null
        val scale = DoubleArray(order) { scaling[it] }
        UmfpackFactors(
            // Row form is the transpose in CSC, and transposing also sorts the rows.
            lower = csc(order, lPtr, lIdx, lVal, lNonzeros.value, transposed = true),
            upper = csc(order, uPtr, uIdx, uVal, uNonzeros.value, transposed = false),
            rowOrder = IntArray(order) { rowPerm[it] },
            columnOrder = IntArray(order) { colPerm[it] },
            // do_recip false means UMFPACK divided by these, so the multiplier is the reciprocal.
            rowScaling = if (reciprocal.value != 0) scale else DoubleArray(order) { 1.0 / scale[it] },
        )
    }
}

private fun csc(
    order: Int,
    pointers: CPointer<IntVar>,
    indices: CPointer<IntVar>,
    values: CPointer<DoubleVar>,
    nonzeros: Int,
    transposed: Boolean,
): F64SparseMatrix {
    val ptr = IntArray(order + 1) { pointers[it] }
    val idx = IntArray(nonzeros) { indices[it] }
    val entries = DoubleArray(nonzeros) { values[it] }
    val once = transposeRaw(order, order, ptr, idx, entries)
    return if (transposed) once else transposeRaw(order, order, once.colPtr, once.rowIdx, once.values)
}
