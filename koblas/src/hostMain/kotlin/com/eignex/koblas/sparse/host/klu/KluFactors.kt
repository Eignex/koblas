@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.internal.transposeRaw
import kotlinx.cinterop.*

/**
 * The factors from a KLU numeric object, or null when this libklu lacks `klu_extract`.
 *
 * KLU permutes to block triangular form and factors the blocks, so the entries outside them are in neither
 * `L` nor `U` and come back as `F`. Its scaling divides, so it is inverted to the multiplier the factor
 * interface documents.
 */
internal fun extractKluFactors(
    functions: KluFunctions,
    symbolic: COpaquePointer?,
    numeric: COpaquePointer?,
    common: CPointer<ByteVar>,
    order: Int,
): KluFactors? {
    val extract = functions.extract ?: return null
    val block = numeric?.reinterpret<ByteVar>() ?: return null
    val lNonzeros = intAt(block, KLU_NUMERIC_LNZ)
    val uNonzeros = intAt(block, KLU_NUMERIC_UNZ)
    val offNonzeros = intAt(block, KLU_NUMERIC_NZOFF)
    val blocks = intAt(block, KLU_NUMERIC_NBLOCKS)
    return memScoped {
        val lPtr = allocArray<IntVar>(order + 1)
        val lIdx = allocArray<IntVar>(maxOf(lNonzeros, 1))
        val lVal = allocArray<DoubleVar>(maxOf(lNonzeros, 1))
        val uPtr = allocArray<IntVar>(order + 1)
        val uIdx = allocArray<IntVar>(maxOf(uNonzeros, 1))
        val uVal = allocArray<DoubleVar>(maxOf(uNonzeros, 1))
        val fPtr = allocArray<IntVar>(order + 1)
        val fIdx = allocArray<IntVar>(maxOf(offNonzeros, 1))
        val fVal = allocArray<DoubleVar>(maxOf(offNonzeros, 1))
        val rowPerm = allocArray<IntVar>(order)
        val colPerm = allocArray<IntVar>(order)
        val scaling = allocArray<DoubleVar>(order)
        val boundaries = allocArray<IntVar>(blocks + 1)
        val ok = extract(
            numeric, symbolic,
            lPtr, lIdx, lVal, uPtr, uIdx, uVal, fPtr, fIdx, fVal,
            rowPerm, colPerm, scaling, boundaries, common,
        )
        if (ok != 1) {
            null
        } else {
            KluFactors(
                lower = csc(order, lPtr, lIdx, lVal, lNonzeros),
                upper = csc(order, uPtr, uIdx, uVal, uNonzeros),
                offDiagonal = csc(order, fPtr, fIdx, fVal, offNonzeros),
                rowOrder = IntArray(order) { rowPerm[it] },
                columnOrder = IntArray(order) { colPerm[it] },
                rowScaling = DoubleArray(order) { if (scaling[it] == 0.0) 1.0 else 1.0 / scaling[it] },
            )
        }
    }
}

private fun csc(
    order: Int,
    pointers: CPointer<IntVar>,
    indices: CPointer<IntVar>,
    values: CPointer<DoubleVar>,
    nonzeros: Int,
): F64SparseMatrix {
    val ptr = IntArray(order + 1) { pointers[it] }
    val idx = IntArray(nonzeros) { indices[it] }
    val entries = DoubleArray(nonzeros) { values[it] }
    val once = transposeRaw(order, order, ptr, idx, entries)
    return transposeRaw(order, order, once.colPtr, once.rowIdx, once.values)
}
