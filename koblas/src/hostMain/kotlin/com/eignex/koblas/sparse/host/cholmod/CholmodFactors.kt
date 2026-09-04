@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.cholmod

import kotlinx.cinterop.*

/**
 * `L` and the ordering from [factor], or null when this libcholmod lacks the conversion symbols.
 *
 * The factor is copied before conversion, because making it simplicial and packed rewrites it in place and
 * the original still has solves to answer. Once packed, its `p`, `i` and `x` are `L` in CSC.
 */
internal fun extractCholmodFactor(
    functions: CholmodFunctions,
    factor: COpaquePointer,
    common: CPointer<ByteVar>,
    asLl: Boolean,
): CholmodFactors? {
    val copy = functions.copyFactor ?: return null
    val change = functions.changeFactor ?: return null
    val duplicate = copy(factor.reinterpret(), common) ?: return null
    val block = duplicate.reinterpret<ByteVar>()
    return try {
        val converted = change(
            CHOLMOD_REAL_XTYPE,
            if (asLl) CHOLMOD_TRUE else 0,
            0,
            CHOLMOD_TRUE,
            CHOLMOD_TRUE,
            block,
            common,
        )
        if (converted != CHOLMOD_TRUE) {
            null
        } else {
            read(block)
        }
    } finally {
        memScoped {
            val slot = alloc<COpaquePointerVar>()
            slot.value = duplicate
            functions.freeFactor(slot.ptr, common)
        }
    }
}

private fun read(block: CPointer<ByteVar>): CholmodFactors {
    val order = sizeAt(block, CHOLMOD_FACTOR_N).toInt()
    val pointers = pointerAt(block, CHOLMOD_FACTOR_P)!!.reinterpret<IntVar>()
    val colPtr = IntArray(order + 1) { pointers[it] }
    val nonzeros = colPtr[order]
    val rowIdx = IntArray(nonzeros)
    val values = DoubleArray(nonzeros)
    if (nonzeros > 0) {
        val indices = pointerAt(block, CHOLMOD_FACTOR_I)!!.reinterpret<IntVar>()
        val entries = pointerAt(block, CHOLMOD_FACTOR_X)!!.reinterpret<DoubleVar>()
        for (k in 0 until nonzeros) {
            rowIdx[k] = indices[k]
            values[k] = entries[k]
        }
    }
    val perm = pointerAt(block, CHOLMOD_FACTOR_PERM)?.reinterpret<IntVar>()
    val permutation = if (perm == null) IntArray(order) { it } else IntArray(order) { perm[it] }
    return CholmodFactors(order, colPtr, rowIdx, values, permutation)
}
