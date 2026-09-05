@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.spqr

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.NativeBlock
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.F64SparseQrFactorization
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_NCOL
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_DENSE_X
import com.eignex.koblas.sparse.host.cholmod.describeGeneral
import com.eignex.koblas.sparse.host.cholmod.freeMatrix
import com.eignex.koblas.sparse.host.cholmod.readCholmodSparse
import kotlinx.cinterop.*

/** SPQR's sparse QR through the 32-bit-index SuiteSparse C interface. */
public actual class SpqrQr actual constructor(config: SpqrConfig) {
    private val loader = SpqrLoader(config)
    private val ordering = config.options.ordering
    private val tolerance = config.options.rankTolerance

    /** Whether both libraries opened and every symbol resolved. */
    public actual val isAvailable: Boolean get() = loader.available

    /** Why SPQR is unusable, or null when it is usable. */
    public actual val unavailableReason: String? get() = loader.unavailableReason

    /** Factor [a], or null when the library is unusable. */
    public actual fun factor(a: F64SparseMatrix): F64SparseQrFactorization? {
        requireShape(a.rows >= a.cols) {
            "qr: A is ${a.rows}x${a.cols}, which is wider than it is tall; factor its transpose instead"
        }
        val functions = loader.functions ?: return null
        val common = loader.common()
        val sparse = describeGeneral(a)
        return try {
            memScoped {
                val rSlot = pointerSlot()
                val eSlot = pointerSlot()
                val hSlot = pointerSlot()
                val hpSlot = pointerSlot()
                val tauSlot = pointerSlot()
                val rank = functions.qr(
                    ordering.code(),
                    tolerance,
                    a.cols,
                    0,
                    sparse,
                    null,
                    null,
                    null,
                    null,
                    rSlot.ptr,
                    eSlot.ptr,
                    hSlot.ptr,
                    hpSlot.ptr,
                    tauSlot.ptr,
                    common,
                )
                try {
                    val r = rSlot.value
                    val h = hSlot.value
                    val tau = tauSlot.value
                    if (rank < 0 || r == null || h == null || tau == null) return@memScoped null
                    SpqrQrFactorization(
                        SpqrFactorData(
                            rank,
                            NativeBlock(r.reinterpret()).readCholmodSparse(),
                            readPermutation(eSlot.value, a.cols),
                            NativeBlock(h.reinterpret()).readCholmodSparse(),
                            readPermutation(hpSlot.value, a.rows),
                            readDense(tau.reinterpret()),
                        ),
                        a.rows,
                        a.cols,
                    )
                } finally {
                    freeSparse(functions, rSlot, common)
                    freePermutation(functions, eSlot, a.cols, common)
                    freeSparse(functions, hSlot, common)
                    freePermutation(functions, hpSlot, a.rows, common)
                    freeDense(functions, tauSlot, common)
                }
            }
        } finally {
            freeMatrix(sparse)
            try {
                functions.finish(common)
            } finally {
                nativeHeap.free(common)
            }
        }
    }
}

private fun MemScope.pointerSlot(): COpaquePointerVar = alloc<COpaquePointerVar>().also { it.value = null }

private fun readPermutation(pointer: COpaquePointer?, size: Int): IntArray {
    if (pointer == null) return IntArray(size) { it }
    return NativeBlock(pointer.reinterpret()).readInts(size)
}

private fun readDense(dense: CPointer<ByteVar>): DoubleArray {
    val descriptor = NativeBlock(dense)
    val size = descriptor.getSize(CHOLMOD_DENSE_NCOL).toInt()
    val values = checkNotNull(descriptor.getPointer(CHOLMOD_DENSE_X, size.toLong() * Double.SIZE_BYTES))
    return values.readDoubles(size)
}

private fun freeSparse(functions: SpqrFunctions, slot: COpaquePointerVar, common: CPointer<ByteVar>) {
    if (slot.value != null) functions.freeSparse(slot.ptr, common)
}

private fun freeDense(functions: SpqrFunctions, slot: COpaquePointerVar, common: CPointer<ByteVar>) {
    if (slot.value != null) functions.freeDense(slot.ptr, common)
}

private fun freePermutation(functions: SpqrFunctions, slot: COpaquePointerVar, size: Int, common: CPointer<ByteVar>) {
    slot.value?.let { functions.free(size.toLong(), Int.SIZE_BYTES.toLong(), it, common) }
}
