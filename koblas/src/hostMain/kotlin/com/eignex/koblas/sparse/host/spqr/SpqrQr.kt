@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.spqr

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.requireShape
import com.eignex.koblas.sparse.F64SparseQrFactorization
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_SPARSE_I
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_SPARSE_NCOL
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_SPARSE_NROW
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_SPARSE_P
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_SPARSE_X
import com.eignex.koblas.sparse.host.cholmod.describeGeneral
import com.eignex.koblas.sparse.host.cholmod.freeMatrix
import com.eignex.koblas.sparse.host.cholmod.pointerAt
import com.eignex.koblas.sparse.host.cholmod.sizeAt
import kotlinx.cinterop.*

/**
 * SPQR's sparse QR, the routine a SuiteSparse backend fills its QR half with rather than a backend of its
 * own: SPQR has no LU and no Cholesky, so one offering it alone would take the seam from one offering all.
 */
public class SpqrQr(config: SpqrConfig = SpqrConfig()) {
    private val loader = SpqrLoader(config)
    private val ordering = config.options.ordering
    private val tolerance = config.options.rankTolerance
    private val orderingName = ordering.name.lowercase()

    /** Whether both libraries opened and every symbol resolved. */
    public val isAvailable: Boolean get() = loader.available

    /** Why SPQR is unusable, or null when it is usable. */
    public val unavailableReason: String? get() = loader.unavailableReason

    /**
     * Factor [a], or null when the library is unusable.
     *
     * @throws IllegalArgumentException if [a] has fewer rows than columns.
     */
    public fun factor(a: F64SparseMatrix): F64SparseQrFactorization? {
        requireShape(a.rows >= a.cols) {
            "qr: A is ${a.rows}x${a.cols}, which is wider than it is tall; factor its transpose instead"
        }
        val functions = loader.functions ?: return null
        val common = loader.common()
        val sparse = describeGeneral(a)
        val factor = try {
            functions.factorize(ordering.code(), tolerance, sparse, common)
        } finally {
            freeMatrix(sparse)
        }
        if (factor == null) {
            nativeHeap.free(common)
            return null
        }
        val handle = SpqrQrFactorization.SpqrHandle(factor, common, functions, a.rows)
        return SpqrQrFactorization(handle, functions, { explicitFactors(functions, a) }, a.rows, a.cols, orderingName)
    }

    private fun explicitFactors(functions: SpqrFunctions, a: F64SparseMatrix): SpqrExplicit {
        val common = loader.common()
        val sparse = describeGeneral(a)
        return try {
            memScoped {
                val rSlot = alloc<COpaquePointerVar>()
                val eSlot = alloc<COpaquePointerVar>()
                val rank = functions.explicitQr(
                    ordering.code(),
                    tolerance,
                    a.cols,
                    sparse,
                    null,
                    rSlot.ptr,
                    eSlot.ptr,
                    common,
                )
                val r = rSlot.value
                check(rank >= 0 && r != null) { "SuiteSparseQR_i_C_QR failed on a matrix it factorized" }
                try {
                    val order = readPermutation(eSlot.value, a.cols)
                    SpqrExplicit(rank, readSparse(r.reinterpret()), order)
                } finally {
                    rSlot.value = r
                    functions.freeSparse(rSlot.ptr, common)
                    eSlot.value?.let { functions.free(a.cols.toLong(), Int.SIZE_BYTES.toLong(), it, common) }
                }
            }
        } finally {
            freeMatrix(sparse)
            nativeHeap.free(common)
        }
    }
}

internal class SpqrExplicit(val rank: Int, val r: F64SparseMatrix, val columnOrder: IntArray)

private fun readPermutation(e: COpaquePointer?, cols: Int): IntArray {
    if (e == null) return IntArray(cols) { it }
    val entries = e.reinterpret<IntVar>()
    return IntArray(cols) { entries[it] }
}

private fun readSparse(sparse: CPointer<ByteVar>): F64SparseMatrix {
    val rows = sizeAt(sparse, CHOLMOD_SPARSE_NROW).toInt()
    val cols = sizeAt(sparse, CHOLMOD_SPARSE_NCOL).toInt()
    val p = pointerAt(sparse, CHOLMOD_SPARSE_P)!!.reinterpret<IntVar>()
    val colPtr = IntArray(cols + 1) { p[it] }
    val nnz = colPtr[cols]
    val rowIdx = IntArray(nnz)
    val values = DoubleArray(nnz)
    if (nnz > 0) {
        val i = pointerAt(sparse, CHOLMOD_SPARSE_I)!!.reinterpret<IntVar>()
        val x = pointerAt(sparse, CHOLMOD_SPARSE_X)!!.reinterpret<DoubleVar>()
        for (k in 0 until nnz) {
            rowIdx[k] = i[k]
            values[k] = x[k]
        }
    }
    return F64SparseMatrix.wrap(rows, cols, colPtr, rowIdx, values)
}

internal fun suiteSparseQr(libraryPath: String?): SpqrQr =
    SpqrQr(SpqrConfig(searchDirectory = libraryPath?.parentDirectory()))

private fun String.parentDirectory(): String? = substringBeforeLast('/', missingDelimiterValue = "").ifEmpty { null }
