@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.openNativeLibrary
import kotlinx.cinterop.*
import platform.posix.dlsym

private typealias Block = CPointer<ByteVar>?
private typealias Handle = COpaquePointer?
private typealias HandleSlot = CPointer<COpaquePointerVar>?

/** The 32-bit-index `cholmod_*` family; the `cholmod_l_*` family takes `int64_t` and is a different binding. */
internal class CholmodFunctions(private val lib: COpaquePointer) {
    private fun required(name: String): COpaquePointer =
        dlsym(lib, name) ?: error("libcholmod is present but lacks $name")

    val start = required("cholmod_start").reinterpret<CFunction<(Block) -> Int>>()

    val analyze = required("cholmod_analyze").reinterpret<CFunction<(Block, Block) -> Handle>>()

    val factorize = required("cholmod_factorize").reinterpret<CFunction<(Block, Block, Block) -> Int>>()

    val solve = required("cholmod_solve").reinterpret<CFunction<(Int, Block, Block, Block) -> Handle>>()

    val rcond = required("cholmod_rcond").reinterpret<CFunction<(Block, Block) -> Double>>()

    val sdmult = required("cholmod_sdmult").reinterpret<
        CFunction<(Block, Int, CPointer<DoubleVar>, CPointer<DoubleVar>, Block, Block, Block) -> Int>,
        >()

    val ssmult = required("cholmod_ssmult").reinterpret<
        CFunction<(Block, Block, Int, Int, Int, Block) -> Handle>,
        >()

    val freeFactor = required("cholmod_free_factor").reinterpret<CFunction<(HandleSlot, Block) -> Int>>()

    val freeDense = required("cholmod_free_dense").reinterpret<CFunction<(HandleSlot, Block) -> Int>>()

    val freeSparse = required("cholmod_free_sparse").reinterpret<CFunction<(HandleSlot, Block) -> Int>>()
}

/** Opens the host CHOLMOD and reports whether every routine these bindings need resolved. */
internal class CholmodLoader(private val config: CholmodConfig) {
    private val library: COpaquePointer? by lazy { openNativeLibrary(candidates(), "cholmod_start") }

    val functions: CholmodFunctions? by lazy { library?.let(::CholmodFunctions) }

    val available: Boolean by lazy { functions != null }

    private fun candidates(): List<String> = buildList {
        config.libraryPath?.let(::add)
        config.searchDirectory?.let { directory ->
            for (soname in CHOLMOD_SONAMES) if ('/' !in soname) add("$directory/$soname")
        }
        addAll(CHOLMOD_SONAMES)
    }

    /**
     * A started `cholmod_common`, asking for silence and for the factorization [finalLl] names. The seam's
     * Cholesky promises `L·Lᵀ` and a raise at the first non-positive pivot, where CHOLMOD's own default
     * leaves an `L·D·Lᵀ` that an indefinite matrix factors into without complaint, which is what the seam's
     * `ldl` asks for instead.
     */
    fun common(finalLl: Int = CHOLMOD_TRUE): CPointer<ByteVar> {
        val resolved = checkNotNull(functions) { "CHOLMOD is not available" }
        val common = nativeHeap.allocArray<ByteVar>(CHOLMOD_COMMON_BYTES)
        check(resolved.start(common) == CHOLMOD_TRUE) { "cholmod_start failed" }
        intAt(common, CHOLMOD_COMMON_FINAL_LL, finalLl)
        intAt(common, CHOLMOD_COMMON_PRINT, 0)
        return common
    }
}

/** Writes one int into a `cholmod_*` block. */
internal fun intAt(block: CPointer<ByteVar>, offset: Long, value: Int) {
    (block + offset)!!.reinterpret<IntVar>().pointed.value = value
}

/** Reads one int out of a `cholmod_*` block. */
internal fun intAt(block: CPointer<ByteVar>, offset: Long): Int = (block + offset)!!.reinterpret<IntVar>().pointed.value

/** Writes one `size_t` into a `cholmod_*` block. */
internal fun sizeAt(block: CPointer<ByteVar>, offset: Long, value: Long) {
    (block + offset)!!.reinterpret<LongVar>().pointed.value = value
}

/** Reads one `size_t` out of a `cholmod_*` block. */
internal fun sizeAt(block: CPointer<ByteVar>, offset: Long): Long =
    (block + offset)!!.reinterpret<LongVar>().pointed.value

/** Writes one pointer into a `cholmod_*` block. */
internal fun pointerAt(block: CPointer<ByteVar>, offset: Long, value: COpaquePointer?) {
    (block + offset)!!.reinterpret<COpaquePointerVar>().pointed.value = value
}

/** Reads one pointer out of a `cholmod_*` block. */
internal fun pointerAt(block: CPointer<ByteVar>, offset: Long): COpaquePointer? =
    (block + offset)!!.reinterpret<COpaquePointerVar>().pointed.value

/**
 * [a]'s lower triangle as a `cholmod_sparse` over a native copy, which the caller frees with [freeMatrix].
 *
 * The copy is what taking operands as a struct costs: a struct field holds an address, and a Kotlin array
 * pinned for the length of one call has none to leave behind.
 */
internal fun describeLowerTriangle(a: F64SparseMatrix): CPointer<ByteVar> = describeMatrix(a, CHOLMOD_STYPE_LOWER)

/** [a] as a general `cholmod_sparse`, including every stored entry. */
internal fun describeGeneral(a: F64SparseMatrix): CPointer<ByteVar> = describeMatrix(a, CHOLMOD_STYPE_GENERAL)

private fun describeMatrix(a: F64SparseMatrix, stype: Int): CPointer<ByteVar> {
    val nnz = maxOf(a.nnz, 1)
    val columnPointers = nativeHeap.allocArray<IntVar>(a.cols + 1)
    for (j in 0..a.cols) columnPointers[j] = a.colPtr[j]
    val rowIndices = nativeHeap.allocArray<IntVar>(nnz)
    for (k in 0 until a.nnz) rowIndices[k] = a.rowIdx[k]
    val values = nativeHeap.allocArray<DoubleVar>(nnz)
    for (k in 0 until a.nnz) values[k] = a.values[k]

    val sparse = nativeHeap.allocArray<ByteVar>(CHOLMOD_SPARSE_BYTES)
    sizeAt(sparse, CHOLMOD_SPARSE_NROW, a.rows.toLong())
    sizeAt(sparse, CHOLMOD_SPARSE_NCOL, a.cols.toLong())
    sizeAt(sparse, CHOLMOD_SPARSE_NZMAX, a.nnz.toLong())
    pointerAt(sparse, CHOLMOD_SPARSE_P, columnPointers)
    pointerAt(sparse, CHOLMOD_SPARSE_I, rowIndices)
    // Null because the columns are packed, so their lengths come from the pointers alone.
    pointerAt(sparse, CHOLMOD_SPARSE_NZ, null)
    pointerAt(sparse, CHOLMOD_SPARSE_X, values)
    pointerAt(sparse, CHOLMOD_SPARSE_Z, null)
    intAt(sparse, CHOLMOD_SPARSE_STYPE, stype)
    intAt(sparse, CHOLMOD_SPARSE_ITYPE, CHOLMOD_INT)
    intAt(sparse, CHOLMOD_SPARSE_XTYPE, CHOLMOD_REAL)
    intAt(sparse, CHOLMOD_SPARSE_DTYPE, CHOLMOD_DOUBLE)
    // koblas keeps rows strictly ascending within a column, which is what sorted claims.
    intAt(sparse, CHOLMOD_SPARSE_SORTED, CHOLMOD_TRUE)
    intAt(sparse, CHOLMOD_SPARSE_PACKED, CHOLMOD_TRUE)
    return sparse
}

/** Frees a matrix built by [describeLowerTriangle], arrays included. */
internal fun freeMatrix(sparse: CPointer<ByteVar>) {
    pointerAt(sparse, CHOLMOD_SPARSE_P)?.let { nativeHeap.free(it) }
    pointerAt(sparse, CHOLMOD_SPARSE_I)?.let { nativeHeap.free(it) }
    pointerAt(sparse, CHOLMOD_SPARSE_X)?.let { nativeHeap.free(it) }
    nativeHeap.free(sparse)
}
