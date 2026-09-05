@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.cholmod

import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.internal.host.HeapBlocks
import com.eignex.koblas.internal.host.NativeBlock
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

    /**
     * Releases the workspace `cholmod_start` left NULL and the routines then filled in.
     *
     * Freeing the common block alone leaks it: `analyze` and `factorize` allocate `Flag`, `Head`, `Iwork`
     * and `Xwork` inside the struct, roughly 40 bytes per row, and nothing else ever frees them.
     */
    val finish = required("cholmod_finish").reinterpret<CFunction<(Block) -> Int>>()

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

    // Optional: without them a factorization still solves and only reading its factors is refused.
    val copyFactor = dlsym(lib, "cholmod_copy_factor")?.reinterpret<CFunction<(Block, Block) -> Handle>>()

    val changeFactor = dlsym(lib, "cholmod_change_factor")
        ?.reinterpret<CFunction<(Int, Int, Int, Int, Int, Block, Block) -> Int>>()
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
        NativeBlock(common).putInt(CHOLMOD_COMMON_FINAL_LL, finalLl)
        NativeBlock(common).putInt(CHOLMOD_COMMON_PRINT, 0)
        return common
    }
}

/**
 * [a]'s lower triangle as a `cholmod_sparse` over a native copy, which the caller frees with [freeMatrix].
 *
 * The copy is what taking operands as a struct costs: a struct field holds an address, and a Kotlin array
 * pinned for the length of one call has none to leave behind.
 */
internal fun describeLowerTriangle(a: F64SparseMatrix): CPointer<ByteVar> =
    HeapBlocks.describeCholmodSparse(a, CHOLMOD_STYPE_LOWER).pointer

/** [a] as a general `cholmod_sparse`, including every stored entry. */
internal fun describeGeneral(a: F64SparseMatrix): CPointer<ByteVar> =
    HeapBlocks.describeCholmodSparse(a, CHOLMOD_STYPE_GENERAL).pointer

/** Frees a matrix built by [describeLowerTriangle], arrays included. */
internal fun freeMatrix(sparse: CPointer<ByteVar>) {
    val descriptor = NativeBlock(sparse)
    // The arrays are freed rather than read, so the span asked for here does not matter.
    for (field in longArrayOf(CHOLMOD_SPARSE_P, CHOLMOD_SPARSE_I, CHOLMOD_SPARSE_X)) {
        descriptor.getPointer(field, 0L)?.let { nativeHeap.free(it.pointer) }
    }
    nativeHeap.free(sparse)
}
