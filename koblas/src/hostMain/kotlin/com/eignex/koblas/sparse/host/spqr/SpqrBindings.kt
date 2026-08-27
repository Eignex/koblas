@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.spqr

import com.eignex.koblas.internal.host.openNativeLibrary
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_COMMON_BYTES
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_COMMON_PRINT
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_SONAMES
import com.eignex.koblas.sparse.host.cholmod.CHOLMOD_TRUE
import com.eignex.koblas.sparse.host.cholmod.intAt
import kotlinx.cinterop.*
import platform.posix.dlsym

private typealias Block = CPointer<ByteVar>?
private typealias Handle = COpaquePointer?
private typealias HandleSlot = CPointer<COpaquePointerVar>?

internal class SpqrFunctions(spqr: COpaquePointer, cholmod: COpaquePointer) {
    private fun required(lib: COpaquePointer, name: String, which: String): COpaquePointer =
        dlsym(lib, name) ?: error("$which is present but lacks $name")

    val start = required(cholmod, "cholmod_start", "libcholmod").reinterpret<CFunction<(Block) -> Int>>()

    val freeDense = required(cholmod, "cholmod_free_dense", "libcholmod")
        .reinterpret<CFunction<(HandleSlot, Block) -> Int>>()

    val freeSparse = required(cholmod, "cholmod_free_sparse", "libcholmod")
        .reinterpret<CFunction<(HandleSlot, Block) -> Int>>()

    val free = required(cholmod, "cholmod_free", "libcholmod")
        .reinterpret<CFunction<(Long, Long, Handle, Block) -> Handle>>()

    val factorize = required(spqr, "SuiteSparseQR_C_factorize", "libspqr")
        .reinterpret<CFunction<(Int, Double, Block, Block) -> Handle>>()

    val explicitQr = required(spqr, "SuiteSparseQR_i_C_QR", "libspqr").reinterpret<
        CFunction<(Int, Double, Int, Block, HandleSlot, HandleSlot, HandleSlot, Block) -> Int>,
        >()

    val qmult = required(spqr, "SuiteSparseQR_C_qmult", "libspqr")
        .reinterpret<CFunction<(Int, Block, Block, Block) -> Handle>>()

    val solve = required(spqr, "SuiteSparseQR_C_solve", "libspqr")
        .reinterpret<CFunction<(Int, Block, Block, Block) -> Handle>>()

    val freeQr = required(spqr, "SuiteSparseQR_C_free", "libspqr")
        .reinterpret<CFunction<(HandleSlot, Block) -> Int>>()
}

internal class SpqrLoader(private val config: SpqrConfig) {
    private val spqrLibrary: COpaquePointer? by lazy {
        openNativeLibrary(candidates(SPQR_SONAMES, config.libraryPath), "SuiteSparseQR_C_factorize")
    }

    private val cholmodLibrary: COpaquePointer? by lazy {
        openNativeLibrary(candidates(CHOLMOD_SONAMES, null), "cholmod_start")
    }

    val functions: SpqrFunctions? by lazy {
        val spqr = spqrLibrary ?: return@lazy null
        val cholmod = cholmodLibrary ?: return@lazy null
        SpqrFunctions(spqr, cholmod)
    }

    val available: Boolean by lazy { functions != null }

    val unavailableReason: String?
        get() = when {
            spqrLibrary == null -> "libspqr could not be opened; SPQR does not appear to be installed"
            cholmodLibrary == null -> "libspqr opened but libcholmod, which it runs its workspace against, did not"
            functions == null -> "libspqr opened but its symbols did not resolve"
            else -> null
        }

    private fun candidates(sonames: List<String>, explicit: String?): List<String> = buildList {
        explicit?.let(::add)
        config.searchDirectory?.let { directory ->
            for (soname in sonames) if ('/' !in soname) add("$directory/$soname")
        }
        addAll(sonames)
    }

    fun common(): CPointer<ByteVar> {
        val resolved = checkNotNull(functions) { "SPQR is not available" }
        val common = nativeHeap.allocArray<ByteVar>(CHOLMOD_COMMON_BYTES)
        check(resolved.start(common) == CHOLMOD_TRUE) { "cholmod_start failed" }
        intAt(common, CHOLMOD_COMMON_PRINT, 0)
        return common
    }
}
