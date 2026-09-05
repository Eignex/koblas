@file:OptIn(ExperimentalForeignApi::class)

package com.eignex.koblas.sparse.host.klu

import com.eignex.koblas.internal.host.NativeBlock
import com.eignex.koblas.internal.host.openNativeLibrary
import kotlinx.cinterop.*
import platform.posix.dlsym
import platform.posix.memset

private typealias Ip = CPointer<IntVar>?
private typealias Dp = CPointer<DoubleVar>?
private typealias Common = CPointer<ByteVar>?
private typealias Handle = COpaquePointer?

/** The 32-bit-index ABI KLU 2 exports; the 64-bit family is named `klu_l_*` and is a different binding. */
internal class KluFunctions(private val lib: COpaquePointer) {
    private fun required(name: String): COpaquePointer = dlsym(lib, name) ?: error("libklu is present but lacks $name")

    val defaults = required("klu_defaults").reinterpret<CFunction<(Common) -> Int>>()

    val analyze = required("klu_analyze").reinterpret<CFunction<(Int, Ip, Ip, Common) -> Handle>>()

    val factor = required("klu_factor").reinterpret<CFunction<(Ip, Ip, Dp, Handle, Common) -> Handle>>()

    val refactor = required("klu_refactor").reinterpret<CFunction<(Ip, Ip, Dp, Handle, Handle, Common) -> Int>>()

    val rcond = required("klu_rcond").reinterpret<CFunction<(Handle, Handle, Common) -> Int>>()

    val solve = required("klu_solve").reinterpret<CFunction<(Handle, Handle, Int, Int, Dp, Common) -> Int>>()

    val transposedSolve =
        required("klu_tsolve").reinterpret<CFunction<(Handle, Handle, Int, Int, Dp, Common) -> Int>>()

    val freeSymbolic =
        required("klu_free_symbolic").reinterpret<CFunction<(CPointer<COpaquePointerVar>?, Common) -> Int>>()

    val freeNumeric =
        required("klu_free_numeric").reinterpret<CFunction<(CPointer<COpaquePointerVar>?, Common) -> Int>>()

    // Optional: without it a factorization still solves and only reading its factors is refused.
    @Suppress("MagicNumber") // the prototype's arity is an ABI fact
    val extract = dlsym(lib, "klu_extract")?.reinterpret<
        CFunction<(Handle, Handle, Ip, Ip, Dp, Ip, Ip, Dp, Ip, Ip, Dp, Ip, Ip, Dp, Ip, Common) -> Int>,
        >()
}

/** Opens the host KLU and reports whether every routine these bindings need resolved. */
internal class KluLoader(private val config: KluConfig) {
    private val library: COpaquePointer? by lazy {
        openNativeLibrary(config.libraryPath?.let(::listOf) ?: KLU_SONAMES, "klu_defaults")
    }

    val functions: KluFunctions? by lazy { library?.let(::KluFunctions) }

    val available: Boolean by lazy { functions != null }

    /** A `klu_common` filled with KLU's defaults and then this configuration, as the JVM binding does. */
    fun common(equilibrate: Boolean): CPointer<ByteVar> {
        val functions = checkNotNull(functions) { "KLU 2 is not available" }
        val common = nativeHeap.allocArray<ByteVar>(KLU_COMMON_BYTES)
        memset(common, 0, KLU_COMMON_BYTES.convert())
        check(functions.defaults(common) == 1) { "klu_defaults failed" }
        NativeBlock(common).applyKluConfig(config, equilibrate)
        return common
    }
}

/** Reads one int out of a `klu_common` or `klu_numeric` block. */
internal fun intAt(block: CPointer<ByteVar>, offset: Long): Int = (block + offset)!!.reinterpret<IntVar>().pointed.value

/** Reads one double out of a `klu_common` block. */
internal fun doubleAt(block: CPointer<ByteVar>, offset: Long): Double =
    (block + offset)!!.reinterpret<DoubleVar>().pointed.value
