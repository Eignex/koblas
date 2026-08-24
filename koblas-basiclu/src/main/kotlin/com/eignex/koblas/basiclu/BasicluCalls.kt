package com.eignex.koblas.basiclu

import com.eignex.koblas.internal.host.FfmLibrary
import com.eignex.koblas.internal.host.FfmLibrary.Companion.doubleOf
import com.eignex.koblas.internal.host.FfmLibrary.Companion.longOf
import com.eignex.koblas.internal.host.FfmLibrary.Companion.pointerOf
import com.eignex.koblas.internal.host.FfmLibrary.Companion.voidOf
import java.lang.foreign.MemorySegment
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.lang.invoke.MethodHandle

internal class BasicluCalls(libraryPath: String?) {
    private val library: FfmLibrary by lazy {
        FfmLibrary.open(
            libraryPath?.let(::listOf) ?: BASICLU_SONAMES,
            "koblas_basiclu_create",
            "BASICLU",
        )
    }
    private val handles: Handles? by lazy { bindAll() }

    private class Handles(
        val create: MethodHandle,
        val free: MethodHandle,
        val factorize: MethodHandle,
        val solve: MethodHandle,
        val prepareUpdate: MethodHandle,
        val update: MethodHandle,
        val info: MethodHandle,
    )

    val available: Boolean get() = handles != null

    private fun bindAll(): Handles? {
        if (!library.present) return null
        return Handles(
            create = library.handleOrNull("koblas_basiclu_create", pointerOf(JAVA_LONG)) ?: return null,
            free = library.handleOrNull("koblas_basiclu_free", voidOf(ADDRESS)) ?: return null,
            factorize = library.handleOrNull(
                "koblas_basiclu_factorize",
                longOf(ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS),
            ) ?: return null,
            solve = library.handleOrNull(
                "koblas_basiclu_solve",
                longOf(ADDRESS, ADDRESS, ADDRESS, JAVA_INT),
            ) ?: return null,
            prepareUpdate = library.handleOrNull(
                "koblas_basiclu_prepare_update",
                longOf(ADDRESS, JAVA_LONG, ADDRESS, ADDRESS, JAVA_LONG),
            ) ?: return null,
            update = library.handleOrNull("koblas_basiclu_update", longOf(ADDRESS)) ?: return null,
            info = library.handleOrNull("koblas_basiclu_info", doubleOf(ADDRESS, JAVA_LONG)) ?: return null,
        )
    }

    private fun MethodHandle.call(vararg arguments: Any?): Any? = invokeWithArguments(*arguments)

    fun factorize(a: Int, begin: LongArray, end: LongArray, rows: LongArray, values: DoubleArray): BasicluHandle? {
        val h = checkNotNull(handles) { "BASICLU is not available" }
        val handle = h.create.call(a.toLong()) as MemorySegment
        check(handle.address() != 0L) { "BASICLU could not allocate a factorization" }
        val status = h.factorize.call(
            handle,
            MemorySegment.ofArray(begin),
            MemorySegment.ofArray(end),
            MemorySegment.ofArray(rows),
            MemorySegment.ofArray(values),
        ) as Long
        return when (status) {
            BASICLU_OK -> BasicluHandle(handle)

            BASICLU_WARNING_SINGULAR -> {
                free(handle)
                null
            }

            else -> {
                free(handle)
                error("BASICLU factorize failed with status $status")
            }
        }
    }

    fun solve(handle: BasicluHandle, rhs: DoubleArray, out: DoubleArray, transpose: Boolean) {
        val h = checkNotNull(handles) { "BASICLU is not available" }
        check(
            h.solve.call(
                handle.address,
                MemorySegment.ofArray(rhs),
                MemorySegment.ofArray(out),
                if (transpose) 1 else 0,
            ) == BASICLU_OK,
        ) {
            "BASICLU solve failed"
        }
    }

    fun replaceColumn(handle: BasicluHandle, entering: LongArray, values: DoubleArray, column: Int): Long {
        val h = checkNotNull(handles) { "BASICLU is not available" }
        val prepared = h.prepareUpdate.call(
            handle.address,
            entering.size.toLong(),
            MemorySegment.ofArray(entering),
            MemorySegment.ofArray(values),
            column.toLong(),
        ) as Long
        return if (prepared == BASICLU_OK) h.update.call(handle.address) as Long else prepared
    }

    fun info(handle: BasicluHandle, index: Int): Double = checkNotNull(handles) { "BASICLU is not available" }
        .info.call(handle.address, index.toLong()) as Double

    fun free(handle: BasicluHandle) = free(handle.address)

    private fun free(handle: MemorySegment) {
        handles?.free?.call(handle)
    }
}

internal class BasicluHandle(val address: MemorySegment)

internal val BASICLU_SONAMES: List<String> = listOf("libkoblas_basiclu.so.1", "libkoblas_basiclu.1.dylib")
internal const val BASICLU_OK = 0L
internal const val BASICLU_REALLOCATE = 1L
internal const val BASICLU_WARNING_SINGULAR = 2L
internal const val BASICLU_ERROR_MAXIMUM_UPDATES = -5L
internal const val BASICLU_ERROR_SINGULAR_UPDATE = -6L
internal const val BASICLU_LNZ = 76
internal const val BASICLU_UNZ = 77
internal const val BASICLU_MIN_PIVOT = 79
internal const val BASICLU_MAX_PIVOT = 80
