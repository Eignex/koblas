package com.eignex.koblas.basiclu

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemoryLayout
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.foreign.ValueLayout.JAVA_LONG
import java.lang.invoke.MethodHandle

internal class BasicluCalls(libraryPath: String?) {
    private val linker: Linker? = runCatching { Linker.nativeLinker() }.getOrNull()
    private val critical = Linker.Option.critical(true)
    private val lookup: SymbolLookup? by lazy { open(libraryPath) }
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

    private fun open(path: String?): SymbolLookup? {
        val paths = path?.let(::listOf) ?: BASICLU_SONAMES
        for (candidate in paths) {
            val lookup = try {
                SymbolLookup.libraryLookup(candidate, Arena.global())
            } catch (_: IllegalArgumentException) {
                continue
            } catch (_: UnsatisfiedLinkError) {
                continue
            }
            if (lookup.find("koblas_basiclu_create").isPresent) return lookup
        }
        return null
    }

    private class MissingSymbol : RuntimeException()

    private fun bindAll(): Handles? = try {
        val found = lookup ?: return null
        fun bind(name: String, result: MemoryLayout, vararg args: MemoryLayout): MethodHandle {
            val address = found.find(name).orElse(null) ?: throw MissingSymbol()
            return checkNotNull(linker).downcallHandle(
                address,
                FunctionDescriptor.of(result, *args),
                critical,
            )
        }
        fun bindVoid(name: String, vararg args: MemoryLayout): MethodHandle {
            val address = found.find(name).orElse(null) ?: throw MissingSymbol()
            return checkNotNull(linker).downcallHandle(
                address,
                FunctionDescriptor.ofVoid(*args),
                critical,
            )
        }
        Handles(
            create = bind("koblas_basiclu_create", ADDRESS, JAVA_LONG),
            free = bindVoid("koblas_basiclu_free", ADDRESS),
            factorize = bind(
                "koblas_basiclu_factorize",
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                ADDRESS,
                ADDRESS,
                ADDRESS,
            ),
            solve = bind("koblas_basiclu_solve", JAVA_LONG, ADDRESS, ADDRESS, ADDRESS, JAVA_INT),
            prepareUpdate = bind(
                "koblas_basiclu_prepare_update",
                JAVA_LONG,
                ADDRESS,
                JAVA_LONG,
                ADDRESS,
                ADDRESS,
                JAVA_LONG,
            ),
            update = bind("koblas_basiclu_update", JAVA_LONG, ADDRESS),
            info = bind("koblas_basiclu_info", JAVA_DOUBLE, ADDRESS, JAVA_LONG),
        )
    } catch (_: MissingSymbol) {
        null
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
