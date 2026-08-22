package com.eignex.koblas.sparse.host.superlu

import com.eignex.koblas.internal.backend.nativeLibraryPaths
import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_DOUBLE
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.invoke.MethodHandle

/** The compact C bridge over SuperLU's factor structures, which are deliberately not part of koblas's ABI. */
internal object SuperluCalls {
    private val linker: Linker? = try {
        Linker.nativeLinker()
    } catch (_: UnsupportedOperationException) {
        null
    }

    private val critical = Linker.Option.critical(true)

    private val lookup: SymbolLookup? by lazy { openSuperlu() }

    @Volatile
    private var bindingFailure: String? = null

    private val handles: Handles? by lazy { bindAll() }

    private class Handles(
        val factorize: MethodHandle,
        val solve: MethodHandle,
        val determinant: MethodHandle,
        val free: MethodHandle,
    )

    val libraryPresent: Boolean get() = linker != null && lookup != null

    val available: Boolean get() = handles != null

    val unavailableReason: String?
        get() = when {
            lookup == null -> "libkoblas-superlu could not be opened"
            handles == null -> "libkoblas-superlu opened but its symbols did not bind: ${bindingFailure ?: "unknown"}"
            else -> null
        }

    private fun openSuperlu(): SymbolLookup? {
        for (soname in nativeLibraryPaths("koblas.superlu.path", "KOBLAS_SUPERLU_PATH", SUPERLU_SONAMES)) {
            val opened = try {
                SymbolLookup.libraryLookup(soname, Arena.global())
            } catch (_: IllegalArgumentException) {
                continue
            } catch (_: UnsatisfiedLinkError) {
                continue
            }
            if (opened.find(FACTORIZE).isPresent) return opened
        }
        return null
    }

    private fun bindAll(): Handles? {
        val found = lookup ?: return null
        val factorize = bind(
            found,
            FACTORIZE,
            FunctionDescriptor.of(ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS),
        )
        val solve = bind(found, SOLVE, FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT))
        val determinant = bind(found, DETERMINANT, FunctionDescriptor.of(JAVA_DOUBLE, ADDRESS))
        val free = bind(found, FREE, FunctionDescriptor.ofVoid(ADDRESS))
        if (factorize == null || solve == null || determinant == null || free == null) {
            bindingFailure = "the host bridge lacks one of the koblas_superlu symbols"
            return null
        }
        return Handles(factorize, solve, determinant, free)
    }

    private fun bind(found: SymbolLookup, name: String, descriptor: FunctionDescriptor): MethodHandle? {
        val downcall = linker ?: return null
        val address = found.find(name).orElse(null) ?: return null
        return downcall.downcallHandle(address, descriptor, critical)
    }

    private fun handlesOrThrow(): Handles =
        checkNotNull(handles) { "SuperLU is not available: ${unavailableReason ?: "unknown reason"}" }

    fun factorize(
        n: Int,
        values: MemorySegment,
        rowIdx: MemorySegment,
        colPtr: MemorySegment,
        status: MemorySegment,
        lnz: MemorySegment,
        unz: MemorySegment,
    ): MemorySegment = handlesOrThrow().factorize.invokeWithArguments(
        n,
        (rowIdx.byteSize() / JAVA_INT.byteSize()).toInt(),
        values,
        rowIdx,
        colPtr,
        status,
        lnz,
        unz,
    ) as MemorySegment

    fun solve(handle: MemorySegment, rhs: MemorySegment, transpose: Boolean): Int =
        handlesOrThrow().solve.invokeWithArguments(handle, rhs, if (transpose) 1 else 0) as Int

    fun determinant(handle: MemorySegment): Double = handlesOrThrow().determinant.invokeWithArguments(handle) as Double

    fun free(handle: MemorySegment) {
        handlesOrThrow().free.invokeWithArguments(handle)
    }
}

private const val FACTORIZE = "koblas_superlu_factorize"
private const val SOLVE = "koblas_superlu_solve"
private const val DETERMINANT = "koblas_superlu_determinant"
private const val FREE = "koblas_superlu_free"

internal val SUPERLU_SONAMES = listOf(
    "libkoblas-superlu.so",
    "libkoblas-superlu.dylib",
    "koblas-superlu.dll",
)
