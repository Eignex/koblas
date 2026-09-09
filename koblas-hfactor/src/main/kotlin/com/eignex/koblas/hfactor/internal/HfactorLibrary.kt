package com.eignex.koblas.hfactor.internal

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemoryLayout
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_INT
import java.lang.invoke.MethodHandle

/** The small FFM loader owned by the optional HFactor implementation. */
internal class HfactorLibrary private constructor(
    private val linker: Linker?,
    private val lookup: SymbolLookup?,
    val unavailableReason: String?,
) {
    val present: Boolean get() = linker != null && lookup != null

    fun handleOrNull(name: String, descriptor: FunctionDescriptor): MethodHandle? {
        val downcall = linker ?: return null
        val address = lookup?.find(name)?.orElse(null) ?: return null
        return downcall.downcallHandle(address, descriptor, Linker.Option.critical(true))
    }

    companion object {
        fun open(candidates: List<String>, keySymbol: String): HfactorLibrary {
            val linker = try {
                Linker.nativeLinker()
            } catch (_: UnsupportedOperationException) {
                return HfactorLibrary(null, null, "this JVM does not provide a native linker")
            }
            val failures = ArrayList<String>()
            for (candidate in candidates) {
                val opened = try {
                    SymbolLookup.libraryLookup(candidate, Arena.global())
                } catch (cause: IllegalArgumentException) {
                    failures += "$candidate: ${cause.message ?: "not found"}"
                    continue
                } catch (cause: UnsatisfiedLinkError) {
                    failures += "$candidate: ${cause.message ?: "not loadable"}"
                    continue
                }
                if (opened.find(keySymbol).isPresent) return HfactorLibrary(linker, opened, null)
                failures += "$candidate: missing $keySymbol"
            }
            return HfactorLibrary(linker, null, failures.joinToString("; ").ifEmpty { "no library candidate" })
        }

        fun voidOf(vararg layouts: MemoryLayout): FunctionDescriptor = FunctionDescriptor.ofVoid(*layouts)
        fun intOf(vararg layouts: MemoryLayout): FunctionDescriptor = FunctionDescriptor.of(JAVA_INT, *layouts)
        fun pointerOf(vararg layouts: MemoryLayout): FunctionDescriptor = FunctionDescriptor.of(ADDRESS, *layouts)
    }
}
