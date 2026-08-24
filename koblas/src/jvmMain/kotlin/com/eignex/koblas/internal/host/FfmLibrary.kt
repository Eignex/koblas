package com.eignex.koblas.internal.host

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

/**
 * One host library reached with `java.lang.foreign`, and the symbols bound out of it.
 *
 * Every JVM binding koblas ships opens its library the same way: try each candidate name in turn, keep the
 * first that both loads and carries a key symbol, and bind handles out of it lazily. This is that, so a
 * binding contributes only its prototypes. The native targets have `openNativeLibrary` for the same reason.
 *
 * Handles are bound with `Linker.Option.critical` by default, which lets a call read and write the Kotlin
 * arrays handed to it in place rather than copying them.
 *
 * Calls on the result use `invokeExact`, a signature-polymorphic call site: the descriptor lands in the
 * bytecode, where `invokeWithArguments` would box every argument into a varargs array and pay a fixed cost
 * per call that a short routine cannot amortise.
 *
 * The exactness is load-bearing. `invokeExact` converts nothing, so an argument whose Kotlin type does not
 * match its layout throws `WrongMethodTypeException` where `invokeWithArguments` would have widened it
 * silently. Two consequences for the call sites: a routine called for effect whose descriptor returns a
 * value still needs the cast, since Kotlin infers `Unit` in statement position and emits a void descriptor
 * that will not match; and a safe-call chain has to be resolved to a local first, because `as Unit?` is the
 * boxed `Unit` rather than void.
 */
public class FfmLibrary private constructor(
    private val linker: Linker?,
    private val lookups: List<SymbolLookup>,
    private val description: String,
) {
    /** Whether a library resolved. A binding still has to check that the symbols it needs are in it. */
    public val present: Boolean get() = linker != null && lookups.isNotEmpty()

    /**
     * Whether [name] is exported, read with `find`, which is a lookup rather than a binding. Discovery asks
     * this instead of binding, because `Linker.downcallHandle` is stack-hungry enough to overflow when it
     * runs at the depth discovery reaches.
     */
    public fun contains(name: String): Boolean = address(name) != null

    /** Whether every one of [names] is exported. */
    public fun containsAll(names: Iterable<String>): Boolean = names.all(::contains)

    /**
     * A handle for [name], or null when the library does not export it.
     *
     * Pass `critical = false` for a routine that may block or start a thread, which a critical downcall is
     * not allowed to do; the numeric routines, which run to completion over the arrays handed to them, are
     * the reason critical is the default.
     */
    public fun handleOrNull(name: String, descriptor: FunctionDescriptor, critical: Boolean = true): MethodHandle? {
        val downcall = linker ?: return null
        val address = address(name) ?: return null
        return if (critical) {
            downcall.downcallHandle(address, descriptor, CRITICAL)
        } else {
            downcall.downcallHandle(address, descriptor)
        }
    }

    /** A handle for [name], which the library is required to export. */
    public fun handle(name: String, descriptor: FunctionDescriptor, critical: Boolean = true): MethodHandle =
        checkNotNull(handleOrNull(name, descriptor, critical)) { "$description is present but lacks $name" }

    /**
     * This library, then [other], searched in that order. For a host that splits one binding's symbols
     * across two libraries, as a build shipping LAPACKE outside its OpenBLAS does.
     */
    public fun withFallback(other: FfmLibrary?): FfmLibrary =
        if (other == null) this else FfmLibrary(linker, lookups + other.lookups, description)

    private fun address(name: String): MemorySegment? = lookups.firstNotNullOfOrNull { it.find(name).orElse(null) }

    /** Opening a library, and the function descriptors a prototype is written with. */
    public companion object {
        /**
         * Opens the first of [candidates] that loads and exports [keySymbol], or a library that reports
         * itself absent. [description] names it in a diagnostic.
         *
         * Only the two exceptions that mean absence are swallowed, so a host without the library does not
         * crash startup while a StackOverflowError is never read as a missing library.
         */
        public fun open(candidates: List<String>, keySymbol: String, description: String): FfmLibrary {
            val linker = nativeLinker()
            if (linker == null) return FfmLibrary(null, emptyList(), description)
            for (candidate in candidates) {
                val opened = try {
                    SymbolLookup.libraryLookup(candidate, Arena.global())
                } catch (_: IllegalArgumentException) {
                    continue // not on this machine
                } catch (_: UnsatisfiedLinkError) {
                    continue // present but unloadable
                }
                if (opened.find(keySymbol).isPresent) return FfmLibrary(linker, listOf(opened), description)
            }
            return FfmLibrary(linker, emptyList(), description)
        }

        /** `void f(...)`. */
        public fun voidOf(vararg layouts: MemoryLayout): FunctionDescriptor = FunctionDescriptor.ofVoid(*layouts)

        /** `int f(...)`. */
        public fun intOf(vararg layouts: MemoryLayout): FunctionDescriptor = FunctionDescriptor.of(JAVA_INT, *layouts)

        /** `double f(...)`. */
        public fun doubleOf(vararg layouts: MemoryLayout): FunctionDescriptor =
            FunctionDescriptor.of(JAVA_DOUBLE, *layouts)

        /** `int64_t f(...)`. */
        public fun longOf(vararg layouts: MemoryLayout): FunctionDescriptor = FunctionDescriptor.of(JAVA_LONG, *layouts)

        /** `void *f(...)`. */
        public fun pointerOf(vararg layouts: MemoryLayout): FunctionDescriptor = FunctionDescriptor.of(
            ADDRESS,
            *layouts,
        )

        /**
         * Pins the on-heap arrays handed over as segments for the call instead of copying them, blocking
         * relocation while it runs.
         */
        private val CRITICAL: Linker.Option = Linker.Option.critical(true)

        /**
         * Null where this platform has no native linker to hand out. Resolved defensively because a binding's
         * initializer runs on the first backend discovery, and an escape from it would leave the portable
         * reference path, which calls out to nothing, unreachable behind a failed class initialization.
         */
        private fun nativeLinker(): Linker? = try {
            Linker.nativeLinker()
        } catch (_: UnsupportedOperationException) {
            null
        }
    }
}
