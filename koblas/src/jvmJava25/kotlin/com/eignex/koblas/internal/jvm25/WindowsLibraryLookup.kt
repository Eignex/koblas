package com.eignex.koblas.internal.jvm25

import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout.ADDRESS
import java.lang.foreign.ValueLayout.JAVA_INT
import java.nio.file.Path

/** Preload with the same dependency search and quiet failure policy as the Kotlin/Native Windows arm. */
internal fun windowsLibraryLookup(candidate: String, linker: Linker): SymbolLookup? {
    // Kernel32 is a Windows KnownDLL, independent of the working directory and PATH.
    val kernel = SymbolLookup.libraryLookup("kernel32.dll", Arena.global())
    val load = linker.downcallHandle(
        kernel.find("LoadLibraryExW").orElseThrow(),
        FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_INT),
    )
    val free = linker.downcallHandle(kernel.find("FreeLibrary").orElseThrow(), FunctionDescriptor.of(JAVA_INT, ADDRESS))
    val getMode = linker.downcallHandle(
        kernel.find("GetThreadErrorMode").orElseThrow(),
        FunctionDescriptor.of(JAVA_INT),
    )
    val setMode = linker.downcallHandle(
        kernel.find("SetThreadErrorMode").orElseThrow(),
        FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS),
    )
    val previous = getMode.invokeExact() as Int
    if ((setMode.invokeExact(previous or FAIL_CRITICAL_ERRORS, MemorySegment.NULL) as Int) == 0) return null
    try {
        Arena.ofConfined().use { arena ->
            val path = arena.allocateFrom(candidate.replace('/', '\\'), Charsets.UTF_16LE)
            val module = load.invokeExact(path, MemorySegment.NULL, SEARCH_DLL_DIRECTORY_AND_DEFAULTS) as MemorySegment
            if (module == MemorySegment.NULL) return null
            try {
                // The FFM lookup retains the already loaded module for the process lifetime.
                return SymbolLookup.libraryLookup(Path.of(candidate), Arena.global())
            } finally {
                val released = free.invokeExact(module) as Int
                check(released != 0) { "Cannot release preloaded Windows library: $candidate" }
            }
        }
    } finally {
        val restored = setMode.invokeExact(previous, MemorySegment.NULL) as Int
        check(restored != 0) { "Cannot restore Windows thread error mode" }
    }
}

private const val FAIL_CRITICAL_ERRORS = 0x0001

// LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS excludes the working directory.
private const val SEARCH_DLL_DIRECTORY_AND_DEFAULTS = 0x1100
