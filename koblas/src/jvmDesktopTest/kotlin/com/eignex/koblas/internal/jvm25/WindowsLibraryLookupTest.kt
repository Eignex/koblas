package com.eignex.koblas.internal.jvm25

import com.eignex.koblas.internal.jvm25.host.windowsLibraryLookup
import com.eignex.koblas.vendor.OperatingSystem
import com.eignex.koblas.vendor.hostPlatform
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.ValueLayout.JAVA_INT
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class WindowsLibraryLookupTest {
    @Test
    fun `windows lookup retains system exports and restores thread error mode after failure`() {
        if (hostPlatform().operatingSystem != OperatingSystem.Windows) return
        val linker = Linker.nativeLinker()
        val system = Path.of(System.getenv("SystemRoot"), "System32")
        val lookup = assertNotNull(windowsLibraryLookup(system.resolve("kernel32.dll").toString(), linker))
        val getMode = linker.downcallHandle(
            lookup.find("GetThreadErrorMode").orElseThrow(),
            FunctionDescriptor.of(JAVA_INT),
        )
        val previous = getMode.invokeExact() as Int

        val missing = windowsLibraryLookup(system.resolve("koblas_missing_library.dll").toString(), linker)

        assertNull(missing)
        assertEquals(previous, getMode.invokeExact() as Int)
        assertTrue(lookup.find("GetCurrentProcessId").isPresent)
    }
}
