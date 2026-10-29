@file:Suppress("FunctionNaming")
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package com.eignex.koblas.vendor

import com.eignex.koblas.internal.host.NativeLibrary
import platform.windows.GetThreadErrorMode
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Exercise the real Windows loader without requiring an installed BLAS. */
class NativeLibraryTest {
    /** A symbol's owning DLL can differ from the requested module when an export is forwarded. */
    @Test
    fun `a windows symbol reports an absolute path for its owning dll`() {
        val library = assertNotNull(NativeLibrary.open("kernel32.dll"))
        try {
            assertNotNull(library.symbol("GetCurrentProcessId"))
            val owner = assertNotNull(library.symbolOwner("GetCurrentProcessId"))

            assertTrue(owner.contains(":\\"), owner)
            assertTrue(owner.endsWith(".dll", ignoreCase = true), owner)
            assertNull(library.symbol("koblas_missing_symbol"))
            assertNull(library.symbolOwner("koblas_missing_symbol"))
        } finally {
            library.close()
        }
    }

    /** An absent optional runtime must leave portable kernels available. */
    @Test
    fun `a missing windows library returns null`() {
        val previous = GetThreadErrorMode()

        assertNull(NativeLibrary.open("koblas_missing_library.dll"))
        assertEquals(previous, GetThreadErrorMode())
    }
}
