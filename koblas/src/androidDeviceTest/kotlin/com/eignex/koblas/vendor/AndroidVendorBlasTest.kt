// detekt exempts test files by path, and its list predates AGP's androidDeviceTest name, so this file would
// otherwise be held to the documentation and naming rules for main sources.
@file:Suppress("UndocumentedPublicClass", "UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.vendor

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class AndroidVendorBlasTest {
    @Test
    fun `the bundled library loads on a device`() {
        // The shared vendor suite skips where no library loads, so a device where this one failed would pass
        // it having run nothing. On a device the library is part of the package, so absence is a failure.
        val blas = assertNotNull(openBlas(), "the bundled OpenBLAS did not load or failed its checks")

        assertEquals(Vendor.OpenBlas, blas.vendor)
    }

    @Test
    fun `the bundled library confirms one compute thread`() {
        val blas = assertNotNull(openBlas())

        assertEquals(ThreadEvidence.Confirmed, blas.threadEvidence)
    }

    @Test
    fun `the resolved library is the one the package carries`() {
        val blas = assertNotNull(openBlas())

        assertTrue(blas.libraryPath.endsWith("lib${AndroidCblas.LIBRARY}.so"), blas.libraryPath)
    }
}
