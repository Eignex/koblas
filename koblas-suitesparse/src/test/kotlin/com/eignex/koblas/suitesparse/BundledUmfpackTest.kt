package com.eignex.koblas.suitesparse

import com.eignex.koblas.sparse.host.umfpack.UmfpackSparseLu
import kotlin.test.Test
import kotlin.test.assertIs
import kotlin.test.assertTrue

class BundledUmfpackTest {
    @Test
    fun `loads bundled OpenBLAS before UMFPACK`() {
        assertTrue(BundledUmfpack().isAvailable)
    }

    /** A caller reaching this library by name gets its own routines only if the two share a type. */
    @Test
    fun `the bundled UMFPACK is the binding rather than a wrapper around it`() {
        assertIs<UmfpackSparseLu>(BundledUmfpack(), "the bundled providers all answer as the type their binding is")
    }
}
