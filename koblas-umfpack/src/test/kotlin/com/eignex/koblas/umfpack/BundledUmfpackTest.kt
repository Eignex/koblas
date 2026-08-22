package com.eignex.koblas.umfpack

import kotlin.test.Test
import kotlin.test.assertTrue

class BundledUmfpackTest {
    @Test
    fun `loads bundled OpenBLAS before UMFPACK`() {
        assertTrue(BundledUmfpack().isAvailable)
    }
}
