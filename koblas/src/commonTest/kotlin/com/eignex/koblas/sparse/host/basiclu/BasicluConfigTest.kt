package com.eignex.koblas.sparse.host.basiclu

import com.eignex.koblas.internal.backend.platformSparseDispatchThresholds
import kotlin.test.Test
import kotlin.test.assertEquals

class BasicluConfigTest {
    @Test
    fun `default metadata reports the sparse factorization gate`() {
        val metadata = BasicluOptions().metadataOptions()

        assertEquals(platformSparseDispatchThresholds.factorize.toString(), metadata["factorizeMin"])
    }
}
