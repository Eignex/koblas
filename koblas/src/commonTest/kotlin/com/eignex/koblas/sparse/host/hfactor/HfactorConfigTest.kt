package com.eignex.koblas.sparse.host.hfactor

import com.eignex.koblas.internal.backend.platformSparseDispatchThresholds
import kotlin.test.Test
import kotlin.test.assertEquals

class HfactorConfigTest {
    @Test
    fun `default metadata reports the sparse factorization gate`() {
        val metadata = HfactorOptions().metadataOptions()

        assertEquals(platformSparseDispatchThresholds.factorize.toString(), metadata["factorizeMin"])
    }
}
