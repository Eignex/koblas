package com.eignex.koblas.sparse.host.hfactor

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertNotNull

class HfactorDirectApiTest {
    @Test
    fun `an unavailable explicit library reports its reason`() {
        val hfactor = HfactorSparseLu(HfactorConfig(libraryPath = "/no/such/hfactor/library"))

        val availability = hfactor.availability

        assertFalse(availability.available)
        assertNotNull(availability.reason)
    }
}
