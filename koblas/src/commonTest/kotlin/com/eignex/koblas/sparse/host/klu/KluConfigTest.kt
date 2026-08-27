package com.eignex.koblas.sparse.host.klu

import kotlin.test.Test
import kotlin.test.assertFailsWith

class KluConfigTest {

    @Test
    fun `rejects an invalid pivot tolerance`() {
        assertFailsWith<IllegalArgumentException> { KluConfig(pivotTolerance = 1.1) }
    }

    @Test
    fun `rejects a nonpositive memory growth`() {
        assertFailsWith<IllegalArgumentException> { KluConfig(memoryGrowth = 0.0) }
    }

    @Test
    fun `rejects a negative sparse QR gate`() {
        assertFailsWith<IllegalArgumentException> { KluConfig(qrFactorizeMin = -1) }
    }
}
