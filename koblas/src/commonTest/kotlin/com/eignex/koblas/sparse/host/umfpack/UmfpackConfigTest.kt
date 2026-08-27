package com.eignex.koblas.sparse.host.umfpack

import kotlin.test.Test
import kotlin.test.assertFailsWith

class UmfpackConfigTest {
    @Test
    fun `rejects a negative sparse QR gate`() {
        assertFailsWith<IllegalArgumentException> { UmfpackConfig(qrFactorizeMin = -1) }
    }
}
