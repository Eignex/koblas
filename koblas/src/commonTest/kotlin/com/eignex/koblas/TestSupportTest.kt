package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertFails

class TestSupportTest {
    @Test
    fun `assertClose accepts only matching infinities`() {
        assertClose(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, "positive infinity")
        assertClose(Double.NEGATIVE_INFINITY, Double.NEGATIVE_INFINITY, "negative infinity")

        assertFails { assertClose(Double.POSITIVE_INFINITY, 0.0, "infinity against finite") }
        assertFails { assertClose(0.0, Double.POSITIVE_INFINITY, "finite against infinity") }
        assertFails { assertClose(Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, "opposite infinities") }
    }

    @Test
    fun `assertClose rejects nan inputs`() {
        assertFails { assertClose(Double.NaN, 0.0, "nan oracle") }
        assertFails { assertClose(0.0, Double.NaN, "nan result") }
    }
}
