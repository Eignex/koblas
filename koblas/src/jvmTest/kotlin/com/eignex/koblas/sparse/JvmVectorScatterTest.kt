package com.eignex.koblas.sparse

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class JvmVectorScatterTest {

    @Test
    fun `the property takes precedence over the environment variable`() {
        assertEquals(
            JvmVectorScatterMode.OFF,
            JvmVectorScatterMode.configured("off", "on"),
        )
    }

    @Test
    fun `the scatter mode accepts its documented values`() {
        assertEquals(JvmVectorScatterMode.AUTO, JvmVectorScatterMode.configured(null, null))
        assertEquals(JvmVectorScatterMode.AUTO, JvmVectorScatterMode.configured(" AUTO ", null))
        assertEquals(JvmVectorScatterMode.ON, JvmVectorScatterMode.configured(null, "on"))
        assertEquals(JvmVectorScatterMode.OFF, JvmVectorScatterMode.configured("off", null))
    }

    @Test
    fun `an unknown scatter mode fails clearly`() {
        assertFailsWith<IllegalStateException> {
            JvmVectorScatterMode.configured("sometimes", null)
        }
    }

    @Test
    fun `on enables indexed stores despite automatic eligibility`() {
        val enabled = jvmVectorScatterEnabled(
            JvmVectorScatterMode.ON,
            vectorApiAvailable = true,
            autoScatterEligible = false,
        )

        assertEquals(true, enabled)
    }

    @Test
    fun `off retains scalar stores despite automatic eligibility`() {
        val enabled = jvmVectorScatterEnabled(
            JvmVectorScatterMode.OFF,
            vectorApiAvailable = true,
            autoScatterEligible = true,
        )

        assertEquals(false, enabled)
    }

    @Test
    fun `on requires the Vector API module`() {
        assertFailsWith<IllegalStateException> {
            jvmVectorScatterEnabled(
                JvmVectorScatterMode.ON,
                vectorApiAvailable = false,
                autoScatterEligible = true,
            )
        }
    }
}
