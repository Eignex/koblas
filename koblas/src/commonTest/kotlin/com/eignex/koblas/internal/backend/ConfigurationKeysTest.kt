package com.eignex.koblas.internal.backend

import kotlin.test.*

class ConfigurationKeysTest {

    @Test
    fun `the property is taken ahead of the environment variable`() {
        assertEquals("openblas", pinnedBackend("openblas", "reference"))
    }

    @Test
    fun `the environment variable answers when no property is set`() {
        assertEquals("reference", pinnedBackend(null, "reference"))
    }

    @Test
    fun `a surrounding space does not become part of the name`() {
        assertEquals("klu", pinnedBackend(null, " klu\n"))
    }

    @Test
    fun `nothing set leaves selection to priority`() {
        assertNull(pinnedBackend(null, null))
    }

    @Test
    fun `a blank setting counts as unset rather than as a name nothing matches`() {
        assertNull(pinnedBackend("", null))
        assertNull(pinnedBackend(null, "   "))
    }
}
