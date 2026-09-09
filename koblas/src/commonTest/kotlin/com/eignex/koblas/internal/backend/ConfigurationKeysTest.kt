package com.eignex.koblas.internal.backend

import kotlin.test.*

class ConfigurationKeysTest {

    @Test
    fun `the property is taken ahead of the environment variable`() {
        assertEquals("hfactor", pinnedBackend("hfactor", "reference"))
    }

    @Test
    fun `the environment variable answers when no property is set`() {
        assertEquals("reference", pinnedBackend(null, "reference"))
    }

    @Test
    fun `a surrounding space does not become part of the name`() {
        assertEquals("hfactor", pinnedBackend(null, " hfactor\n"))
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

    @Test
    fun `an absolute path is accepted in either spelling`() {
        assertTrue(isAbsolutePath("/opt/hfactor/lib/libhfactor.so"))
        assertTrue(isAbsolutePath("C:/libs/hfactor.dll"))
    }

    @Test
    fun `a relative path is not a configured path`() {
        assertFalse(isAbsolutePath("build/libhfactor.so"))
        assertFalse(isAbsolutePath("libhfactor.so"))
        assertFalse(isAbsolutePath(""))
    }

    @Test
    fun `a pin matches a bundled provider by its canonical name`() {
        assertTrue(matchesRequested(bundledProvider("hfactor-bundled", "hfactor"), "hfactor"))
        assertTrue(matchesRequested(bundledProvider("hfactor-bundled", "hfactor"), "hfactor-bundled"))
        assertTrue(matchesRequested(namedProvider("hfactor"), "hfactor"))
        assertFalse(matchesRequested(namedProvider("hfactor"), "other"))
        assertFalse(
            matchesRequested(namedProvider("hfactor-bundled"), "hfactor"),
            "the name no longer carries the meaning; a provider declares what it bundles",
        )
    }
}
