package com.eignex.koblas.dense.host.jvm

import kotlin.test.Test
import kotlin.test.assertEquals

class HostLibraryPathTest {
    @Test
    fun `uses the environment path without a property`() {
        assertEquals(
            "/native/from/environment",
            preferredLibraryPath("koblas.test.openblas.path", "KOBLAS_TEST_OPENBLAS_PATH", "/native/from/environment"),
        )
    }

    @Test
    fun `uses the property path before the environment`() {
        val property = "koblas.test.openblas.path"
        val previous = System.getProperty(property)
        try {
            System.setProperty(property, "/native/from/property")

            assertEquals(
                "/native/from/property",
                preferredLibraryPath(property, "KOBLAS_TEST_OPENBLAS_PATH", "/native/from/environment"),
            )
        } finally {
            if (previous == null) System.clearProperty(property) else System.setProperty(property, previous)
        }
    }
}
