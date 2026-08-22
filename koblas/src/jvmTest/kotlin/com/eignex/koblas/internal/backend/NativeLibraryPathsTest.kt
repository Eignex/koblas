package com.eignex.koblas.internal.backend

import kotlin.test.Test
import kotlin.test.assertEquals

class NativeLibraryPathsTest {
    private val defaults = listOf("libnative.so.1", "libnative.so")

    @Test
    fun `uses the absolute property path before the environment`() {
        assertEquals(
            listOf("/native/from/property") + defaults,
            nativeLibraryPaths(
                "koblas.test.path",
                "KOBLAS_TEST_PATH",
                defaults,
                "/native/from/property",
                "/native/from/environment",
            ),
        )
    }

    @Test
    fun `uses the absolute environment path without a property`() {
        assertEquals(
            listOf("/native/from/environment") + defaults,
            nativeLibraryPaths(
                "koblas.test.path",
                "KOBLAS_TEST_PATH",
                defaults,
                null,
                "/native/from/environment",
            ),
        )
    }

    @Test
    fun `ignores relative configured paths`() {
        assertEquals(
            defaults,
            nativeLibraryPaths(
                "koblas.test.path",
                "KOBLAS_TEST_PATH",
                defaults,
                "libnative.so",
                "other.so",
            ),
        )
    }
}
