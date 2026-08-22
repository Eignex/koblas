package com.eignex.koblas.sparse.host.umfpack

import kotlin.test.Test
import kotlin.test.assertEquals

class UmfpackCallsTest {

    @Test
    fun `an absolute configured library path precedes system lookup`() {
        withUmfpackPath("/opt/koblas/libumfpack.so") {
            assertEquals(
                "/opt/koblas/libumfpack.so",
                UmfpackCalls.umfpackPaths(environmentPath = "/opt/bundle/libumfpack.so").first(),
            )
        }
    }

    @Test
    fun `the environment path precedes system lookup when no property is configured`() {
        assertEquals(
            "/opt/bundle/libumfpack.so",
            UmfpackCalls.umfpackPaths(
                propertyPath = null,
                environmentPath = "/opt/bundle/libumfpack.so",
            ).first(),
        )
    }

    @Test
    fun `a relative configured library path is ignored`() {
        withUmfpackPath("libumfpack.so") {
            assertEquals(
                UMFPACK_SONAMES,
                UmfpackCalls.umfpackPaths(environmentPath = "libumfpack-from-environment.so"),
            )
        }
    }

    private fun withUmfpackPath(value: String, block: () -> Unit) {
        val property = "koblas.umfpack.path"
        val previous = System.getProperty(property)
        try {
            System.setProperty(property, value)
            block()
        } finally {
            if (previous == null) System.clearProperty(property) else System.setProperty(property, previous)
        }
    }
}
