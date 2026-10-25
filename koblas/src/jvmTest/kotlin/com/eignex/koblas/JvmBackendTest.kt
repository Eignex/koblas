package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertTrue

class JvmBackendTest {
    @Test
    fun `desktop implementation is loaded from the versioned jar section`() {
        val resource = JvmBackend::class.java.getResource("JvmBackend.class").toString()

        assertTrue(resource.contains(".jar!/META-INF/versions/25/com/eignex/koblas/JvmBackend.class"), resource)
    }
}
