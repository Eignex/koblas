package com.eignex.koblas

import com.eignex.koblas.vendor.Vendor
import kotlin.test.Test
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

class JvmBackendTest {
    @Test
    fun `runtime loads the implementation from its packaged section`() {
        val resource = JvmBackend::class.java.getResource("JvmBackend.class").toString()
        val section = if (Runtime.version().feature() >= 25) "META-INF/versions/25/" else ""

        assertTrue(resource.contains(".jar!/${section}com/eignex/koblas/JvmBackend.class"), resource)
    }

    @Test
    fun `baseline runtime selects scalar without opening a host library`() {
        if (Runtime.version().feature() >= 25) return

        assertSame(BuiltinEngines.scalar, koblas)
        assertNull(BuiltinEngines.simd)
        assertNull(Vendor.entries.firstNotNullOfOrNull { BuiltinEngines.vendor(it) })
    }
}
