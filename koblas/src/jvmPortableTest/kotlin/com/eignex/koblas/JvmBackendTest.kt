package com.eignex.koblas

import com.eignex.koblas.vendor.openBlas
import kotlin.test.Test
import kotlin.test.assertNull
import kotlin.test.assertSame
import kotlin.test.assertTrue

// Kbuild's test-path exemptions do not yet recognize a custom JVM test compilation.
@Suppress("UndocumentedPublicClass", "UndocumentedPublicFunction", "FunctionNaming")
class JvmBackendTest {
    @Test
    fun `portable runtime selects scalar without opening a host library`() {
        assertSame(BuiltinEngines.scalar, koblas)
        assertNull(BuiltinEngines.simd)
        assertNull(openBlas())
    }

    @Test
    fun `portable implementation is loaded from the jar root`() {
        val resource = JvmBackend::class.java.getResource("JvmBackend.class").toString()

        assertTrue(resource.contains(".jar!/com/eignex/koblas/JvmBackend.class"), resource)
    }
}
