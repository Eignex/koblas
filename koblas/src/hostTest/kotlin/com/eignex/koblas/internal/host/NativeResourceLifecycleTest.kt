// detekt's default test exclusions cover the standard source-set names, not this custom one.
@file:Suppress("UndocumentedPublicFunction", "FunctionNaming")

package com.eignex.koblas.internal.host

import kotlin.test.*

/** Checks the native lifecycle guard independently of a particular binding. */
class NativeResourceLifecycleTest {
    @Test
    fun `close releases once and rejects later calls`() {
        var releases = 0
        val lifecycle = NativeResourceLifecycle("test resource") { releases++ }

        lifecycle.withResource {}
        lifecycle.close()
        lifecycle.close()

        assertEquals(1, releases)
        assertFailsWith<IllegalStateException> { lifecycle.withResource {} }
    }
}
