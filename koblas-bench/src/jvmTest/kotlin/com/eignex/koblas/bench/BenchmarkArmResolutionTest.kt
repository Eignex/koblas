package com.eignex.koblas.bench

import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class BenchmarkArmResolutionTest {
    @Test
    fun `jvm c mode resolves the exact built in engine`() {
        val (_, identity) = resolveEngine("jvm-c")

        assertTrue(identity.startsWith("jvm-c/c/"), identity)
    }

    @Test
    fun `unknown mode fails instead of selecting a fallback`() {
        assertFailsWith<IllegalStateException> { resolveEngine("openblas") }
    }
}
