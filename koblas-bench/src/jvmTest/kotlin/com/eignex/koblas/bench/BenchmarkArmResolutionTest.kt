package com.eignex.koblas.bench

import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class BenchmarkArmResolutionTest {
    @Test
    fun `jvm scalar mode resolves the exact built in engine`() {
        val (engine, identity) = resolveEngine("jvm-scalar")

        assertTrue(identity.startsWith("jvm-scalar/scalar/"), identity)
        assertTrue(engine === com.eignex.koblas.BuiltinEngines.scalar)
    }

    @Test
    fun `jvm c mode resolves the built in native policy`() {
        val (engine, identity) = resolveEngine("jvm-c")
        val packLeft = Cases.parse("pack-left+4x32+uniform+packed=4x4").single()

        assertTrue(engine === com.eignex.koblas.BuiltinEngines.c)
        assertTrue(identity.startsWith("jvm-c/${engine.vectorKernels.name}/"), identity)
        assertTrue(denseWork(packLeft, engine) != null)
    }

    @Test
    fun `unknown mode fails instead of selecting a fallback`() {
        assertFailsWith<IllegalStateException> { resolveEngine("openblas") }
    }
}
