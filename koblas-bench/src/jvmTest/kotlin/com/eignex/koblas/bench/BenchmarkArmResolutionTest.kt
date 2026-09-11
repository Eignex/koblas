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
    fun `jvm c mode resolves the exact built in engine`() {
        val (engine, identity) = resolveEngine("jvm-c")
        val packLeft = Cases.parse("pack-left+4x32+uniform+physical=4x4+work=pack-left-v1+leftLayout=depth-rows-v1+rightLayout=depth-columns-v1+leftGroup=4+rightGroup=4+leftStride=4+rightStride=4+padding=zero+alignment=8+block=4x32x32+panel=32+diagonal=0+rhs=0+batch=1+variant=current-tile-v1+timing=packing-only").single()

        assertTrue(identity.startsWith("jvm-c/c/"), identity)
        assertTrue(denseWork(packLeft, engine) != null)
    }

    @Test
    fun `unknown mode fails instead of selecting a fallback`() {
        assertFailsWith<IllegalStateException> { resolveEngine("openblas") }
    }
}
