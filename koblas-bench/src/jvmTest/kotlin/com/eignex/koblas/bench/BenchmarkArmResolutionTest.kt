package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.koblas
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class BenchmarkArmResolutionTest {
    @Test
    fun `jvm scalar mode resolves the exact built in engine`() {
        val (engine, identity) = resolveEngine("jvm-scalar")

        assertTrue(identity.startsWith("jvm-scalar/scalar/"), identity)
        assertTrue(engine === BuiltinEngines.scalar)
    }

    @Test
    fun `jvm simd mode resolves the vector api engine or refuses to stand in for it`() {
        val available = BuiltinEngines.simd
        if (available == null) {
            assertFailsWith<IllegalArgumentException> { resolveEngine("jvm-simd") }
            return
        }

        val (engine, identity) = resolveEngine("jvm-simd")

        assertTrue(engine === available, "jvm-simd resolved an engine other than the Vector API one")
        assertTrue(identity.startsWith("jvm-simd/${engine.vectorKernels.name}/"), identity)
    }

    /**
     * The default arm is the platform's own policy, which is what the generic entry points use, so naming it
     * separately keeps a default-policy row from being read as a measurement of an exact arm. Where the
     * module resolved, the two arms name the same object; where it did not, `jvm-simd` refuses to run.
     */
    @Test
    fun `jvm default mode resolves the platform selection`() {
        val (engine, identity) = resolveEngine("jvm-default")

        assertTrue(engine === koblas, "jvm-default resolved an engine other than the platform default")
        assertTrue(identity.startsWith("jvm-default/"), identity)
        assertEquals(
            BuiltinEngines.simd ?: BuiltinEngines.scalar,
            engine,
            "the platform default is neither the Vector API engine nor the portable one",
        )
    }

    @Test
    fun `unknown mode fails instead of selecting a fallback`() {
        assertFailsWith<IllegalStateException> { resolveEngine("openblas") }
    }

    @Test
    fun `the generic product is built by the measured fork on the default arm`() = withGenericCase { case, path ->
        val work = JvmBenchmarkBridge.create("jvm-default", case.id, path)
        try {
            assertEquals("default-policy", work.comparisonKind)
            assertTrue(!work.kernel.isNullOrEmpty(), "the default arm's generic row named no route")
        } finally {
            work.close()
        }
    }

    @Test
    fun `the generic product is declined by the measured fork on an exact arm`() = withGenericCase { case, path ->
        if (BuiltinEngines.simd === koblas) return@withGenericCase

        val declined = assertFailsWith<IllegalStateException> {
            JvmBenchmarkBridge.create("jvm-simd", case.id, path)
        }

        assertTrue("declined" in declined.message.orEmpty(), declined.message.orEmpty())
    }

    private fun withGenericCase(body: (BenchCase, String) -> Unit) {
        // The bridge reads its case file, but its admission rule is independent of the benchmark workload size.
        val case = Cases.parse("gemm-generic+15x7x31+uniform").single()
        val path = Files.createTempFile("koblas-generic-case-", ".txt")
        try {
            Files.writeString(path, case.id)
            body(case, path.toString())
        } finally {
            Files.deleteIfExists(path)
        }
    }
}
