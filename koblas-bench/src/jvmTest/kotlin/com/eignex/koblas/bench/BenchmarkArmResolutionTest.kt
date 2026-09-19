package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.koblas
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
     * The default arm is the platform's own policy, which is not the same object as any exact arm.
     *
     * The generic entry points use the engine this platform selected, so a row for one of them has to be
     * published on an arm whose engine is that. Naming it separately is also what keeps a default-policy
     * measurement from being read as a measurement of the kernels an exact arm holds.
     */
    @Test
    fun `jvm default mode resolves the platform engine rather than an exact arm`() {
        val (engine, identity) = resolveEngine("jvm-default")

        assertTrue(engine === koblas, "jvm-default resolved an engine other than the platform default")
        assertTrue(identity.startsWith("jvm-default/"), identity)
        assertEquals(
            "scalar-panel",
            engine.panelKernels.name,
            "the platform default schedules panels other than the portable ones",
        )
    }

    @Test
    fun `unknown mode fails instead of selecting a fallback`() {
        assertFailsWith<IllegalStateException> { resolveEngine("openblas") }
    }

    /**
     * The generic product reaches the measured fork through the bridge, not only through the scan.
     *
     * The scan and the measurement are two different seams: the scan decides which cases a fork is asked
     * for, and the bridge builds the work inside it. A case admitted by one and dropped by the other looks
     * like an unsupported row with no reason, which is what happened to `gemm-generic` before the bridge
     * learned the dense arm. This holds both seams to the same answer.
     */
    @Test
    fun `the generic product is built by the measured fork on the default arm and declined on an exact one`() {
        val case = Cases.parse(readTextFile(CASES)).single { it.operation == "gemm-generic" }

        val work = JvmBenchmarkBridge.create("jvm-default", case.id, CASES)

        assertEquals("default-policy", work.comparisonKind)
        assertTrue(!work.kernel.isNullOrEmpty(), "the default arm's generic row named no route")
        work.close()

        if (BuiltinEngines.simd !== koblas) {
            val declined = assertFailsWith<IllegalStateException> {
                JvmBenchmarkBridge.create("jvm-simd", case.id, CASES)
            }
            assertTrue("declined" in declined.message.orEmpty(), declined.message.orEmpty())
        }
    }

    private companion object {
        /** The workload the benchmark tasks read, resolved from the module directory the tests run in. */
        const val CASES = "cases.txt"
    }
}
