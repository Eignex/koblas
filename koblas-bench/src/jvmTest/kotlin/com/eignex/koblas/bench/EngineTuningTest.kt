package com.eignex.koblas.bench

import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.engine.*
import kotlin.test.*

class EngineTuningTest {
    @Test fun `losing larger lengths delay a vector threshold`() {
        val data = listOf(32 to 0.5, 64 to 1.2, 128 to 0.7, 256 to 0.8).flatMap { (n, cost) ->
            List(3) { timing(n, cost) }
        }

        val rule = fitProfile("onemkl", data).single()

        assertEquals(128..Int.MAX_VALUE, rule.rows)
    }

    @Test fun `one cpu model cannot hide another model losing`() {
        val data = List(30) { timing(128, 0.5) } + List(3) { timing(128, 1.1).copy(cpu = "B") }

        assertTrue(fitProfile("onemkl", data).isEmpty())
    }

    @Test fun `a winning variant cannot hide a losing variant of the same shape`() {
        val data = List(3) { timing(128, 0.5) } +
            List(3) { timing(128, 1.1).copy(case = "dot+128+uniform+stride=1") }

        assertTrue(fitProfile("onemkl", data).isEmpty())
    }

    @Test fun `matrix regions exclude skinny losses and separate borrowed layouts`() {
        fun matrix(n: Int, m: Int, k: Int, ratio: Double, borrowed: Boolean = false) =
            timing(n, ratio).copy(shape = RoutingShape(EngineImplementation.Gemm, n, m, k, borrowed, 3, true))
        val data = listOf(matrix(8, 8, 8, 1.5), matrix(4, 512, 512, 1.5), matrix(32, 32, 32, 0.5),
            matrix(64, 64, 64, 0.5), matrix(32, 32, 32, 1.5, true))

        val rule = fitProfile("onemkl", data).single()

        assertEquals(32..Int.MAX_VALUE, rule.rows)
        assertEquals(32768L, rule.work)
        assertEquals(false, rule.borrowed)
    }

    @Test fun `composition orders vendors using overlapping paired regions`() {
        fun capture(cost: Double) = CalibrationCapture(emptyList(), setOf("dot"), List(3) { timing(128, cost) }, emptyMap(), emptyList())
        val captures = mapOf("onemkl" to capture(0.8), "openblas" to capture(0.5))

        val rules = composeProfiles(captures)

        assertEquals(listOf("openblas", "onemkl"), rules.map { it.source })
    }

    @Test fun `emitted configuration is kotlin with typed capabilities`() {
        val code = renderProfile("fixture", listOf("native", "Linux", "X86_64", "*", "Avx2", "*"),
            listOf(EngineRule("onemkl", setOf(EngineImplementation.Dot), rows = 128..Int.MAX_VALUE)))

        assertTrue("EngineRuntime.Native" in code)
        assertTrue("CpuClass.Avx2" in code)
        assertTrue("setOf(Dot)" in code)
        assertFalse("tsv" in code)
    }

    private fun timing(n: Int, ratio: Double) = PairedTiming("dot+$n+uniform", "A", "onemkl",
        RoutingShape(EngineImplementation.Dot, n, 1, 1, false, 1, true), ratio)
}
