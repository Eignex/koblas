package com.eignex.koblas.bench

import com.eignex.koblas.engine.*
import kotlin.test.*

class EngineTuningTest {
    @Test fun `losing larger lengths delay a vector threshold`() {
        val data = listOf(32 to 0.5, 64 to 1.2, 128 to 0.7, 256 to 0.8).flatMap { (n, cost) ->
            List(3) { timing(n, cost) }
        }

        val rule = fitCutoffs(data).single()

        assertEquals(128, rule.rows)
    }

    @Test fun `one cpu model cannot hide another model losing`() {
        val data = List(30) { timing(128, 0.5) } + List(3) { timing(128, 1.1).copy(cpu = "B") }

        assertTrue(fitCutoffs(data).isEmpty())
    }

    @Test fun `a winning variant cannot hide a losing variant of the same shape`() {
        val data = List(3) { timing(128, 0.5) } +
            List(3) { timing(128, 1.1).copy(case = "dot+128+uniform+stride=1") }

        assertTrue(fitCutoffs(data).isEmpty())
    }

    @Test fun `matrix regions exclude skinny losses and separate borrowed layouts`() {
        fun matrix(n: Int, m: Int, k: Int, ratio: Double, borrowed: Boolean = false) =
            timing(n, ratio).copy(shape = RoutingShape(EngineImplementation.Gemm, n, m, k, borrowed, 3, true))
        val data = listOf(matrix(8, 8, 8, 1.5), matrix(4, 512, 512, 1.5), matrix(32, 32, 32, 0.5),
            matrix(64, 64, 64, 0.5), matrix(32, 32, 32, 1.5, true))

        val rule = fitCutoffs(data).single()

        assertEquals(32, rule.rows)
        assertEquals(32768L, rule.work)
        assertEquals(false, rule.borrowed)
    }

    @Test fun `cutoff report includes operation and layout bounds`() {
        val report = renderCutoffs(listOf(FittedCutoff(EngineImplementation.Dot, rows = 128, contiguous = true)))

        assertEquals("operation\trows\tcolumns\tdepth\twork\tcontiguous\tborrowed\nDot\t128\t0\t0\t0\ttrue\t*\n", report)
    }

    private fun timing(n: Int, ratio: Double) = PairedTiming("dot+$n+uniform", "A", "onemkl",
        RoutingShape(EngineImplementation.Dot, n, 1, 1, false, 1, true), ratio)
}
