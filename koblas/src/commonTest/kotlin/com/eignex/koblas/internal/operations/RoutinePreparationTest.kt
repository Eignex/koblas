package com.eignex.koblas.internal.operations

import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.kernels.KernelProvider
import kotlin.test.Test
import kotlin.test.assertContentEquals

class RoutinePreparationTest {
    @Test
    fun `trivial beta factors need no backend arithmetic`() {
        val engine = object : KoblasEngine {
            override val name: String = "no kernels"
        }
        val provider = object : KernelProvider {
            override val name: String = "no vector kernels"
        }
        for (stride in intArrayOf(1, 2, -2)) {
            for (beta in doubleArrayOf(0.0, -0.0, 1.0)) {
                val offset = if (stride < 0) 5 else 1
                val before = doubleArrayOf(99.0, Double.NaN, -99.0, -0.0, 99.0, Double.NaN, -99.0)
                val expected = before.copyOf()
                if (beta == 0.0) for (i in 0 until 3) expected[offset + i * stride] = 0.0
                val typed = before.copyOf()
                val primitive = before.copyOf()

                applyBeta(engine, typed, offset, 3, beta, stride)
                applyBeta(provider, primitive, offset, 3, beta, stride)

                assertContentEquals(expected, typed, "typed beta=$beta stride=$stride")
                assertContentEquals(expected, primitive, "primitive beta=$beta stride=$stride")
            }
        }
    }
}
