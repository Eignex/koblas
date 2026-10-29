package com.eignex.koblas.bench

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.KoblasEngineApi
import com.eignex.koblas.dense.DenseCall
import com.eignex.koblas.dense.DenseMatrixOperation
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class ShrinkingRankUpdateTest {
    @Test
    fun `the primary fixture has disjoint shrinking supports and the measured density`() {
        val fixture = ShrinkingRankUpdate(6)

        assertEquals(31, fixture.matrix.values.count { it != 0.0 })
        assertEquals(5, fixture.x.size)
        for (step in fixture.x.indices) {
            val x = fixture.x[step]
            val y = fixture.y[step]
            assertEquals(5 - step, x.count { it != 0.0 })
            assertEquals(5 - step, y.count { it != 0.0 })
            assertTrue((0..step).all { x[it] == 0.0 && y[it] == 0.0 })
            assertTrue(x !== y && x !== fixture.matrix.values && y !== fixture.matrix.values)
        }
    }

    @OptIn(KoblasEngineApi::class)
    @Test
    fun `timed batches agree with the reference across repeated calls`() {
        val engines = listOfNotNull<KoblasEngine>(BuiltinEngines.scalar, BuiltinEngines.simd)
        for (n in listOf(5, 6, 7)) for (operation in listOf("ger-shrinking", "ger-window-shrinking")) {
            val fixture = ShrinkingRankUpdate(n)
            for (engine in engines) {
                val work = assertNotNull(denseWork(Cases.parse("$operation+$n+uniform").single(), engine))
                var expected = fixture.matrix.values.copyOf()

                repeat(3) {
                    for (step in fixture.x.indices) {
                        expected = DenseReference.ger(-1.0, fixture.x[step], fixture.y[step], DenseMatrix.wrap(n, n, expected))
                    }
                    work.run()
                    assertRankUpdateAgreesWithReference(expected, assertNotNull(work.result), "$operation n=$n ${engine.name}")
                }

                assertEquals("arithmetic", work.timingMode)
            }
        }
    }

    @OptIn(KoblasEngineApi::class)
    @Test
    fun `full calls and shrinking panels name the bodies they reach`() {
        for (engine in listOfNotNull<KoblasEngine>(BuiltinEngines.scalar, BuiltinEngines.simd)) {
            for (n in intArrayOf(6, 34)) {
                val full = assertNotNull(denseWork(Cases.parse("ger-shrinking+$n+uniform").single(), engine))
                val publicWindow = assertNotNull(denseWork(Cases.parse("ger-window-shrinking+$n+uniform").single(), engine))
                val kernel = assertNotNull(publicWindow.kernel)
                val groups = kernel.substringAfter("/ger-window@").split('+').map(String::toInt)

                assertEquals(matrixKernel(engine.routeOf(DenseMatrixOperation.Ger, DenseCall(n, n, alpha = -1.0))), full.kernel)
                for (extent in n - 1 downTo 1) {
                    val route = engine.routeOf(DenseMatrixOperation.GerWindow, DenseCall(extent, extent, alpha = -1.0))
                    for (body in route.components) {
                        assertTrue(body in kernel, "${engine.name} n=$n omitted $body at extent $extent: $kernel")
                    }
                    assertTrue(route.executionGroup in groups, "${engine.name} n=$n omitted group ${route.executionGroup}")
                }
            }
        }
    }

    private fun assertRankUpdateAgreesWithReference(expected: DoubleArray, actual: DoubleArray, context: String) {
        DenseReference.check(expected, actual, context)
    }
}
