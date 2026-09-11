package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngine
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * The C and SIMD compositions directly, bypassing automatic platform selection, so both
 * backends are exercised even on a JVM where one shadows the other for the platform-dispatched tests.
 */
class ExplicitKernelsTest {
    @Test
    fun `available explicit kernels preserve vector contracts`() {
        for (engine in availableEngines()) {
            assertModifiedGivensKernelsAgreeWithReference(engine.vectorKernels)
            assertRotKernelAgreesWithReference(engine.vectorKernels)
        }
    }

    @Test
    fun `available explicit kernels preserve exceptional arithmetic`() {
        for (engine in availableEngines()) {
            assertAxpyArithmeticPreservesOverflow(engine.panelKernels)
            assertSymvPreservesFiniteCancellation(engine)
            assertSymvPreservesOverflow(engine)
        }
    }

    private fun assertAxpyArithmeticPreservesOverflow(kernels: DensePanelKernels) {
        val x = DoubleArray(64) { Double.MAX_VALUE }
        val y = DoubleArray(64) { Double.MAX_VALUE }

        kernels.axpyArithmetic(y, 0, -2.0, x, 0, y.size)

        assertEquals(Double.NEGATIVE_INFINITY, y[0])
    }

    private fun availableEngines(): List<KoblasEngine> = listOfNotNull(BuiltinEngines.c, BuiltinEngines.simd)

    private fun assertSymvPreservesFiniteCancellation(engine: KoblasEngine) =
        assertSymvColumnResult(engine, second = -1e308, common = 1e308, expectedValue = 1e308)

    private fun assertSymvPreservesOverflow(engine: KoblasEngine) =
        assertSymvColumnResult(engine, second = 1e308, common = -1e308, expectedValue = Double.POSITIVE_INFINITY)

    private fun assertSymvColumnResult(engine: KoblasEngine, second: Double, common: Double, expectedValue: Double) {
        val n = DenseTuning.symvFourColumnCrossover
        val column = n - 8
        val a = DenseMatrix(n, n)
        a[column + 1, column] = 1e308
        a[column + 2, column] = second
        a[column + 4, column] = common
        val x = DoubleArray(n)
        x[column + 1] = 1.0
        x[column + 2] = 1.0
        x[column + 4] = 1.0
        val expected = DoubleArray(n)
        val actual = DoubleArray(n)

        ReferenceBlas.symv(1.0, a, x, 0.0, expected, lower = true)
        engine.symv(1.0, a, x, 0.0, actual, lower = true)

        assertEquals(expectedValue, expected[column])
        assertEquals(expected[column], actual[column], engine.vectorKernels.name)
    }
}
