package com.eignex.koblas

import com.eignex.koblas.dense.DenseMatrixOperation
import com.eignex.koblas.dense.RecordingPanels
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class DenseOperandRoutesTest {
    @Test
    fun `operand vector routes name the bodies the recorded windows reach`() {
        for (exact in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)) {
            for (stride in intArrayOf(1, 2, -2)) {
                for (operation in listOf(
                    DenseMatrixOperation.Gemv,
                    DenseMatrixOperation.GemvTransposed,
                    DenseMatrixOperation.Symv,
                )) {
                    val panels = RecordingPanels(exact.panelKernels)
                    val engine = KoblasEngine(
                        exact.vectorKernels,
                        exact.indexedSparseKernels,
                        null,
                        panelKernels = panels,
                    )
                    val a = DenseMatrix.wrap(37, 36, DoubleArray(37 * 36) { (it % 7) / 9.0 }).view(2, 1, 33, 33)
                    val x = StridedVector(DoubleArray(70) { (it % 5) / 11.0 }, if (stride < 0) 66 else 2, 33, stride)
                    val y = StridedVector(DoubleArray(70), if (stride < 0) 66 else 2, 33, stride)
                    val route = engine.routeOf(operation, 0.875, a, x, 0.0, y)

                    if (operation == DenseMatrixOperation.Symv) {
                        engine.symv(0.875, a, x, 0.0, y)
                    } else {
                        engine.gemv(0.875, a, x, 0.0, y, operation == DenseMatrixOperation.GemvTransposed)
                    }

                    assertEquals(panels.bodies.distinct(), route.components.map { it.substringBeforeLast('/') })
                }
            }
        }
    }

    @Test
    fun `operand routes validate their operation before a zero multiplier shortcut`() {
        val a = DenseMatrix.zero(2, 3)
        val vector = DenseVector.zero(2)

        assertFailsWith<DimensionMismatch> {
            BuiltinEngines.scalar.routeOf(DenseMatrixOperation.Symv, 0.0, a, vector, 0.0, vector)
        }
    }
}
