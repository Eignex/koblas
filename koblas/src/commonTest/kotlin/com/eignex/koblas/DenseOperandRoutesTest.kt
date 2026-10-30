package com.eignex.koblas

import com.eignex.koblas.dense.DenseMatrixOperation
import com.eignex.koblas.dense.DenseOperation
import com.eignex.koblas.dense.DenseVectorKernels
import com.eignex.koblas.dense.RecordingPanels
import com.eignex.koblas.internal.dense.DensePanelKernels
import com.eignex.koblas.internal.dense.PanelWork
import com.eignex.koblas.internal.dense.PortablePanelKernels
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class DenseOperandRoutesTest {
    @Test
    fun `generic product routes retain padded or staged coefficient strides`() {
        val addressed = object : DensePanelKernels by PortablePanelKernels {
            override fun implementationFor(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): String =
                if (contiguous) "contiguous-panel" else "strided-panel"
        }
        val backends = listOf(addressed) +
            listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd).map { it.panelKernels }
        for (backend in backends) {
            for (aliased in booleanArrayOf(false, true)) {
                val panels = RecordingPanels(backend)
                val engine = KoblasEngine(
                    BuiltinEngines.scalar.vectorKernels,
                    BuiltinEngines.scalar.indexedSparseKernels,
                    null,
                    panelKernels = panels,
                )
                val a = DenseMatrix.wrap(68, 3, DoubleArray(68 * 3) { 0.25 }).view(1, 1, 65, 1)
                val parent = DenseMatrix.wrap(4, 67, DoubleArray(4 * 67) { 0.5 })
                val b = parent.view(1, 1, 1, 65)
                val c = if (aliased) parent.view(1, 1, 1, 1) else DenseMatrix.zero(1)
                val route = engine.routeOf(0.875, a as Matrix, true, b as Matrix, true, 0.0, c)

                engine.gemm(0.875, a as Matrix, true, b as Matrix, true, 0.0, c)

                assertEquals(panels.bodies.distinct(), route.components.map { it.substringBeforeLast('/') })
            }
        }
    }

    @Test
    fun `symmetric product routes name the scaling runs of the destination layout`() {
        val exact = BuiltinEngines.scalar
        val scaled = mutableListOf<Int>()
        val vectors = object : DenseVectorKernels by exact.vectorKernels {
            override fun implementationFor(operation: DenseOperation, length: Int, contiguous: Boolean): String =
                if (operation == DenseOperation.Scale && length >= 64) "wide" else "narrow"

            override fun scale(v: DoubleArray, vOff: Int, alpha: Double, len: Int, vStride: Int) {
                scaled.add(len)
                exact.vectorKernels.scale(v, vOff, alpha, len, vStride)
            }
        }
        val engine = KoblasEngine(vectors, exact.indexedSparseKernels, null)
        for (padding in intArrayOf(0, 4)) {
            for (alpha in doubleArrayOf(0.0, 1.0)) {
                for (beta in doubleArrayOf(0.0, 0.5, 1.0)) {
                    scaled.clear()
                    val a = DenseMatrix.diagonal(8)
                    val b = DenseMatrix.wrap(8, 8, DoubleArray(64) { 1.0 })
                    val c = DenseMatrix.zero(8 + padding, 10).view(0, 1, 8, 8)
                    val route = engine.routeOf(DenseMatrixOperation.Symm, alpha, a, b, false, false, beta, c)

                    engine.symm(alpha, a, b, beta, c)

                    val actual = scaled
                        .map { "${vectors.implementationFor(DenseOperation.Scale, it)}/scale" }
                        .distinct()
                    assertEquals(actual, route.components.filter { it.endsWith("/scale") })
                }
            }
        }
    }

    @Test
    fun `selected product routes retain padded or staged coefficient strides`() {
        for (exact in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd)) {
            for (aliased in listOf(false, true)) {
                val panels = RecordingPanels(exact.panelKernels)
                val engine = KoblasEngine(exact.vectorKernels, exact.indexedSparseKernels, null, panelKernels = panels)
                val a = DenseMatrix.wrap(68, 3, DoubleArray(68 * 3) { 0.25 }).view(1, 1, 65, 1)
                val parent = DenseMatrix.wrap(4, 67, DoubleArray(4 * 67) { 0.5 })
                val b = parent.view(1, 1, 1, 65)
                val c = if (aliased) parent.view(1, 1, 1, 1) else DenseMatrix.zero(1)
                val route = engine.routeOf(DenseMatrixOperation.Gemmt, 0.875, a, b, true, true, 0.0, c)

                engine.gemmt(0.875, a, true, b, true, 0.0, c)

                assertEquals(panels.bodies.distinct(), route.components.map { it.substringBeforeLast('/') })
            }
        }
    }

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
