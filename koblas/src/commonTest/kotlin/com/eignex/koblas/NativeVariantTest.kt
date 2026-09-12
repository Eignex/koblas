package com.eignex.koblas

import com.eignex.koblas.dense.*
import kotlin.test.*

class NativeVariantTest {
    private fun engines(): List<KoblasEngine> = BuiltinEngines.nativeVariants.map(BuiltinEngines::exactC)

    @Test
    fun `exact native variants preserve vector arithmetic`() {
        engines().forEach { assertLevel1KernelsAgreeWithReference(it.vectorKernels) }
    }

    @Test
    fun `exact native variants preserve robust reductions`() {
        engines().forEach { assertReductionsAgreeWithReference(it.vectorKernels) }
    }

    @Test
    fun `exact native variants preserve iamax ties and special values`() {
        engines().forEach { assertIamaxAgreesWithReference(it.vectorKernels) }
    }

    @Test
    fun `exact native variants preserve scaling storage`() {
        engines().forEach { assertScaleAgreesWithReference(it.vectorKernels) }
    }

    @Test
    fun `exact native variants preserve swaps and rotations`() {
        engines().forEach {
            assertSwapAgreesWithReference(it.vectorKernels)
            assertRotKernelAgreesWithReference(it.vectorKernels)
            assertModifiedGivensKernelsAgreeWithReference(it.vectorKernels)
        }
    }

    @Test
    fun `exact native variants preserve packed products`() {
        engines().forEach {
            assertPackedGemmAgreesWithWrittenOutProduct(it.packedKernels, intArrayOf(1, 4, 5, 9), intArrayOf(1, 17, 65))
        }
    }

    @Test
    fun `exact native variants preserve panel input aliases`() {
        engines().forEach { engine ->
            for (length in listOf(1, 3, 4, 15, 16, 17, 65)) {
                val expected = DoubleArray(4 * length + 11) { it * 0.125 - 2.0 }
                val actual = expected.copyOf()
                val x = DoubleArray(length + 3) { it * -0.2 + 1.0 }
                ScalarPanelKernels.dot4(expected, 3, length, x, 1, length, expected, 4)
                engine.panelKernels.dot4(actual, 3, length, x, 1, length, actual, 4)
                assertClose(expected, actual, "aliased dot panel ${engine.name} length=$length")
            }
        }
    }

    @Test
    fun `exact native solves preserve triangles padding and unit diagonals`() {
        engines().forEach { engine -> assertSolveTilesAgreeWithReference(engine.packedKernels) }
    }

    private fun assertSolveTilesAgreeWithReference(kernels: PackedKernels) {
        for (rows in 1..4) {
            for (order in 1..4) {
                for (lower in listOf(false, true)) {
                    for (unit in listOf(false, true)) {
                        val triangle = DoubleArray(23) { Double.NaN }
                        for (j in 0 until order) {
                            for (i in 0 until order) {
                                if ((lower && i >= j) || (!lower && i <= j)) {
                                    triangle[3 + i * 4 + j] = if (i == j) {
                                        if (unit) Double.NaN else 2.0
                                    } else {
                                        0.125
                                    }
                                }
                            }
                        }
                        for (depth in listOf(0, 1, 3, 17)) {
                            val left = DoubleArray(depth * 4 + 5) { it * 0.125 - 1.0 }
                            val right = DoubleArray(depth * 4 + 7) { 2.0 - it * 0.25 }
                            val expected = DoubleArray(27) { if (it % 3 == 0) -0.0 else it * 0.25 }
                            val actual = expected.copyOf()
                            PortablePackedKernels.gemmTrsmTile(
                                depth, rows, order, left, 2, right, 3, triangle, 3, lower, unit, expected, 5,
                            )
                            kernels.gemmTrsmTile(
                                depth, rows, order, left, 2, right, 3, triangle, 3, lower, unit, actual, 5,
                            )
                            assertClose(
                                expected,
                                actual,
                                "solve rows=$rows order=$order depth=$depth lower=$lower unit=$unit",
                            )
                            for (index in actual.indices) {
                                val offset = index - 5
                                if (offset < 0 || offset / 4 >= order || offset % 4 >= rows) {
                                    assertEquals(expected[index].toBits(), actual[index].toBits())
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `exact native zero work avoids empty input storage`() {
        engines().forEach { engine ->
            val empty = doubleArrayOf()
            assertEquals(0.0, engine.vectorKernels.dot(empty, 0, empty, 0, 0))
            assertEquals(-1, engine.vectorKernels.iamax(empty, 0, 0))
            engine.vectorKernels.axpy(empty, 0, 0.0, empty, 0, 100)
            engine.vectorKernels.scale(empty, 0, 1.0, 100)
            engine.panelKernels.axpy4(empty, 0, empty, 0, 0, 0.0, 0.0, 0.0, 0.0, 0)
            assertEquals(0.0, engine.panelKernels.dotAxpy(empty, 0, 0.0, empty, 0, empty, 0, 0))
            val output = DoubleArray(6) { 7.0 }
            engine.panelKernels.dot4(empty, 0, 0, empty, 0, 0, output, 1)
            assertContentEquals(doubleArrayOf(7.0, 0.0, 0.0, 0.0, 0.0, 7.0), output)
        }
    }

    @Test
    fun `an unavailable exact variant fails before execution`() {
        NativeVariant.entries.filterNot { it in BuiltinEngines.nativeVariants }.forEach {
            assertFailsWith<IllegalArgumentException> { BuiltinEngines.exactC(it) }
        }
    }
}
