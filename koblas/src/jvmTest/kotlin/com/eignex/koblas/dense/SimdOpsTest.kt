package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.assertClose
import com.eignex.koblas.internal.jvm25.SimdOps
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SimdOpsTest {
    @Test
    fun `reductions agree with the scalar oracle for windows in large backing arrays`() {
        val kernels = BuiltinEngines.simd?.vectorKernels ?: return
        val a = DoubleArray(65_560)
        val b = DoubleArray(65_560) { (it % 11 - 5) / 8.0 }

        for (length in (lengths() + listOf(65_535, 65_536, 65_537)).distinct()) {
            for (offset in 0..7) {
                a.fill(Double.NaN)
                for (i in 0 until length) a[offset + i] = (i % 17 - 8) / 16.0
                if (length > 0) a[offset + length / 2] = 1.0
                val bOffset = 7 - offset
                val expectedSum = ScalarVectorKernels.asum(a, offset, length)
                val expectedDot = ScalarVectorKernels.dot(a, offset, b, bOffset, length)
                val expectedIndex = ScalarVectorKernels.iamax(a, offset, length)

                val actualSum = kernels.asum(a, offset, length)
                val actualDot = kernels.dot(a, offset, b, bOffset, length)
                val actualIndex = kernels.iamax(a, offset, length)

                val context = "large backing length=$length offsets=$offset $bOffset"
                assertAsumAgreesWithReference(expectedSum, actualSum, context)
                assertDotAgreesWithReference(expectedDot, actualDot, context)
                assertIamaxAgreesWithReference(expectedIndex, actualIndex, context)
            }
        }
    }

    @Test
    fun `reductions agree with the scalar oracle for finite scales in large backing arrays`() {
        if (BuiltinEngines.simd == null) return
        assertFiniteReductionsAgreeWithReference(32 * SimdOps.lanes() + 1)
    }

    internal fun assertFiniteReductionsAgreeWithReference(length: Int) {
        val kernels = BuiltinEngines.simd?.vectorKernels ?: return
        val values = DoubleArray(65_560)
        val other = DoubleArray(65_560) { 0.5 }

        for (offset in 0..7) {
            for (value in doubleArrayOf(-1.25e-250, Double.MIN_VALUE, 1.25e200)) {
                for (position in intArrayOf(0, SimdOps.lanes() - 1, length - 1)) {
                    values.fill(Double.NaN)
                    values.fill(-0.0, offset, offset + length)
                    values[offset + position] = value
                    val expectedSum = ScalarVectorKernels.asum(values, offset, length)
                    val expectedDot = ScalarVectorKernels.dot(values, offset, other, 7 - offset, length)
                    val expectedIndex = ScalarVectorKernels.iamax(values, offset, length)

                    val actualSum = kernels.asum(values, offset, length)
                    val actualDot = kernels.dot(values, offset, other, 7 - offset, length)
                    val actualIndex = kernels.iamax(values, offset, length)

                    val context = "aligned length=$length offset=$offset position=$position value=$value"
                    assertAsumAgreesWithReference(expectedSum, actualSum, context)
                    if (expectedDot.isNaN()) {
                        assertTrue(actualDot.isNaN(), context)
                    } else {
                        assertDotAgreesWithReference(expectedDot, actualDot, context)
                    }
                    assertIamaxAgreesWithReference(expectedIndex, actualIndex, context)
                }
            }
        }
    }

    @Test
    fun `absolute sum agrees with the scalar oracle across offsets and vector boundaries`() {
        val kernels = BuiltinEngines.simd?.vectorKernels ?: return

        for (length in lengths()) {
            for (offset in intArrayOf(0, 1, 3, 7)) {
                val values = DoubleArray(offset + length + 3) { Double.NaN }
                for (i in 0 until length) values[offset + i] = (i % 17 - 8) / 16.0
                val expected = ScalarVectorKernels.asum(values, offset, length)

                val actual = kernels.asum(values, offset, length)

                assertAsumAgreesWithReference(expected, actual, "length=$length offset=$offset")
            }
        }
    }

    @Test
    fun `absolute sum agrees with the scalar oracle for finite scales`() {
        val kernels = BuiltinEngines.simd?.vectorKernels ?: return

        for (length in lengths().filter { it > 0 }) {
            for (value in doubleArrayOf(-1.25e-250, Double.MIN_VALUE, 1.25e200)) {
                for (position in intArrayOf(0, length / 2, length - 1).distinct()) {
                    val values = DoubleArray(length + 6) { Double.NaN }
                    values.fill(-0.0, 3, 3 + length)
                    values[3 + position] = value
                    val expected = ScalarVectorKernels.asum(values, 3, length)

                    val actual = kernels.asum(values, 3, length)

                    assertAsumAgreesWithReference(expected, actual, "length=$length position=$position value=$value")
                }
            }
        }
    }

    @Test
    fun `dot agrees with the scalar oracle across independent offsets and vector boundaries`() {
        val kernels = BuiltinEngines.simd?.vectorKernels ?: return

        for (length in lengths()) {
            for (aOffset in intArrayOf(0, 1, 3, 7)) {
                for (bOffset in intArrayOf(0, 2, 5)) {
                    val a = DoubleArray(aOffset + length + 3) { Double.NaN }
                    val b = DoubleArray(bOffset + length + 3) { Double.NaN }
                    for (i in 0 until length) {
                        a[aOffset + i] = (i % 17 - 8) / 16.0
                        b[bOffset + i] = (i % 11 - 5) / 8.0
                    }
                    val expected = ScalarVectorKernels.dot(a, aOffset, b, bOffset, length)

                    val actual = kernels.dot(a, aOffset, b, bOffset, length)

                    assertDotAgreesWithReference(expected, actual, "length=$length offsets=$aOffset $bOffset")
                }
            }
        }
    }

    @Test
    fun `iamax agrees with the scalar oracle when blocks keep improving`() {
        if (BuiltinEngines.simd == null) return
        val lanes = SimdOps.lanes()

        for (length in intArrayOf(32 * lanes - 1, 32 * lanes, 32 * lanes + 1, 64 * lanes + 1)) {
            for (offset in intArrayOf(0, 1, 7)) {
                for (fixture in 0..3) {
                    val values = DoubleArray(offset + length + 3) { Double.POSITIVE_INFINITY }
                    for (i in 0 until length) {
                        values[offset + i] = when (fixture) {
                            0 -> -i.toDouble()
                            1 -> if (i < 16 * lanes) i.toDouble() else -i.toDouble()
                            2 -> if (i == length - 1) -length.toDouble() else minOf(i, 16 * lanes - 1).toDouble()
                            else -> if (i % (4 * lanes) == 0) (i + 1) * Double.MIN_VALUE else 0.0
                        }
                    }
                    val expected = ScalarVectorKernels.iamax(values, offset, length)

                    val actual = SimdOps.iamax(values, offset, length)

                    assertIamaxAgreesWithReference(expected, actual, "length=$length offset=$offset fixture=$fixture")
                }
            }
        }
    }

    @Test
    fun `iamax distinguishes magnitudes sharing a high word`() {
        if (BuiltinEngines.simd == null) return

        for (length in lengths().filter { it > 0 }) {
            for (offset in intArrayOf(0, 1, 7)) {
                for (position in intArrayOf(0, length / 2, length - 1).distinct()) {
                    val values = DoubleArray(offset + length + 3) { Double.NaN }
                    values.fill(1.0, offset, offset + length)
                    values[offset + position] = Double.fromBits(1.0.toRawBits() + 1)
                    val expected = ScalarVectorKernels.iamax(values, offset, length)

                    val actual = SimdOps.iamax(values, offset, length)

                    assertIamaxAgreesWithReference(expected, actual, "length=$length offset=$offset position=$position")
                }
            }
        }
    }

    private fun assertIamaxAgreesWithReference(expected: Int, actual: Int, context: String) {
        assertEquals(expected, actual, context)
    }
    private fun lengths(): List<Int> {
        val lanes = SimdOps.lanes()
        return listOf(0, 1, lanes - 1, lanes, lanes + 1, 4 * lanes - 1, 4 * lanes, 4 * lanes + 1) +
            listOf(8 * lanes - 1, 8 * lanes, 8 * lanes + 1) +
            listOf(32 * lanes - 1, 32 * lanes, 32 * lanes + 1, 36 * lanes - 1, 36 * lanes, 36 * lanes + 1) +
            listOf(40 * lanes - 1, 40 * lanes, 40 * lanes + 1)
    }

    private fun assertAsumAgreesWithReference(expected: Double, actual: Double, context: String) {
        if (expected.isNaN()) assertTrue(actual.isNaN(), context) else assertClose(expected, actual, context)
    }

    private fun assertDotAgreesWithReference(expected: Double, actual: Double, context: String) {
        assertClose(expected, actual, context)
    }
}
