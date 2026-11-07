package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.*
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import org.junit.Assume
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * The Vector API Level 1 bodies against the scalar oracle. Short windows call [SimdOps] directly, because the
 * engine hands a window shorter than a lane block to the portable loop and would compare it with itself.
 */
class SimdOpsTest {
    // Offsets matter only where the aligned bodies peel a prefix: dot and asum from 4,096 entries and iamax from
    // 65,536. The arrays are large so a collector that aligns only big arrays still exercises the prefix.
    @Test
    fun `reductions agree with the scalar oracle for windows in large backing arrays`() {
        val kernels = simdEngine()
        val a = DoubleArray(65_560)
        val b = DoubleArray(65_560) { (it % 11 - 5) / 8.0 }

        for (length in intArrayOf(4_095, 4_096, 65_535, 65_536, 65_537)) {
            for (offset in 0..7) {
                a.fill(Double.NaN)
                for (i in 0 until length) a[offset + i] = (i % 17 - 8) / 16.0
                a[offset + length / 2] = 1.0
                val bOffset = 7 - offset
                val expectedSum = ScalarEngine.asum(a, offset, length)
                val expectedDot = ScalarEngine.dot(a, offset, b, bOffset, length)
                val expectedIndex = ScalarEngine.iamax(a, offset, length)

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
        assumeSimd()
        assertFiniteReductionsAgreeWithReference(32 * SimdOps.lanes() + 1)
    }

    internal fun assertFiniteReductionsAgreeWithReference(length: Int) {
        val kernels = simdEngine()
        val values = DoubleArray(8 + length + 3)
        val other = DoubleArray(8 + length + 3) { 0.5 }

        for (offset in 0..7) {
            for (value in doubleArrayOf(-1.25e-250, Double.MIN_VALUE, 1.25e200)) {
                for (position in intArrayOf(0, SimdOps.lanes() - 1, length - 1)) {
                    values.fill(Double.NaN)
                    values.fill(-0.0, offset, offset + length)
                    values[offset + position] = value
                    val expectedSum = ScalarEngine.asum(values, offset, length)
                    val expectedDot = ScalarEngine.dot(values, offset, other, 7 - offset, length)
                    val expectedIndex = ScalarEngine.iamax(values, offset, length)

                    val actualSum = kernels.asum(values, offset, length)
                    val actualDot = kernels.dot(values, offset, other, 7 - offset, length)
                    val actualIndex = kernels.iamax(values, offset, length)

                    val context = "aligned length=$length offset=$offset position=$position value=$value"
                    assertAsumAgreesWithReference(expectedSum, actualSum, context)
                    assertDotAgreesWithReference(expectedDot, actualDot, context)
                    assertIamaxAgreesWithReference(expectedIndex, actualIndex, context)
                }
            }
        }
    }

    @Test
    fun `absolute sum agrees with the scalar oracle across offsets and vector boundaries`() {
        assumeSimd()

        for (length in lengths()) {
            for (offset in intArrayOf(0, 1, 3, 7)) {
                val values = DoubleArray(offset + length + 3) { Double.NaN }
                for (i in 0 until length) values[offset + i] = (i % 17 - 8) / 16.0
                val expected = ScalarEngine.asum(values, offset, length)

                val actual = SimdOps.asum(values, offset, length)

                assertAsumAgreesWithReference(expected, actual, "length=$length offset=$offset")
            }
        }
    }

    @Test
    fun `absolute sum agrees with the scalar oracle for finite scales`() {
        assumeSimd()

        for (length in lengths().filter { it > 0 }) {
            for (value in doubleArrayOf(-1.25e-250, Double.MIN_VALUE, 1.25e200)) {
                for (position in intArrayOf(0, length / 2, length - 1).distinct()) {
                    val values = DoubleArray(length + 6) { Double.NaN }
                    values.fill(-0.0, 3, 3 + length)
                    values[3 + position] = value
                    val expected = ScalarEngine.asum(values, 3, length)

                    val actual = SimdOps.asum(values, 3, length)

                    assertAsumAgreesWithReference(expected, actual, "length=$length position=$position value=$value")
                }
            }
        }
    }

    @Test
    fun `dot agrees with the scalar oracle across independent offsets and vector boundaries`() {
        assumeSimd()

        for (length in lengths()) {
            for (aOffset in intArrayOf(0, 1, 3, 7)) {
                for (bOffset in intArrayOf(0, 2, 5)) {
                    val a = DoubleArray(aOffset + length + 3) { Double.NaN }
                    val b = DoubleArray(bOffset + length + 3) { Double.NaN }
                    for (i in 0 until length) {
                        a[aOffset + i] = (i % 17 - 8) / 16.0
                        b[bOffset + i] = (i % 11 - 5) / 8.0
                    }
                    val expected = ScalarEngine.dot(a, aOffset, b, bOffset, length)

                    val actual = SimdOps.dot(a, aOffset, b, bOffset, length)

                    assertDotAgreesWithReference(expected, actual, "length=$length offsets=$aOffset $bOffset")
                }
            }
        }
    }

    // Each length ends a different body: the short loop's vector and scalar tails, the streaming body's
    // unrolled blocks alone, and the streaming body with both of its tails. There is no unit-multiplier body.
    @Test
    fun `axpy agrees with the scalar oracle on aliased and disjoint windows`() {
        assumeSimd()
        val lane = SimdOps.lanes()
        val values = doubleArrayOf(-0.125, 0.0, 0.25, -1.25, -2.5, 3.0)
        val lengths = listOf(0, lane - 1, lane, lane + 1, 2 * lane + 1, 65_535, 65_536, 65_536 + 5 * lane + 1)

        for (length in lengths.distinct()) {
            for (alpha in doubleArrayOf(0.0, -0.75)) {
                for (sourceOffset in intArrayOf(3, length + 6)) {
                    val expected = DoubleArray(2 * length + 9) { values[it % values.size] }
                    val actual = expected.copyOf()
                    ScalarEngine.axpy(expected, 3, alpha, expected, sourceOffset, length)

                    SimdOps.axpy(actual, 3, alpha, actual, sourceOffset, length)

                    val layout = if (sourceOffset == 3) "aliased" else "disjoint"
                    assertAxpyAgreesWithReference(expected, actual, "length=$length alpha=$alpha $layout")
                }
            }
        }
    }

    @Test
    fun `iamax agrees with the scalar oracle when blocks keep improving`() {
        assumeSimd()
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
                    val expected = ScalarEngine.iamax(values, offset, length)

                    val actual = SimdOps.iamax(values, offset, length)

                    assertIamaxAgreesWithReference(expected, actual, "length=$length offset=$offset fixture=$fixture")
                }
            }
        }
    }

    @Test
    fun `iamax distinguishes magnitudes sharing a high word`() {
        assumeSimd()

        for (length in lengths().filter { it > 0 }) {
            for (offset in intArrayOf(0, 1, 7)) {
                for (position in intArrayOf(0, length / 2, length - 1).distinct()) {
                    val values = DoubleArray(offset + length + 3) { Double.NaN }
                    values.fill(1.0, offset, offset + length)
                    values[offset + position] = Double.fromBits(1.0.toRawBits() + 1)
                    val expected = ScalarEngine.iamax(values, offset, length)

                    val actual = SimdOps.iamax(values, offset, length)

                    assertIamaxAgreesWithReference(expected, actual, "length=$length offset=$offset position=$position")
                }
            }
        }
    }

    @Test
    fun `iamax finds the largest finite magnitude beside a NaN in the same lane`() {
        assumeSimd()
        val lanes = SimdOps.lanes()

        // Past the first block, which every scan reads while its threshold is still zero.
        for (length in lengths().filter { it >= 16 * lanes }) {
            for (vector in intArrayOf(0, 3, 6)) {
                val values = DoubleArray(length) { 1.0 }
                val nan = 8 * lanes + vector * lanes
                values[nan] = Double.NaN
                values[nan + lanes] = 4.0
                val expected = ScalarEngine.iamax(values, 0, length)

                val actual = SimdOps.iamax(values, 0, length)

                assertIamaxAgreesWithReference(expected, actual, "length=$length nan=$nan")
            }
        }
    }

    /** A runtime without the module skips rather than passes, since there is no vector body to compare. */
    private fun assumeSimd() {
        Assume.assumeTrue("the Vector API module is unavailable", simdAvailable)
    }

    /** The exact Vector API engine, which hands each window to [SimdOps] once it fills a lane block. */
    private fun simdEngine(): EngineKernels {
        assumeSimd()
        return checkNotNull(KoblasEngine.simd).kernels
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
        assertClose(expected, actual, context)
    }

    private fun assertDotAgreesWithReference(expected: Double, actual: Double, context: String) {
        assertClose(expected, actual, context)
    }

    private fun assertAxpyAgreesWithReference(expected: DoubleArray, actual: DoubleArray, context: String) {
        assertContentEquals(expected, actual, context)
    }
}
