package com.eignex.koblas.sparse

import com.eignex.koblas.assertClose
import com.eignex.koblas.internal.jvm25.simd.SparseSimd
import com.eignex.koblas.internal.jvm25.simd.simdAvailable
import com.eignex.koblas.internal.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.internal.sparse.axpy
import com.eignex.koblas.internal.sparse.dotDense
import com.eignex.koblas.internal.sparse.gather
import com.eignex.koblas.internal.sparse.gatherZero
import com.eignex.koblas.internal.sparse.scatter
import org.junit.Assume
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

class SparseSimdTest {
    @Test
    fun `indexed scatter agrees with the portable implementation`() {
        forEachPattern(::assertScatterAgreesWithReference)
    }

    @Test
    fun `indexed gather agrees with the portable implementation`() {
        forEachPattern(::assertGatherAgreesWithReference)
    }

    @Test
    fun `indexed gather zero agrees with the portable implementation`() {
        forEachPattern(::assertGatherZeroAgreesWithReference)
    }

    @Test
    fun `indexed axpy agrees with the portable implementation`() {
        forEachPattern { x -> assertAxpyAgreesWithReference(x, -0.75) }
    }

    @Test
    fun `indexed dot dense agrees with the portable implementation`() {
        forEachPattern(::assertDotDenseAgreesWithReference)
    }

    @Test
    fun `indexed norm agrees with the portable implementation across scaling`() {
        forEachPattern { x ->
            for (scale in listOf(1.0, 1e200, 1e-200)) {
                assertNormAgreesWithReference(x, scale)
            }
        }
    }

    @Test
    fun `indexed axpy rounds multiplication before addition`() {
        Assume.assumeTrue("the Vector API module is unavailable", simdAvailable)
        val count = 256
        val x = Pattern(
            IntArray(count) { it },
            0,
            DoubleArray(count) { 1e308 },
            0,
            count,
            DoubleArray(count) { -1e308 },
        )

        assertAxpyAgreesWithReference(x, 2.0)
    }

    private fun assertScatterAgreesWithReference(x: Pattern) {
        val expected = x.dense.copyOf()
        ScalarIndexedSparseKernels.scatter(x.indices, x.indexOffset, x.values, x.valueOffset, x.count, expected)
        val actual = x.dense.copyOf()

        SparseSimd.scatter(x.indices, x.indexOffset, x.values, x.valueOffset, x.count, actual)

        assertContentEquals(expected, actual, "scatter ${x.context}")
    }

    private fun assertGatherAgreesWithReference(x: Pattern) {
        val expected = x.values.copyOf()
        ScalarIndexedSparseKernels.gather(x.indices, x.indexOffset, expected, x.valueOffset, x.count, x.dense)
        val actual = x.values.copyOf()
        val source = x.dense.copyOf()

        SparseSimd.gather(x.indices, x.indexOffset, actual, x.valueOffset, x.count, source)

        assertContentEquals(expected, actual, "gather values ${x.context}")
        assertContentEquals(x.dense, source, "gather source ${x.context}")
    }

    private fun assertGatherZeroAgreesWithReference(x: Pattern) {
        val expectedValues = x.values.copyOf()
        val expectedDense = x.dense.copyOf()
        ScalarIndexedSparseKernels.gatherZero(
            x.indices,
            x.indexOffset,
            expectedValues,
            x.valueOffset,
            x.count,
            expectedDense,
        )
        val actualValues = x.values.copyOf()
        val actualDense = x.dense.copyOf()

        SparseSimd.gatherZero(x.indices, x.indexOffset, actualValues, x.valueOffset, x.count, actualDense)

        assertContentEquals(expectedValues, actualValues, "gatherZero values ${x.context}")
        assertContentEquals(expectedDense, actualDense, "gatherZero dense ${x.context}")
    }

    private fun assertAxpyAgreesWithReference(x: Pattern, alpha: Double) {
        val expected = x.dense.copyOf()
        ScalarIndexedSparseKernels.axpy(x.indices, x.indexOffset, x.values, x.valueOffset, x.count, alpha, expected)
        val actual = x.dense.copyOf()

        SparseSimd.axpy(x.indices, x.indexOffset, x.values, x.valueOffset, x.count, alpha, actual)

        assertClose(expected, actual, "axpy ${x.context}")
    }

    private fun assertDotDenseAgreesWithReference(x: Pattern) {
        val expected = ScalarIndexedSparseKernels.dotDense(
            x.indices,
            x.indexOffset,
            x.values,
            x.valueOffset,
            x.count,
            x.dense,
        )

        val actual = SparseSimd.dotDense(x.indices, x.indexOffset, x.values, x.valueOffset, x.count, x.dense)

        assertClose(expected, actual, "dotDense ${x.context}")
    }

    private fun assertNormAgreesWithReference(x: Pattern, scale: Double) {
        val dense = DoubleArray(x.dense.size) { Double.NaN }
        for (k in 0 until x.count) {
            val index = x.indices[x.indexOffset + k]
            dense[index] = (0.25 + abs(x.dense[index])) * scale
        }
        val expected = ScalarIndexedSparseKernels.nrm2(x.indices, x.indexOffset, x.count, dense)

        val actual = SparseSimd.nrm2(x.indices, x.indexOffset, x.count, dense)

        assertTrue(
            abs(expected - actual) <= 1e-12 * abs(expected),
            "nrm2 ${x.context} scale=$scale expected=$expected actual=$actual",
        )
    }

    private fun forEachPattern(block: (Pattern) -> Unit) {
        Assume.assumeTrue("the Vector API module is unavailable", simdAvailable)
        val lanes = SparseSimd.lanes
        val counts = (
            listOf(lanes - 1, lanes, lanes + 1, 2 * lanes + 1) +
                listOf(1, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 255, 256, 257)
            ).distinct()
        for (count in counts) {
            for ((indexOffset, valueOffset) in listOf(0 to 0, 1 to 3, 3 to 1)) {
                val random = Random(count)
                val indices = IntArray(indexOffset + count + 3) { -1 }
                val values = DoubleArray(valueOffset + count + 3) { Double.NaN }
                for (k in 0 until count) {
                    indices[indexOffset + k] = 2 * k + 1
                    values[valueOffset + k] = random.nextDouble(-1.0, 1.0)
                }
                block(
                    Pattern(
                        indices,
                        indexOffset,
                        values,
                        valueOffset,
                        count,
                        DoubleArray(2 * count + 1) { random.nextDouble(-1.0, 1.0) },
                    ),
                )
            }
        }
    }

    private data class Pattern(
        val indices: IntArray,
        val indexOffset: Int,
        val values: DoubleArray,
        val valueOffset: Int,
        val count: Int,
        val dense: DoubleArray,
    ) {
        val context: String get() = "count=$count indexOffset=$indexOffset valueOffset=$valueOffset"
    }
}
