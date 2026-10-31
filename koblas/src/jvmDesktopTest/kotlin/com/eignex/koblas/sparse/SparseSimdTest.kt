package com.eignex.koblas.sparse

import com.eignex.koblas.assertClose
import com.eignex.koblas.internal.jvm25.simd.SparseSimd
import com.eignex.koblas.internal.jvm25.simd.simdAvailable
import com.eignex.koblas.internal.sparse.ScalarIndexedSparseKernels
import com.eignex.koblas.internal.sparse.dotDense
import org.junit.Assume
import kotlin.random.Random
import kotlin.test.Test

class SparseSimdTest {
    @Test
    fun `indexed dot dense agrees with the portable implementation`() {
        forEachPattern(::assertDotDenseAgreesWithReference)
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
