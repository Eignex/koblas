package com.eignex.koblas.internal.jvm25.simd

import com.eignex.koblas.*
import com.eignex.koblas.internal.engine.implementationFor
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import org.junit.Assume
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

class SimdIndexedKernelsTest {
    // The indexed leaf registers the dense dot alone, and only on two-lane units; every other indexed operation
    // stays scalar at any width.
    @Test
    fun `only an indexed dense dot from its minimum length reaches the indexed vector leaf`() {
        Assume.assumeTrue("the Vector API module is unavailable", simdAvailable)
        val engine = checkNotNull(KoblasEngine.simd)
        val minimum = SimdIndexedKernels.minimumLength(EngineImplementation.IndexedDot)
        val indexed = EngineImplementation.entries.filter {
            it.ordinal in EngineImplementation.IndexedDot.ordinal until EngineImplementation.Panels.ordinal
        }

        for (operation in indexed) {
            for (count in intArrayOf(minimum - 1, minimum, WIDE)) {
                val expected =
                    operation == EngineImplementation.IndexedDot && SimdIndexedKernels.vectorDot && count >= minimum

                val reached = engine.implementationFor(operation, count, true)

                assertEquals(expected, reached == SimdIndexedKernels.name, "$operation at $count reached $reached")
            }
        }
    }

    @Test
    fun `indexed dot dense agrees with the portable implementation`() {
        forEachPattern(::assertDotDenseAgreesWithReference)
    }

    private fun assertDotDenseAgreesWithReference(x: Pattern) {
        val expected = ScalarEngine.dot(x.indices, x.indexOffset, x.values, x.valueOffset, x.count, x.dense)

        val actual = SimdIndexedKernels.dot(x.indices, x.indexOffset, x.values, x.valueOffset, x.count, x.dense)

        assertClose(expected, actual, "dot ${x.context}")
    }

    private fun forEachPattern(block: (Pattern) -> Unit) {
        Assume.assumeTrue("the Vector API module is unavailable", simdAvailable)
        Assume.assumeTrue("the indexed vector leaf requires two lanes", SimdIndexedKernels.vectorDot)
        val lanes = SimdIndexedKernels.lanes
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

    private companion object {
        /** A support wider than any lane block, so a vector body that exists is reached. */
        const val WIDE = 1 shl 12
    }
}
