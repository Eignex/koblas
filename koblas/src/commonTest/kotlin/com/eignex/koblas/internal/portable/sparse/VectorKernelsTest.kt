package com.eignex.koblas.internal.portable.sparse

import com.eignex.koblas.internal.portable.dense.euclideanNorm
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue

class VectorKernelsTest {
    @Test
    fun `indexed norm agrees with reference across scaling and tails`() {
        for (count in COUNTS) {
            for (scale in listOf(0.0, 1.0, 1e200, 1e-200)) {
                val indices = IntArray(count + 5) { -1 }
                val values = DoubleArray(37) { (it % 11 - 5) * scale }
                for (k in 0 until count) indices[k + 2] = (k * 13) % values.size
                val gathered = DoubleArray(count) { values[indices[it + 2]] }
                val expected = euclideanNorm(gathered, 0, 1, count)

                val actual = indexedNrm2(indices, 2, count, values)

                assertNormAgreesWithReference(expected, actual, "count=$count scale=$scale")
            }
        }
    }

    private fun assertNormAgreesWithReference(expected: Double, actual: Double, context: String) {
        assertTrue(abs(expected - actual) <= 1e-12 * abs(expected), context)
    }

    private companion object {
        /** Each side of the unrolled square sum's threshold, and every tail its four-way unroll leaves. */
        val COUNTS = listOf(0, 1, 63, 64, 65, 66, 67)
    }
}
