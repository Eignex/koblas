package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.internal.jvm25.SimdOps
import kotlin.test.Test
import kotlin.test.assertEquals

class SimdOpsTest {
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
                            1 -> if (i < 16 * lanes) i.toDouble() else Double.NaN
                            2 -> if (i < 16 * lanes) i.toDouble() else -(16 * lanes - 1).toDouble()
                            else -> if (i % (4 * lanes) == 0) Double.MIN_VALUE else -0.0
                        }
                    }
                    val expected = ScalarVectorKernels.iamax(values, offset, length)

                    val actual = SimdOps.iamax(values, offset, length)

                    assertIamaxAgreesWithReference(expected, actual, "length=$length offset=$offset fixture=$fixture")
                }
            }
        }
    }

    private fun assertIamaxAgreesWithReference(expected: Int, actual: Int, context: String) {
        assertEquals(expected, actual, context)
    }
}
