package com.eignex.koblas.internal.operations

import com.eignex.koblas.*
import com.eignex.koblas.internal.kernels.kernels
import kotlin.math.abs
import kotlin.test.*

class ReductionWindowsTest {
    @Test
    fun `built in reduction windows agree with the scalar oracle`() {
        val values = doubleArrayOf(99.0, -2.0, 99.0, 3.0, 99.0, -4.0, 99.0)
        val oracle = KoblasEngine.scalar.kernels
        for (engine in listOfNotNull(KoblasEngine.scalar, KoblasEngine.simd)) {
            for (stride in intArrayOf(2, -2)) {
                val offset = if (stride > 0) 1 else 5
                for (size in intArrayOf(0, 3)) {
                    assertReductionAgreesWithReference(
                        oracle.sum(values, offset, size, stride),
                        engine.sumWindow(values, offset, size, stride),
                    )
                    assertReductionAgreesWithReference(
                        oracle.asum(values, offset, size, stride),
                        engine.absoluteSumWindow(values, offset, size, stride),
                    )
                    assertReductionAgreesWithReference(
                        oracle.nrm2(values, offset, size, stride),
                        engine.euclideanNormWindow(values, offset, size, stride),
                    )
                    val index = oracle.iamax(values, offset, size, stride)
                    val expected = if (index < 0) 0.0 else abs(values[offset + index * stride])
                    assertReductionAgreesWithReference(expected, engine.maximumMagnitude(values, offset, size, stride))
                }
            }
        }
    }

    @Test
    fun `custom engines receive typed windows with the original addressing`() {
        val values = doubleArrayOf(99.0, -2.0, 99.0, 3.0, 99.0, -4.0, 99.0)
        val calls = mutableListOf<String>()
        val oracle = KoblasEngine.scalar
        val engine = object : KoblasEngine by oracle {
            private fun record(name: String, v: DenseVector) {
                assertSame(values, v.values)
                assertEquals(5, v.offset)
                assertEquals(3, v.size)
                assertEquals(-2, v.stride)
                calls += name
            }
            override fun sum(v: DenseVector): Double {
                record("sum", v)
                return oracle.sum(v)
            }
            override fun asum(v: DenseVector): Double {
                record("asum", v)
                return oracle.asum(v)
            }
            override fun nrm2(v: DenseVector): Double {
                record("nrm2", v)
                return oracle.nrm2(v)
            }
            override fun iamax(v: DenseVector): Int {
                record("iamax", v)
                return oracle.iamax(v)
            }
        }

        engine.sumWindow(values, 5, 3, -2)
        engine.absoluteSumWindow(values, 5, 3, -2)
        engine.euclideanNormWindow(values, 5, 3, -2)
        engine.maximumMagnitude(values, 5, 3, -2)

        assertEquals(listOf("sum", "asum", "nrm2", "iamax"), calls)
    }

    private fun assertReductionAgreesWithReference(expected: Double, actual: Double) {
        assertClose(expected, actual, "window reduction")
    }
}
