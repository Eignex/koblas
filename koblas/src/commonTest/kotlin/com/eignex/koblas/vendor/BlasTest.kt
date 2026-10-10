package com.eignex.koblas.vendor

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.StridedVector
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class BlasTest {
    private val scalar = BuiltinEngines.scalar.vectorKernels

    @Test
    fun `singleton norm accepts extreme strides`() {
        val vendor = vendor ?: return skipped()
        for (stride in STRIDES) {
            val values = doubleArrayOf(Double.NaN, -3.25, Double.NaN)
            val vector = StridedVector(values, 1, 1, stride)
            val expected = scalar.nrm2(values, 1, 1, stride)

            val actual = vendor.nrm2(vector)

            assertNormAgreesWithReference(expected, actual, "${vendor.vendor}, stride=$stride")
        }
    }

    @Test
    fun `singleton absolute sum accepts extreme strides`() {
        val vendor = vendor ?: return skipped()
        for (stride in STRIDES) {
            val values = doubleArrayOf(Double.NaN, -3.25, Double.NaN)
            val vector = StridedVector(values, 1, 1, stride)
            val expected = scalar.asum(values, 1, 1, stride)

            val actual = vendor.asum(vector)

            assertAsumAgreesWithReference(expected, actual, "${vendor.vendor}, stride=$stride")
        }
    }

    @Test
    fun `singleton scaling accepts extreme strides without writing padding`() {
        val vendor = vendor ?: return skipped()
        for (stride in STRIDES) {
            val values = doubleArrayOf(Double.NaN, -3.25, Double.NaN)
            val expected = values.copyOf()
            val vector = StridedVector(values, 1, 1, stride)
            scalar.scale(expected, 1, -0.75, 1, stride)

            vendor.scal(-0.75, vector)

            assertScalAgreesWithReference(expected, values, "${vendor.vendor}, stride=$stride")
        }
    }

    private fun skipped() {
        println("SKIPPED: no CBLAS library installed; singleton vendor calls were not verified")
    }

    private fun assertNormAgreesWithReference(expected: Double, actual: Double, context: String) =
        assertEquals(expected, actual, context)

    private fun assertAsumAgreesWithReference(expected: Double, actual: Double, context: String) =
        assertEquals(expected, actual, context)

    private fun assertScalAgreesWithReference(expected: DoubleArray, actual: DoubleArray, context: String) =
        assertContentEquals(expected, actual, context)

    private companion object {
        // Some vendors special-case singletons and hide a negative increment; prefer the affected binding.
        val vendor = openBlas(Vendor.OpenBlas) ?: openBlas()
        val STRIDES = intArrayOf(Int.MIN_VALUE, -3, -1, 1, 3, Int.MAX_VALUE)
    }
}
