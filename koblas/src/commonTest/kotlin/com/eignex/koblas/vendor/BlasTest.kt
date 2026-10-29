package com.eignex.koblas.vendor

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.StridedVector
import com.eignex.koblas.assertClose
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

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

    @Test
    fun `dot agrees with the scalar oracle under independent strides`() {
        val vendor = vendor ?: return skipped()
        forEachStridePair { x, y ->
            val expected = scalar.dot(x.values, x.offset, y.values, y.offset, x.size, x.stride, y.stride)

            val actual = vendor.dot(x, y)

            assertAgreesWithReference(expected, actual, context(vendor, "dot", x, y))
        }
    }

    @Test
    fun `norm agrees with the scalar oracle under positive and negative strides`() {
        val vendor = vendor ?: return skipped()
        forEachStride { x ->
            val expected = scalar.nrm2(x.values, x.offset, x.size, x.stride)

            val actual = vendor.nrm2(x)

            assertAgreesWithReference(expected, actual, context(vendor, "nrm2", x))
        }
    }

    @Test
    fun `absolute sum agrees with the scalar oracle under positive and negative strides`() {
        val vendor = vendor ?: return skipped()
        forEachStride { x ->
            val expected = scalar.asum(x.values, x.offset, x.size, x.stride)

            val actual = vendor.asum(x)

            assertAgreesWithReference(expected, actual, context(vendor, "asum", x))
        }
    }

    @Test
    fun `maximum index selects a largest magnitude under positive and negative strides`() {
        val vendor = vendor ?: return skipped()
        forEachStride { x ->
            val expected = scalar.iamax(x.values, x.offset, x.size, x.stride)

            val actual = vendor.iamax(x)

            assertIamaxHonoursItsContract(expected, actual, x, context(vendor, "iamax", x))
        }
    }

    @Test
    fun `a backward vector reports the last of several equal maxima`() {
        val vendor = vendor ?: return skipped()
        for (stride in intArrayOf(1, 2)) {
            val forward = strideVector(stride, 3, 24)
            val entries = doubleArrayOf(1.0, 5.0, 2.0, -5.0, 3.0, 0.5)
            for (i in entries.indices) forward.values[forward.offset + i * stride] = entries[i]
            val backward = StridedVector(
                forward.values,
                forward.offset + (forward.size - 1) * stride,
                forward.size,
                -stride,
            )
            val expected = scalar.iamax(forward.values, forward.offset, forward.size, stride)

            val actualForward = vendor.iamax(forward)
            val actualBackward = vendor.iamax(backward)

            assertEquals(expected, actualForward, context(vendor, "iamax forward tie", forward))
            // The binding walks backward vectors in physical ascending order, then remaps the index.
            assertEquals(backward.size - 1 - expected, actualBackward, context(vendor, "iamax backward tie", backward))
        }
    }

    @Test
    fun `scaling agrees with the scalar oracle without writing padding`() {
        val vendor = vendor ?: return skipped()
        forEachStride { x ->
            val expected = x.values.copyOf()
            scalar.scale(expected, x.offset, -0.75, x.size, x.stride)

            vendor.scal(-0.75, x)

            assertAgreesWithReference(expected, x.values, context(vendor, "scal", x))
        }
    }

    @Test
    fun `axpy agrees with the scalar oracle under independent strides without writing padding`() {
        val vendor = vendor ?: return skipped()
        forEachStridePair { x, y ->
            val originalX = x.values.copyOf()
            val expectedY = y.values.copyOf()
            scalar.axpy(expectedY, y.offset, 0.5, x.values, x.offset, x.size, y.stride, x.stride)

            vendor.axpy(0.5, x, y)

            assertContentEquals(originalX, x.values, context(vendor, "axpy source", x, y))
            assertAgreesWithReference(expectedY, y.values, context(vendor, "axpy destination", x, y))
        }
    }

    @Test
    fun `copy preserves logical order under independent strides without writing padding`() {
        val vendor = vendor ?: return skipped()
        forEachStridePair { x, y ->
            val originalX = x.values.copyOf()
            val expectedY = y.values.copyOf()
            for (i in 0 until x.size) expectedY[y.offset + i * y.stride] = x[i]

            vendor.copy(x, y)

            assertContentEquals(originalX, x.values, context(vendor, "copy source", x, y))
            assertContentEquals(expectedY, y.values, context(vendor, "copy destination", x, y))
        }
    }

    @Test
    fun `swap agrees with the scalar oracle under independent strides without writing padding`() {
        val vendor = vendor ?: return skipped()
        forEachStridePair { x, y ->
            val expectedX = x.values.copyOf()
            val expectedY = y.values.copyOf()
            scalar.swap(expectedX, x.offset, expectedY, y.offset, x.size, x.stride, y.stride)

            vendor.swap(x, y)

            assertAgreesWithReference(expectedX, x.values, context(vendor, "swap first", x, y))
            assertAgreesWithReference(expectedY, y.values, context(vendor, "swap second", x, y))
        }
    }

    @Test
    fun `rotation agrees with the scalar oracle under independent strides without writing padding`() {
        val vendor = vendor ?: return skipped()
        forEachStridePair { x, y ->
            val expectedX = x.values.copyOf()
            val expectedY = y.values.copyOf()
            val packedX = x.toDoubleArray()
            val packedY = y.toDoubleArray()
            scalar.rot(packedX, 0, packedY, 0, x.size, 0.6, 0.8)
            for (i in 0 until x.size) {
                expectedX[x.offset + i * x.stride] = packedX[i]
                expectedY[y.offset + i * y.stride] = packedY[i]
            }

            vendor.rot(x, y, 0.6, 0.8)

            assertAgreesWithReference(expectedX, x.values, context(vendor, "rot first", x, y))
            assertAgreesWithReference(expectedY, y.values, context(vendor, "rot second", x, y))
        }
    }

    private fun forEachStride(body: (StridedVector) -> Unit) {
        for (stride in MULTI_ENTRY_STRIDES) body(strideVector(stride, 3, 24))
    }

    private fun forEachStridePair(body: (StridedVector, StridedVector) -> Unit) {
        for (xStride in MULTI_ENTRY_STRIDES) {
            for (yStride in MULTI_ENTRY_STRIDES) {
                body(strideVector(xStride, 3, 24), strideVector(yStride, 7, 25))
            }
        }
    }

    private fun strideVector(stride: Int, padding: Int, seed: Int): StridedVector {
        val size = 6
        val offset = padding + if (stride < 0) (size - 1) * -stride else 0
        return StridedVector(vendorValues(32, seed), offset, size, stride)
    }

    private fun context(vendor: Blas, operation: String, x: StridedVector, y: StridedVector? = null): String =
        "${vendor.vendor} $operation x offset=${x.offset} stride=${x.stride}" +
            y?.let { " y offset=${it.offset} stride=${it.stride}" }.orEmpty()

    private fun assertAgreesWithReference(expected: Double, actual: Double, context: String) =
        assertClose(expected, actual, context)

    private fun assertAgreesWithReference(expected: DoubleArray, actual: DoubleArray, context: String) =
        assertClose(expected, actual, context)

    private fun assertIamaxHonoursItsContract(expected: Int, actual: Int, x: StridedVector, context: String) {
        assertTrue(actual in 0 until x.size, "$context returned index $actual")
        assertEquals(abs(x[expected]), abs(x[actual]), context)
    }

    private fun skipped() {
        println("SKIPPED: no CBLAS library installed; direct vendor calls were not verified")
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

        // Direct Blas calls bypass the native engine crossover even for these six-entry windows.
        val MULTI_ENTRY_STRIDES = intArrayOf(1, 2, -1, -2)
        val STRIDES = intArrayOf(Int.MIN_VALUE, -3, -1, 1, 3, Int.MAX_VALUE)
    }
}
