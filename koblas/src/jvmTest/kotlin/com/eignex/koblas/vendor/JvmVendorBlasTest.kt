package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.StridedVector
import com.eignex.koblas.copyOf
import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.dense.ScalarVectorKernels
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class JvmVendorBlasTest {
    private fun matrix(rows: Int, cols: Int, seed: Int) = DenseMatrix.wrap(rows, cols, vendorValues(rows * cols, seed))

    private fun vector(size: Int, seed: Int) = DenseVector.wrap(vendorValues(size, seed))

    @Test
    fun `gemm transposes each operand the flag selects`() = withVendor { blas ->
        // Transposition is passed through flags, so every combination must reach the binding correctly.
        for (transposeA in listOf(false, true)) {
            for (transposeB in listOf(false, true)) {
                val a = if (transposeA) matrix(4, 3, 5) else matrix(3, 4, 5)
                val b = if (transposeB) matrix(5, 4, 6) else matrix(4, 5, 6)
                val c = matrix(3, 5, 7)
                val expected = c.copyOf()
                ReferenceBlas.gemm(1.0, a, transposeA, b, transposeB, 0.5, expected)

                blas.gemm(1.0, a, transposeA, b, transposeB, 0.5, c)

                assertAgreesWithReference(expected.values, c.values, "gemm transA=$transposeA transB=$transposeB")
            }
        }
    }

    @Test
    fun `beta zero overwrites the destination without reading it`() = withVendor { blas ->
        val a = matrix(3, 4, 11)
        val b = matrix(4, 5, 12)
        val c = DenseMatrix.wrap(3, 5, DoubleArray(15) { Double.NaN })
        val clean = DenseMatrix.zero(3, 5)
        ReferenceBlas.gemm(1.0, a, false, b, false, 0.0, clean)

        blas.gemm(1.0, a, false, b, false, 0.0, c)

        assertAgreesWithReference(clean.values, c.values, "gemm beta zero")
    }

    @Test
    fun `an empty operation leaves the destination alone`() = withVendor { blas ->
        val a = DenseMatrix.wrap(3, 0, DoubleArray(0))
        val b = DenseMatrix.wrap(0, 5, DoubleArray(0))
        val storage = vendorValues(15, 13)
        val original = storage.copyOf()

        blas.gemm(1.0, a, false, b, false, 1.0, DenseMatrix.wrap(3, 5, storage))

        assertContentEquals(original, storage)
    }

    @Test
    fun `gemv agrees with the reference under either transpose`() = withVendor { blas ->
        for (transpose in listOf(false, true)) {
            val a = matrix(4, 3, 14)
            val x = vector(if (transpose) 4 else 3, 15)
            val y = vector(if (transpose) 3 else 4, 16)
            val expected = y.values.copyOf()
            ReferenceBlas.gemv(0.5, a, x.values, 2.0, expected, transpose)

            blas.gemv(0.5, a, transpose, x, 2.0, y)

            assertAgreesWithReference(expected, y.values, "gemv transA=$transpose")
        }
    }

    @Test
    fun `a symmetric operand reads only its stored triangle`() = withVendor { blas ->
        val order = 5
        val stored = vendorValues(order * order, 17)
        for (column in 0 until order) {
            for (row in 0 until column) stored[row + column * order] = Double.NaN
        }
        val a = DenseMatrix.wrap(order, order, stored)
        val x = vector(order, 18)
        val y = DenseVector.zero(order)
        val expected = DoubleArray(order)
        ReferenceBlas.symv(1.0, a, x.values, 0.0, expected)

        blas.symv(1.0, a, MatrixStructure.SymmetricLower, x, 0.0, y)

        assertAgreesWithReference(expected, y.values, "symv")
    }

    @Test
    fun `a unit diagonal is never loaded from storage`() = withVendor { blas ->
        val order = 4
        val stored = vendorValues(order * order, 19)
        for (index in 0 until order) stored[index + index * order] = Double.NaN
        val a = DenseMatrix.wrap(order, order, stored)
        val rightHandSide = vendorValues(order, 20)
        val expected = rightHandSide.copyOf()
        ReferenceBlas.trsv(a, expected, lower = true, unitDiag = true)
        val x = DenseVector.wrap(rightHandSide.copyOf())

        blas.trsv(a, MatrixStructure.UnitLower, false, x)

        assertAgreesWithReference(expected, x.values, "trsv unit diagonal")
    }

    @Test
    fun `a triangular solve inverts its own product`() = withVendor { blas ->
        val order = 4
        val stored = vendorValues(order * order, 21)
        for (index in 0 until order) stored[index + index * order] = 3.0 + index
        val triangular = DenseMatrix.wrap(order, order, stored)
        val original = vendorValues(order, 22)
        val x = DenseVector.wrap(original.copyOf())

        blas.trsv(triangular, MatrixStructure.TriangularLower, false, x)
        blas.trmv(triangular, MatrixStructure.TriangularLower, false, x)

        assertAgreesWithReference(original, x.values, "trsv then trmv")
    }

    @Test
    fun `syrk fills only the selected triangle`() = withVendor { blas ->
        val order = 4
        val a = matrix(order, 3, 23)
        val c = DenseMatrix.wrap(order, order, DoubleArray(order * order) { 7.0 })
        // The oracle skips the unselected triangle, so its untouched 7.0 is part of what agreement means.
        val expected = c.copyOf()
        ReferenceBlas.syrk(1.0, a, false, 0.0, expected)

        blas.syrk(1.0, a, false, 0.0, c, MatrixStructure.SymmetricLower)

        assertAgreesWithReference(expected.values, c.values, "syrk")
    }

    @Test
    fun `dot agrees with the scalar oracle under positive and negative strides`() = withVendor { blas ->
        forEachStride { storage, offset, size, stride ->
            val other = vendorValues(storage.size, 25)
            val x = StridedVector(storage, offset, size, stride)
            val y = StridedVector(other, offset, size, stride)
            val expected = ScalarVectorKernels.dot(storage, offset, other, offset, size, stride, stride)

            val actual = blas.dot(x, y)

            assertAgreesWithReference(expected, actual, "dot at $stride")
        }
    }

    @Test
    fun `norm agrees with the scalar oracle under positive and negative strides`() = withVendor { blas ->
        forEachStride { storage, offset, size, stride ->
            val x = StridedVector(storage, offset, size, stride)
            val expected = ScalarVectorKernels.nrm2(storage, offset, size, stride)

            val actual = blas.nrm2(x)

            assertAgreesWithReference(expected, actual, "nrm2 at $stride")
        }
    }

    @Test
    fun `absolute sum agrees with the scalar oracle under positive and negative strides`() = withVendor { blas ->
        forEachStride { storage, offset, size, stride ->
            val x = StridedVector(storage, offset, size, stride)
            val expected = ScalarVectorKernels.asum(storage, offset, size, stride)

            val actual = blas.asum(x)

            assertAgreesWithReference(expected, actual, "asum at $stride")
        }
    }

    @Test
    fun `maximum index selects a largest magnitude under positive and negative strides`() = withVendor { blas ->
        forEachStride { storage, offset, size, stride ->
            val x = StridedVector(storage, offset, size, stride)
            val expected = ScalarVectorKernels.iamax(storage, offset, size, stride)

            val actual = blas.iamax(x)

            assertEquals(abs(x[expected]), abs(x[actual]), "iamax at $stride")
        }
    }

    private fun forEachStride(body: (DoubleArray, Int, Int, Int) -> Unit) {
        val size = 6
        for (stride in listOf(1, 2, -1, -2)) {
            val storage = vendorValues(2 * size, 24)
            val offset = if (stride > 0) 0 else (size - 1) * -stride
            body(storage, offset, size, stride)
        }
    }

    @Test
    fun `a backward vector reports the last of several equal maxima`() = withVendor { blas ->
        val storage = doubleArrayOf(1.0, 5.0, 2.0, 5.0, 3.0)

        val forward = blas.iamax(StridedVector(storage, 0, 5, 1))
        val backward = blas.iamax(StridedVector(storage, 4, 5, -1))

        // The backward vector reads [3, 5, 2, 5, 1], whose equal maxima sit at logical 1 and 3.
        assertEquals(1, forward, "a forward vector reports the first largest entry")
        assertEquals(3, backward, "a backward vector reports the last of the equal maxima")
    }

    @Test
    fun `axpy writes only through the vector it was given`() = withVendor { blas ->
        val size = 4
        val storage = vendorValues(20, 26)
        val expected = storage.copyOf()
        val x = vector(size, 27)
        val y = StridedVector(storage, 3, size, 4)
        ScalarVectorKernels.axpy(expected, 3, 0.5, x.values, 0, size, yStride = 4)

        blas.axpy(0.5, x, y)

        assertAgreesWithReference(expected, storage, "axpy backing buffer")
    }

    @Test
    fun `an empty vector call returns the identity without touching storage`() = withVendor { blas ->
        val empty = DenseVector.wrap(DoubleArray(0))

        assertEquals(0.0, blas.dot(empty, empty))
        assertEquals(0.0, blas.nrm2(empty))
        assertEquals(0.0, blas.asum(empty))
        assertEquals(0, blas.iamax(empty))
    }

    @Test
    fun `gemmt writes one triangle whether it is bound or composed`() = withVendor { blas ->
        val order = 4
        val a = matrix(order, 3, 28)
        val b = matrix(3, order, 29)
        val c = DenseMatrix.wrap(order, order, DoubleArray(order * order) { 5.0 })
        val expected = c.copyOf()
        ReferenceBlas.gemmt(1.0, a, false, b, false, 0.0, expected)

        blas.gemmt(1.0, a, false, b, false, 0.0, c, MatrixStructure.SymmetricLower)

        assertAgreesWithReference(expected.values, c.values, "gemmt")
    }

    @Test
    fun `symmetric rank two update agrees with the reference under its shared transpose flag`() = withVendor { blas ->
        for (transpose in listOf(false, true)) {
            val order = 3
            val depth = 2
            val a = if (transpose) matrix(depth, order, 34) else matrix(order, depth, 34)
            val b = if (transpose) matrix(depth, order, 35) else matrix(order, depth, 35)
            val c = DenseMatrix.wrap(order, order, vendorValues(order * order, 36))
            val expected = c.copyOf()
            ReferenceBlas.syr2k(1.25, a, b, transpose, 0.5, expected)

            blas.syr2k(1.25, a, b, transpose, 0.5, c, MatrixStructure.SymmetricLower)

            assertAgreesWithReference(expected.values, c.values, "syr2k transA=$transpose")
        }
    }

    @Test
    fun `scal scales a negatively strided vector`() = withVendor { blas ->
        // BLAS returns without doing anything for a non-positive increment, so the sign must not be passed on.
        val size = 5
        val storage = vendorValues(size, 30)
        val expected = storage.copyOf()
        val x = StridedVector(storage, size - 1, size, -1)
        ScalarVectorKernels.scale(expected, size - 1, 2.0, size, -1)

        blas.scal(2.0, x)

        assertAgreesWithReference(expected, storage, "scal negative stride")
    }

    @Test
    fun `swap exchanges two interleaved vectors of one array`() = withVendor { blas ->
        // Two rows of a column-major matrix interleave in one array, which is what a pivot swap works on.
        val size = 4
        val lda = 3
        val storage = vendorValues(lda * size, 31)
        val expected = storage.copyOf()
        val first = StridedVector(storage, 0, size, lda)
        val second = StridedVector(storage, 1, size, lda)
        ScalarVectorKernels.swap(expected, 0, expected, 1, size, lda, lda)

        blas.swap(first, second)

        assertContentEquals(expected, storage, "swap backing buffer")
    }

    @Test
    fun `a symmetric rank one update keeps the triangle the call never wrote`() = withVendor { blas ->
        val order = 3
        val a = DenseMatrix.wrap(order, order, vendorValues(order * order, 32))
        val x = vector(order, 33)
        val expected = a.copyOf()
        ReferenceBlas.syr(1.5, x, expected)

        blas.syr(1.5, x, a, MatrixStructure.SymmetricLower)

        assertAgreesWithReference(expected.values, a.values, "syr")
    }

    @Test
    fun `the resolved library and version identify what actually ran`() = withVendor { blas ->
        assertTrue(blas.version.isNotEmpty(), "no version string")
        assertTrue(BlasOperation.Gemm in blas.directlyImplemented, "gemm should be directly implemented")
    }

    @Test
    fun `the reported library is the file the symbol came from not the name that was asked for`() {
        for (vendor in Vendor.entries) {
            val blas = openBlas(vendor) ?: continue

            assertTrue(
                blas.libraryPath.startsWith("/"),
                "${vendor.vendorName} reported ${blas.libraryPath}, which is a candidate name rather than a file",
            )
            assertEquals(vendor, blas.vendor, "a library resolved under the wrong vendor")
        }
    }
}
