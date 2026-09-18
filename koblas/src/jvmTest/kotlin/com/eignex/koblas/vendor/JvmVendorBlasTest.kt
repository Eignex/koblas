package com.eignex.koblas.vendor

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.StridedVector
import com.eignex.koblas.dense.MatrixStructure
import com.eignex.koblas.dense.ReferenceBlas
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class JvmVendorBlasTest {
    private fun values(size: Int, seed: Int = 1): DoubleArray {
        var state = seed
        return DoubleArray(size) {
            state = state * 1_103_515_245 + 12_345
            ((state ushr 8) % 1000) / 250.0 - 2.0
        }
    }

    private fun matrix(rows: Int, cols: Int, seed: Int) = DenseMatrix.wrap(rows, cols, values(rows * cols, seed))

    private fun vector(size: Int, seed: Int) = DenseVector.wrap(values(size, seed))

    /** A destination the oracle can write while the vendor writes the original. */
    private fun DenseMatrix.copy() = DenseMatrix.wrap(rows, cols, values.copyOf())

    @Test
    fun `gemm over whole matrices agrees with the reference`() = withVendor { blas ->
        val a = matrix(3, 4, 1)
        val b = matrix(4, 5, 2)
        val c = matrix(3, 5, 3)
        val expected = c.copy()
        ReferenceBlas.gemm(0.75, a, false, b, false, -0.25, expected)

        blas.gemm(0.75, a, false, b, false, -0.25, c)

        assertAgreesWithReference(expected.values, c.values, "gemm")
    }

    @Test
    fun `gemm transposes each operand the flag selects`() = withVendor { blas ->
        // The flags are the whole of what a transposed operand costs now, so each combination has to land.
        for (transposeA in listOf(false, true)) {
            for (transposeB in listOf(false, true)) {
                val a = if (transposeA) matrix(4, 3, 5) else matrix(3, 4, 5)
                val b = if (transposeB) matrix(5, 4, 6) else matrix(4, 5, 6)
                val c = matrix(3, 5, 7)
                val expected = c.copy()
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
        blas.gemm(1.0, a, false, b, false, 0.0, clean)

        blas.gemm(1.0, a, false, b, false, 0.0, c)

        assertAgreesWithReference(clean.values, c.values, "gemm beta zero")
    }

    @Test
    fun `an empty operation leaves the destination alone`() = withVendor { blas ->
        val a = DenseMatrix.wrap(3, 0, DoubleArray(0))
        val b = DenseMatrix.wrap(0, 5, DoubleArray(0))
        val storage = values(15, 13)
        val original = storage.copyOf()

        blas.gemm(1.0, a, false, b, false, 1.0, DenseMatrix.wrap(3, 5, storage))

        assertEquals(original.toList(), storage.toList())
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
        val stored = values(order * order, 17)
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
        val stored = values(order * order, 19)
        for (index in 0 until order) stored[index + index * order] = Double.NaN
        val a = DenseMatrix.wrap(order, order, stored)
        val rightHandSide = values(order, 20)
        val expected = rightHandSide.copyOf()
        ReferenceBlas.trsv(a, expected, lower = true, unitDiag = true)
        val x = DenseVector.wrap(rightHandSide.copyOf())

        blas.trsv(a, MatrixStructure.UnitLower, false, x)

        assertAgreesWithReference(expected, x.values, "trsv unit diagonal")
    }

    @Test
    fun `a triangular solve inverts its own product`() = withVendor { blas ->
        val order = 4
        val stored = values(order * order, 21)
        for (index in 0 until order) stored[index + index * order] = 3.0 + index
        val triangular = DenseMatrix.wrap(order, order, stored)
        val original = values(order, 22)
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
        val expected = c.copy()
        ReferenceBlas.syrk(1.0, a, false, 0.0, expected)

        blas.syrk(1.0, a, false, 0.0, c, MatrixStructure.SymmetricLower)

        assertAgreesWithReference(expected.values, c.values, "syrk")
    }

    @Test
    fun `level one calls agree with the reference including negative strides`() = withVendor { blas ->
        val size = 6
        val xs = values(2 * size, 24)
        val ys = values(2 * size, 25)
        for (stride in listOf(1, 2, -1, -2)) {
            val offset = if (stride > 0) 0 else (size - 1) * -stride
            val x = StridedVector(xs, offset, size, stride)
            val y = StridedVector(ys.copyOf(), offset, size, stride)
            val expectedDot = (0 until size).sumOf { x[it] * y[it] }

            assertTrue(abs(blas.dot(x, y) - expectedDot) <= 1e-12 * maxOf(1.0, abs(expectedDot)), "dot at $stride")

            val largest = (0 until size).maxOf { abs(x[it]) }
            assertEquals(largest, abs(x[blas.iamax(x)]), "iamax at $stride did not land on the largest entry")

            val expectedNorm = sqrt((0 until size).sumOf { x[it] * x[it] })
            assertTrue(abs(blas.nrm2(x) - expectedNorm) <= 1e-12 * maxOf(1.0, expectedNorm), "nrm2 at $stride")

            val expectedSum = (0 until size).sumOf { abs(x[it]) }
            assertTrue(abs(blas.asum(x) - expectedSum) <= 1e-12 * maxOf(1.0, expectedSum), "asum at $stride")
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
        val storage = values(20, 26)
        val original = storage.copyOf()
        val x = vector(size, 27)
        val y = StridedVector(storage, 3, size, 4)
        val expected = DoubleArray(size) { y[it] + 0.5 * x[it] }

        blas.axpy(0.5, x, y)

        assertAgreesWithReference(expected, DoubleArray(size) { y[it] }, "axpy")
        for (index in storage.indices) {
            if ((index - 3) % 4 != 0 || index < 3 || index > 3 + 3 * 4) {
                assertEquals(original[index], storage[index], "storage outside the vector changed at $index")
            }
        }
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
        val expected = c.copy()
        ReferenceBlas.gemmt(1.0, a, false, b, false, 0.0, expected)

        blas.gemmt(1.0, a, false, b, false, 0.0, c, MatrixStructure.SymmetricLower)

        assertAgreesWithReference(expected.values, c.values, "gemmt")
    }

    @Test
    fun `syr2k agrees with the reference under its shared transpose flag`() = withVendor { blas ->
        for (transpose in listOf(false, true)) {
            val order = 3
            val depth = 2
            val a = if (transpose) matrix(depth, order, 34) else matrix(order, depth, 34)
            val b = if (transpose) matrix(depth, order, 35) else matrix(order, depth, 35)
            val c = DenseMatrix.wrap(order, order, values(order * order, 36))
            val expected = c.copy()
            ReferenceBlas.syr2k(1.25, a, b, transpose, 0.5, expected)

            blas.syr2k(1.25, a, b, transpose, 0.5, c, MatrixStructure.SymmetricLower)

            assertAgreesWithReference(expected.values, c.values, "syr2k transA=$transpose")
        }
    }

    @Test
    fun `scal scales a negatively strided vector`() = withVendor { blas ->
        // BLAS returns without doing anything for a non-positive increment, so the sign must not be passed on.
        val size = 5
        val storage = values(size, 30)
        val original = storage.copyOf()
        val x = StridedVector(storage, size - 1, size, -1)

        blas.scal(2.0, x)

        assertAgreesWithReference(DoubleArray(size) { 2.0 * original[it] }, storage, "scal negative stride")
    }

    @Test
    fun `swap exchanges two interleaved vectors of one array`() = withVendor { blas ->
        // Two rows of a column-major matrix interleave in one array, which is what a pivot swap works on.
        val size = 4
        val lda = 3
        val storage = values(lda * size, 31)
        val original = storage.copyOf()
        val first = StridedVector(storage, 0, size, lda)
        val second = StridedVector(storage, 1, size, lda)

        blas.swap(first, second)

        for (index in 0 until size) {
            assertEquals(original[1 + index * lda], storage[index * lda], "row 0 entry $index")
            assertEquals(original[index * lda], storage[1 + index * lda], "row 1 entry $index")
            assertEquals(original[2 + index * lda], storage[2 + index * lda], "row 2 entry $index")
        }
    }

    @Test
    fun `a symmetric rank one update keeps the triangle the call never wrote`() = withVendor { blas ->
        val order = 3
        val a = DenseMatrix.wrap(order, order, values(order * order, 32))
        val x = vector(order, 33)
        val expected = a.copy()
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
