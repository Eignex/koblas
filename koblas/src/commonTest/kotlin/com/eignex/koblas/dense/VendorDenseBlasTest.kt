package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.StridedVector
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * What Koblas hands the library, rather than what comes back. A flag mapped the wrong way is invisible to an
 * agreement test whenever the fixture is symmetric, the operand square, or the operation its own inverse.
 */
class VendorDenseBlasTest {
    private fun recorded(): Pair<RecordingBlas, DenseBlas> {
        val recorder = RecordingBlas()
        return recorder to VendorDenseBlas(recorder)
    }

    private fun matrix(rows: Int, cols: Int = rows) = DenseMatrix.zero(rows, cols)

    @Test
    fun `the transpose flag reaches the call unchanged`() {
        for (transpose in booleanArrayOf(false, true)) {
            val (recorder, blas) = recorded()
            val a = matrix(3, 4)

            blas.gemv(
                1.0,
                a,
                DoubleArray(if (transpose) 3 else 4),
                0.0,
                DoubleArray(if (transpose) 4 else 3),
                transpose,
            )

            assertEquals(transpose, recorder.single().transposeA, "gemv transpose=$transpose")
        }
    }

    @Test
    fun `each gemm operand carries its own transpose flag`() {
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val (recorder, blas) = recorded()
                val a = if (transposeA) matrix(4, 3) else matrix(3, 4)
                val b = if (transposeB) matrix(5, 4) else matrix(4, 5)
                val c = matrix(3, 5)

                blas.gemm(1.0, a, transposeA, b, transposeB, 0.0, c)

                val call = recorder.single()
                assertEquals(transposeA, call.transposeA, "gemm transA")
                assertEquals(transposeB, call.transposeB, "gemm transB")
                // Order matters as much as the flags: a swap would still typecheck and still produce a product.
                assertSame(a, call.matrices[0], "A was not the first operand")
                assertSame(b, call.matrices[1], "B was not the second operand")
                assertSame(c, call.matrices[2], "C was not the destination")
            }
        }
    }

    @Test
    fun `a symmetric operand declares the triangle it stores`() {
        for (lower in booleanArrayOf(true, false)) {
            val expected = if (lower) MatrixStructure.SymmetricLower else MatrixStructure.SymmetricUpper
            val (recorder, blas) = recorded()
            val a = matrix(3)

            blas.symv(1.0, a, DoubleArray(3), 0.0, DoubleArray(3), lower)
            blas.symm(1.0, a, matrix(3, 2), 0.0, matrix(3, 2), lower)
            blas.syr(1.0, DenseVector.zero(3), a, lower)
            blas.syr2(1.0, DenseVector.zero(3), DenseVector.zero(3), a, lower)
            blas.syrk(1.0, matrix(3, 2), false, 0.0, matrix(3), lower)
            blas.syr2k(1.0, matrix(3, 2), matrix(3, 2), false, 0.0, matrix(3), lower)
            blas.gemmt(1.0, matrix(3, 2), false, matrix(2, 3), false, 0.0, matrix(3), lower)

            for (call in recorder.calls) {
                assertEquals(expected, call.structure, "${call.operation} lower=$lower")
            }
        }
    }

    @Test
    fun `a triangular operand declares its triangle and whether its diagonal is implied`() {
        for (lower in booleanArrayOf(true, false)) {
            for (unitDiag in booleanArrayOf(false, true)) {
                val expected = when {
                    unitDiag && lower -> MatrixStructure.UnitLower
                    unitDiag -> MatrixStructure.UnitUpper
                    lower -> MatrixStructure.TriangularLower
                    else -> MatrixStructure.TriangularUpper
                }
                val (recorder, blas) = recorded()
                val a = matrix(3)

                blas.trsv(a, DoubleArray(3), lower, false, unitDiag)
                blas.trmv(a, DoubleArray(3), lower, false, unitDiag)
                blas.trsm(a, matrix(3, 2), lower, false, unitDiag)
                blas.trmm(a, matrix(3, 2), lower, false, unitDiag)

                for (call in recorder.calls) {
                    assertEquals(expected, call.structure, "${call.operation} lower=$lower unit=$unitDiag")
                }
            }
        }
    }

    @Test
    fun `the side of a two sided call reaches the library`() {
        for (right in booleanArrayOf(false, true)) {
            val (recorder, blas) = recorded()
            val order = 3
            fun wide() = if (right) matrix(2, order) else matrix(order, 2)

            blas.trsm(matrix(order), wide(), lower = true, right = right)
            blas.trmm(matrix(order), wide(), lower = true, right = right)
            blas.symm(1.0, matrix(order), wide(), 0.0, wide(), lower = true, right = right)

            for (call in recorder.calls) assertEquals(right, call.rightSide, "${call.operation} right=$right")
        }
    }

    @Test
    fun `alpha and beta reach the library in that order`() {
        val (recorder, blas) = recorded()

        blas.gemm(0.75, matrix(3, 4), false, matrix(4, 5), false, -0.25, matrix(3, 5))

        assertEquals(listOf(0.75, -0.25), recorder.single().scalars)
    }

    @Test
    fun `a strided operand keeps its own spacing rather than being flattened`() {
        val (recorder, blas) = recorded()
        val storage = DoubleArray(12)
        val x = StridedVector(storage, 1, 4, 3)

        blas.syr(1.0, x, matrix(4), lower = true)

        val handed = recorder.single().vectors.single()
        assertSame(storage, handed.values, "the backing array was copied instead of passed")
        assertEquals(1, handed.offset, "the origin was lost")
        assertEquals(3, handed.stride, "the step was lost, so the call would read the wrong entries")
        assertEquals(4, handed.size)
    }

    @Test
    fun `an array operand is handed over without being copied`() {
        val (recorder, blas) = recorded()
        val x = DoubleArray(4)
        val y = DoubleArray(3)

        blas.gemv(1.0, matrix(3, 4), x, 0.0, y, transpose = false)

        val call = recorder.single()
        assertSame(x, call.vectors[0].values, "x was copied on the way to the library")
        assertSame(y, call.vectors[1].values, "y was copied, so the result would land in the copy")
        assertTrue(call.vectors.all { it.offset == 0 && it.stride == 1 }, "an array is adjacent from its start")
    }

    /** The allocating overloads choose a result shape and supply the scalars, neither of which a caller sees. */
    @Test
    fun `the allocating overloads supply a unit alpha and a zero beta`() {
        for (transpose in booleanArrayOf(false, true)) {
            val (recorder, blas) = recorded()

            val y = blas.gemv(matrix(3, 4), DoubleArray(if (transpose) 3 else 4), transpose)

            assertEquals(if (transpose) 4 else 3, y.size, "gemv transpose=$transpose sized its result wrongly")
            assertEquals(listOf(1.0, 0.0), recorder.single().scalars, "gemv transpose=$transpose")
        }

        val (recorder, blas) = recorded()

        val c = blas.gemm(matrix(3, 4), matrix(4, 5))

        assertEquals(3, c.rows)
        assertEquals(5, c.cols)
        assertEquals(listOf(1.0, 0.0), recorder.single().scalars, "gemm")
    }

    @Test
    fun `a call a host cannot serve raises rather than reaching a substitute`() {
        val blas = VendorDenseBlas(null)

        assertFailsWithMissingVendor { blas.gemm(1.0, matrix(2), false, matrix(2), false, 0.0, matrix(2)) }
        assertFailsWithMissingVendor { blas.symv(1.0, matrix(2), DoubleArray(2), 0.0, DoubleArray(2), true) }
        assertFailsWithMissingVendor { blas.trsm(matrix(2), matrix(2, 1), lower = true) }
    }

    // A vendorless host must answer a shape error with the shape error rather than with the missing library.
    @Test
    fun `a shape is rejected as a shape even where no library is installed`() {
        val blas = VendorDenseBlas(null)

        assertFailsWith<DimensionMismatch> { blas.gemv(1.0, matrix(3, 4), DoubleArray(2), 0.0, DoubleArray(3)) }
        assertFailsWith<DimensionMismatch> { blas.gemm(1.0, matrix(3, 4), false, matrix(3, 3), false, 0.0, matrix(3)) }
        assertFailsWith<DimensionMismatch> { blas.symv(1.0, matrix(3, 4), DoubleArray(4), 0.0, DoubleArray(3), true) }
        assertFailsWith<DimensionMismatch> { blas.trsv(matrix(3, 4), DoubleArray(3), lower = true) }
        assertFailsWith<DimensionMismatch> { blas.trsm(matrix(3), matrix(2, 2), lower = true) }
        assertFailsWith<DimensionMismatch> { blas.syrk(1.0, matrix(3, 2), false, 0.0, matrix(2), lower = true) }
        assertFailsWith<DimensionMismatch> { blas.ger(1.0, DoubleArray(2), DoubleArray(4), matrix(3, 4)) }
    }

    private fun assertFailsWithMissingVendor(body: () -> Unit) {
        val failed = try {
            body()
            false
        } catch (_: com.eignex.koblas.vendor.MissingVendorException) {
            true
        }
        assertTrue(failed, "a vendorless seam answered instead of raising")
    }
}
