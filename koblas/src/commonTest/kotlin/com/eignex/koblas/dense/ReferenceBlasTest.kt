package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The oracle's own arithmetic, pinned so it cannot drift under the tests that compare against it.
 *
 * Everything else in the dense suite asks whether a vendor agrees with [ReferenceBlas]; nothing else asks
 * whether [ReferenceBlas] is still what it claims. These cases fix the parts a plausible-looking edit could
 * change silently: which entries a triangle reads, and where a zero multiplier does and does not suppress a
 * product. They state the textbook definition, not any vendor's choices, which are the vendor's to make.
 */
class ReferenceBlasTest {
    @Test
    fun `gemv forms zero products from nonzero alpha`() {
        val a = DenseMatrix(2, 2, doubleArrayOf(Double.POSITIVE_INFINITY, 1.0, 2.0, 3.0))
        val y = DoubleArray(2)

        ReferenceBlas.gemv(1.0, a, doubleArrayOf(0.0, 1.0), 0.0, y)

        assertTrue(y[0].isNaN(), "zero times infinity was ${y[0]}")
        assertEquals(3.0, y[1])
    }

    @Test
    fun `gemm forms zero products in every transpose mode`() {
        val a = DenseMatrix.diagonal(2).also { it[0, 0] = Double.POSITIVE_INFINITY }
        val b = DenseMatrix(2, 2)
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val c = DenseMatrix(2, 2)

                ReferenceBlas.gemm(1.0, a, transposeA, b, transposeB, 0.0, c)

                assertTrue(c[0, 0].isNaN(), "transposeA=$transposeA transposeB=$transposeB produced ${c[0, 0]}")
            }
        }
    }

    @Test
    fun `trsv divides by a zero diagonal instead of reporting it`() {
        val singular = DenseMatrix.ofRows(arrayOf(doubleArrayOf(1.0, 0.0), doubleArrayOf(3.0, 0.0)))
        val x = doubleArrayOf(1.0, 1.0)

        ReferenceBlas.trsv(singular, x, lower = true)

        assertTrue(!x[1].isFinite(), "expected a non-finite entry from the zero pivot, got ${x[1]}")
    }

    @Test
    fun `transposed trsv divides a zero right hand side pivot`() {
        val singular = DenseMatrix.ofRows(arrayOf(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0, 1.0)))
        val x = DoubleArray(2)

        ReferenceBlas.trsv(singular, x, lower = true, transpose = true)

        assertTrue(x[0].isNaN())
    }

    @Test
    fun `transposed trsv forms products with zero triangle entries`() {
        val triangle = DenseMatrix.diagonal(2)
        val x = doubleArrayOf(0.0, Double.POSITIVE_INFINITY)

        ReferenceBlas.trsv(triangle, x, lower = true, transpose = true)

        assertTrue(x[0].isNaN())
    }

    @Test
    fun `right trsm skips zero triangle coefficients`() {
        val triangle = DenseMatrix.diagonal(2)
        val b = DenseMatrix(1, 2, doubleArrayOf(Double.POSITIVE_INFINITY, 1.0))

        ReferenceBlas.trsm(triangle, b, lower = true, right = true)

        assertEquals(Double.POSITIVE_INFINITY, b[0, 0])
        assertEquals(1.0, b[0, 1])
    }

    @Test
    fun `right trmm skips zero triangle coefficients`() {
        val triangle = DenseMatrix.diagonal(2)
        val b = DenseMatrix(1, 2, doubleArrayOf(Double.POSITIVE_INFINITY, 1.0))

        ReferenceBlas.trmm(triangle, b, lower = true, right = true)

        assertEquals(Double.POSITIVE_INFINITY, b[0, 0])
        assertEquals(1.0, b[0, 1])
    }
}
