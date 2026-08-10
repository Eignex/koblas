package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import com.eignex.koblas.norm1
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The fluent spellings of the dense factorizations. Each one has to agree with the member it forwards to,
 * which is what makes it a spelling rather than a second implementation.
 */
class DenseExtensionsTest {

    private val square = DenseMatrix.of(
        arrayOf(
            doubleArrayOf(4.0, 1.0, 0.0),
            doubleArrayOf(1.0, 3.0, 1.0),
            doubleArrayOf(0.0, 1.0, 2.0),
        ),
    )
    private val tall = DenseMatrix.of(
        arrayOf(
            doubleArrayOf(1.0, 0.0),
            doubleArrayOf(1.0, 1.0),
            doubleArrayOf(1.0, 2.0),
        ),
    )
    private val b3 = doubleArrayOf(1.0, 2.0, 3.0)

    @Test
    fun `every factorize extension agrees with the member`() {
        assertEquals(koblas.factor(square).n, square.lu().n)
        assertEquals(koblas.ldl(square).n, square.ldl().n)
        assertEquals(koblas.qr(tall).n, tall.qr().n)
        assertEquals(koblas.qrPivoted(tall).rank, tall.qrPivoted().rank)
    }

    @Test
    fun `qrPivoted passes the tolerance through rather than defaulting it`() {
        // A tolerance loose enough to call the second column dependent must lower the reported rank; if
        // the extension dropped the argument, both calls would agree.
        val automatic = tall.qrPivoted().rank
        val loose = tall.qrPivoted(tolerance = 0.9).rank
        assertEquals(2, automatic)
        assertTrue(loose < automatic, "a loose tolerance should reduce the rank, got $loose")
    }

    @Test
    fun `the LU family solves and inverts and estimates through the extensions`() {
        val lu = square.lu()
        assertClose(koblas.solve(lu, b3), lu.solve(b3), "lu solve")
        assertClose(koblas.invert(lu).data, lu.invert().data, "lu invert")
        val anorm = norm1(square)
        assertEquals(koblas.rcond(lu, anorm), lu.rcond(anorm), "lu rcond")
        // The transposed direction and the blocked form reach the same routines.
        assertClose(koblas.solve(lu, b3, transpose = true), lu.solve(b3, transpose = true), "lu solve transposed")
        val rhs = DenseMatrix.ofColumns(arrayOf(b3, doubleArrayOf(0.0, 1.0, 0.0)))
        assertClose(koblas.solve(lu, rhs).data, lu.solve(rhs).data, "lu blocked solve")
    }

    @Test
    fun `the LDL and QR families solve through the extensions`() {
        val ldl = square.ldl()
        assertClose(koblas.solve(ldl, b3), ldl.solve(b3), "ldl solve")

        val qr = tall.qr()
        assertClose(koblas.solveLeastSquares(qr, b3), qr.solveLeastSquares(b3), "qr least squares")
        assertClose(koblas.applyQ(qr, b3), qr.applyQ(b3), "applyQ")
        assertClose(
            koblas.applyQ(qr, b3, transpose = true),
            qr.applyQ(b3, transpose = true),
            "applyQ transposed",
        )
        assertClose(
            koblas.solveLeastSquares(tall.qrPivoted(), b3),
            tall.qrPivoted().solveLeastSquares(b3),
            "pivoted least squares",
        )
    }

    @Test
    fun `the LU solve actually solves the system`() {
        // The extensions are thin, but thin is not the same as correct: check one residual end to end.
        val x = square.lu().solve(b3)
        for (i in 0 until 3) {
            var s = 0.0
            for (j in 0 until 3) s += square[i, j] * x[j]
            assertEquals(b3[i], s, 1e-12, "residual at $i")
        }
    }
}
