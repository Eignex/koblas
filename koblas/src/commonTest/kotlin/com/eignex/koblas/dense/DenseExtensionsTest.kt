package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import kotlin.test.*

class DenseExtensionsTest {

    private val square = F64DenseMatrix.of(
        arrayOf(
            doubleArrayOf(4.0, 1.0, 0.0),
            doubleArrayOf(1.0, 3.0, 1.0),
            doubleArrayOf(0.0, 1.0, 2.0),
        ),
    )
    private val tall = F64DenseMatrix.of(
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
        assertEquals(
            koblas.pivotedSymmetricIndefinite(square).n,
            square.pivotedSymmetricIndefinite().n,
        )
        assertEquals(koblas.qr(tall).n, tall.qr().n)
        assertEquals(koblas.qrPivoted(tall).rank, tall.qrPivoted().rank)
    }

    @Test
    fun `qrPivoted passes the tolerance through rather than defaulting it`() {
        // The loose tolerance is what lowers the rank, so an extension dropping the argument would make both agree.
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
        val anorm = square.norm1()
        assertEquals(koblas.rcond(lu, anorm), lu.rcond(anorm), "lu rcond")
        assertClose(koblas.solve(lu, b3, transpose = true), lu.solve(b3, transpose = true), "lu solve transposed")
        val rhs = F64DenseMatrix.ofColumns(arrayOf(b3, doubleArrayOf(0.0, 1.0, 0.0)))
        assertClose(koblas.solve(lu, rhs).data, lu.solve(rhs).data, "lu blocked solve")
    }

    @Test
    fun `the pivoted symmetric indefinite and QR families solve through the extensions`() {
        val factor = square.pivotedSymmetricIndefinite()
        assertClose(koblas.solve(factor, b3), factor.solve(b3), "pivoted symmetric indefinite solve")

        @Suppress("DEPRECATION")
        run {
            val deprecatedFactor = square.pivotedSymmetricIndefinite()
            assertEquals(factor.n, deprecatedFactor.n, "the LDL alias stays compatible")
            assertClose(koblas.solve(factor, b3), deprecatedFactor.solve(b3), "the LDL alias solves the same way")
        }

        val qr = tall.qr()
        assertClose(koblas.solve(qr, b3), qr.solve(b3), "qr least squares")
        assertClose(koblas.applyQ(qr, b3), qr.applyQ(b3), "applyQ")
        assertClose(
            koblas.applyQ(qr, b3, transpose = true),
            qr.applyQ(b3, transpose = true),
            "applyQ transposed",
        )
        assertClose(
            koblas.solve(tall.qrPivoted(), b3),
            tall.qrPivoted().solve(b3),
            "pivoted least squares",
        )
    }

    @Test
    fun `factor owned Into operations retain their destinations and agree with the seam`() {
        val lu = square.lu()
        val luOut = DoubleArray(lu.n)
        assertSame(luOut, lu.solveInto(b3, luOut), "lu vector destination")
        assertClose(koblas.solve(lu, b3), luOut, "lu vector solve")
        val block = F64DenseMatrix.ofColumns(arrayOf(b3, doubleArrayOf(0.0, 1.0, 0.0)))
        val luBlockOut = F64DenseMatrix(lu.n, block.cols)
        assertSame(luBlockOut, lu.solveInto(block, luBlockOut), "lu block destination")
        assertClose(koblas.solve(lu, block).data, luBlockOut.data, "lu block solve")

        val ldl = square.ldl()
        val ldlOut = DoubleArray(ldl.n)
        assertSame(ldlOut, ldl.solveInto(b3, ldlOut), "ldl vector destination")
        assertClose(koblas.solve(ldl, b3), ldlOut, "ldl vector solve")
        val ldlBlockOut = F64DenseMatrix(ldl.n, block.cols)
        assertSame(ldlBlockOut, ldl.solveInto(block, ldlBlockOut), "ldl block destination")
        assertClose(koblas.solve(ldl, block).data, ldlBlockOut.data, "ldl block solve")

        val chol = square.cholesky()
        val cholOut = DoubleArray(chol.n)
        assertSame(cholOut, chol.solveInto(b3, cholOut), "cholesky vector destination")
        assertClose(koblas.solve(chol, b3), cholOut, "cholesky vector solve")
        val cholBlockOut = F64DenseMatrix(chol.n, block.cols)
        assertSame(cholBlockOut, chol.solveInto(block, cholBlockOut), "cholesky block destination")
        assertClose(koblas.solve(chol, block).data, cholBlockOut.data, "cholesky block solve")

        val qr = tall.qr()
        val qrOut = DoubleArray(qr.n)
        assertSame(qrOut, qr.solveInto(b3, qrOut), "qr destination")
        assertClose(koblas.solve(qr, b3), qrOut, "qr solve")
        val qOut = DoubleArray(qr.m)
        assertSame(qOut, qr.applyQInto(b3, qOut), "applyQ destination")
        assertClose(koblas.applyQ(qr, b3), qOut, "applyQ")

        val pivoted = tall.qrPivoted()
        val pivotedOut = DoubleArray(pivoted.n)
        assertSame(pivotedOut, pivoted.solveInto(b3, pivotedOut), "pivoted qr destination")
        assertClose(koblas.solve(pivoted, b3), pivotedOut, "pivoted qr solve")
    }

    @Test
    fun `an LU refactors through its factor owned reusable buffers`() {
        val factor = square.lu()
        val factors = factor.lu
        val pivots = factor.piv
        val next = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(2.0, 1.0, 0.0),
                doubleArrayOf(1.0, 2.0, 1.0),
                doubleArrayOf(0.0, 1.0, 2.0),
            ),
        )

        assertSame(factor, factor.refactorInto(next), "refactor destination")
        assertSame(factors, factor.lu, "refactor factor buffer")
        assertSame(pivots, factor.piv, "refactor pivot buffer")
        assertClose(koblas.solve(koblas.factor(next), b3), factor.solve(b3), "refactored solve")
    }
}
