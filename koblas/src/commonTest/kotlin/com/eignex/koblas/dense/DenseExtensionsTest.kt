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
    private val squareNext = F64DenseMatrix.of(
        arrayOf(
            doubleArrayOf(2.0, 1.0, 0.0),
            doubleArrayOf(1.0, 2.0, 1.0),
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
    private val tallNext = F64DenseMatrix.of(
        arrayOf(
            doubleArrayOf(2.0, 1.0),
            doubleArrayOf(1.0, 1.0),
            doubleArrayOf(0.0, 2.0),
        ),
    )
    private val b3 = doubleArrayOf(1.0, 2.0, 3.0)
    private val multiRhs = F64DenseMatrix.ofColumns(arrayOf(b3, doubleArrayOf(0.0, 1.0, 0.0)))

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
        assertClose(koblas.solve(lu, multiRhs).data, lu.solve(multiRhs).data, "lu blocked solve")
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
    fun `lu solveInto retains its destination and agrees with the seam`() {
        val lu = square.lu()
        val out = DoubleArray(lu.n)
        assertSame(out, lu.solveInto(b3, out), "vector destination")
        assertClose(koblas.solve(lu, b3), out, "vector solve")
        val blockOut = F64DenseMatrix(lu.n, multiRhs.cols)
        assertSame(blockOut, lu.solveInto(multiRhs, blockOut), "block destination")
        assertClose(koblas.solve(lu, multiRhs).data, blockOut.data, "block solve")
    }

    @Test
    fun `ldl solveInto retains its destination and agrees with the seam`() {
        val ldl = square.ldl()
        val out = DoubleArray(ldl.n)
        assertSame(out, ldl.solveInto(b3, out), "vector destination")
        assertClose(koblas.solve(ldl, b3), out, "vector solve")
        val blockOut = F64DenseMatrix(ldl.n, multiRhs.cols)
        assertSame(blockOut, ldl.solveInto(multiRhs, blockOut), "block destination")
        assertClose(koblas.solve(ldl, multiRhs).data, blockOut.data, "block solve")
    }

    @Test
    fun `cholesky solveInto retains its destination and agrees with the seam`() {
        val chol = square.cholesky()
        val out = DoubleArray(chol.n)
        assertSame(out, chol.solveInto(b3, out), "vector destination")
        assertClose(koblas.solve(chol, b3), out, "vector solve")
        val blockOut = F64DenseMatrix(chol.n, multiRhs.cols)
        assertSame(blockOut, chol.solveInto(multiRhs, blockOut), "block destination")
        assertClose(koblas.solve(chol, multiRhs).data, blockOut.data, "block solve")
    }

    @Test
    fun `qr solveInto and applyQInto retain their destinations and agree with the seam`() {
        val qr = tall.qr()
        val solveOut = DoubleArray(qr.n)
        assertSame(solveOut, qr.solveInto(b3, solveOut), "solve destination")
        assertClose(koblas.solve(qr, b3), solveOut, "solve")
        val applyOut = DoubleArray(qr.m)
        assertSame(applyOut, qr.applyQInto(b3, applyOut), "applyQ destination")
        assertClose(koblas.applyQ(qr, b3), applyOut, "applyQ")
    }

    @Test
    fun `pivoted qr solveInto retains its destination and agrees with the seam`() {
        val pivoted = tall.qrPivoted()
        val out = DoubleArray(pivoted.n)
        assertSame(out, pivoted.solveInto(b3, out), "destination")
        assertClose(koblas.solve(pivoted, b3), out, "solve")
    }

    @Test
    fun `an LU refactors through its factor owned reusable buffers`() {
        val factor = square.lu()
        val factors = factor.lu
        val pivots = factor.piv

        assertSame(factor, factor.refactorInto(squareNext), "refactor destination")
        assertSame(factors, factor.lu, "refactor factor buffer")
        assertSame(pivots, factor.piv, "refactor pivot buffer")
        assertClose(koblas.solve(koblas.factor(squareNext), b3), factor.solve(b3), "refactored solve")
    }

    @Test
    fun `a cholesky refactors through its factor owned reusable buffer`() {
        val factor = square.cholesky()
        val l = factor.l

        assertSame(factor, factor.refactorInto(squareNext), "refactor destination")
        assertSame(l, factor.l, "refactor factor buffer")
        assertClose(koblas.solve(koblas.cholesky(squareNext), b3), factor.solve(b3), "refactored solve")
    }

    @Test
    fun `an LDL refactors through its factor owned reusable buffers`() {
        val factor = square.ldl()
        val factors = factor.ldl
        val pivots = factor.ipiv

        assertSame(factor, factor.refactorInto(squareNext), "refactor destination")
        assertSame(factors, factor.ldl, "refactor factor buffer")
        assertSame(pivots, factor.ipiv, "refactor pivot buffer")
        assertClose(koblas.solve(koblas.ldl(squareNext), b3), factor.solve(b3), "refactored solve")
    }

    @Test
    fun `a QR refactors through its factor owned reusable buffers`() {
        val factor = tall.qr()
        val factors = factor.qr
        val taus = factor.tau

        assertSame(factor, factor.refactorInto(tallNext), "refactor destination")
        assertSame(factors, factor.qr, "refactor factor buffer")
        assertSame(taus, factor.tau, "refactor tau buffer")
        assertClose(koblas.solve(koblas.qr(tallNext), b3), factor.solve(b3), "refactored solve")
    }
}
