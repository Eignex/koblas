package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * The [DenseVector] spellings of the seam. Each has to agree with its `DoubleArray` sibling, and the
 * in-place forms must write through to the vector rather than to a copy — which is the whole point of
 * forwarding `DenseVector.data` instead of materialising it.
 */
class DenseVectorOverloadsTest {

    private val a = DenseMatrix.of(
        arrayOf(
            doubleArrayOf(4.0, 1.0, 0.0),
            doubleArrayOf(1.0, 3.0, 1.0),
            doubleArrayOf(0.0, 1.0, 2.0),
        ),
    )
    private val lower = DenseMatrix.of(
        arrayOf(
            doubleArrayOf(2.0, 0.0, 0.0),
            doubleArrayOf(1.0, 3.0, 0.0),
            doubleArrayOf(0.0, 1.0, 4.0),
        ),
    )
    private val raw = doubleArrayOf(1.0, 2.0, 3.0)

    @Test
    fun `the dense solves agree with their array siblings`() {
        val v = DenseVector.of(raw)
        assertClose(koblas.solve(a.lu(), raw), koblas.solve(a.lu(), v).data, "lu solve")
        assertClose(koblas.solve(a.ldl(), raw), koblas.solve(a.ldl(), v).data, "ldl solve")
        assertClose(koblas.solve(a.cholesky(), raw), koblas.solve(a.cholesky(), v).data, "cholesky solve")
        assertClose(
            koblas.solve(a.lu(), raw, transpose = true),
            koblas.solve(a.lu(), v, transpose = true).data,
            "lu solve transposed",
        )
    }

    @Test
    fun `the QR family agrees with its array siblings`() {
        val tall = DenseMatrix.of(
            arrayOf(doubleArrayOf(1.0, 0.0), doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 2.0)),
        )
        val v = DenseVector.of(raw)
        assertClose(
            koblas.solveLeastSquares(tall.qr(), raw),
            koblas.solveLeastSquares(tall.qr(), v).data,
            "qr least squares",
        )
        assertClose(
            koblas.solveLeastSquares(tall.qrPivoted(), raw),
            koblas.solveLeastSquares(tall.qrPivoted(), v).data,
            "pivoted least squares",
        )
        assertClose(koblas.applyQ(tall.qr(), raw), koblas.applyQ(tall.qr(), v).data, "applyQ")
    }

    @Test
    fun `the BLAS routines agree with their array siblings`() {
        val v = DenseVector.of(raw)
        assertClose(koblas.gemv(a, raw), koblas.gemv(a, v).data, "gemv")
        assertClose(
            koblas.gemv(a, raw, transpose = true),
            koblas.gemv(a, v, transpose = true).data,
            "gemv transposed",
        )

        val expectedTrsv = raw.copyOf()
        koblas.trsv(lower, expectedTrsv, lower = true)
        val actualTrsv = DenseVector.of(raw)
        koblas.trsv(lower, actualTrsv, lower = true)
        assertClose(expectedTrsv, actualTrsv.data, "trsv")

        val expectedTrmv = raw.copyOf()
        koblas.trmv(lower, expectedTrmv, lower = true)
        val actualTrmv = DenseVector.of(raw)
        koblas.trmv(lower, actualTrmv, lower = true)
        assertClose(expectedTrmv, actualTrmv.data, "trmv")
    }

    @Test
    fun `the sparse routines agree with their array siblings`() {
        val s = SparseMatrix.ofTriplets(
            rows = 3,
            cols = 3,
            rowIdx = intArrayOf(0, 1, 1, 2),
            colIdx = intArrayOf(0, 0, 1, 2),
            values = doubleArrayOf(2.0, 1.0, 3.0, 4.0),
        )
        val v = DenseVector.of(raw)
        assertClose(koblas.gemv(s, raw), koblas.gemv(s, v).data, "sparse gemv")

        val expected = raw.copyOf()
        koblas.trsv(s, expected, lower = true)
        val actual = DenseVector.of(raw)
        koblas.trsv(s, actual, lower = true)
        assertClose(expected, actual.data, "sparse trsv")
    }

    @Test
    fun `the in-place forms write through to the vector rather than a copy`() {
        // Forwarding `.data` is what makes this zero-copy; materialising would leave the operand untouched.
        val x = DenseVector.of(raw)
        val before = x.data
        koblas.trsv(lower, x, lower = true)
        assertSame(before, x.data, "trsv must not replace the backing")
        assertTrue(!raw.contentEquals(x.data), "trsv should have solved in place")

        val y = DenseVector.zero(3)
        koblas.gemv(1.0, a, DenseVector.of(raw), 0.0, y)
        assertClose(koblas.gemv(a, raw), y.data, "gemv should have written into y")

        val out = DenseVector.zero(3)
        val returned = koblas.solveInto(a.lu(), DenseVector.of(raw), out)
        assertSame(out, returned, "solveInto returns the destination it was given")
        assertClose(koblas.solve(a.lu(), raw), out.data, "solveInto should have written into out")
    }
}
