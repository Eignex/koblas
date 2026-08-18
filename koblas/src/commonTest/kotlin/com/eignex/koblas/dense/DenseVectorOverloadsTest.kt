package com.eignex.koblas.dense

import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.F64DenseVector
import com.eignex.koblas.F64SparseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import kotlin.test.Test
import kotlin.test.assertSame
import kotlin.test.assertTrue

class DenseVectorOverloadsTest {

    private val a = F64DenseMatrix.of(
        arrayOf(
            doubleArrayOf(4.0, 1.0, 0.0),
            doubleArrayOf(1.0, 3.0, 1.0),
            doubleArrayOf(0.0, 1.0, 2.0),
        ),
    )
    private val lower = F64DenseMatrix.of(
        arrayOf(
            doubleArrayOf(2.0, 0.0, 0.0),
            doubleArrayOf(1.0, 3.0, 0.0),
            doubleArrayOf(0.0, 1.0, 4.0),
        ),
    )
    private val raw = doubleArrayOf(1.0, 2.0, 3.0)

    @Test
    fun `the dense solves agree with their array siblings`() {
        val v = F64DenseVector.of(raw)
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
        val tall = F64DenseMatrix.of(
            arrayOf(doubleArrayOf(1.0, 0.0), doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 2.0)),
        )
        val v = F64DenseVector.of(raw)
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
        val v = F64DenseVector.of(raw)
        assertClose(koblas.gemv(a, raw), koblas.gemv(a, v).data, "gemv")
        assertClose(
            koblas.gemv(a, raw, transpose = true),
            koblas.gemv(a, v, transpose = true).data,
            "gemv transposed",
        )

        val expectedTrsv = raw.copyOf()
        koblas.trsv(lower, expectedTrsv, lower = true)
        val actualTrsv = F64DenseVector.of(raw)
        koblas.trsv(lower, actualTrsv, lower = true)
        assertClose(expectedTrsv, actualTrsv.data, "trsv")

        val expectedTrmv = raw.copyOf()
        koblas.trmv(lower, expectedTrmv, lower = true)
        val actualTrmv = F64DenseVector.of(raw)
        koblas.trmv(lower, actualTrmv, lower = true)
        assertClose(expectedTrmv, actualTrmv.data, "trmv")
    }

    @Test
    fun `the sparse routines agree with their array siblings`() {
        val s = F64SparseMatrix.ofTriplets(
            rows = 3,
            cols = 3,
            rowIdx = intArrayOf(0, 1, 1, 2),
            colIdx = intArrayOf(0, 0, 1, 2),
            values = doubleArrayOf(2.0, 1.0, 3.0, 4.0),
        )
        val v = F64DenseVector.of(raw)
        assertClose(koblas.gemv(s, raw), koblas.gemv(s, v).data, "sparse gemv")

        val expected = raw.copyOf()
        koblas.trsv(s, expected, lower = true)
        val actual = F64DenseVector.of(raw)
        koblas.trsv(s, actual, lower = true)
        assertClose(expected, actual.data, "sparse trsv")
    }

    @Test
    fun `the in-place forms write through to the vector rather than a copy`() {
        val x = F64DenseVector.of(raw)
        val before = x.data
        koblas.trsv(lower, x, lower = true)
        assertSame(before, x.data, "trsv must not replace the backing")
        assertTrue(!raw.contentEquals(x.data), "trsv should have solved in place")

        val y = F64DenseVector.zero(3)
        koblas.gemv(1.0, a, F64DenseVector.of(raw), 0.0, y)
        assertClose(koblas.gemv(a, raw), y.data, "gemv should have written into y")

        val out = F64DenseVector.zero(3)
        val returned = koblas.solveInto(a.lu(), F64DenseVector.of(raw), out)
        assertSame(out, returned, "solveInto returns the destination it was given")
        assertClose(koblas.solve(a.lu(), raw), out.data, "solveInto should have written into out")
    }
}
