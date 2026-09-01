package com.eignex.koblas.dense

import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.core.F64DenseMatrix
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse

class FactorSnapshotsTest {

    @Test
    fun `LU factor snapshots reconstruct the pivoted matrix and stay independent`() {
        val a = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(0.0, 2.0, 1.0),
                doubleArrayOf(3.0, 1.0, 4.0),
                doubleArrayOf(2.0, 5.0, 6.0),
            ),
        )
        val factor = koblas.factor(a)

        val l = factor.lowerFactor()
        val u = factor.upperFactor()
        val rebuilt = koblas.gemm(l, u)
        val order = factor.rowOrder()

        for (row in 0 until factor.n) {
            for (column in 0 until factor.n) assertClose(a[order[row], column], rebuilt[row, column], "P A")
        }
        l[0, 0] = 99.0
        order[0] = 99
        assertEquals(1.0, factor.lowerFactor()[0, 0])
        assertFalse(factor.rowOrder().contains(99))
    }

    @Test
    fun `LU signed log determinant avoids product overflow`() {
        val huge = koblas.factor(F64DenseMatrix(200, 200).also { matrix ->
            for (i in 0 until 200) matrix[i, i] = -100.0
        })

        assertEquals(1, huge.determinantSign())
        assertClose(200.0 * kotlin.math.ln(100.0), huge.logAbsDeterminant(), "log absolute determinant")
        assertEquals(Double.POSITIVE_INFINITY, huge.determinant())

        val singular = koblas.factor(F64DenseMatrix(2, 2))
        assertEquals(0, singular.determinantSign())
        assertEquals(Double.NEGATIVE_INFINITY, singular.logAbsDeterminant())
    }

    @Test
    fun `Cholesky and LDL snapshots are independent`() {
        val a = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(6.0, 2.0, 1.0, 0.0),
                doubleArrayOf(2.0, 5.0, 2.0, 1.0),
                doubleArrayOf(1.0, 2.0, 5.0, 2.0),
                doubleArrayOf(0.0, 1.0, 2.0, 4.0),
            ),
        )
        val chol = a.cholesky()
        val lower = chol.lowerFactor()
        val rebuilt = F64DenseMatrix(4, 4)
        koblas.gemm(1.0, lower, false, lower, true, 0.0, rebuilt)
        assertClose(a, rebuilt, "L L transpose", tolerance = 1e-11)
        lower[0, 0] = 99.0
        assertFalse(chol.lowerFactor()[0, 0] == 99.0)

        val ldl = a.ldl()
        val packed = ldl.packedFactor()
        val pivots = ldl.pivotBlocks()
        packed[0, 0] = 99.0
        pivots[0] = 99
        assertFalse(ldl.packedFactor()[0, 0] == 99.0)
        assertFalse(ldl.pivotBlocks().contains(99))
    }

    @Test
    fun `explicit QR factors reconstruct and pivot inspection is independent`() {
        val a = randomMatrix(7, 4, Random(20261001))
        val qr = a.qr()
        assertClose(a, koblas.gemm(qr.explicitQ(), qr.explicitR()), "Q R", tolerance = 1e-11)

        val pivoted = a.qrPivoted()
        val order = pivoted.columnOrder()
        val expected = F64DenseMatrix(a.rows, a.cols)
        for (column in order.indices) for (row in 0 until a.rows) expected[row, column] = a[row, order[column]]
        assertClose(expected, koblas.gemm(pivoted.explicitQ(), pivoted.explicitR()), "Q R pivoted", tolerance = 1e-11)
        order[0] = 99
        assertFalse(pivoted.columnOrder().contains(99))
    }
}
