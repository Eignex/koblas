package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** QR at the ends of the double range, where the result must still be the unscaled one times the scale. */
class LinearAlgebraQrScalingTest {

    private val scales = doubleArrayOf(1e-200, 1e-170, 1e-8, 1.0, 1e8, 1e170, 1e200)

    private val base = arrayOf(
        doubleArrayOf(3.0, 4.0, 1.0),
        doubleArrayOf(1.0, 2.0, 7.0),
        doubleArrayOf(5.0, 1.0, 2.0),
        doubleArrayOf(2.0, 8.0, 3.0),
    )

    private fun scaled(scale: Double) = DenseMatrix.of(
        Array(base.size) { i -> DoubleArray(3) { j -> base[i][j] * scale } },
    )

    private fun assertReconstructs(a: DenseMatrix, f: QrDecomposition, scale: Double) {
        for (j in 0 until a.cols) {
            val rebuilt = koblas.applyQ(f, rColumn(f, j))
            for (i in 0 until a.rows) {
                val expected = a[i, j]
                assertTrue(
                    abs(rebuilt[i] - expected) <= 1e-12 * scale,
                    "scale $scale entry ($i,$j): got ${rebuilt[i]}, want $expected",
                )
            }
        }
    }

    @Test
    fun `Q times R reconstructs the matrix at every scale`() {
        for (scale in scales) {
            val a = scaled(scale)
            val f = koblas.qr(a)
            for (t in f.tau) assertTrue(t.isFinite(), "scale $scale: tau $t")
            assertTrue(f.qr.all { it.isFinite() }, "scale $scale: non-finite R")
            assertReconstructs(a, f, scale)
        }
    }

    @Test
    fun `the pivoted factorization keeps its rank and reconstructs the permuted matrix at every scale`() {
        for (scale in scales) {
            val a = scaled(scale)
            val f = koblas.qrPivoted(a)
            assertEquals(3, f.rank, "scale $scale")
            val permuted = DenseMatrix.of(Array(a.rows) { i -> DoubleArray(a.cols) { j -> a[i, f.pivots[j]] } })
            assertReconstructs(permuted, f.factorization, scale)
        }
    }

    @Test
    fun `a rank deficient matrix is detected at every scale`() {
        for (scale in scales) {
            val a = DenseMatrix.of(
                Array(base.size) { i ->
                    doubleArrayOf(base[i][0] * scale, base[i][1] * scale, 2.0 * base[i][0] * scale)
                },
            )
            assertEquals(2, koblas.qrPivoted(a).rank, "scale $scale")
        }
    }

    @Test
    fun `least squares is invariant to scaling both sides`() {
        val rhs = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val reference = koblas.solveLeastSquares(koblas.qr(scaled(1.0)), rhs.copyOf())
        for (scale in scales) {
            val b = DoubleArray(rhs.size) { rhs[it] * scale }
            val x = koblas.solveLeastSquares(koblas.qr(scaled(scale)), b)
            assertClose(reference, x, "scale $scale", tolerance = 1e-10)
        }
    }

    @Test
    fun `a column that is exactly zero still yields no reflector`() {
        val a = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(0.0, 1.0),
                doubleArrayOf(0.0, 2.0),
            ),
        )
        val f = koblas.qr(a)
        assertEquals(0.0, f.tau[0])
    }
}
