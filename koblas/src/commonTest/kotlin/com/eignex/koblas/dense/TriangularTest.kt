package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.DenseMatrix
import kotlin.random.Random
import kotlin.test.*

class TriangularTest {

    @Test
    fun `trsv solves all eight flag combinations and never reads the opposite triangle`() = withDenseBlas { blas ->
        val rng = Random(20260728)
        for (n in intArrayOf(1, 3, 8, 25, 65, 130)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(true, false)) {
                    for (unitDiag in booleanArrayOf(true, false)) {
                        val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                        dominateStrictPart(t, explicit, n, lower)
                        val xTrue = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
                        val x = DoubleArray(n)
                        ReferenceBlas.gemv(1.0, explicit, xTrue, 0.0, x, transpose)

                        blas.trsv(t, x, lower, transpose, unitDiag)

                        assertClose(
                            xTrue,
                            x,
                            "trsv n=$n lower=$lower t=$transpose unit=$unitDiag",
                            tolerance = 1e-9,
                        )
                    }
                }
            }
        }
    }

    @Test
    fun `trmv matches gemv on the explicit triangle for all flag combos`() = withDenseBlas { blas ->
        val rng = Random(20260915)
        for (n in intArrayOf(1, 2, 7, 16, 65, 130)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                        val x = randomVector(n, rng)
                        val expected = DoubleArray(n)
                        ReferenceBlas.gemv(1.0, explicit, x, 0.0, expected, transpose)
                        val actual = x.copyOf()

                        blas.trmv(t, actual, lower, transpose, unitDiag)

                        assertClose(expected, actual, "trmv n=$n lower=$lower t=$transpose unit=$unitDiag")
                    }
                }
            }
        }
    }

    @Test
    fun `trsv and trsm validate shapes`() {
        assertFailsWith<DimensionMismatch> { koblas.trsv(DenseMatrix(2, 3), DoubleArray(2), lower = true) }
        assertFailsWith<DimensionMismatch> { koblas.trsv(DenseMatrix(3, 3), DoubleArray(2), lower = true) }
        assertFailsWith<DimensionMismatch> { koblas.trsm(DenseMatrix(3, 3), DenseMatrix(2, 4), lower = true) }
    }

    /**
     * Shrinks the strict triangle by the order so the diagonal dominates. A random unit triangle is
     * exponentially ill-conditioned in its order, so a solve past a few dozen steps would test the matrix
     * rather than the routine.
     */
    private fun dominateStrictPart(t: DenseMatrix, explicit: DenseMatrix, n: Int, lower: Boolean) {
        for (j in 0 until n) {
            for (i in 0 until n) {
                if (if (lower) i > j else i < j) {
                    t[i, j] /= n
                    explicit[i, j] /= n
                }
            }
        }
    }
}
