package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.DenseMatrix
import kotlin.random.Random
import kotlin.test.*

class TriangularTest {

    @Test
    fun `trsv solves all eight flag combinations and never reads the opposite triangle`() = withDenseBlas { blas ->
        val rng = Random(20260728)
        for (n in intArrayOf(1, 3, 8, 25)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(true, false)) {
                    for (unitDiag in booleanArrayOf(true, false)) {
                        val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
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
        for (n in intArrayOf(1, 2, 7, 16)) {
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
        assertFailsWith<DimensionMismatch> { DenseMatrix(2, 3).trsv(DoubleArray(2), lower = true) }
        assertFailsWith<DimensionMismatch> { DenseMatrix(3, 3).trsv(DoubleArray(2), lower = true) }
        assertFailsWith<DimensionMismatch> { DenseMatrix(3, 3).trsm(DenseMatrix(2, 4), lower = true) }
    }
}
