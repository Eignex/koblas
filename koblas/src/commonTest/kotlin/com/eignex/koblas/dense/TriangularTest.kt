package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import com.eignex.koblas.poisonedTriangle
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class TriangularTest {

    /** `op(explicit) x` by definition, so the solve checks do not depend on the library's own kernels. */
    private fun naiveMultiply(explicit: DenseMatrix, transpose: Boolean, x: DoubleArray): DoubleArray {
        val n = explicit.rows
        val y = DoubleArray(n)
        for (i in 0 until n) {
            for (j in 0 until n) y[i] += (if (transpose) explicit[j, i] else explicit[i, j]) * x[j]
        }
        return y
    }

    @Test
    fun `trsv solves all eight flag combinations and never reads the opposite triangle`() {
        val rng = Random(20260728)
        for (n in intArrayOf(1, 3, 8, 25)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(true, false)) {
                    for (unitDiag in booleanArrayOf(true, false)) {
                        val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                        val xTrue = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
                        val x = naiveMultiply(explicit, transpose, xTrue)
                        trsv(t, x, lower, transpose, unitDiag)
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
    fun `trsm matches per-column trsv across the flag combinations`() {
        val rng = Random(20260729)
        val n = 12
        val nrhs = 5
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(true, false)) {
                for (unitDiag in booleanArrayOf(true, false)) {
                    val (t, _) = poisonedTriangle(rng, n, lower, unitDiag)
                    val b = randomMatrix(n, nrhs, rng)
                    val viaTrsm = DenseMatrix(n, nrhs, b.data.copyOf())
                    trsm(t, viaTrsm, lower, transpose, unitDiag)
                    for (c in 0 until nrhs) {
                        val col = DoubleArray(n) { b[it, c] }
                        trsv(t, col, lower, transpose, unitDiag)
                        for (i in 0 until n) {
                            assertTrue(
                                abs(viaTrsm[i, c] - col[i]) < 1e-12,
                                "trsm lower=$lower t=$transpose unit=$unitDiag col $c row $i",
                            )
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `trsm solves the multi-RHS system against a gemm residual check`() {
        val rng = Random(20260730)
        val n = 10
        val nrhs = 3
        val (t, explicit) = poisonedTriangle(rng, n, lower = true, unitDiag = false)
        val xTrue = randomMatrix(n, nrhs, rng)
        val b = explicit.matMul(xTrue)
        trsm(t, b, lower = true)
        assertClose(xTrue, b, "trsm multi-RHS", tolerance = 1e-9)
    }

    @Test
    fun `trmv matches gemv on the explicit triangle for all flag combos`() {
        val rng = Random(20260915)
        for (n in intArrayOf(1, 2, 7, 16)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                        val x = randomVector(n, rng)
                        val expected = koblas.gemv(explicit, x, transpose)
                        val actual = x.copyOf()
                        trmv(t, actual, lower, transpose, unitDiag)
                        assertClose(expected, actual, "trmv n=$n lower=$lower t=$transpose unit=$unitDiag")
                    }
                }
            }
        }
    }

    @Test
    fun `trmm matches gemm on the explicit triangle for all flag combos`() {
        val rng = Random(20260916)
        for (n in intArrayOf(1, 5, 12)) {
            val p = 3
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                        val b = randomMatrix(n, p, rng)
                        val expected = DenseMatrix(n, p)
                        koblas.gemm(1.0, explicit, transpose, b, false, 0.0, expected)
                        val actual = DenseMatrix(n, p, b.data.copyOf())
                        trmm(t, actual, lower, transpose, unitDiag)
                        assertClose(expected, actual, "trmm n=$n lower=$lower t=$transpose unit=$unitDiag")
                    }
                }
            }
        }
    }

    @Test
    fun `trmm right multiplies from the right and matches gemm`() {
        val rng = Random(20260940)
        val n = 5
        val rows = 4
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(true, false)) {
                for (unitDiag in booleanArrayOf(true, false)) {
                    val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                    val b = randomMatrix(rows, n, rng)
                    val expected = DenseMatrix(rows, n)
                    koblas.gemm(1.0, b, false, explicit, transpose, 0.0, expected)
                    val actual = DenseMatrix.wrap(rows, n, b.data.copyOf())
                    trmm(t, actual, lower, transpose, unitDiag, right = true)
                    assertClose(expected, actual, "trmm right l=$lower t=$transpose u=$unitDiag", tolerance = 1e-11)
                }
            }
        }
    }

    @Test
    fun `trsm right inverts trmm right`() {
        val rng = Random(20260941)
        val n = 6
        val rows = 3
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(true, false)) {
                for (unitDiag in booleanArrayOf(true, false)) {
                    val (t, _) = poisonedTriangle(rng, n, lower, unitDiag)
                    val x = randomMatrix(rows, n, rng)
                    val b = DenseMatrix.wrap(rows, n, x.data.copyOf())
                    trmm(t, b, lower, transpose, unitDiag, right = true)
                    trsm(t, b, lower, transpose, unitDiag, right = true)
                    assertClose(x, b, "trsm right l=$lower t=$transpose u=$unitDiag", tolerance = 1e-11)
                }
            }
        }
    }

    @Test
    fun `trmv round-trips with trsv`() {
        val rng = Random(20260917)
        val n = 9
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                val (t, _) = poisonedTriangle(rng, n, lower, unitDiag = false)
                val x0 = randomVector(n, rng)
                val x = x0.copyOf()
                trmv(t, x, lower, transpose)
                trsv(t, x, lower, transpose)
                assertClose(x0, x, "roundtrip lower=$lower t=$transpose")
            }
        }
    }

    @Test
    fun `trsv and trsm validate shapes`() {
        assertFailsWith<IllegalArgumentException> { trsv(DenseMatrix(2, 3), DoubleArray(2), lower = true) }
        assertFailsWith<IllegalArgumentException> { trsv(DenseMatrix(3, 3), DoubleArray(2), lower = true) }
        assertFailsWith<IllegalArgumentException> { trsm(DenseMatrix(3, 3), DenseMatrix(2, 4), lower = true) }
    }
}
