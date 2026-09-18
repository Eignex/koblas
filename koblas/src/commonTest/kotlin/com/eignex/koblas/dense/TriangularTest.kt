package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.DenseMatrix
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.*

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

    /** The product the right-hand sides of a solve are formed from, by the oracle rather than by the seam. */
    private fun product(a: DenseMatrix, b: DenseMatrix): DenseMatrix =
        DenseMatrix(a.rows, b.cols).also { ReferenceBlas.gemm(1.0, a, false, b, false, 0.0, it) }

    @Test
    fun `trsv solves all eight flag combinations and never reads the opposite triangle`() = withDenseBlas { blas ->
        val rng = Random(20260728)
        for (n in intArrayOf(1, 3, 8, 25)) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(true, false)) {
                    for (unitDiag in booleanArrayOf(true, false)) {
                        val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                        val xTrue = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
                        val x = naiveMultiply(explicit, transpose, xTrue)
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
    fun `trsm matches per-column trsv across the flag combinations`() = withDenseBlas { blas ->
        val rng = Random(20260729)
        val n = 12
        val nrhs = 5
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(true, false)) {
                for (unitDiag in booleanArrayOf(true, false)) {
                    val (t, _) = poisonedTriangle(rng, n, lower, unitDiag)
                    val b = randomMatrix(n, nrhs, rng)
                    val viaTrsm = DenseMatrix(n, nrhs, b.values.copyOf())
                    blas.trsm(t, viaTrsm, lower, transpose, unitDiag)
                    for (c in 0 until nrhs) {
                        val col = DoubleArray(n) { b[it, c] }
                        blas.trsv(t, col, lower, transpose, unitDiag)
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
    fun `trsm solves the multi-RHS system against a gemm residual check`() = withDenseBlas { blas ->
        val rng = Random(20260730)
        val n = 10
        val nrhs = 3
        val (t, explicit) = poisonedTriangle(rng, n, lower = true, unitDiag = false)
        val xTrue = randomMatrix(n, nrhs, rng)
        val b = product(explicit, xTrue)

        blas.trsm(t, b, lower = true)

        assertClose(xTrue, b, "trsm multi-RHS", tolerance = 1e-9)
    }

    @Test
    fun `trsm applies alpha to the solved matrix`() = withDenseBlas { blas ->
        val rng = Random(20261021)
        val alpha = -0.75
        val n = 10
        val nrhs = 3
        val (t, explicit) = poisonedTriangle(rng, n, lower = true, unitDiag = false)
        val x = randomMatrix(n, nrhs, rng)
        val b = product(explicit, x)

        blas.trsm(t, b, lower = true, alpha = alpha)

        val expected = DenseMatrix(n, nrhs, DoubleArray(n * nrhs) { alpha * x.values[it] })
        assertClose(expected, b, "trsm alpha", tolerance = 1e-9)
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
    fun `trmm matches gemm on the explicit triangle for all flag combos`() = withDenseBlas { blas ->
        val rng = Random(20260916)
        val alpha = -0.75
        for (n in intArrayOf(1, 5, 12)) {
            val p = 3
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                        val b = randomMatrix(n, p, rng)
                        val expected = DenseMatrix(n, p)
                        ReferenceBlas.gemm(alpha, explicit, transpose, b, false, 0.0, expected)
                        val actual = DenseMatrix(n, p, b.values.copyOf())

                        blas.trmm(t, actual, lower, transpose, unitDiag, alpha = alpha)

                        assertClose(expected, actual, "trmm n=$n lower=$lower t=$transpose unit=$unitDiag")
                    }
                }
            }
        }
    }

    @Test
    fun `trmm right multiplies from the right and matches gemm`() = withDenseBlas { blas ->
        val rng = Random(20260940)
        val n = 5
        val rows = 4
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(true, false)) {
                for (unitDiag in booleanArrayOf(true, false)) {
                    val (t, explicit) = poisonedTriangle(rng, n, lower, unitDiag)
                    val b = randomMatrix(rows, n, rng)
                    val expected = DenseMatrix(rows, n)
                    ReferenceBlas.gemm(1.0, b, false, explicit, transpose, 0.0, expected)
                    val actual = DenseMatrix.wrap(rows, n, b.values.copyOf())

                    blas.trmm(t, actual, lower, transpose, unitDiag, right = true)

                    assertClose(expected, actual, "trmm right l=$lower t=$transpose u=$unitDiag", tolerance = 1e-11)
                }
            }
        }
    }

    @Test
    fun `trsm right inverts trmm right`() = withDenseBlas { blas ->
        val rng = Random(20260941)
        val n = 6
        val rows = 3
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(true, false)) {
                for (unitDiag in booleanArrayOf(true, false)) {
                    val (t, _) = poisonedTriangle(rng, n, lower, unitDiag)
                    val x = randomMatrix(rows, n, rng)
                    val b = DenseMatrix.wrap(rows, n, x.values.copyOf())

                    blas.trmm(t, b, lower, transpose, unitDiag, right = true)
                    blas.trsm(t, b, lower, transpose, unitDiag, right = true)

                    assertClose(x, b, "trsm right l=$lower t=$transpose u=$unitDiag", tolerance = 1e-11)
                }
            }
        }
    }

    @Test
    fun `trmv round-trips with trsv`() = withDenseBlas { blas ->
        val rng = Random(20260917)
        val n = 9
        for (lower in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                val (t, _) = poisonedTriangle(rng, n, lower, unitDiag = false)
                val x0 = randomVector(n, rng)
                val x = x0.copyOf()

                blas.trmv(t, x, lower, transpose)
                blas.trsv(t, x, lower, transpose)

                assertClose(x0, x, "roundtrip lower=$lower t=$transpose")
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
