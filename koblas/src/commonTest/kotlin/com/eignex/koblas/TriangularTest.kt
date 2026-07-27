package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class TriangularTest {

    // A well-conditioned matrix with distinct lower and upper triangles; the untouched triangle is
    // filled with garbage to verify only the selected triangle is read.
    private fun triangularTestMatrix(n: Int, lower: Boolean, unitDiag: Boolean, rng: Random): DenseMatrix {
        val t = DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                val inTriangle = if (lower) j < i else j > i
                t[i, j] = when {
                    i == j -> if (unitDiag) Double.NaN else rng.nextDouble(2.0, 4.0)

                    // NaN diag must be ignored
                    inTriangle -> rng.nextDouble(-1.0, 1.0)

                    else -> Double.NaN // opposite triangle must never be read
                }
            }
        }
        return t
    }

    // Naive reference: materialize the effective triangle (unit/actual diagonal, zeros outside),
    // then apply op(T)·x with a plain double loop.
    private fun opT(
        t: DenseMatrix,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        x: DoubleArray,
    ): DoubleArray {
        val n = t.rows
        val e = DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                val stored = if (lower) j < i else j > i
                e[i, j] = when {
                    i == j -> if (unitDiag) 1.0 else t[i, i]
                    stored -> t[i, j]
                    else -> 0.0
                }
            }
        }
        val y = DoubleArray(n)
        for (i in 0 until n) {
            for (j in 0 until n) y[i] += (if (transpose) e[j, i] else e[i, j]) * x[j]
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
                        val t = triangularTestMatrix(n, lower, unitDiag, rng)
                        val xTrue = DoubleArray(n) { rng.nextDouble(-2.0, 2.0) }
                        val b = opT(t, lower, transpose, unitDiag, xTrue)
                        val x = b.copyOf()
                        trsv(t, x, lower, transpose, unitDiag)
                        for (i in 0 until n) {
                            assertTrue(
                                abs(x[i] - xTrue[i]) < 1e-9,
                                "trsv n=$n lower=$lower t=$transpose unit=$unitDiag at $i: ${x[i]} vs ${xTrue[i]}",
                            )
                        }
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
                    val t = triangularTestMatrix(n, lower, unitDiag, rng)
                    val b = DenseMatrix(n, nrhs, DoubleArray(n * nrhs) { rng.nextDouble(-2.0, 2.0) })
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
        val t = triangularTestMatrix(n, lower = true, unitDiag = false, rng = rng)
        // Zero the NaN garbage so gemm can verify T·X = B.
        for (i in 0 until n) for (j in i + 1 until n) t[i, j] = 0.0
        val xTrue = DenseMatrix(n, nrhs, DoubleArray(n * nrhs) { rng.nextDouble(-2.0, 2.0) })
        val b = t.matMul(xTrue)
        trsm(t, b, lower = true)
        for (idx in b.data.indices) {
            assertTrue(abs(b.data[idx] - xTrue.data[idx]) < 1e-9, "trsm multi-RHS at $idx")
        }
    }

    @Test
    fun `trsv and trsm validate shapes`() {
        assertFailsWith<IllegalArgumentException> { trsv(DenseMatrix(2, 3), DoubleArray(2), lower = true) }
        assertFailsWith<IllegalArgumentException> { trsv(DenseMatrix(3, 3), DoubleArray(2), lower = true) }
        assertFailsWith<IllegalArgumentException> { trsm(DenseMatrix(3, 3), DenseMatrix(2, 4), lower = true) }
    }
}
