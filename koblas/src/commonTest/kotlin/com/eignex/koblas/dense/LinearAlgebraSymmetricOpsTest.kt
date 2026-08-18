package com.eignex.koblas.dense

import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import com.eignex.koblas.poisonedSymmetric
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class LinearAlgebraSymmetricOpsTest {

    @Test
    fun `symv matches gemv on the full matrix and reads only the selected triangle`() {
        val rng = Random(20260910)
        for (lower in booleanArrayOf(true, false)) {
            for (n in intArrayOf(1, 2, 7, 16)) {
                for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                        val x = randomVector(n, rng)
                        val y0 = DoubleArray(n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                        val expected = y0.copyOf()
                        koblas.gemv(alpha, full, x, beta, expected)
                        val actual = y0.copyOf()
                        koblas.symv(alpha, poisoned, x, beta, actual, lower)
                        assertClose(expected, actual, "symv n=$n a=$alpha b=$beta lower=$lower")
                    }
                }
            }
        }
    }

    /**
     * `dsymv` derives its extent from one dimension, so a non-square matrix would have the host backend read
     * `n²` entries from a shorter array. The sizes straddle every backend's level-2 gate.
     */
    @Test
    fun `symv refuses a non-square matrix at every size`() {
        for (n in intArrayOf(2, 17, 64, 129)) {
            val wide = F64DenseMatrix.zero(n, n + 1)
            val tall = F64DenseMatrix.zero(n + 1, n)
            assertFailsWith<DimensionMismatch>("symv ${n}x${n + 1}") {
                koblas.symv(1.0, wide, DoubleArray(n), 0.0, DoubleArray(n))
            }
            assertFailsWith<DimensionMismatch>("symv ${n + 1}x$n") {
                koblas.symv(1.0, tall, DoubleArray(n + 1), 0.0, DoubleArray(n + 1))
            }
        }
    }

    @Test
    fun `symm matches gemm on the full matrix and reads only the selected triangle`() {
        val rng = Random(20260911)
        for (lower in booleanArrayOf(true, false)) {
            for (n in intArrayOf(1, 5, 12)) {
                val p = 4
                for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                        val b = randomMatrix(n, p, rng)
                        val c0 = DoubleArray(n * p) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                        val expected = F64DenseMatrix(n, p, c0.copyOf())
                        koblas.gemm(alpha, full, transposeA = false, b, transposeB = false, beta, expected)
                        val actual = F64DenseMatrix(n, p, c0.copyOf())
                        koblas.symm(alpha, poisoned, b, beta, actual, lower)
                        assertClose(expected, actual, "symm n=$n a=$alpha b=$beta lower=$lower")
                    }
                }
            }
        }
    }

    @Test
    fun `symm right multiplies from the right and reads only the selected triangle`() {
        val rng = Random(20260942)
        val n = 5
        val rows = 4
        for (lower in booleanArrayOf(true, false)) {
            for (alpha in doubleArrayOf(0.0, 0.75)) {
                for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                    val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                    val b = randomMatrix(rows, n, rng)
                    val c0 = DoubleArray(rows * n) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                    val expected = F64DenseMatrix.wrap(rows, n, c0.copyOf())
                    koblas.gemm(alpha, b, false, full, false, beta, expected)
                    val actual = F64DenseMatrix.wrap(rows, n, c0.copyOf())
                    koblas.symm(alpha, poisoned, b, beta, actual, lower, right = true)
                    assertClose(expected, actual, "symm right l=$lower a=$alpha b=$beta", tolerance = 1e-11)
                }
            }
        }
    }

    @Test
    fun `syrk triangle modes write only the selected triangle with strict beta semantics`() {
        val rng = Random(20260933)
        for (uplo in listOf(Uplo.LOWER, Uplo.UPPER)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (alpha in doubleArrayOf(0.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        checkSyrk(rng, uplo, transpose, alpha, beta)
                    }
                }
            }
        }
    }

    @Suppress("CyclomaticComplexMethod")
    private fun checkSyrk(rng: Random, uplo: Uplo, transpose: Boolean, alpha: Double, beta: Double) {
        val a = randomMatrix(6, 4, rng)
        val n = if (transpose) a.cols else a.rows
        val term = F64DenseMatrix(n, n)
        koblas.syrk(1.0, a, transpose, 0.0, term)
        // The output is NaN where beta == 0 must overwrite without reading, and the unselected triangle stays NaN.
        val c = F64DenseMatrix(n, n)
        val c0 = F64DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0 until n) {
                val selected = if (uplo == Uplo.LOWER) j <= i else j >= i
                val v = if (!selected || beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0)
                c[i, j] = v
                c0[i, j] = v
            }
        }
        koblas.syrk(alpha, a, transpose, beta, c, uplo)
        val context = "syrk $uplo t=$transpose a=$alpha b=$beta"
        for (i in 0 until n) {
            for (j in 0 until n) {
                val selected = if (uplo == Uplo.LOWER) j <= i else j >= i
                if (selected) {
                    val expected = (if (beta == 0.0) 0.0 else beta * c0[i, j]) + alpha * term[i, j]
                    assertClose(expected, c[i, j], "$context ($i;$j)")
                } else {
                    assertTrue(c[i, j].isNaN(), "$context ($i;$j): untouched triangle was written")
                }
            }
        }
    }

    @Test
    fun `syrk through a reused workspace does not accumulate the previous call`() {
        val rng = Random(20260946)
        val n = 8
        val k = 5
        val ws = Workspace()
        for (uplo in listOf(Uplo.FULL, Uplo.LOWER, Uplo.UPPER)) {
            val a = randomMatrix(n, k, rng)
            val first = F64DenseMatrix(n, n)
            koblas.syrk(0.75, a, transpose = false, beta = 0.0, c = first, uplo = uplo, workspace = ws)
            val second = F64DenseMatrix(n, n)
            koblas.syrk(0.75, a, transpose = false, beta = 0.0, c = second, uplo = uplo, workspace = ws)
            val fresh = F64DenseMatrix(n, n)
            koblas.syrk(0.75, a, transpose = false, beta = 0.0, c = fresh, uplo = uplo)
            for (i in 0 until n) {
                for (j in 0 until n) {
                    val f = first[i, j]
                    if (f.isNaN()) continue // an untouched triangle stays untouched; checked elsewhere
                    assertClose(f, second[i, j], "syrk $uplo reused workspace ($i;$j)")
                    assertClose(f, fresh[i, j], "syrk $uplo against no workspace ($i;$j)")
                }
            }
        }
    }

    @Test
    fun `syr2k matches the gemm expansion in both orientations`() {
        val rng = Random(20260809)
        for (transpose in booleanArrayOf(false, true)) {
            val n = 4
            val k = 3
            val a = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
            val b = if (transpose) randomMatrix(k, n, rng) else randomMatrix(n, k, rng)
            val c = F64DenseMatrix(n, n)
            koblas.syr2k(0.5, a, b, transpose, 0.0, c)
            for (i in 0 until n) {
                for (j in 0 until n) {
                    var s = 0.0
                    for (p in 0 until k) {
                        val ai = if (transpose) a[p, i] else a[i, p]
                        val aj = if (transpose) a[p, j] else a[j, p]
                        val bi = if (transpose) b[p, i] else b[i, p]
                        val bj = if (transpose) b[p, j] else b[j, p]
                        s += ai * bj + bi * aj
                    }
                    assertClose(0.5 * s, c[i, j], "transpose=$transpose at [$i,$j]")
                }
            }
        }
    }

    @Test
    fun `symmetric ops handle empty shapes`() {
        koblas.symm(1.0, F64DenseMatrix(0, 0), F64DenseMatrix(0, 3), 0.0, F64DenseMatrix(0, 3))
        val c = F64DenseMatrix(2, 0)
        koblas.symm(1.0, F64DenseMatrix(2, 2), F64DenseMatrix(2, 0), 0.0, c)
        val y = DoubleArray(2) { Double.NaN }
        koblas.symv(0.0, F64DenseMatrix(2, 2), DoubleArray(2), 0.0, y)
        assertTrue(y.all { it == 0.0 }, "symv alpha=0 beta=0 left ${y.toList()}")
    }
}
