@file:Suppress("PropertyName") // math convention: single-letter matrices in tests

package com.eignex.koblas

import com.eignex.koblas.dense.Uplo
import com.eignex.koblas.dense.invert
import com.eignex.koblas.dense.lu
import com.eignex.koblas.dense.trtri
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class CoverageGapTest {

    @Test
    fun `invert produces the inverse of a general matrix`() {
        val rng = Random(20260805)
        for (n in intArrayOf(1, 2, 5, 9)) {
            val a = wellConditioned(n, rng)
            val inv = a.lu().invert()
            for (i in 0 until n) {
                for (j in 0 until n) {
                    var s = 0.0
                    for (k in 0 until n) s += a[i, k] * inv[k, j]
                    assertEquals(if (i == j) 1.0 else 0.0, s, 1e-9, "n=$n A·Ainv at [$i,$j]")
                }
            }
        }
    }

    @Test
    fun `invert refuses a singular factorization and names the pivot`() {
        val rank1 = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
        val failure = assertFailsWith<IllegalArgumentException> { rank1.lu().invert() }
        assertTrue("pivot 1" in failure.message!!, "should name the failing pivot: ${failure.message}")
    }

    @Test
    fun `trtri inverts a triangle in both orientations`() {
        val rng = Random(20260806)
        for (n in intArrayOf(1, 4, 7)) {
            for (lower in booleanArrayOf(true, false)) {
                val t = DenseMatrix(n, n)
                for (j in 0 until n) {
                    for (i in 0 until n) {
                        val inTriangle = if (lower) i >= j else i <= j
                        if (inTriangle) t[i, j] = if (i == j) 2.0 + j else rng.nextDouble(-1.0, 1.0)
                    }
                }
                val inv = trtri(t, lower)
                for (i in 0 until n) {
                    for (j in 0 until n) {
                        var s = 0.0
                        for (k in 0 until n) s += t[i, k] * inv[k, j]
                        assertEquals(if (i == j) 1.0 else 0.0, s, 1e-9, "n=$n lower=$lower at [$i,$j]")
                    }
                    for (j in 0 until n) {
                        val outside = if (lower) i < j else i > j
                        if (outside) assertEquals(0.0, inv[i, j], "n=$n lower=$lower leaked at [$i,$j]")
                    }
                }
            }
        }
    }

    @Test
    fun `trtri rejects a zero on the diagonal`() {
        val singular = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 0.0), doubleArrayOf(3.0, 0.0)))
        val failure = assertFailsWith<IllegalArgumentException> { trtri(singular, lower = true) }
        assertTrue("entry 1" in failure.message!!, "should name the zero position: ${failure.message}")
        trtri(singular, lower = true, unitDiag = true)
    }

    @Test
    fun `normInf and normFro agree with direct sums`() {
        val a = DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, -2.0, 3.0),
                doubleArrayOf(-4.0, 5.0, -6.0),
            ),
        )
        // Row sums are 6 and 15. Column sums are 5, 7 and 9.
        assertEquals(15.0, normInf(a), 1e-12)
        assertEquals(9.0, norm1(a), 1e-12, "norm1 must be unaffected")
        assertEquals(sqrt(1.0 + 4 + 9 + 16 + 25 + 36), normFro(a), 1e-12)

        val ws = Workspace().apply { reserve(a.rows, count = 2) }
        assertEquals(normInf(a), normInf(a, ws), 1e-12)
        assertEquals(normInf(a), normInf(a, ws), 1e-12)

        val empty = DenseMatrix(0, 0)
        assertEquals(0.0, normInf(empty))
        assertEquals(0.0, normFro(empty))
    }

    @Test
    fun `normFro survives entries that square out of range`() {
        val big = DenseMatrix.of(arrayOf(doubleArrayOf(3e200, 4e200)))
        assertEquals(5e200, normFro(big), 1e188)
    }

    @Test
    fun `syr and syr2 match the equivalent ger sweeps`() {
        val rng = Random(20260807)
        val n = 6
        val x = DenseVector.of(randomVector(n, rng))
        val y = DenseVector.of(randomVector(n, rng))

        val viaSyr = DenseMatrix(n, n)
        syr(1.5, x, viaSyr)
        val viaGer = DenseMatrix(n, n)
        ger(1.5, x, x, viaGer)
        assertClose(viaGer, viaSyr, "syr against ger")

        val viaSyr2 = DenseMatrix(n, n)
        syr2(-0.75, x, y, viaSyr2)
        val viaGer2 = DenseMatrix(n, n)
        ger(-0.75, x, y, viaGer2)
        ger(-0.75, y, x, viaGer2)
        assertClose(viaGer2, viaSyr2, "syr2 against two gers")
    }

    @Test
    fun `the symmetric updates are exactly symmetric and honour uplo`() {
        val rng = Random(20260808)
        val n = 5
        val x = DenseVector.of(randomVector(n, rng))
        val y = DenseVector.of(randomVector(n, rng))

        val full = DenseMatrix(n, n)
        syr2(0.3, x, y, full)
        for (i in 0 until n) {
            for (j in 0 until n) {
                assertTrue(full[i, j] == full[j, i], "syr2 FULL is not exactly symmetric at [$i,$j]")
            }
        }

        val lower = DenseMatrix(n, n)
        syr(1.0, x, lower, Uplo.LOWER)
        for (i in 0 until n) {
            for (j in 0 until n) {
                if (i < j) assertEquals(0.0, lower[i, j], "syr LOWER wrote the upper triangle at [$i,$j]")
                if (i >= j) assertEquals(x[i] * x[j], lower[i, j], 1e-12, "[$i,$j]")
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
            val c = DenseMatrix(n, n)
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
                    assertEquals(0.5 * s, c[i, j], 1e-12, "transpose=$transpose at [$i,$j]")
                }
            }
        }
    }

    @Test
    fun `the sparse transpose round-trips and preserves stored zeros`() {
        val a = SparseMatrix.ofColumns(
            3,
            2,
            listOf(
                listOf(0 to 4.0, 1 to 0.0, 2 to -1.0),
                listOf(1 to 7.0),
            ),
        )
        val t = a.transpose()
        assertEquals(2, t.rows)
        assertEquals(3, t.cols)
        assertEquals(a.nnz, t.nnz, "an explicitly stored zero was dropped")
        for (i in 0 until a.rows) {
            for (j in 0 until a.cols) assertEquals(a[i, j], t[j, i], "transpose at [$i,$j]")
        }
        assertEquals(a, t.transpose(), "transpose twice is not the original")
    }

    @Test
    fun `the sparse transpose handles degenerate shapes`() {
        val empty = SparseMatrix.ofColumns(0, 0, emptyList())
        assertEquals(empty, empty.transpose().transpose())
        val noEntries = SparseMatrix.ofColumns(3, 2, listOf(emptyList(), emptyList()))
        val t = noEntries.transpose()
        assertEquals(2, t.rows)
        assertEquals(3, t.cols)
        assertEquals(0, t.nnz)
    }

    @Test
    fun `rotg generates the rotation that zeroes the second component`() {
        val pairs = listOf(3.0 to 4.0, -3.0 to 4.0, 3.0 to -4.0, 1.0 to 0.0, 0.0 to 1.0, -2.0 to 0.0)
        for ((a, b) in pairs) {
            val g = rotg(a, b)
            assertEquals(sqrt(a * a + b * b), abs(g.r), 1e-12, "r is not the pair's length for ($a, $b)")
            assertEquals(1.0, g.c * g.c + g.s * g.s, 1e-12, "not a rotation for ($a, $b)")
            assertEquals(g.r, g.c * a + g.s * b, 1e-12, "rotation does not produce r for ($a, $b)")
            assertEquals(0.0, g.c * b - g.s * a, 1e-12, "rotation does not zero b for ($a, $b)")
        }
    }

    @Test
    fun `rotg on an all-zero pair is the identity rotation`() {
        val g = rotg(0.0, 0.0)
        assertEquals(1.0, g.c)
        assertEquals(0.0, g.s)
        assertEquals(0.0, g.r)
    }

    @Test
    fun `rotg survives components that square out of range`() {
        val g = rotg(3e200, 4e200)
        assertEquals(5e200, abs(g.r), 1e188)
        assertEquals(1.0, g.c * g.c + g.s * g.s, 1e-12)
        val tiny = rotg(3e-200, 4e-200)
        assertEquals(5e-200, abs(tiny.r), 1e-212)
        assertEquals(1.0, tiny.c * tiny.c + tiny.s * tiny.s, 1e-12)
    }

    @Test
    fun `rot applies the rotation elementwise and preserves length`() {
        val rng = Random(20260810)
        val n = 7
        val x = DenseVector.of(randomVector(n, rng))
        val y = DenseVector.of(randomVector(n, rng))
        val x0 = x.data.copyOf()
        val y0 = y.data.copyOf()
        val g = rotg(2.0, 1.0)

        rot(x, y, g)
        for (i in 0 until n) {
            assertEquals(g.c * x0[i] + g.s * y0[i], x.data[i], 1e-12, "x at $i")
            assertEquals(g.c * y0[i] - g.s * x0[i], y.data[i], 1e-12, "y at $i")
        }
        for (i in 0 until n) {
            val before = x0[i] * x0[i] + y0[i] * y0[i]
            val after = x.data[i] * x.data[i] + y.data[i] * y.data[i]
            assertEquals(before, after, 1e-12, "length changed at $i")
        }
    }

    @Test
    fun `rotg and rot together zero the target entry`() {
        val x = DenseVector.of(doubleArrayOf(3.0, 1.0, 2.0))
        val y = DenseVector.of(doubleArrayOf(4.0, -1.0, 0.5))
        val g = rotg(x[0], y[0])
        rot(x, y, g)
        assertEquals(0.0, y[0], 1e-12, "the leading entry of y was not eliminated")
        assertEquals(5.0, x[0], 1e-12)
    }

    private fun assertClose(expected: DenseMatrix, actual: DenseMatrix, context: String) {
        assertEquals(expected.rows, actual.rows, "$context: rows")
        assertEquals(expected.cols, actual.cols, "$context: cols")
        for (i in 0 until expected.rows) {
            for (j in 0 until expected.cols) {
                assertEquals(expected[i, j], actual[i, j], 1e-12, "$context at [$i,$j]")
            }
        }
    }
}
