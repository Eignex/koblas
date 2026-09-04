@file:Suppress("PropertyName") // math convention: single-letter matrices (A, L, V) in tests

package com.eignex.koblas.dense

import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.koblas
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.*

@Suppress("VariableNaming") // single-letter matrix/vector names track math conventions
class CholeskyUpdateTest {

    /** A symmetric positive-definite matrix whose conditioning is set by [spread] decades of eigenvalue. */
    private fun spd(n: Int, rng: Random, spread: Double = 0.0): F64DenseMatrix {
        val q = F64DenseMatrix(n, n)
        for (j in 0 until n) for (i in 0 until n) q[i, j] = rng.nextDouble(-1.0, 1.0)
        val a = F64DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0..i) {
                var s = 0.0
                for (t in 0 until n) {
                    val weight = if (spread == 0.0) 1.0 else pow10(-spread * t / maxOf(1, n - 1))
                    s += weight * q[i, t] * q[j, t]
                }
                if (i == j) s += n.toDouble()
                a[i, j] = s
                a[j, i] = s
            }
        }
        return a
    }

    private fun pow10(e: Double): Double {
        var r = 1.0
        var k = e
        while (k <= -1.0) {
            r /= 10.0
            k += 1.0
        }
        return r * (1.0 - 0.9 * (-k))
    }

    /** `A + sigma * V * Vt`, reading [v]'s columns as the update vectors. */
    private fun updated(a: F64DenseMatrix, v: F64DenseMatrix, sigma: Double = 1.0): F64DenseMatrix {
        val out = F64DenseMatrix(a.rows, a.cols, a.data.copyOf())
        for (c in 0 until v.cols) {
            for (j in 0 until a.cols) {
                for (i in 0 until a.rows) out[i, j] += sigma * v[i, c] * v[j, c]
            }
        }
        return out
    }

    private fun columnOf(v: DoubleArray) = F64DenseMatrix(v.size, 1, v.copyOf())

    /** Adds `sigma * v * vt` into [a], the accumulation a long run of updates is checked against. */
    private fun accumulate(a: F64DenseMatrix, v: DoubleArray) {
        for (j in 0 until a.cols) {
            for (i in 0 until a.rows) a[i, j] += v[i] * v[j]
        }
    }

    private fun assertFactors(expected: F64DenseMatrix, actual: F64CholeskyDecomposition, context: String) {
        val n = expected.rows
        for (i in 0 until n) {
            assertTrue(actual.l[i, i] > 0.0, "$context: diagonal $i is ${actual.l[i, i]}, expected positive")
            for (j in i + 1 until n) assertEquals(0.0, actual.l[i, j], "$context: L[$i,$j] above the diagonal")
        }
        for (i in 0 until n) {
            for (j in 0..i) {
                var s = 0.0
                for (t in 0..minOf(i, j)) s += actual.l[i, t] * actual.l[j, t]
                assertClose(expected[i, j], s, "$context: reconstruction at ($i,$j)", tolerance = 1e-8)
            }
        }
    }

    /**
     * The rotation takes its length in a scaled form, so a pair whose squares would overflow still rotates.
     * A one by one factor is the whole rotation: the updated diagonal is exactly `hypot(l, v)`.
     */
    @Test
    fun `an update whose squares would overflow keeps its magnitude`() {
        val huge = 1e200
        val chol = F64CholeskyDecomposition(F64DenseMatrix(1, 1, doubleArrayOf(huge)))

        val updated = chol.rankUpdate(doubleArrayOf(huge))

        assertEquals(sqrt(2.0) * huge, updated.l[0, 0], huge * 1e-12, "squaring 1e200 would overflow")
    }

    /** And a pair whose squares would vanish keeps it, rather than collapsing the diagonal to zero. */
    @Test
    fun `an update whose squares would vanish keeps its magnitude`() {
        val tiny = 1e-200
        val chol = F64CholeskyDecomposition(F64DenseMatrix(1, 1, doubleArrayOf(tiny)))

        val updated = chol.rankUpdate(doubleArrayOf(tiny))

        assertEquals(sqrt(2.0) * tiny, updated.l[0, 0], tiny * 1e-12, "squaring 1e-200 would vanish")
    }

    /** The diagonal stays positive by construction, which is what lets the factor skip a sign correction. */
    @Test
    fun `an update by a larger negative vector keeps a positive diagonal`() {
        val chol = F64CholeskyDecomposition(F64DenseMatrix(1, 1, doubleArrayOf(1.0)))

        val updated = chol.rankUpdate(doubleArrayOf(-8.0))

        assertEquals(sqrt(65.0), updated.l[0, 0], 1e-12)
    }

    @Test
    fun `a rank-1 update reconstructs the updated matrix`() {
        val rng = Random(20260903)
        for (n in intArrayOf(1, 2, 5, 17)) {
            val A = spd(n, rng)
            val v = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            val factor = A.cholesky().rankUpdate(v)
            assertFactors(updated(A, columnOf(v)), factor, "n=$n")
        }
    }

    @Test
    fun `a rank-1 update of an ill-conditioned factor agrees with refactorization`() {
        val rng = Random(20260904)
        for (spread in doubleArrayOf(4.0, 8.0)) {
            val n = 24
            val A = spd(n, rng, spread)
            val v = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            val expected = updated(A, columnOf(v)).cholesky()
            val actual = A.cholesky().rankUpdate(v)
            assertClose(expected.l.data, actual.l.data, "ill-conditioned spread=$spread", tolerance = 1e-7)
        }
    }

    @Test
    fun `thousands of sequential updates do not drift from a refactorization`() {
        val rng = Random(20260905)
        val n = 64
        val A = spd(n, rng)
        val factor = A.cholesky()
        val accumulated = F64DenseMatrix(n, n, A.data.copyOf())
        repeat(2000) {
            val v = DoubleArray(n) { rng.nextDouble(-0.05, 0.05) }
            factor.rankUpdate(v)
            accumulate(accumulated, v)
        }
        assertFactors(accumulated, factor, "after 2000 updates")
    }

    @Test
    fun `a rank-k update matches the same columns applied one at a time`() {
        val rng = Random(20260906)
        val n = 12
        val k = 5
        val A = spd(n, rng)
        val V = F64DenseMatrix(n, k)
        for (c in 0 until k) for (i in 0 until n) V[i, c] = rng.nextDouble(-1.0, 1.0)

        val blocked = A.cholesky().rankUpdate(V)
        val oneAtATime = A.cholesky()
        for (c in 0 until k) {
            oneAtATime.rankUpdate(DoubleArray(n) { V[it, c] })
        }

        assertClose(oneAtATime.l.data, blocked.l.data, "rank-$k against $k rank-1 updates", tolerance = 1e-9)
        assertFactors(updated(A, V), blocked, "rank-$k")
    }

    @Test
    fun `sigma scales the outer product and zero leaves the factor alone`() {
        val rng = Random(20260907)
        val n = 9
        val A = spd(n, rng)
        val v = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

        val scaled = A.cholesky().rankUpdate(v, sigma = 2.5)
        assertFactors(updated(A, columnOf(v), sigma = 2.5), scaled, "sigma=2.5")

        // A zero scale is not the same input as a zero vector, and both have to leave the factor untouched.
        val untouched = A.cholesky()
        val before = untouched.l.data.copyOf()
        untouched.rankUpdate(v, sigma = 0.0)
        assertContentEquals(before, untouched.l.data, "sigma=0 changed the factor")
        untouched.rankUpdate(DoubleArray(n))
        assertContentEquals(before, untouched.l.data, "a zero vector changed the factor")
    }

    @Test
    fun `a basis vector update leaves the leading rotations as the identity`() {
        val rng = Random(20260908)
        val n = 7
        val A = spd(n, rng)
        for (k in intArrayOf(0, 3, 6)) {
            val v = DoubleArray(n).also { it[k] = 1.5 }
            val factor = A.cholesky().rankUpdate(v)
            assertFactors(updated(A, columnOf(v)), factor, "e_$k")
        }
    }

    @Test
    fun `an empty factor and a single entry update`() {
        val empty = F64DenseMatrix(0, 0).cholesky()
        assertSame(empty, empty.rankUpdate(DoubleArray(0)))
        assertEquals(0, empty.n)

        val one = F64DenseMatrix.of(arrayOf(doubleArrayOf(4.0))).cholesky()
        one.rankUpdate(doubleArrayOf(3.0))
        assertClose(sqrt(13.0), one.l[0, 0], "1x1 update should carry sqrt(4 + 9)")
    }

    @Test
    fun `the update keeps the factor buffer and leaves the vector alone`() {
        val rng = Random(20260909)
        val n = 11
        val factor = spd(n, rng).cholesky()
        val buffer = factor.l.data
        val v = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val vBefore = v.copyOf()

        val returned = factor.rankUpdate(v)

        assertSame(factor, returned, "rankUpdate must return its receiver")
        assertSame(buffer, returned.l.data, "rankUpdate must retain the factor buffer")
        assertContentEquals(vBefore, v, "rankUpdate must not consume the update vector")
    }

    @Test
    fun `updating by the factor itself scales it by the square root of two`() {
        val rng = Random(20260910)
        val n = 4
        val A = spd(n, rng)
        val factor = A.cholesky()
        val before = factor.l.data.copyOf()
        // One buffer behind both the factor and the update block. Column c of L is zero above row c, so the
        // rotations before step c are the identity, and at step c the column meets an identical vector:
        // it scales by sqrt(2) and zeroes the working vector, leaving every later step the identity too.
        // So each column is read before its own sweep touches it, and the whole factor scales exactly.
        factor.rankUpdate(F64DenseMatrix(n, n, factor.l.data))

        for (i in 0 until n) {
            for (j in 0..i) {
                assertClose(sqrt(2.0) * before[i + j * n], factor.l[i, j], "scaled factor at ($i,$j)")
            }
        }
        assertFactors(updated(A, F64DenseMatrix(n, n, before)), factor, "factor as its own update block")
    }

    @Test
    fun `a negative or non-finite sigma is rejected`() {
        val rng = Random(20260911)
        val factor = spd(3, rng).cholesky()
        assertFailsWith<IllegalArgumentException> { factor.rankUpdate(DoubleArray(3), sigma = -1.0) }
        assertFailsWith<IllegalArgumentException> { factor.rankUpdate(DoubleArray(3), sigma = Double.NaN) }
    }

    @Test
    fun `an update vector of the wrong length is rejected`() {
        val rng = Random(20260912)
        val factor = spd(3, rng).cholesky()
        assertFailsWith<com.eignex.koblas.DimensionMismatch> { factor.rankUpdate(DoubleArray(4)) }
        assertFailsWith<com.eignex.koblas.DimensionMismatch> { koblas.choleskyRankUpdate(factor, F64DenseMatrix(4, 2)) }
    }

    @Test
    fun `a zero-column update block returns the factor unchanged`() {
        val rng = Random(20260913)
        val factor = spd(5, rng).cholesky()
        val before = factor.l.data.copyOf()
        assertSame(factor, factor.rankUpdate(F64DenseMatrix(5, 0)))
        assertContentEquals(before, factor.l.data)
    }

    @Test
    fun `the update never puts a negative entry on the diagonal`() {
        val rng = Random(20260914)
        val n = 16
        val factor = spd(n, rng).cholesky()
        // Vectors far larger than the factor's diagonal are what drive rotg's sign convention negative.
        repeat(30) {
            factor.rankUpdate(DoubleArray(n) { -50.0 * rng.nextDouble(0.5, 1.0) })
            for (i in 0 until n) assertTrue(factor.l[i, i] > 0.0, "diagonal $i went to ${factor.l[i, i]}")
        }
        assertTrue(abs(factor.l[0, 0]) > 0.0)
    }
}
