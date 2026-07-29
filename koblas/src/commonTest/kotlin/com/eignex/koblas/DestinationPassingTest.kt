package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * The destination-passing forms exist so a steady-state loop allocates nothing. They therefore have to
 * agree with the allocating forms exactly, tolerate a destination that aliases the input, and survive
 * being driven repeatedly with one reused [Workspace] — a shared buffer that two nested operations both
 * wrote would show up here as a wrong answer on the second call.
 */
class DestinationPassingTest {

    private fun wellConditioned(n: Int, rng: Random): DenseMatrix {
        val a = DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0 until n) a[i, j] = rng.nextDouble(-1.0, 1.0) + if (i == j) n.toDouble() else 0.0
        }
        return a
    }

    private fun assertClose(expected: DoubleArray, actual: DoubleArray, context: String) {
        assertEquals(expected.size, actual.size, context)
        for (i in expected.indices) {
            assertTrue(
                abs(expected[i] - actual[i]) <= 1e-12 * maxOf(1.0, abs(expected[i])),
                "$context index $i: expected ${expected[i]} actual ${actual[i]}",
            )
        }
    }

    @Test
    fun `lu solveInto matches solve and returns its destination`() {
        val rng = Random(20260730)
        for (n in intArrayOf(1, 5, 17)) {
            val lu = wellConditioned(n, rng).lu()
            val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            for (transpose in booleanArrayOf(false, true)) {
                val out = DoubleArray(n)
                val returned = koblas.solveInto(lu, b, out, transpose)
                assertSame(out, returned, "solveInto must return its destination")
                assertClose(koblas.solve(lu, b, transpose), out, "lu n=$n t=$transpose")
            }
        }
    }

    @Test
    fun `lu solveInto tolerates the destination aliasing the input`() {
        val rng = Random(20260731)
        val n = 12
        val lu = wellConditioned(n, rng).lu()
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        for (transpose in booleanArrayOf(false, true)) {
            val expected = koblas.solve(lu, b, transpose)
            val inPlace = b.copyOf()
            koblas.solveInto(lu, inPlace, inPlace, transpose)
            assertClose(expected, inPlace, "aliased t=$transpose")
        }
    }

    @Test
    fun `ldl solveInto matches solve aliased or not`() {
        val rng = Random(20260732)
        for (n in intArrayOf(1, 2, 9)) {
            val a = DenseMatrix(n, n)
            for (i in 0 until n) {
                for (j in 0..i) {
                    var v = rng.nextDouble(-1.0, 1.0)
                    if (i == j) v += if (i % 2 == 0) 2.0 else -2.0
                    a[i, j] = v
                    a[j, i] = v
                }
            }
            val f = koblas.ldl(a)
            val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            val expected = koblas.solve(f, b)
            assertClose(expected, koblas.solveInto(f, b, DoubleArray(n)), "ldl n=$n")
            val aliased = b.copyOf()
            assertClose(expected, koblas.solveInto(f, aliased, aliased), "ldl aliased n=$n")
        }
    }

    @Test
    fun `factorInto reuses the buffers and matches a fresh factorization`() {
        val rng = Random(20260733)
        val n = 10
        val first = wellConditioned(n, rng)
        val reused = koblas.factor(first)
        val luBuffer = reused.lu
        val pivBuffer = reused.piv
        // Refactorize a different matrix into the same decomposition.
        val second = wellConditioned(n, rng)
        val returned = koblas.factorInto(second, reused)
        assertSame(reused, returned, "factorInto must return its destination")
        assertSame(luBuffer, reused.lu, "factorInto must not replace the factor buffer")
        assertSame(pivBuffer, reused.piv, "factorInto must not replace the pivot buffer")
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        assertClose(koblas.solve(koblas.factor(second), b), koblas.solve(reused, b), "refactorized solve")
        // A singular refactorization updates the flag rather than leaving the previous one.
        val singular = DenseMatrix(n, n) // all zeros
        koblas.factorInto(singular, reused)
        assertTrue(reused.singular, "singular flag must follow the refactorization")
    }

    @Test
    fun `rcond with a workspace matches rcond without one when repeated`() {
        val rng = Random(20260734)
        val ws = Workspace()
        for (n in intArrayOf(1, 6, 24)) {
            val a = wellConditioned(n, rng)
            val anorm = norm1(a)
            val lu = a.lu()
            val plain = koblas.rcond(lu, anorm)
            // Twice, to catch a workspace that a previous call left in a bad state.
            assertEquals(plain, koblas.rcond(lu, anorm, ws), "rcond n=$n first")
            assertEquals(plain, koblas.rcond(lu, anorm, ws), "rcond n=$n second")
        }
    }

    @Test
    fun `block solves into a destination match the allocating forms`() {
        val rng = Random(20260750)
        val ws = Workspace()
        for (n in intArrayOf(2, 9)) {
            // Both sides of the narrow/wide dispatch: column-by-column below the threshold, blocked above.
            for (nrhs in intArrayOf(1, 3, 40)) {
                val lu = wellConditioned(n, rng).lu()
                val b = DenseMatrix(n, nrhs)
                for (i in 0 until n) for (j in 0 until nrhs) b[i, j] = rng.nextDouble(-1.0, 1.0)
                for (transpose in booleanArrayOf(false, true)) {
                    val expected = koblas.solve(lu, b, transpose)
                    val out = DenseMatrix(n, nrhs)
                    val returned = koblas.solveInto(lu, b, out, transpose, ws)
                    assertSame(out, returned, "solveInto must return its destination")
                    assertClose(expected.data, out.data, "block n=$n nrhs=$nrhs t=$transpose")
                    // Aliased: solving in place over the caller's own matrix.
                    val aliased = DenseMatrix(n, nrhs, b.data.copyOf())
                    koblas.solveInto(lu, aliased, aliased, transpose, ws)
                    assertClose(expected.data, aliased.data, "block aliased n=$n nrhs=$nrhs t=$transpose")
                }
            }
        }
    }

    @Test
    fun `ldl block solves into a destination match the allocating form`() {
        val rng = Random(20260751)
        val n = 8
        val a = DenseMatrix(n, n)
        for (i in 0 until n) {
            for (j in 0..i) {
                var v = rng.nextDouble(-1.0, 1.0)
                if (i == j) v += if (i % 2 == 0) 2.0 else -2.0
                a[i, j] = v
                a[j, i] = v
            }
        }
        val f = koblas.ldl(a)
        for (nrhs in intArrayOf(1, 3, 40)) {
            val b = DenseMatrix(n, nrhs)
            for (i in 0 until n) for (j in 0 until nrhs) b[i, j] = rng.nextDouble(-1.0, 1.0)
            val expected = koblas.solve(f, b)
            val out = DenseMatrix(n, nrhs)
            assertClose(expected.data, koblas.solveInto(f, b, out).data, "ldl block nrhs=$nrhs")
            val aliased = DenseMatrix(n, nrhs, b.data.copyOf())
            koblas.solveInto(f, aliased, aliased)
            assertClose(expected.data, aliased.data, "ldl block aliased nrhs=$nrhs")
        }
    }

    @Test
    fun `the QR family writes into destinations`() {
        val rng = Random(20260752)
        val ws = Workspace()
        for ((m, n) in listOf(7 to 4, 6 to 6)) {
            val a = DenseMatrix(m, n)
            for (i in 0 until m) for (j in 0 until n) a[i, j] = rng.nextDouble(-1.0, 1.0)
            val f = koblas.qr(a, ws)
            val y = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
            for (transpose in booleanArrayOf(false, true)) {
                val expected = koblas.applyQ(f, y, transpose)
                assertClose(expected, koblas.applyQInto(f, y, DoubleArray(m), transpose), "applyQ t=$transpose")
                val aliased = y.copyOf()
                koblas.applyQInto(f, aliased, aliased, transpose)
                assertClose(expected, aliased, "applyQ aliased t=$transpose")
            }
            val ls = koblas.solveLeastSquares(f, y)
            assertClose(ls, koblas.solveLeastSquaresInto(f, y, DoubleArray(n), ws), "least squares ${m}x$n")
            // The minimum-norm solve reads the QR of a transpose, so it consumes a length-n right side.
            val bWide = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            val mn = koblas.solveMinimumNorm(f, bWide)
            assertClose(mn, koblas.solveMinimumNormInto(f, bWide, DoubleArray(m), ws), "min norm ${m}x$n")
        }
    }

    @Test
    fun `sparse solves into a destination match the allocating forms`() {
        val rng = Random(20260735)
        val m = 14
        val columns = List(m) { j ->
            val entries = ArrayList<Pair<Int, Double>>()
            entries.add(j to (rng.nextDouble(-1.0, 1.0) + m))
            for (i in 0 until m) if (i != j && rng.nextDouble() < 0.15) entries.add(i to rng.nextDouble(-1.0, 1.0))
            entries
        }
        val sparse = SparseMatrix.ofColumns(m, m, columns)
        val lu = SparseLu.factorize(sparse)!!
        val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
        val ws = Workspace()
        assertClose(lu.ftran(b), lu.ftranInto(b, DoubleArray(m), ws), "sparse ftran")
        assertClose(lu.btran(b), lu.btranInto(b, DoubleArray(m), ws), "sparse btran")
        // Reusing the same workspace must not carry state between the two directions.
        assertClose(lu.ftran(b), lu.ftranInto(b, DoubleArray(m), ws), "sparse ftran after btran")

        val eta = EtaBasis.of(lu, m)
        val spike = eta.ftran(DoubleArray(m) { rng.nextDouble(-1.0, 1.0) })
        eta.update(pivotRow = 3, spike = spike)
        assertClose(eta.ftran(b), eta.ftranInto(b, DoubleArray(m)), "eta ftran")
        assertClose(eta.btran(b), eta.btranInto(b, DoubleArray(m)), "eta btran")
        val aliased = b.copyOf()
        assertClose(eta.ftran(b), eta.ftranInto(aliased, aliased), "eta ftran aliased")
    }
}
