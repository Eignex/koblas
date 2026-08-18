package com.eignex.koblas.dense

import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.F64SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import com.eignex.koblas.norm1
import com.eignex.koblas.poisonedIndefinite
import com.eignex.koblas.sparse.lu
import com.eignex.koblas.wellConditioned
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertSame
import kotlin.test.assertTrue

class DestinationPassingTest {

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
            val (a, _) = poisonedIndefinite(rng, n)
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
        val second = wellConditioned(n, rng)
        val returned = koblas.factorInto(second, reused)
        assertSame(reused, returned, "factorInto must return its destination")
        assertSame(luBuffer, reused.lu, "factorInto must not replace the factor buffer")
        assertSame(pivBuffer, reused.piv, "factorInto must not replace the pivot buffer")
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        assertClose(koblas.solve(koblas.factor(second), b), koblas.solve(reused, b), "refactorized solve")
        val singular = F64DenseMatrix(n, n) // all zeros
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
            assertEquals(plain, koblas.rcond(lu, anorm, ws), "rcond n=$n first")
            assertEquals(plain, koblas.rcond(lu, anorm, ws), "rcond n=$n second")
        }
    }

    @Test
    fun `block solves into a destination match the allocating forms`() {
        val rng = Random(20260750)
        val ws = Workspace()
        for (n in intArrayOf(2, 9)) {
            // Both sides of the dispatch, column by column below the threshold and blocked above.
            for (nrhs in intArrayOf(1, 3, 40)) {
                val lu = wellConditioned(n, rng).lu()
                val b = F64DenseMatrix(n, nrhs)
                for (i in 0 until n) for (j in 0 until nrhs) b[i, j] = rng.nextDouble(-1.0, 1.0)
                for (transpose in booleanArrayOf(false, true)) {
                    val expected = koblas.solve(lu, b, transpose)
                    val out = F64DenseMatrix(n, nrhs)
                    val returned = koblas.solveInto(lu, b, out, transpose, ws)
                    assertSame(out, returned, "solveInto must return its destination")
                    assertClose(expected.data, out.data, "block n=$n nrhs=$nrhs t=$transpose")
                    val aliased = F64DenseMatrix(n, nrhs, b.data.copyOf())
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
        val (a, _) = poisonedIndefinite(rng, n)
        val f = koblas.ldl(a)
        for (nrhs in intArrayOf(1, 3, 40)) {
            val b = F64DenseMatrix(n, nrhs)
            for (i in 0 until n) for (j in 0 until nrhs) b[i, j] = rng.nextDouble(-1.0, 1.0)
            val expected = koblas.solve(f, b)
            val out = F64DenseMatrix(n, nrhs)
            assertClose(expected.data, koblas.solveInto(f, b, out).data, "ldl block nrhs=$nrhs")
            val aliased = F64DenseMatrix(n, nrhs, b.data.copyOf())
            koblas.solveInto(f, aliased, aliased)
            assertClose(expected.data, aliased.data, "ldl block aliased nrhs=$nrhs")
        }
    }

    @Test
    fun `the QR family writes into destinations`() {
        val rng = Random(20260752)
        val ws = Workspace()
        for ((m, n) in listOf(7 to 4, 6 to 6)) {
            val a = F64DenseMatrix(m, n)
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
        val sparse = F64SparseMatrix.ofColumns(m, m, columns)
        val lu = sparse.lu()
        val b = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
        val ws = Workspace()
        assertClose(lu.solve(b), lu.solveInto(b, DoubleArray(m), workspace = ws), "sparse forward solve")
        assertClose(
            lu.solve(b, transpose = true),
            lu.solveInto(b, DoubleArray(m), transpose = true, workspace = ws),
            "sparse transposed solve",
        )
        assertClose(
            lu.solve(b),
            lu.solveInto(b, DoubleArray(m), workspace = ws),
            "sparse forward solve after a transposed one",
        )
    }

    @Test
    fun `lu block solve tolerates a destination sharing the input's buffer`() {
        val rng = Random(20260753)
        val ws = Workspace()
        for (n in intArrayOf(2, 9)) {
            for (nrhs in intArrayOf(1, 3, 40)) {
                // Row-swapped so the factorization actually permutes: an identity pivot would make the
                // gather a no-op and the test blind to the aliasing it exists to cover.
                val a = wellConditioned(n, rng)
                for (j in 0 until n) {
                    val top = a[0, j]
                    a[0, j] = a[n - 1, j]
                    a[n - 1, j] = top
                }
                val lu = a.lu()
                assertTrue(lu.piv.withIndex().any { (k, p) -> k != p }, "n=$n must permute to exercise the gather")
                val b = F64DenseMatrix(n, nrhs)
                for (i in 0 until n) for (j in 0 until nrhs) b[i, j] = rng.nextDouble(-1.0, 1.0)
                for (transpose in booleanArrayOf(false, true)) {
                    val expected = koblas.solve(lu, b, transpose)
                    // Two views over one buffer: not the same object, but the same storage.
                    val shared = b.data.copyOf()
                    val rhs = F64DenseMatrix.wrap(n, nrhs, shared)
                    val out = F64DenseMatrix.wrap(n, nrhs, shared)
                    koblas.solveInto(lu, rhs, out, transpose, ws)
                    assertClose(expected.data, shared, "shared buffer n=$n nrhs=$nrhs t=$transpose")
                }
            }
        }
    }
}
