package com.eignex.koblas.dense

import com.eignex.koblas.*
import com.eignex.koblas.core.F64DenseMatrix
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.lu
import kotlin.random.Random
import kotlin.test.*

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
            val f = koblas.pivotedSymmetricIndefinite(a)
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
        val pivBuffer = reused.mutablePivots
        val second = wellConditioned(n, rng)
        val returned = koblas.factorInto(second, reused)
        assertSame(reused, returned, "factorInto must return its destination")
        assertSame(luBuffer, reused.lu, "factorInto must not replace the factor buffer")
        assertSame(pivBuffer, reused.mutablePivots, "factorInto must not replace the pivot buffer")
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
            val anorm = a.norm1()
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
        val f = koblas.pivotedSymmetricIndefinite(a)
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
            val ls = koblas.solve(f, y)
            assertClose(ls, koblas.solveInto(f, y, DoubleArray(n), workspace = ws), "least squares ${m}x$n")
            val bWide = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
            val mn = koblas.solve(f, bWide, minimumNorm = true)
            assertClose(
                mn,
                koblas.solveInto(f, bWide, DoubleArray(m), minimumNorm = true, workspace = ws),
                "min norm ${m}x$n",
            )
        }
    }

    @Test
    fun `LU decomposition-receiver solveInto matches the koblas dispatch form aliased or not`() {
        val rng = Random(20260760)
        val n = 11
        val lu = wellConditioned(n, rng).lu()
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        for (transpose in booleanArrayOf(false, true)) {
            val expected = koblas.solve(lu, b, transpose)
            assertClose(expected, lu.solveInto(b, DoubleArray(n), transpose), "lu receiver n=$n t=$transpose")
            val aliased = b.copyOf()
            lu.solveInto(aliased, aliased, transpose)
            assertClose(expected, aliased, "lu receiver aliased t=$transpose")
        }
        val ws = Workspace()
        val nrhs = 3
        val block = F64DenseMatrix(n, nrhs)
        for (i in 0 until n) for (j in 0 until nrhs) block[i, j] = rng.nextDouble(-1.0, 1.0)
        val expectedBlock = koblas.solve(lu, block)
        val out = F64DenseMatrix(n, nrhs)
        assertSame(out, lu.solveInto(block, out, workspace = ws), "lu receiver block must return its destination")
        assertClose(expectedBlock.data, out.data, "lu receiver block")
        val aliasedBlock = F64DenseMatrix(n, nrhs, block.data.copyOf())
        lu.solveInto(aliasedBlock, aliasedBlock, workspace = ws)
        assertClose(expectedBlock.data, aliasedBlock.data, "lu receiver block aliased")
    }

    @Test
    fun `LDL decomposition-receiver solveInto matches the koblas dispatch form aliased or not`() {
        val rng = Random(20260761)
        val n = 9
        val (a, _) = poisonedIndefinite(rng, n)
        val f = koblas.pivotedSymmetricIndefinite(a)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val expected = koblas.solve(f, b)
        assertClose(expected, f.solveInto(b, DoubleArray(n)), "ldl receiver n=$n")
        val aliased = b.copyOf()
        f.solveInto(aliased, aliased)
        assertClose(expected, aliased, "ldl receiver aliased n=$n")

        val nrhs = 3
        val block = F64DenseMatrix(n, nrhs)
        for (i in 0 until n) for (j in 0 until nrhs) block[i, j] = rng.nextDouble(-1.0, 1.0)
        val expectedBlock = koblas.solve(f, block)
        val out = F64DenseMatrix(n, nrhs)
        assertSame(out, f.solveInto(block, out), "ldl receiver block must return its destination")
        assertClose(expectedBlock.data, out.data, "ldl receiver block")
    }

    @Test
    fun `Cholesky decomposition-receiver solveInto matches the koblas dispatch form aliased or not`() {
        val rng = Random(20260762)
        val n = 8
        val chol = wellConditioned(n, rng).cholesky()
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val expected = koblas.solve(chol, b)
        assertClose(expected, chol.solveInto(b, DoubleArray(n)), "cholesky receiver n=$n")
        val aliased = b.copyOf()
        chol.solveInto(aliased, aliased)
        assertClose(expected, aliased, "cholesky receiver aliased n=$n")

        val nrhs = 3
        val block = F64DenseMatrix(n, nrhs)
        for (i in 0 until n) for (j in 0 until nrhs) block[i, j] = rng.nextDouble(-1.0, 1.0)
        val expectedBlock = koblas.solve(chol, block)
        val out = F64DenseMatrix(n, nrhs)
        assertSame(out, chol.solveInto(block, out), "cholesky receiver block must return its destination")
        assertClose(expectedBlock.data, out.data, "cholesky receiver block")
    }

    @Test
    fun `QR decomposition-receiver applyQInto and solveInto match the koblas dispatch form`() {
        val rng = Random(20260763)
        val m = 7
        val n = 4
        val a = F64DenseMatrix(m, n)
        for (i in 0 until m) for (j in 0 until n) a[i, j] = rng.nextDouble(-1.0, 1.0)
        val f = koblas.qr(a)
        val y = DoubleArray(m) { rng.nextDouble(-1.0, 1.0) }
        for (transpose in booleanArrayOf(false, true)) {
            val expected = koblas.applyQ(f, y, transpose)
            assertClose(expected, f.applyQInto(y, DoubleArray(m), transpose), "qr receiver applyQ t=$transpose")
            val aliased = y.copyOf()
            f.applyQInto(aliased, aliased, transpose)
            assertClose(expected, aliased, "qr receiver applyQ aliased t=$transpose")
        }
        val expectedSolve = koblas.solve(f, y)
        assertClose(expectedSolve, f.solveInto(y, DoubleArray(n)), "qr receiver solveInto")

        val pivoted = a.qrPivoted()
        val expectedPivoted = koblas.solve(pivoted, y)
        assertClose(expectedPivoted, pivoted.solveInto(y, DoubleArray(n)), "pivoted qr receiver solveInto")
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
                assertTrue(
                    lu.rowOrder().withIndex().any { (k, p) -> k != p },
                    "n=$n must permute to exercise the gather",
                )
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

    @Test
    fun `choleskyInto tolerates the destination sharing the input buffer`() {
        val rng = Random(20260805)
        for (n in intArrayOf(1, 7, 16)) {
            val a = wellConditioned(n, rng)
            val expected = a.cholesky()
            // One buffer behind both the input and the factor, which is the in-place dpotrf call shape.
            val shared = a.data.copyOf()
            val input = F64DenseMatrix.wrap(n, n, shared)
            val destination = F64CholeskyDecomposition(F64DenseMatrix.wrap(n, n, shared))
            assertSame(destination, koblas.choleskyInto(input, destination), "choleskyInto returns its destination")
            assertClose(expected.l.data, destination.l.data, "aliased choleskyInto n=$n")
        }
    }

    @Test
    fun `qrInto tolerates the destination sharing the input buffer`() {
        val rng = Random(20260806)
        val m = 9
        val n = 5
        val a = F64DenseMatrix(m, n)
        for (i in 0 until m) for (j in 0 until n) a[i, j] = rng.nextDouble(-1.0, 1.0)
        val expected = koblas.qr(a)
        val shared = a.data.copyOf()
        val input = F64DenseMatrix.wrap(m, n, shared)
        val destination = F64QrDecomposition(m, n, shared, DoubleArray(minOf(m, n)))
        assertSame(destination, koblas.qrInto(input, destination), "qrInto returns its destination")
        assertClose(expected.qr, destination.qr, "aliased qrInto")
        assertClose(expected.tau, destination.tau, "aliased qrInto tau")
    }
}
