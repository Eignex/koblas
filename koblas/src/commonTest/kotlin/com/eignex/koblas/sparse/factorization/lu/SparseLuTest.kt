package com.eignex.koblas.sparse.factorization.lu

import com.eignex.koblas.*
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.sparse.*
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.*

class SparseLuTest {

    private fun square(vararg rows: DoubleArray): F64SparseMatrix {
        val n = rows.size
        val cols = List(n) { j -> (0 until n).mapNotNull { i -> if (rows[i][j] != 0.0) i to rows[i][j] else null } }
        return F64SparseMatrix.ofColumns(n, n, cols)
    }

    private fun randomSparseSquare(
        n: Int,
        rng: Random,
        density: Double = 0.3,
        dominance: Double = 5.0,
    ): F64SparseMatrix {
        val columns = List(n) { j ->
            val entries = ArrayList<Pair<Int, Double>>()
            entries.add(j to (rng.nextDouble(-2.0, 2.0) + n * dominance))
            for (i in 0 until n) if (i != j && rng.nextDouble() < density) entries.add(i to rng.nextDouble(-2.0, 2.0))
            entries
        }
        return F64SparseMatrix.ofColumns(n, n, columns)
    }

    @Test
    fun `the seam solves from a factorization the way the dense side does`() {
        val a = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0)))
        val f = a.lu()
        val b = doubleArrayOf(3.0, 4.0)
        assertClose(f.solve(b), koblas.solve(f, b), "seam solve", tolerance = 1e-12)
        val out = DoubleArray(2)
        assertClose(f.solve(b), koblas.solveInto(f, b, out), "seam solveInto", tolerance = 1e-12)
    }

    @Test
    fun `ftran and btran solve a random sparse system`() {
        val rng = Random(20260716)
        repeat(150) {
            val n = rng.nextInt(1, 10)
            val a = randomSparseSquare(n, rng)
            // The reference explicitly: this is a test of its elimination, and the fill it is compared against
            // below is the reference's own. Through the seam a host backend would answer with different fill.
            val lu = F64ReferenceSparseDecompositions(equilibrate = true).factor(a)
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            assertClose(x, lu.solve(koblas.gemv(a, x)), "forward solve n=$n", tolerance = 1e-7)
            assertClose(
                x,
                lu.solve(koblas.gemv(a, x, transpose = true), transpose = true),
                "transposed solve n=$n",
                tolerance = 1e-7,
            )
        }
    }

    @Test
    fun `bounded pivot search stays accurate and sparse on a larger random system`() {
        val rng = Random(20260727)
        val n = 300
        val a = randomSparseSquare(n, rng, density = 0.02, dominance = 1.0)
        // The reference explicitly: this is a test of its elimination, and the fill it is compared against
        // below is the reference's own. Through the seam a host backend would answer with different fill.
        val lu = F64ReferenceSparseDecompositions(equilibrate = true).factor(a)
        val b = randomVector(n, rng)
        assertClose(b, koblas.gemv(a, lu.solve(b)), "ftran residual", tolerance = 1e-8)
        assertClose(
            b,
            koblas.gemv(a, lu.solve(b, transpose = true), transpose = true),
            "btran residual",
            tolerance = 1e-8,
        )
        // The bound is loose because random sparsity gives the ordering no structure to exploit.
        assertTrue(lu.nnz < 15 * a.nnz, "fill blew up: factor nnz=${lu.nnz} vs input nnz=${a.nnz}")
        val dropped = F64ReferenceSparseDecompositions(equilibrate = true, dropTolerance = 1e-9).factor(a)
        assertTrue(
            dropped.nnz < lu.nnz / 2,
            "the drop tolerance should cut fill sharply here: ${dropped.nnz} against ${lu.nnz}",
        )
    }

    @Test
    fun `equilibration leaves a row it cannot scale into range alone`() {
        val a = F64SparseMatrix(2, 2, intArrayOf(0, 1, 2), intArrayOf(0, 1), doubleArrayOf(1e-320, 2e-320))
        val plain = assertNotNull(F64ReferenceSparseDecompositions().factor(a))
        val equilibrated = assertNotNull(F64ReferenceSparseDecompositions(equilibrate = true).factor(a))
        assertEquals(plain.singular, equilibrated.singular, "equilibration changed the singularity verdict")
        assertEquals(plain.failedAt, equilibrated.failedAt, "equilibration changed the reported position")
    }

    @Test
    fun `solve is correct with equilibration off`() {
        val rng = Random(22)
        repeat(60) {
            val n = rng.nextInt(1, 8)
            val a = randomSparseSquare(n, rng, density = 1.0)
            val lu = assertNotNull(F64ReferenceSparseDecompositions().factor(a))
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            assertClose(x, lu.solve(koblas.gemv(a, x)), "unequilibrated n=$n", tolerance = 1e-7)
        }
    }

    @Test
    fun `the reciprocal pivot condition estimate reads the U diagonal range`() {
        val matrix = square(doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 4.0))

        assertEquals(0.25, F64ReferenceSparseLinearAlgebra.factor(matrix).rcond)
    }

    @Test
    fun `a singular matrix factors to a singular factorization`() {
        // A column of zeros and a duplicated column, neither with a full set of acceptable pivots.
        for (a in listOf(
            F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf())),
            square(doubleArrayOf(1.0, 1.0), doubleArrayOf(2.0, 2.0)),
        )) {
            val f = a.lu()
            assertTrue(f.singular, "expected singular for $a")
            assertEquals(2, f.n)
            assertEquals(0, f.nnz, "a singular factorization has no fill")
            assertEquals(0.0, f.rcond)
            assertFailsWith<SingularMatrix> { f.solve(doubleArrayOf(1.0, 2.0)) }
        }
        // The failing pivot position is koblas's own, so this uses the portable factorization. UMFPACK reports only
        // that the matrix is singular, so asserting a position against an installed backend would assert the machine.
        val singular = F64ReferenceSparseLinearAlgebra.factor(
            F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf())),
        )
        assertTrue(singular is F64SingularSparseFactorization)
        assertEquals(1, singular.failedAt, "the second pivot is the one with no candidate")
    }

    @Test
    fun `factorize rejects a non-square matrix`() {
        val a = F64SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 1.0), listOf(0 to 1.0)))
        assertFailsWith<IllegalArgumentException> { a.lu() }
    }

    @Test
    fun `factorization is invariant to the matrix scale`() {
        val rng = Random(20260807)
        val base = randomSparseSquare(6, rng, dominance = 1.5)
        val rhs = randomVector(6, rng)
        // These are koblas's own tolerances, so this uses the portable factorization rather than an installed UMFPACK.
        val reference = F64ReferenceSparseLinearAlgebra.factor(base)
        for (scale in doubleArrayOf(POW_2_MINUS_60, POW_2_60)) {
            val scaled = F64SparseMatrix.ofColumns(
                6,
                6,
                List(6) { j -> buildList { base.forEachInColumn(j) { i, v -> add(i to v * scale) } } },
            )
            val f = F64ReferenceSparseLinearAlgebra.factor(scaled)
            assertTrue(!f.singular, "scaling by $scale must not make a matrix singular")
            assertEquals(reference.nnz, f.nnz, "scaling by $scale changed the fill")
            val expected = reference.solve(rhs)
            val actual = f.solve(rhs).map { it * scale }.toDoubleArray()
            assertClose(expected, actual, "solve at scale $scale", tolerance = 1e-9)
        }
    }

    @Test
    fun `the drop tolerance trades fill for accuracy and defaults to off`() {
        val rng = Random(20260807)
        val a = randomSparseSquare(40, rng, density = 0.2, dominance = 1.0)
        val rhs = randomVector(40, rng)
        // A drop tolerance falls back to the portable path, so the baseline takes it too rather than a host backend.
        val exact = F64ReferenceSparseLinearAlgebra.factor(a)
        val dropped = F64ReferenceSparseDecompositions(dropTolerance = 1e-3).factor(a)
        assertTrue(
            dropped.nnz < exact.nnz,
            "a drop tolerance must discard fill: ${dropped.nnz} against ${exact.nnz}",
        )
        assertTrue(residualOf(a, exact, rhs) < 1e-12, "the default factorization is a real one")
        assertTrue(
            residualOf(a, dropped, rhs) > residualOf(a, exact, rhs),
            "an incomplete factorization cannot be as accurate as the complete one",
        )
    }

    @Test
    fun `a negative drop tolerance is rejected`() {
        assertFailsWith<IllegalArgumentException> {
            F64ReferenceSparseDecompositions(dropTolerance = -1e-9).factor(randomSparseSquare(3, Random(1)))
        }
    }

    @Test
    fun `entries that cancel exactly are not stored`() {
        val a = square(
            doubleArrayOf(1.0, 1.0, 0.0),
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(0.0, 1.0, 1.0),
        )
        val lu = F64ReferenceSparseLinearAlgebra.factor(a)
        assertEquals(6, lu.nnz, "the cancelled entry must not be stored")
        assertClose(doubleArrayOf(1.0, 2.0, 3.0), lu.solve(koblas.gemv(a, doubleArrayOf(1.0, 2.0, 3.0))), "solve")
    }

    private fun residualOf(a: F64SparseMatrix, f: F64SparseFactorization, rhs: DoubleArray): Double {
        val residual = koblas.gemv(a, f.solve(rhs))
        var worst = 0.0
        for (i in rhs.indices) worst = maxOf(worst, abs(residual[i] - rhs[i]))
        return worst
    }

    @Test
    fun `L rows stay ordered when the pivot order permutes them`() {
        // lRowsOf transposes the per-step multipliers into row space by counting rather than by hashing and
        // sorting, which relies on the step loop running ascending. A permuting pivot order is what tells a
        // correct transpose from one that only looks right when invPerm is the identity.
        val rng = Random(20260960)
        val n = 40
        val columns = List(n) { j ->
            val rows = (listOf((j + n / 2) % n, j) + List(3) { rng.nextInt(n) }).distinct().sorted()
            rows.map { i -> i to if (i == (j + n / 2) % n) 8.0 + rng.nextDouble() else rng.nextDouble(-1.0, 1.0) }
        }
        val a = F64SparseMatrix.ofColumns(n, n, columns)
        val factorization = F64SparseMarkowitzLu.factorCsc(a)
        assertTrue(!factorization.singular, "the fixture should factor")

        // Building `l` validates CSC, so an out-of-order transpose cannot survive this.
        val l = factorization.l
        for (j in 0 until l.cols) {
            var previous = -1
            l.forEachInColumn(j) { i, _ ->
                assertTrue(i > previous, "column $j of L has row $i after $previous")
                previous = i
            }
        }
        val expected = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }
        val b = DoubleArray(n)
        for (j in 0 until n) a.forEachInColumn(j) { i, v -> b[i] += v * expected[j] }

        assertClose(expected, factorization.solve(b), "permuted solve", tolerance = 1e-8)
    }
}

/** 2^-60 and 2^60, exact in floating point and far enough from 1 to expose an absolute tolerance. */
private const val POW_2_MINUS_60 = 8.673617379884035e-19
private const val POW_2_60 = 1.152921504606847e18
