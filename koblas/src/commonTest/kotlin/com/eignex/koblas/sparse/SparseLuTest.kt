package com.eignex.koblas.sparse

import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.randomVector
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * The sparse LU: factorization, the FTRAN and BTRAN solves, the determinant, and the singular and malformed cases.
 */
class SparseLuTest {

    /** A square CSC matrix from dense rows, dropping the zeros. */
    private fun square(vararg rows: DoubleArray): SparseMatrix {
        val n = rows.size
        val cols = List(n) { j -> (0 until n).mapNotNull { i -> if (rows[i][j] != 0.0) i to rows[i][j] else null } }
        return SparseMatrix.ofColumns(n, n, cols)
    }

    /**
     * A random square matrix with [density] fill off the diagonal and a diagonal dominant by a factor of [dominance],
     * which makes it non-singular. Lower dominance leaves the pivot search more work to do.
     */
    private fun randomSparseSquare(n: Int, rng: Random, density: Double = 0.3, dominance: Double = 5.0): SparseMatrix {
        val columns = List(n) { j ->
            val entries = ArrayList<Pair<Int, Double>>()
            entries.add(j to (rng.nextDouble(-2.0, 2.0) + n * dominance))
            for (i in 0 until n) if (i != j && rng.nextDouble() < density) entries.add(i to rng.nextDouble(-2.0, 2.0))
            entries
        }
        return SparseMatrix.ofColumns(n, n, columns)
    }

    @Test
    fun `ftran and btran solve a random sparse system`() {
        val rng = Random(20260716)
        repeat(150) {
            val n = rng.nextInt(1, 10)
            val a = randomSparseSquare(n, rng)
            val lu = assertNotNull(a.lu(equilibrate = true))
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            assertClose(x, lu.solve(a.gemv(x)), "forward solve n=$n", tolerance = 1e-7)
            assertClose(
                x,
                lu.solve(a.gemv(x, transpose = true), transpose = true),
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
        val lu = assertNotNull(a.lu(equilibrate = true))
        val b = randomVector(n, rng)
        assertClose(b, a.gemv(lu.solve(b)), "ftran residual", tolerance = 1e-8)
        assertClose(b, a.gemv(lu.solve(b, transpose = true), transpose = true), "btran residual", tolerance = 1e-8)
        // The bounded candidate search must stay far from dense. Dense here is n² = 90000, roughly 44x the
        // input; the search holds it near 12x. The bound is generous because random sparsity is the hard
        // case for any ordering — there is no structure to exploit — so this catches a search that has
        // stopped reducing fill, not a few per cent of drift.
        assertTrue(lu.nnz < 15 * a.nnz, "fill blew up: factor nnz=${lu.nnz} vs input nnz=${a.nnz}")
        // And what the drop tolerance buys on the same matrix, which is the trade a caller is choosing
        // between: markedly sparser factors for a residual four orders worse.
        val dropped = a.lu(equilibrate = true, dropTolerance = 1e-9)
        assertTrue(
            dropped.nnz < lu.nnz / 2,
            "the drop tolerance should cut fill sharply here: ${dropped.nnz} against ${lu.nnz}",
        )
    }

    @Test
    fun `solve is correct with equilibration off`() {
        val rng = Random(22)
        repeat(60) {
            val n = rng.nextInt(1, 8)
            val a = randomSparseSquare(n, rng, density = 1.0)
            val lu = assertNotNull(a.lu(equilibrate = false))
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            assertClose(x, lu.solve(a.gemv(x)), "unequilibrated n=$n", tolerance = 1e-7)
        }
    }

    @Test
    fun `determinant matches the hand value and tracks sign under row order`() {
        // [[2, 1], [1, 3]] → det 5.
        val det = square(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)).lu().determinant()
        assertClose(5.0, det, "det", tolerance = 1e-9)
        // A row swap negates the determinant: [[1,3],[2,1]] → det = 1 - 6 = -5.
        val det2 = square(doubleArrayOf(1.0, 3.0), doubleArrayOf(2.0, 1.0)).lu().determinant()
        assertClose(-5.0, det2, "det after row swap", tolerance = 1e-9)
    }

    @Test
    fun `determinant is invariant to equilibration`() {
        val rng = Random(11)
        repeat(30) {
            val n = rng.nextInt(2, 6)
            val a = randomSparseSquare(n, rng, density = 1.0, dominance = 4.0)
            val d0 = assertNotNull(a.lu(equilibrate = false)).determinant()
            val d1 = assertNotNull(a.lu(equilibrate = true)).determinant()
            assertTrue(abs(d0 - d1) <= 1e-6 * (abs(d0) + 1.0), "det differs: $d0 vs $d1")
        }
    }

    /** A singular matrix comes back as a factorization reporting `singular`, not as null. */
    @Test
    fun `a singular matrix factors to a singular factorization`() {
        // A column of zeros, and a duplicated column: neither has a full set of acceptable pivots.
        for (a in listOf(
            SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf())),
            square(doubleArrayOf(1.0, 1.0), doubleArrayOf(2.0, 2.0)),
        )) {
            val f = a.lu()
            assertTrue(f.singular, "expected singular for $a")
            assertEquals(2, f.n)
            assertEquals(0, f.nnz, "a singular factorization has no fill")
            assertEquals(0.0, f.determinant())
            assertFailsWith<SingularMatrix> { f.solve(doubleArrayOf(1.0, 2.0)) }
        }
        // The failing pivot POSITION is koblas's own to report, so this goes to the portable factorization
        // directly rather than through the seam. A host backend need not know where it failed — UMFPACK
        // reports only that the matrix is singular, hence SINGULAR_POSITION_UNKNOWN — and asserting a
        // position against whichever backend happens to be installed would be asserting the machine.
        val singular = ReferenceSparseLinearAlgebra.factor(
            SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf())),
        )
        assertTrue(singular is SingularSparseFactorization)
        assertEquals(1, singular.failedAt, "the second pivot is the one with no candidate")
    }

    @Test
    fun `factorize rejects a non-square matrix`() {
        val a = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 1.0), listOf(0 to 1.0)))
        assertFailsWith<IllegalArgumentException> { a.lu() }
    }

    /**
     * The same matrix at three scales must factor the same way, which an absolute pivot floor cannot do: a
     * well-conditioned matrix whose entries all sit near `1e-12` is not singular, and one near `1e12` is not
     * automatically well-pivoted. Both tolerances are fractions of the largest magnitude present for this reason, so
     * scaling by a power of two — exact in floating point — changes nothing but the scale.
     */
    @Test
    fun `factorization is invariant to the matrix scale`() {
        val rng = Random(20260807)
        val base = randomSparseSquare(6, rng, dominance = 1.5)
        val rhs = randomVector(6, rng)
        // Through the portable factorization directly: these are koblas's own tolerances, and a machine
        // with SuiteSparse installed would otherwise be measuring UMFPACK's scale handling instead.
        val reference = ReferenceSparseLinearAlgebra.factor(base)
        // Powers of two, so the scaling itself is exact and any difference is the factorization's. Far
        // enough out that an absolute tolerance would break: at 2^-60 every entry sits below 1e-14, which
        // an absolute zero test would call singular, and at 2^60 an absolute drop would discard nothing.
        for (scale in doubleArrayOf(POW_2_MINUS_60, POW_2_60)) {
            val scaled = SparseMatrix.ofColumns(
                6,
                6,
                List(6) { j -> buildList { base.forEachInColumn(j) { i, v -> add(i to v * scale) } } },
            )
            val f = ReferenceSparseLinearAlgebra.factor(scaled)
            assertTrue(!f.singular, "scaling by $scale must not make a matrix singular")
            assertEquals(reference.nnz, f.nnz, "scaling by $scale changed the fill")
            // A·x = b scaled by s solves to x/s, so the scaled solve times the scale is the original.
            val expected = reference.solve(rhs)
            val actual = f.solve(rhs).map { it * scale }.toDoubleArray()
            assertClose(expected, actual, "solve at scale $scale", tolerance = 1e-9)
        }
    }

    /** The drop tolerance is off unless asked for, and asking for it trades accuracy for sparsity. */
    @Test
    fun `the drop tolerance trades fill for accuracy and defaults to off`() {
        val rng = Random(20260807)
        val a = randomSparseSquare(40, rng, density = 0.2, dominance = 1.0)
        val rhs = randomVector(40, rng)
        // Both through the portable path: a drop tolerance falls back to it, so an installed host backend
        // would make the baseline a different factorization and the comparison meaningless.
        val exact = ReferenceSparseLinearAlgebra.factor(a)
        val dropped = ReferenceSparseLinearAlgebra.factor(a, dropTolerance = 1e-3)
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
        assertFailsWith<IllegalArgumentException> { randomSparseSquare(3, Random(1)).lu(dropTolerance = -1e-9) }
    }

    /** A value that cancels to exactly zero is dropped from the pattern, with no drop tolerance involved. */
    @Test
    fun `entries that cancel exactly are not stored`() {
        // The leading 2x2 block is all ones, so whichever of its four entries the pivot search takes, the
        // update against it drives another of them to exactly zero. Built that way on purpose: a matrix where
        // the cancellation depends on which pivot is chosen makes this test a hostage to the search's
        // tie-breaking, which is what it was until the bounded search began exiting earlier.
        val a = square(
            doubleArrayOf(1.0, 1.0, 0.0),
            doubleArrayOf(1.0, 1.0, 1.0),
            doubleArrayOf(0.0, 1.0, 1.0),
        )
        val lu = ReferenceSparseLinearAlgebra.factor(a)
        assertEquals(6, lu.nnz, "the cancelled entry must not be stored")
        assertClose(doubleArrayOf(1.0, 2.0, 3.0), lu.solve(a.gemv(doubleArrayOf(1.0, 2.0, 3.0))), "solve")
    }

    /** `max |A·x − b|` for the solve this factorization gives. */
    private fun residualOf(a: SparseMatrix, f: SparseFactorization, rhs: DoubleArray): Double {
        val residual = a.gemv(f.solve(rhs))
        var worst = 0.0
        for (i in rhs.indices) worst = maxOf(worst, abs(residual[i] - rhs[i]))
        return worst
    }
}

/** `2^-60` and `2^60`: exact in floating point, and far enough from 1 to expose an absolute tolerance. */
private const val POW_2_MINUS_60 = 8.673617379884035e-19
private const val POW_2_60 = 1.152921504606847e18
