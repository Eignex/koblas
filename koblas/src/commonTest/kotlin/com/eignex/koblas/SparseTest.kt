package com.eignex.koblas

import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class SparseTest {

    @Test
    fun `CSC mat-vec multiplies a matrix and its transpose`() {
        // [[1, 0, 2], [0, 3, 0]]  (2x3), CSC by column.
        val a = SparseMatrix.ofColumns(
            rows = 2,
            cols = 3,
            columns = listOf(
                listOf(0 to 1.0),
                listOf(1 to 3.0),
                listOf(0 to 2.0),
            ),
        )
        assertTrue(doubleArrayOf(7.0, 9.0).contentEquals(a.gemv(doubleArrayOf(1.0, 3.0, 3.0))))
        assertTrue(doubleArrayOf(1.0, 6.0, 2.0).contentEquals(a.gemv(doubleArrayOf(1.0, 2.0), transpose = true)))
    }

    private fun randomSparseSquare(n: Int, rng: Random): SparseMatrix {
        val columns = List(n) { j ->
            val entries = ArrayList<Pair<Int, Double>>()
            entries.add(j to (rng.nextDouble(-2.0, 2.0) + n * 5.0)) // dominant diagonal ⇒ non-singular
            for (i in 0 until n) if (i != j && rng.nextDouble() < 0.3) entries.add(i to rng.nextDouble(-2.0, 2.0))
            entries
        }
        return SparseMatrix.ofColumns(n, n, columns)
    }

    @Test
    fun `SparseLu ftran and btran solve a random sparse system`() {
        val rng = Random(20260716)
        repeat(150) {
            val n = rng.nextInt(1, 10)
            val a = randomSparseSquare(n, rng)
            val lu = assertNotNull(SparseLu.factorize(a, equilibrate = true))
            val x = DoubleArray(n) { rng.nextDouble(-3.0, 3.0) }
            val b = a.gemv(x) // A x = b
            val solved = lu.ftran(b)
            for (i in 0 until n) assertTrue(abs(solved[i] - x[i]) < 1e-7, "ftran component $i: ${solved[i]} != ${x[i]}")
            val bt = a.gemv(x, transpose = true) // Aᵀ x = bt
            val solvedT = lu.btran(bt)
            for (i in 0 until n) {
                assertTrue(abs(solvedT[i] - x[i]) < 1e-7, "btran component $i: ${solvedT[i]} != ${x[i]}")
            }
        }
    }

    @Test
    fun `SparseLu reports a singular matrix`() {
        // A column of zeros ⇒ singular.
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf()))
        assertNull(SparseLu.factorize(a))
    }

    @Test
    fun `EtaBasis solves after a rank-1 basis update`() {
        // Start from the identity basis, then replace column p with a new column q; solve B x = b.
        val n = 4
        val base = assertNotNull(SparseLu.factorize(identity(n)))
        val eta = EtaBasis.of(base, n)

        val p = 2
        val q = doubleArrayOf(0.5, -1.0, 3.0, 2.0) // new column entering slot p
        val spike = eta.ftran(q) // spike = B⁻¹ q, computed before the update (B is still identity)
        eta.update(p, spike)
        assertEquals(1, eta.etaCount)

        // The updated basis B has identity columns except column p = q. Solve B x = b for a known x.
        val x = doubleArrayOf(1.0, -2.0, 4.0, 0.5)
        val b = DoubleArray(n)
        for (i in 0 until n) b[i] = if (i == p) 0.0 else x[i]
        for (i in 0 until n) b[i] += q[i] * x[p] // column p contributes q * x[p]
        val solved = eta.ftran(b)
        for (i in 0 until n) assertTrue(abs(solved[i] - x[i]) < 1e-9, "component $i: ${solved[i]} != ${x[i]}")
    }

    private fun identity(n: Int): SparseMatrix = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to 1.0) })
}
