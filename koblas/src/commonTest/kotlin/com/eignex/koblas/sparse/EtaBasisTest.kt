package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.assertClose
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

/**
 * The product-form basis: a sparse LU plus a chain of rank-1 eta factors, which is how a simplex solver
 * carries a basis forward without refactorizing on every iteration.
 *
 * Each update replaces one basis column, and the eta factor is built from the spike `B⁻¹ q` computed
 * against the basis as it stood *before* the update. Getting that order wrong still produces a solvable
 * chain, so these tests track a dense copy of the basis alongside the chain and require the eta solves to
 * agree with it after several updates.
 */
class EtaBasisTest {

    private fun identity(n: Int): SparseMatrix = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to 1.0) })

    @Test
    fun `solves after a single rank-1 basis update`() {
        // Start from the identity basis, then replace column p with a new column q; solve B x = b.
        val n = 4
        val eta = EtaBasis.of(assertNotNull(SparseLu.factorize(identity(n))), n)

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
        assertClose(x, eta.ftran(b), "single update", tolerance = 1e-9)
    }

    @Test
    fun `solves and transpose-solves after multiple rank-1 updates`() {
        val n = 4
        val eta = EtaBasis.of(assertNotNull(SparseLu.factorize(identity(n))), n)

        // Track the dense basis alongside the eta chain: start from identity, replace columns 1 then 3.
        val basis = Array(n) { j -> DoubleArray(n) { i -> if (i == j) 1.0 else 0.0 } }
        val updates = listOf(1 to doubleArrayOf(2.0, 1.0, 0.0, -1.0), 3 to doubleArrayOf(0.5, 0.0, 3.0, 2.0))
        for ((p, q) in updates) {
            eta.update(p, eta.ftran(q))
            for (i in 0 until n) basis[i][p] = q[i]
        }
        assertEquals(2, eta.etaCount)

        val x = doubleArrayOf(1.0, -2.0, 4.0, 0.5)
        val b = DoubleArray(n) { i -> (0 until n).sumOf { j -> basis[i][j] * x[j] } } // B x
        assertClose(x, eta.ftran(b), "ftran after two updates", tolerance = 1e-9)

        val bt = DoubleArray(n) { j -> (0 until n).sumOf { i -> basis[i][j] * x[i] } } // Bᵀ x
        assertClose(x, eta.btran(bt), "btran after two updates", tolerance = 1e-9)
    }
}
