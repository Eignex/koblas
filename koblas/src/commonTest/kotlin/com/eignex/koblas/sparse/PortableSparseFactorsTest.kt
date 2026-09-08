package com.eignex.koblas.sparse

import com.eignex.koblas.SparseMatrix
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class PortableSparseFactorsTest {
    @Test
    fun `the LU factors reproduce the permuted matrix`() {
        val rng = Random(20260827)
        for (n in intArrayOf(1, 5, 24, 60)) {
            val a = sparseDominantSystem(n, rng)

            assertLuFactorsReproduce(a, F64ReferenceSparseLinearAlgebra.factor(a), "n=$n")
        }
    }

    @Test
    fun `an equilibrating LU reports the scaling its factors are of`() {
        val rng = Random(20260901)
        val a = sparseDominantSystem(30, rng)

        val lu = F64ReferenceSparseDecompositions(equilibrate = true).factor(a)

        assertLuFactorsReproduce(a, lu, "equilibrated")
    }

    @Test
    fun `the Cholesky factor reproduces the matrix`() {
        val rng = Random(20260902)
        for (n in intArrayOf(1, 6, 30)) {
            val a = sparseSymmetricConformanceSystem(n, rng)

            assertCholeskyFactorReproduces(a, F64ReferenceSparseLinearAlgebra.cholesky(a), "n=$n")
        }
    }

    @Test
    fun `the LDL factors reproduce the matrix`() {
        val rng = Random(20260903)
        for (n in intArrayOf(1, 6, 30)) {
            val a = sparseSymmetricConformanceSystem(n, rng)

            assertLdlFactorsReproduce(a, F64ReferenceSparseLinearAlgebra.quasiDefiniteLdl(a), "n=$n")
        }
    }

    @Test
    fun `a singular factorization has no factors to give`() {
        // Column 1 repeats column 0, so no acceptable pivot remains at the second step.
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0, 1 to 2.0), listOf(0 to 1.0, 1 to 2.0)))

        val lu = F64ReferenceSparseLinearAlgebra.factor(a)

        assertEquals(true, lu.singular)
        assertFailsWith<com.eignex.koblas.SingularMatrix> { lu.l }
        assertFailsWith<com.eignex.koblas.SingularMatrix> { lu.u }
        assertFailsWith<com.eignex.koblas.SingularMatrix> { lu.offDiagonal }
    }

    @Test
    fun `a singular LDL factorization has no factors to give`() {
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), emptyList()))

        val ldl = F64ReferenceSparseLinearAlgebra.quasiDefiniteLdl(a)

        assertEquals(true, ldl.singular)
        assertFailsWith<com.eignex.koblas.SingularMatrix> { ldl.l }
        assertFailsWith<com.eignex.koblas.SingularMatrix> { ldl.d }
        assertFailsWith<com.eignex.koblas.SingularMatrix> { ldl.order }
    }
}
