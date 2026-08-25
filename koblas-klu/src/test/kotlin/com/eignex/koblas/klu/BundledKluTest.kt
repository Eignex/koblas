package com.eignex.koblas.klu

import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import kotlin.test.*

class BundledKluTest {
    @Test
    fun `the bundled KLU is available`() {
        assertTrue(BundledKlu(factorizeMin = 0).isAvailable)
    }

    @Test
    fun `the bundled KLU solves sparse systems in both directions`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(1 to 2.0), listOf(0 to 3.0)))

        val factorization = BundledKlu(factorizeMin = 0).factor(matrix)

        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(9.0, 4.0)))
        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(6.0, 6.0), transpose = true))
    }

    @Test
    fun `the bundled KLU reports reciprocal pivot condition`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 4.0)))

        assertEquals(0.25, BundledKlu(factorizeMin = 0).factor(matrix).rcond)
    }

    @Test
    fun `the bundled KLU honors equilibration`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 8.0), listOf(1 to 0.25)))

        val factorization = BundledKlu(factorizeMin = 0).factor(matrix, equilibrate = true)

        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(16.0, 0.75)))
    }

    @Test
    fun `the bundled KLU reports a singular matrix`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(0 to 2.0)))

        assertTrue(BundledKlu(factorizeMin = 0).factor(matrix).singular)
    }

    @Test
    fun `the bundled KLU refactors a basis for another solve`() {
        val first = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 3.0)))
        val second = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 4.0), listOf(1 to 5.0)))
        val klu = BundledKlu(factorizeMin = 0)
        val factorization = klu.factor(first)

        val refactored = klu.refactor(factorization, second)

        val rhs = doubleArrayOf(12.0, 20.0)
        assertContentEquals(
            F64ReferenceSparseLinearAlgebra.factor(second).solve(rhs),
            refactored.solve(rhs),
        )
    }

    @Test
    fun `the bundled KLU marks a singular refactorization unsolvable`() {
        val nonsingular = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0)))
        val singular = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 4.0, 1 to 2.0)))
        val klu = BundledKlu(factorizeMin = 0)
        val factorization = klu.factor(nonsingular)

        val refactored = klu.refactor(factorization, singular)

        assertTrue(refactored.singular)
        assertFailsWith<SingularMatrix> { refactored.solve(doubleArrayOf(1.0, 1.0)) }
    }

    @Test
    fun `the bundled KLU retains a factorization when refactorization changes its pattern`() {
        val diagonal = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 3.0)))
        val full = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0, 1 to 1.0), listOf(0 to 1.0, 1 to 3.0)))
        val klu = BundledKlu(factorizeMin = 0)
        val factorization = klu.factor(diagonal)

        val refactored = klu.refactor(factorization, full)

        val rhs = doubleArrayOf(4.0, 9.0)
        val expected = F64ReferenceSparseLinearAlgebra.factor(full).solve(rhs)
        val actual = refactored.solve(rhs)
        for (i in expected.indices) assertEquals(expected[i], actual[i], 1e-12)
    }
}
