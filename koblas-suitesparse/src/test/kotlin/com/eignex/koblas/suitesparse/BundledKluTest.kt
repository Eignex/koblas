package com.eignex.koblas.suitesparse

import com.eignex.koblas.*
import com.eignex.koblas.sparse.F64ReferenceSparseLinearAlgebra
import com.eignex.koblas.sparse.host.klu.*
import com.eignex.koblas.sparse.host.klu.KluSparseLu
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

        val factorization = BundledKlu(factorizeMin = 0, equilibrate = true).factor(matrix)

        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(16.0, 0.75)))
    }

    @Test
    fun `shared options control bundled KLU routing and diagnostics`() {
        val options = KluOptions(
            factorizeMin = 100,
            pivotTolerance = 0.25,
            useBtf = false,
            ordering = KluOrdering.COLAMD,
            haltIfSingular = true,
        )
        val backend = BundledKlu(options)

        val route = assertNotNull(backend.route(F64RouteQuery.SparseLu(storedEntries = 16)))

        assertEquals(BackendExecution.PORTABLE, route.execution)
        assertEquals("0.25", backend.backendMetadata.options["pivotTolerance"])
        assertEquals("false", backend.backendMetadata.options["useBtf"])
        assertEquals("COLAMD", backend.backendMetadata.options["ordering"])
        assertEquals("true", backend.backendMetadata.options["haltIfSingular"])
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

    /** A caller reaching this library by name gets its own routines only if the two share a type. */
    @Test
    fun `the bundled KLU is the binding rather than a wrapper around it`() {
        assertIs<KluSparseLu>(BundledKlu(), "refactor is on the binding rather than on a seam")
    }
}
