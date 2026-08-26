package com.eignex.koblas.sparse

import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseMatrix
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.withColumn
import kotlin.random.Random
import kotlin.test.*

class F64SparseFactorizationTest {
    private val reference = F64ReferenceSparseLinearAlgebra

    /** The fallback any backend can be given, for one that cannot update its own factors. */
    private fun refactoring(basis: F64SparseMatrix): F64BasisFactorization =
        F64RefactoringBasisFactorization(reference, basis, reference.factor(basis))

    @Test
    fun `the refactoring fallback rejects a rectangular basis`() {
        val wide = F64SparseMatrix.ofColumns(1, 2, listOf(listOf(0 to 1.0), listOf(0 to 2.0)))
        assertFailsWith<DimensionMismatch> { refactoring(wide) }
    }

    @Test
    fun `the refactoring fallback solves as the basis it started from`() {
        val rng = Random(20260825)
        val n = 9
        val basis = sparseConformanceSystem(n, rng)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

        val fromBasis = refactoring(basis).solve(b)

        assertClose(reference.factor(basis).solve(b), fromBasis, "the factored basis", tolerance = 1e-12)
    }

    @Test
    fun `the refactoring fallback follows a column replacement`() {
        val rng = Random(20260826)
        val n = 9
        val basis = sparseConformanceSystem(n, rng)
        val entering = F64SparseVector.of(n, intArrayOf(0, 4, 7), doubleArrayOf(3.0, -2.0, 11.0))
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

        val updated = refactoring(basis).replaceColumn(4, entering)

        assertEquals(basis.withColumn(4, entering), updated.basis)
        assertClose(reference.factor(updated.basis).solve(b), updated.solve(b), "one replacement", 1e-12)
    }

    @Test
    fun `the refactoring fallback follows a chain of replacements in both directions`() {
        val rng = Random(20260827)
        val n = 7
        var expected = sparseConformanceSystem(n, rng)
        var factorization = refactoring(expected)
        val b = DoubleArray(n) { rng.nextDouble(-1.0, 1.0) }

        for (column in 0 until n) {
            val entering = F64SparseVector.of(
                n,
                intArrayOf(column, (column + 3) % n),
                doubleArrayOf(n + 5.0, 1.0),
            )
            expected = expected.withColumn(column, entering)
            factorization = factorization.replaceColumn(column, entering)
        }

        for (transpose in booleanArrayOf(false, true)) {
            assertClose(
                reference.factor(expected).solve(b, transpose),
                factorization.solve(b, transpose),
                "chain transpose=$transpose",
                1e-12,
            )
        }
    }

    @Test
    fun `a replacement that makes the basis singular is reported rather than thrown`() {
        val basis = F64SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(1 to 1.0)))
        val duplicate = F64SparseVector.of(2, intArrayOf(0), doubleArrayOf(1.0))

        val updated = refactoring(basis).replaceColumn(1, duplicate)

        assertTrue(updated.singular, "a basis with two equal columns should have been called singular")
    }
}
