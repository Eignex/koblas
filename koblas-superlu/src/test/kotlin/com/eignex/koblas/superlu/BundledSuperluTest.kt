package com.eignex.koblas.superlu

import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.sparse.host.superlu.SuperluDriver
import com.eignex.koblas.sparse.lu
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BundledSuperluTest {
    @Test
    fun `the bundled SuperLU is available`() {
        assertTrue(BundledSuperlu().isAvailable)
    }

    @Test
    fun `the bundled SuperLU solves a sparse system`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 3.0)))

        val solution = BundledSuperlu().factor(matrix).solve(doubleArrayOf(4.0, 9.0))

        assertContentEquals(doubleArrayOf(2.0, 3.0), solution)
    }

    @Test
    fun `the bundled SuperLU computes a permuted determinant`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(1 to 2.0), listOf(0 to 3.0)))

        val determinant = matrix.lu().determinant()

        assertEquals(-6.0, determinant)
    }

    @Test
    fun `the expert driver honors equilibration`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 8.0), listOf(1 to 0.25)))

        val factorization = BundledSuperlu(SuperluDriver.Expert).factor(matrix, equilibrate = true)

        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(16.0, 0.75)))
        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(16.0, 0.75), transpose = true))
        assertEquals(2.0, factorization.determinant())
    }

    @Test
    fun `the standard driver remains selectable`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 2.0), listOf(1 to 3.0)))

        val solution = BundledSuperlu(SuperluDriver.Standard).factor(matrix).solve(doubleArrayOf(4.0, 9.0))

        assertContentEquals(doubleArrayOf(2.0, 3.0), solution)
    }

    @Test
    fun `the bundled SuperLU reports a singular matrix`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(0 to 2.0)))

        val factorization = matrix.lu()

        assertTrue(factorization.singular)
    }
}
