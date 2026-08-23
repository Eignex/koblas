package com.eignex.koblas.klu

import com.eignex.koblas.SparseMatrix
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertTrue

class BundledKluTest {
    @Test
    fun `the bundled KLU is available`() {
        assertTrue(BundledKlu().isAvailable)
    }

    @Test
    fun `the bundled KLU solves sparse systems in both directions`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(1 to 2.0), listOf(0 to 3.0)))

        val factorization = BundledKlu().factor(matrix)

        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(9.0, 4.0)))
        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(6.0, 6.0), transpose = true))
    }

    @Test
    fun `the bundled KLU honors equilibration`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 8.0), listOf(1 to 0.25)))

        val factorization = BundledKlu().factor(matrix, equilibrate = true)

        assertContentEquals(doubleArrayOf(2.0, 3.0), factorization.solve(doubleArrayOf(16.0, 0.75)))
    }

    @Test
    fun `the bundled KLU reports a singular matrix`() {
        val matrix = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0), listOf(0 to 2.0)))

        assertTrue(BundledKlu().factor(matrix).singular)
    }
}
