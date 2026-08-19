package com.eignex.koblas.dense

import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.F64SparseMatrix
import com.eignex.koblas.assertClose
import com.eignex.koblas.sparse.lu
import com.eignex.koblas.wellConditioned
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class LinearAlgebraDeterminantTest {

    @Test
    fun `dense LU determinant matches hand values and tracks row-swap sign`() {
        val a = F64DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
        assertEquals(5.0, a.lu().determinant(), 1e-12)
        val b = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 3.0), doubleArrayOf(2.0, 1.0)))
        assertEquals(-5.0, b.lu().determinant(), 1e-12)
        // The 3x3 determinant by hand is 1*(50-48) - 2*(40-42) + 3*(32-35) = -3.
        val c = F64DenseMatrix.of(
            arrayOf(
                doubleArrayOf(1.0, 2.0, 3.0),
                doubleArrayOf(4.0, 5.0, 6.0),
                doubleArrayOf(7.0, 8.0, 10.0),
            ),
        )
        assertEquals(-3.0, c.lu().determinant(), 1e-9)
        assertEquals(1.0, F64DenseMatrix(0, 0).lu().determinant())
        assertEquals(1.0, F64DenseMatrix.diagonal(4).lu().determinant(), 1e-12)
    }

    @Test
    fun `dense and sparse determinants agree on random matrices`() {
        val rng = Random(20260803)
        for (n in intArrayOf(1, 3, 7, 15)) {
            val a = wellConditioned(n, rng)
            val cols = List(n) { j -> (0 until n).map { i -> i to a[i, j] } }
            val sparseDet = F64SparseMatrix.ofColumns(n, n, cols).lu().determinant()
            assertClose(a.lu().determinant(), sparseDet, "n=$n dense vs sparse determinant", tolerance = 1e-9)
        }
    }

    @Test
    fun `the determinant saturates without that meaning singular`() {
        // The product of n pivots leaves the double range long before n is large, in both directions. A
        // returned zero therefore says nothing about singularity, which is what `singular` is for.
        val tiny = F64DenseMatrix.diagonal(200, 0.01).lu()
        assertEquals(0.0, tiny.determinant(), "0.01^200 underflows")
        assertFalse(tiny.singular, "an underflowed determinant is not a singular factorization")
        val huge = F64DenseMatrix.diagonal(200, 100.0).lu()
        assertTrue(huge.determinant().isInfinite(), "100^200 overflows")
        assertFalse(huge.singular, "an overflowed determinant is not a singular factorization")
        // And the solve is unaffected, which is the point: the factorization is perfectly usable.
        val x = tiny.solve(DoubleArray(200) { 1.0 })
        assertClose(DoubleArray(200) { 100.0 }, x, "the underflowed determinant did not spoil the solve")
    }
}
