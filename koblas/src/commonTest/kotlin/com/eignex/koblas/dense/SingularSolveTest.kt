package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.lu
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SingularSolveTest {

    private fun singular() = DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))

    private val b = doubleArrayOf(1.0, 1.0)

    @Test
    fun `the LU factorization reports itself singular`() {
        val lu = singular().lu()
        assertTrue(lu.singular)
        assertEquals(0.0, lu.determinant(), "a singular factorization has determinant zero")
    }

    @Test
    fun `every LU solve refuses a singular factorization`() {
        val lu = singular().lu()
        val rhs = DenseMatrix.ofColumns(arrayOf(b))
        val cases = listOf<Pair<String, () -> Unit>>(
            "solve" to { koblas.solve(lu, b) },
            "solve transposed" to { koblas.solve(lu, b, transpose = true) },
            "solveInto" to { koblas.solveInto(lu, b, DoubleArray(2)) },
            "solveInto transposed" to { koblas.solveInto(lu, b, DoubleArray(2), transpose = true) },
            "blocked solve" to { koblas.solve(lu, rhs) },
            "blocked solve transposed" to { koblas.solve(lu, rhs, transpose = true) },
            "blocked solveInto" to { koblas.solveInto(lu, rhs, DenseMatrix.zero(2, 1)) },
            "vector solve" to { koblas.solve(lu, DenseVector.of(b)) },
            "vector solveInto" to { koblas.solveInto(lu, DenseVector.of(b), DenseVector.zero(2)) },
            "extension solve" to { lu.solve(b) },
            "invert" to { lu.invert() },
        )
        for ((name, block) in cases) {
            val e = assertFailsWith<SingularMatrix>("$name should refuse a singular factorization") { block() }
            assertEquals(lu.failedAt, e.position, "$name should report the failing pivot")
        }
    }

    @Test
    fun `a non-singular factorization still solves`() {
        val a = DenseMatrix.of(arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0)))
        val x = a.lu().solve(doubleArrayOf(3.0, 5.0))
        assertEquals(0.8, x[0], 1e-12)
        assertEquals(1.4, x[1], 1e-12)
    }

    @Test
    fun `the sparse solve refuses on the same terms`() {
        val s = SparseMatrix.ofTriplets(
            rows = 2,
            cols = 2,
            rowIdx = intArrayOf(0, 1, 0, 1),
            colIdx = intArrayOf(0, 0, 1, 1),
            values = doubleArrayOf(1.0, 2.0, 2.0, 4.0),
        )
        val lu = s.lu()
        assertTrue(lu.singular)
        assertFailsWith<IllegalArgumentException> { lu.solve(b) }
    }
}
