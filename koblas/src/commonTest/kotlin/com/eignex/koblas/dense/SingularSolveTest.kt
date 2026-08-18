package com.eignex.koblas.dense

import com.eignex.koblas.F64DenseMatrix
import com.eignex.koblas.F64DenseVector
import com.eignex.koblas.F64SparseMatrix
import com.eignex.koblas.SingularMatrix
import com.eignex.koblas.koblas
import com.eignex.koblas.sparse.lu
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class SingularSolveTest {

    private fun singular() = F64DenseMatrix.of(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))

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
        val rhs = F64DenseMatrix.ofColumns(arrayOf(b))
        val cases = listOf<Pair<String, () -> Unit>>(
            "solve" to { koblas.solve(lu, b) },
            "solve transposed" to { koblas.solve(lu, b, transpose = true) },
            "solveInto" to { koblas.solveInto(lu, b, DoubleArray(2)) },
            "solveInto transposed" to { koblas.solveInto(lu, b, DoubleArray(2), transpose = true) },
            "blocked solve" to { koblas.solve(lu, rhs) },
            "blocked solve transposed" to { koblas.solve(lu, rhs, transpose = true) },
            "blocked solveInto" to { koblas.solveInto(lu, rhs, F64DenseMatrix.zero(2, 1)) },
            "vector solve" to { koblas.solve(lu, F64DenseVector.of(b)) },
            "vector solveInto" to { koblas.solveInto(lu, F64DenseVector.of(b), F64DenseVector.zero(2)) },
            "extension solve" to { lu.solve(b) },
            "invert" to { lu.invert() },
        )
        for ((name, block) in cases) {
            val e = assertFailsWith<SingularMatrix>("$name should refuse a singular factorization") { block() }
            assertEquals(lu.failedAt, e.position, "$name should report the failing pivot")
        }
    }

    /** Four right-hand sides, the width from which the host backend takes its native `dsytrs` path. */
    @Test
    fun `every LDL solve refuses a singular factorization`() {
        val ldl = koblas.ldl(F64DenseMatrix.zero(3, 3))
        assertTrue(ldl.singular)
        val rhs = F64DenseMatrix.zero(3, 4)
        val cases = listOf<Pair<String, () -> Unit>>(
            "solve" to { koblas.solve(ldl, DoubleArray(3)) },
            "solveInto" to { koblas.solveInto(ldl, DoubleArray(3), DoubleArray(3)) },
            "blocked solve" to { koblas.solve(ldl, rhs) },
            "blocked solveInto" to { koblas.solveInto(ldl, rhs, F64DenseMatrix.zero(3, 4)) },
            "vector solve" to { koblas.solve(ldl, F64DenseVector.zero(3)) },
            "vector solveInto" to { koblas.solveInto(ldl, F64DenseVector.zero(3), F64DenseVector.zero(3)) },
            "extension solve" to { ldl.solve(DoubleArray(3)) },
        )
        for ((name, block) in cases) {
            val e = assertFailsWith<SingularMatrix>("$name should refuse a singular factorization") { block() }
            assertEquals(ldl.failedAt, e.position, "$name should report the failing pivot")
        }
    }

    @Test
    fun `the sparse solve refuses on the same terms`() {
        val s = F64SparseMatrix.ofTriplets(
            rows = 2,
            cols = 2,
            rowIdx = intArrayOf(0, 1, 0, 1),
            colIdx = intArrayOf(0, 0, 1, 1),
            values = doubleArrayOf(1.0, 2.0, 2.0, 4.0),
        )
        val lu = s.lu()
        assertTrue(lu.singular)
        assertFailsWith<SingularMatrix> { lu.solve(b) }
    }
}
