package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.gemvInto
import com.eignex.koblas.koblas
import com.eignex.koblas.prepare
import com.eignex.koblas.trmm
import com.eignex.koblas.trsm
import com.eignex.koblas.trsv
import kotlin.test.Test
import kotlin.test.assertContentEquals

/**
 * Operands that share the buffer a call writes into.
 *
 * `*Into` destinations may alias their inputs, which means an operand is worth exactly what it held when the
 * call began. Every case here runs the same call twice, once over independent buffers and once over shared
 * ones, and requires the two to agree. The independent run is the oracle, so what is asserted is the contract
 * rather than any particular staging.
 */
class SparseAliasTest {

    private fun example(): SparseMatrix =
        SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 4), intArrayOf(0, 1, 0, 1), doubleArrayOf(2.0, 3.0, 0.0, 4.0))

    @Test
    fun `gemv reads the input vector it was given when that vector is the destination`() {
        for (transpose in booleanArrayOf(false, true)) {
            val a = example()
            val expected = doubleArrayOf(1.0, 2.0)
            koblas.gemv(1.0, a, doubleArrayOf(1.0, 2.0), 0.0, expected, transpose)

            val shared = doubleArrayOf(1.0, 2.0)
            koblas.gemv(1.0, a, shared, 0.0, shared, transpose)

            assertContentEquals(expected, shared, "transpose=$transpose")
        }
    }

    /**
     * A square matrix storing one entry per column, so its coefficient array is exactly as long as the
     * destination vector and the two can share one buffer. The entries are off the diagonal, so the two
     * orientations of the product read different positions of it.
     */
    private fun permutation(): SparseMatrix = SparseMatrix.ofColumns(
        4,
        4,
        listOf(listOf(1 to 2.0), listOf(3 to -3.0), listOf(0 to 5.0), listOf(2 to 7.0)),
    )

    @Test
    fun `gemv reads the coefficients it was given when they are the destination`() {
        for (transpose in booleanArrayOf(false, true)) {
            val x = doubleArrayOf(1.0, -2.0, 0.5, 3.0)
            val independent = permutation()
            val expected = independent.values.copyOf()
            koblas.gemv(0.5, independent, x, -1.5, expected, transpose)

            val shared = permutation()
            koblas.gemv(0.5, shared, x, -1.5, shared.values, transpose)

            assertContentEquals(expected, shared.values, "transpose=$transpose")
        }
    }

    @Test
    fun `the root gemv extension stages its operands like the seam does`() {
        val a = example()
        val expected = doubleArrayOf(1.0, 2.0)
        koblas.gemv(1.0, a, doubleArrayOf(1.0, 2.0), 0.0, expected)

        val shared = doubleArrayOf(1.0, 2.0)
        a.gemvInto(1.0, shared, 0.0, shared)

        assertContentEquals(expected, shared)
    }

    @Test
    fun `a prepared product reads the input vector it was given when it is the destination`() {
        val prepared = example().prepare()
        val expected = doubleArrayOf(1.0, 2.0)
        prepared.gemv(1.0, doubleArrayOf(1.0, 2.0), 0.0, expected)

        val shared = doubleArrayOf(1.0, 2.0)
        prepared.gemv(1.0, shared, 0.0, shared)

        assertContentEquals(expected, shared)
    }

    @Test
    fun `trsv solves against the triangle it was given when the right-hand side is its values`() {
        // A column whose diagonal is missing, so the solve divides by zero and every later column depends on
        // a coefficient the substitution would already have overwritten.
        val pointers = intArrayOf(0, 2, 3, 3)
        val rows = intArrayOf(0, 2, 1)
        val values = doubleArrayOf(2.0, 3.0, 4.0)
        val independent = SparseMatrix.wrap(3, 3, pointers.copyOf(), rows.copyOf(), values.copyOf())
        val expected = values.copyOf()
        independent.trsv(expected, lower = true)

        val shared = SparseMatrix.wrap(3, 3, pointers.copyOf(), rows.copyOf(), values.copyOf())
        shared.trsv(shared.values, lower = true)

        assertContentEquals(expected, shared.values)
    }

    @Test
    fun `trsm solves against the triangle it was given when the block is its values`() {
        for (transpose in booleanArrayOf(false, true)) {
            for (unitDiag in booleanArrayOf(false, true)) {
                val values = doubleArrayOf(2.0, 3.0, 0.0, 4.0)
                val independent = SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 4), intArrayOf(0, 1, 0, 1), values.copyOf())
                val expected = DenseMatrix.wrap(2, 2, values.copyOf())
                independent.trsm(expected, lower = true, transpose = transpose, unitDiag = unitDiag, alpha = 2.0)

                val shared = SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 4), intArrayOf(0, 1, 0, 1), values.copyOf())
                shared.trsm(
                    DenseMatrix.wrap(2, 2, shared.values),
                    lower = true,
                    transpose = transpose,
                    unitDiag = unitDiag,
                    alpha = 2.0,
                    workspace = Workspace(),
                )

                assertContentEquals(
                    expected.values,
                    shared.values,
                    "transpose=$transpose unit=$unitDiag",
                )
            }
        }
    }

    /**
     * A triangle of order three whose stored entries fill its selected half exactly, so its coefficient array
     * is also a valid three-by-two block and the two really do share one buffer. Six entries is the smallest
     * size where a right-hand side and a left-hand one can both be formed from the same array.
     */
    private fun filledTriangle(lower: Boolean, values: DoubleArray): SparseMatrix = SparseMatrix.ofColumns(
        3,
        3,
        List(3) { j ->
            (0 until 3).mapNotNull { i ->
                val inTriangle = if (lower) i >= j else i <= j
                if (inTriangle) i to values[triangleSlot(i, j, lower)] else null
            }
        },
    )

    private fun triangleSlot(i: Int, j: Int, lower: Boolean): Int {
        var slot = 0
        for (column in 0 until 3) {
            for (row in 0 until 3) {
                val inTriangle = if (lower) row >= column else row <= column
                if (!inTriangle) continue
                if (row == i && column == j) return slot
                slot++
            }
        }
        return 0
    }

    @Test
    fun `every triangular side and direction reads the triangle it was given`() {
        val coefficients = doubleArrayOf(3.0, -0.5, 0.75, 2.5, -1.25, 4.0)
        for (lower in booleanArrayOf(false, true)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (right in booleanArrayOf(false, true)) {
                    for (solve in booleanArrayOf(false, true)) {
                        val what = "lower=$lower transpose=$transpose right=$right solve=$solve"
                        val rows = if (right) 2 else 3
                        val columns = if (right) 3 else 2
                        val independent = filledTriangle(lower, coefficients)
                        val expected = DenseMatrix.wrap(rows, columns, coefficients.copyOf())
                        if (solve) {
                            independent.trsm(expected, lower, transpose, right = right, alpha = -0.75)
                        } else {
                            independent.trmm(expected, lower, transpose, right = right, alpha = -0.75)
                        }

                        val shared = filledTriangle(lower, coefficients)
                        val block = DenseMatrix.wrap(rows, columns, shared.values)
                        if (solve) {
                            shared.trsm(
                                block,
                                lower,
                                transpose,
                                right = right,
                                alpha = -0.75,
                                workspace = Workspace(),
                            )
                        } else {
                            shared.trmm(
                                block,
                                lower,
                                transpose,
                                right = right,
                                alpha = -0.75,
                                workspace = Workspace(),
                            )
                        }

                        assertClose(expected, block, what, tolerance = 1e-9)
                    }
                }
            }
        }
    }
}
