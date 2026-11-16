package com.eignex.koblas.internal.operations.sparse

import com.eignex.koblas.*
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class StagingTest {
    // Aliased operands: a destination may share an input's buffer, and the call must agree with one made on
    // independent buffers holding the values present at entry.

    @Test
    fun `gemv reads the input vector it was given when that vector is the destination`() {
        val values = doubleArrayOf(2.0, 3.0, 0.0, 4.0)
        for (transpose in booleanArrayOf(false, true)) {
            val a = SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 4), intArrayOf(0, 1, 0, 1), values.copyOf())
            val expected = doubleArrayOf(1.0, 2.0)
            koblas.gemv(
                1.0,
                a,
                DenseVector.wrap(doubleArrayOf(1.0, 2.0)),
                0.0,
                DenseVector.wrap(expected),
                transpose = transpose,
            )

            val shared = doubleArrayOf(1.0, 2.0)
            koblas.gemv(1.0, a, DenseVector.wrap(shared), 0.0, DenseVector.wrap(shared), transpose = transpose)

            assertContentEquals(expected, shared, "transpose=$transpose")
        }
    }

    @Test
    fun `gemv reads the coefficients it was given when they are the destination`() {
        for (transpose in booleanArrayOf(false, true)) {
            val x = doubleArrayOf(1.0, -2.0, 0.5, 3.0)
            val independent = permutation()
            val expected = independent.values.copyOf()
            koblas.gemv(0.5, independent, DenseVector.wrap(x), -1.5, DenseVector.wrap(expected), transpose = transpose)

            val shared = permutation()
            koblas.gemv(0.5, shared, DenseVector.wrap(x), -1.5, DenseVector.wrap(shared.values), transpose = transpose)

            assertContentEquals(expected, shared.values, "transpose=$transpose")
        }
    }

    @Test
    fun `symv reads the vector and coefficients it was given when both are the destination`() {
        val a = SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 2), intArrayOf(0, 1), doubleArrayOf(2.0, 5.0))
        val shared = a.values

        koblas.symv(1.0, a, DenseVector.wrap(shared), 0.0, DenseVector.wrap(shared), lower = true)

        assertContentEquals(doubleArrayOf(4.0, 25.0), shared)
    }

    @Test
    fun `symm reads the operands it was given when all three share one buffer`() {
        val allShared = doubleArrayOf(2.0, 3.0, 7.0, 5.0)
        val sharedA = SparseMatrix.wrap(2, 2, intArrayOf(0, 2, 4), intArrayOf(0, 1, 0, 1), allShared)
        val sharedB = DenseMatrix.wrap(2, 2, allShared)
        val sharedC = DenseMatrix.wrap(2, 2, allShared)

        koblas.symm(1.0, sharedA, sharedB, 0.0, sharedC, lower = true, scratch = Scratch())

        assertContentEquals(doubleArrayOf(13.0, 21.0, 29.0, 46.0), sharedC.values)
    }

    @Test
    fun `trsv solves against the triangle it was given when the right hand side is its values`() {
        // A column whose diagonal is missing, so the solve divides by zero and every later column depends on
        // a coefficient the substitution would already have overwritten.
        val independent = aliasTriangle()
        val expected = independent.values.copyOf()
        koblas.trsv(independent, DenseVector.wrap(expected), lower = true)

        val shared = aliasTriangle()
        koblas.trsv(shared, DenseVector.wrap(shared.values), lower = true)

        assertContentEquals(expected, shared.values)
    }

    @Test
    fun `every triangular side and direction reads the triangle it was given`() {
        val coefficients = doubleArrayOf(3.0, -0.5, 0.75, 2.5, -1.25, 4.0)
        for (lower in booleanArrayOf(false, true)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (unitDiag in booleanArrayOf(false, true)) {
                    for (right in booleanArrayOf(false, true)) {
                        for (solve in booleanArrayOf(false, true)) {
                            val what = "lower=$lower transpose=$transpose unit=$unitDiag right=$right solve=$solve"
                            val rows = if (right) 2 else 3
                            val columns = if (right) 3 else 2
                            val independent = filledTriangle(lower, coefficients)
                            val expected = DenseMatrix.wrap(rows, columns, coefficients.copyOf())
                            val shared = filledTriangle(lower, coefficients)
                            val block = DenseMatrix.wrap(rows, columns, shared.values)
                            val scratch = Scratch()

                            if (solve) {
                                koblas.trsm(independent, expected, lower, transpose, unitDiag, right, -0.75)
                                koblas.trsm(shared, block, lower, transpose, unitDiag, right, -0.75, scratch = scratch)
                            } else {
                                koblas.trmm(independent, expected, lower, transpose, unitDiag, right, -0.75)
                                koblas.trmm(shared, block, lower, transpose, unitDiag, right, -0.75, scratch = scratch)
                            }

                            assertClose(expected, block, what, tolerance = 1e-9)
                        }
                    }
                }
            }
        }
    }

    // Alias snapshots borrow from the workspace. The tests above check their numerical results, which do not
    // depend on whether a snapshot was allocated or borrowed, so these check only the loans.

    @Test
    fun `every level two routine stages its alias from the workspace`() {
        val n = 4

        assertStaged(n, "gemv over the vector it writes") { w ->
            val y = doubleArrayOf(1.0, -2.0, 0.5, 3.0)
            koblas.gemv(
                0.5,
                permutation(),
                DenseVector.wrap(y),
                -1.5,
                DenseVector.wrap(y),
                transpose = false,
                scratch = w,
            )
        }
        assertStaged(n, "gemv over the coefficients it writes") { w ->
            val a = permutation()
            koblas.gemv(
                0.5,
                a,
                DenseVector.wrap(doubleArrayOf(1.0, -2.0, 0.5, 3.0)),
                -1.5,
                DenseVector.wrap(a.values),
                transpose = false,
                scratch = w,
            )
        }
        assertStaged(n, "symv over the vector it writes") { w ->
            val y = doubleArrayOf(1.0, -2.0, 0.5, 3.0)
            koblas.symv(0.5, permutation(), DenseVector.wrap(y), -1.5, DenseVector.wrap(y), lower = true, scratch = w)
        }
        assertStaged(3, "trsv over the coefficients that are its right-hand side") { w ->
            val a = aliasTriangle()
            koblas.trsv(a, DenseVector.wrap(a.values), lower = true, scratch = w)
        }
        assertStaged(3, "trmv over the coefficients that are its right-hand side") { w ->
            val a = aliasTriangle()
            koblas.trmv(a, DenseVector.wrap(a.values), lower = true, scratch = w)
        }
    }

    /** Sparse staging copies only coefficients: structural arrays cannot alias a dense destination. */
    @Test
    fun `a staged sparse operand copies its coefficients and shares its structure`() {
        val workspace = Scratch()
        val a = permutation()

        koblas.gemv(
            0.5,
            a,
            DenseVector.wrap(doubleArrayOf(1.0, -2.0, 0.5, 3.0)),
            -1.5,
            DenseVector.wrap(a.values),
            transpose = false,
            scratch = workspace,
        )

        assertEquals(1, workspace.available(a.nnz), "the coefficients were not staged from the workspace")
        assertEquals(0, workspace.availableI32(a.cols + 1), "the column pointers were copied")
        assertEquals(0, workspace.availableI32(a.nnz), "the row indices were copied")
    }

    /**
     * A square matrix storing one entry per column, so its coefficient array is exactly as long as the
     * vectors and either of them can be the buffer a call writes. The entries are off the diagonal, so the
     * two orientations of the product read different positions of it.
     */
    private fun permutation(): SparseMatrix = SparseMatrix.ofColumns(
        4,
        4,
        listOf(listOf(1 to 2.0), listOf(3 to -3.0), listOf(0 to 5.0), listOf(2 to 7.0)),
    )

    /**
     * A lower triangle storing as many coefficients as it has rows, which is what lets its values be the
     * right-hand side a triangular routine writes.
     */
    private fun aliasTriangle(): SparseMatrix =
        SparseMatrix.wrap(3, 3, intArrayOf(0, 2, 3, 3), intArrayOf(0, 2, 1), doubleArrayOf(2.0, 3.0, 4.0))

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

    /** That [call] took exactly one loan, of [length], and returned it. */
    private fun assertStaged(length: Int, what: String, call: (Scratch) -> Unit) {
        val workspace = Scratch()

        call(workspace)

        assertEquals(1, workspace.available(length), "$what did not stage from the workspace")
        assertEquals(1, workspace.idleLengths(), "$what borrowed scratch besides its snapshot")
    }
}
