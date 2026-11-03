package com.eignex.koblas.internal.portable.sparse

import com.eignex.koblas.*
import com.eignex.koblas.assertClose
import com.eignex.koblas.engine.koblas
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.ReferenceSparseBlas
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// The traversal is portable for every engine, and the one leaf with a vector body, the indexed dot, is checked
// against the portable one in SparseSimdTest, so the selected engine is enough here.
class Level2OperationsTest {
    @Test
    fun `gemv agrees across transpose alpha and beta variants`() {
        val matrix = SparseMatrix.ofTriplets(
            rows = 5,
            cols = 5,
            rowIndices = intArrayOf(0, 2, 1, 3, 4, 2, 0),
            colIndices = intArrayOf(0, 0, 1, 1, 2, 3, 4),
            values = doubleArrayOf(2.0, -1.0, 3.0, 0.5, 4.0, -2.0, 1.5),
        )
        val rng = Random(20260822)
        for (transpose in booleanArrayOf(false, true)) {
            for (alpha in doubleArrayOf(0.0, 1.0, -0.75)) {
                for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                    val x = DoubleArray(5) { rng.nextDouble(-1.0, 1.0) }
                    val initial = DoubleArray(5) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                    val expected = initial.copyOf()
                    ReferenceSparseBlas.gemv(alpha, matrix, x, beta, expected, transpose)

                    val actual = initial.copyOf()
                    koblas.gemv(alpha, matrix, x, beta, actual, transpose)

                    assertClose(expected, actual, "transpose=$transpose alpha=$alpha beta=$beta")
                }
            }
        }
    }

    @Test
    fun `transposed gemv leaves empty columns at their scaled value`() {
        val matrix = SparseMatrix.ofColumns(3, 3, listOf(emptyList(), listOf(1 to 0.0), listOf(2 to 3.0)))
        val x = DoubleArray(3) { 2.0 }
        for (alpha in doubleArrayOf(1.0, Double.POSITIVE_INFINITY)) {
            for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                val initial = DoubleArray(3) { if (beta == 0.0) Double.NaN else -0.0 }
                val expected = initial.copyOf()
                ReferenceSparseBlas.gemv(alpha, matrix, x, beta, expected, transpose = true)

                val actual = initial.copyOf()
                koblas.gemv(alpha, matrix, x, beta, actual, transpose = true)

                assertContentEquals(expected, actual, "alpha=$alpha beta=$beta")
            }
        }
    }

    @Test
    fun `gemv forms zero products for stored entries only`() {
        val a = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to Double.POSITIVE_INFINITY)))
        val y = DoubleArray(2)

        koblas.gemv(1.0, a, doubleArrayOf(0.0), 0.0, y)

        assertTrue(y[0].isNaN(), "the stored infinity produced ${y[0]}")
        assertEquals(0.0, y[1], "the missing entry was not structural zero")
    }

    // The opposite triangle is NaN, so reading it would show in the product.
    @Test
    fun `symv reads only the selected triangle`() {
        val x = doubleArrayOf(2.0, -3.0, 5.0)
        for (lower in booleanArrayOf(false, true)) {
            val a = if (lower) {
                SparseMatrix.ofColumns(
                    3,
                    3,
                    listOf(
                        listOf(0 to 2.0, 1 to -1.0, 2 to 4.0),
                        listOf(0 to Double.NaN, 1 to 3.0, 2 to 5.0),
                        listOf(0 to Double.NaN, 1 to Double.NaN, 2 to 7.0),
                    ),
                )
            } else {
                SparseMatrix.ofColumns(
                    3,
                    3,
                    listOf(
                        listOf(0 to 2.0, 1 to Double.NaN, 2 to Double.NaN),
                        listOf(0 to -1.0, 1 to 3.0, 2 to Double.NaN),
                        listOf(0 to 4.0, 1 to 5.0, 2 to 7.0),
                    ),
                )
            }
            val y = DoubleArray(3)

            koblas.symv(1.0, a, x, 0.0, y, lower)

            assertContentEquals(doubleArrayOf(27.0, 14.0, 28.0), y, "lower=$lower")
        }
    }

    @Test
    fun `symv agrees with the independent sparse oracle`() {
        val a = SparseMatrix.ofColumns(
            4,
            4,
            listOf(
                listOf(0 to 2.0, 2 to -1.5),
                listOf(1 to 3.25, 3 to 0.5),
                listOf(2 to 4.0),
                listOf(3 to -2.25),
            ),
        )
        val x = doubleArrayOf(1.5, -2.0, 0.25, 3.0)
        for (lower in booleanArrayOf(false, true)) {
            val expected = DoubleArray(4) { 0.5 * it }
            val actual = expected.copyOf()
            ReferenceSparseBlas.symv(0.75, a, x, -1.5, expected, lower)

            koblas.symv(0.75, a, x, -1.5, actual, lower)

            assertContentEquals(expected, actual, "lower=$lower")
        }
    }
}
