package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PortableSparseBlasTest {
    private val engines: List<KoblasEngine> = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd).distinct()
    private val emptyTrmmTriangles = listOf(
        SparseMatrix.ofColumns(3, 3, List(3) { listOf(it to 2.0) }),
        SparseMatrix.wrap(3, 3, IntArray(4), IntArray(0), DoubleArray(0)),
    )

    @Test
    fun `trmm on the left with no right hand sides borrows nothing`() {
        assertEmptyTrmmBorrowsNothing(right = false)
    }

    @Test
    fun `trmm on the right with no right hand sides borrows nothing`() {
        assertEmptyTrmmBorrowsNothing(right = true)
    }

    private fun assertEmptyTrmmBorrowsNothing(right: Boolean) {
        for (engine in engines) {
            for (a in emptyTrmmTriangles) {
                val order = a.rows
                for (lower in booleanArrayOf(false, true)) {
                    for (transpose in booleanArrayOf(false, true)) {
                        for (unitDiag in booleanArrayOf(false, true)) {
                            // An empty triangle shares the backing to exercise alias staging as well.
                            val backing = if (a.nnz == 0) a.values else DoubleArray(0)
                            val b = if (right) {
                                DenseMatrix.wrap(0, order, backing)
                            } else {
                                DenseMatrix.wrap(order, 0, backing)
                            }
                            val workspace = Workspace()
                            val context = "${engine.name} nnz=${a.nnz} right=$right lower=$lower " +
                                "transpose=$transpose unitDiag=$unitDiag"

                            engine.trmm(a, b, lower, transpose, unitDiag, right, 1.0, workspace)

                            assertContentEquals(DoubleArray(0), backing, context)
                            assertEquals(0, workspace.idleLengths(), context)
                            assertEquals(0, workspace.idleI32Lengths(), context)
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `trmm with no right hand sides still validates dimensions`() {
        val order = 3
        val triangles = listOf(
            SparseMatrix.wrap(order, order, IntArray(order + 1), IntArray(0), DoubleArray(0)),
            SparseMatrix.wrap(order, order + 1, IntArray(order + 2), IntArray(0), DoubleArray(0)),
        )
        for (engine in engines) {
            for (a in triangles) {
                for (right in booleanArrayOf(false, true)) {
                    val dimension = if (a.rows == a.cols) order + 1 else order
                    val b = if (right) DenseMatrix.zero(0, dimension) else DenseMatrix.zero(dimension, 0)
                    val workspace = Workspace()

                    assertFailsWith<DimensionMismatch>("${engine.name} shape=${a.rows} by ${a.cols} right=$right") {
                        engine.trmm(a, b, lower = true, right = right, workspace = workspace)
                    }

                    assertEquals(0, workspace.idleLengths())
                    assertEquals(0, workspace.idleI32Lengths())
                }
            }
        }
    }

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
                    for (engine in engines) {
                        val actual = initial.copyOf()
                        engine.gemv(alpha, matrix, x, beta, actual, transpose)
                        assertClose(expected, actual, "${engine.name} transpose=$transpose alpha=$alpha beta=$beta")
                    }
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

                for (engine in engines) {
                    val actual = initial.copyOf()
                    engine.gemv(alpha, matrix, x, beta, actual, transpose = true)

                    assertGemvAgreesWithReference(expected, actual, "${engine.name} alpha=$alpha beta=$beta")
                }
            }
        }
    }

    private fun assertGemvAgreesWithReference(expected: DoubleArray, actual: DoubleArray, context: String) {
        assertContentEquals(expected, actual, context)
    }

    @Test
    fun `scaled addition agrees for matching disjoint and overlapping patterns`() {
        val a = SparseMatrix.ofColumns(
            5,
            4,
            listOf(listOf(0 to 2.0, 3 to 0.0), emptyList(), listOf(1 to -1.0), listOf(3 to 4.0)),
        )
        val matching = SparseMatrix.ofColumns(
            5,
            4,
            listOf(listOf(0 to -2.0, 3 to 1.0), emptyList(), listOf(1 to 0.0), listOf(3 to 2.0)),
        )
        val disjoint = SparseMatrix.ofColumns(
            5,
            4,
            listOf(listOf(1 to 3.0), emptyList(), listOf(2 to 0.0), listOf(4 to -2.0)),
        )
        val overlapping = SparseMatrix.ofColumns(
            5,
            4,
            listOf(listOf(0 to 3.0, 4 to 1.0), emptyList(), listOf(2 to 0.0), listOf(3 to -2.0)),
        )
        for (b in listOf(matching, disjoint, overlapping)) {
            for (alpha in doubleArrayOf(0.0, 1.0, -0.75)) {
                val left = SparseMatrix.wrap(
                    a.rows,
                    a.cols,
                    a.copyColumnPointers(),
                    a.copyRowIndices(),
                    DoubleArray(a.nnz) { if (alpha == 0.0) Double.NaN else a.values[it] },
                )
                for (transpose in booleanArrayOf(false, true)) {
                    val source = if (transpose) ReferenceSparseBlas.transpose(left) else left
                    val expected = ReferenceSparseBlas.addScaled(alpha, source, transpose, b)

                    for (engine in engines) {
                        val actual = engine.addScaled(alpha, source, transpose, b)

                        assertAddScaledAgreesWithReference(
                            expected,
                            actual,
                            "${engine.name} alpha=$alpha transpose=$transpose",
                        )
                    }
                }
            }
        }
    }

    private fun assertAddScaledAgreesWithReference(expected: SparseMatrix, actual: SparseMatrix, context: String) {
        assertEquals(expected.rows, actual.rows, context)
        assertEquals(expected.cols, actual.cols, context)
        assertContentEquals(expected.copyColumnPointers(), actual.copyColumnPointers(), context)
        assertContentEquals(expected.copyRowIndices(), actual.copyRowIndices(), context)
        assertClose(expected.values, actual.values, context)
    }

    @Test
    fun `transposed gemv preserves csc reduction order`() {
        val values = DoubleArray(256)
        val matrix = singleColumn(values)
        val x = DoubleArray(256) { 1.0 }

        values[0] = 1e308
        values[1] = -1e308
        values[8] = 1e308
        for (engine in engines) {
            assertEquals(1e308, engine.gemv(matrix, x, transpose = true)[0], engine.name)
        }

        values[1] = 1e308
        values[8] = -1e308
        for (engine in engines) {
            assertEquals(Double.POSITIVE_INFINITY, engine.gemv(matrix, x, transpose = true)[0], engine.name)
        }
    }

    @Test
    fun `gemv updates do not fuse multiplication and addition`() {
        val matrix = singleColumn(DoubleArray(256) { 1e308 })
        for (engine in engines) {
            val y = DoubleArray(256) { -1e308 }
            engine.gemv(1.0, matrix, doubleArrayOf(2.0), 1.0, y, transpose = false)
            assertTrue(y.all { it == Double.POSITIVE_INFINITY }, engine.name)
        }
    }

    @Test
    fun `gemv evaluates zero coefficient updates`() {
        val matrix = singleColumn(doubleArrayOf(Double.POSITIVE_INFINITY))
        for (engine in engines) {
            assertTrue(engine.gemv(matrix, doubleArrayOf(0.0))[0].isNaN(), engine.name)

            val y = doubleArrayOf(0.0)
            engine.gemv(Double.MIN_VALUE, matrix, doubleArrayOf(0.5), 0.0, y, transpose = false)
            assertTrue(y[0].isNaN(), engine.name)
        }
    }

    private fun singleColumn(values: DoubleArray): SparseMatrix = SparseMatrix.wrap(
        values.size,
        1,
        intArrayOf(0, values.size),
        IntArray(values.size) { it },
        values,
    )
}
