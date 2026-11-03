package com.eignex.koblas.internal.portable.sparse

import com.eignex.koblas.*
import com.eignex.koblas.assertClose
import com.eignex.koblas.copyOf
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.portable.sparse.sweepsTouchedRows
import com.eignex.koblas.internal.sparse.withSymmetricRankScratch
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.ReferenceSparseBlas
import com.eignex.koblas.times
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotSame
import kotlin.test.assertTrue

@OptIn(UnsafeKoblasApi::class)
class SparseProductTest {

    private fun sparseAndDense(
        rows: Int,
        cols: Int,
        rng: Random,
        density: Double = 0.35,
    ): Pair<SparseMatrix, DenseMatrix> {
        val dense = DenseMatrix.zero(rows, cols)
        val columns = ArrayList<List<Pair<Int, Double>>>(cols)
        for (j in 0 until cols) {
            val column = ArrayList<Pair<Int, Double>>()
            for (i in 0 until rows) {
                if (rng.nextDouble() >= density) continue
                val v = rng.nextDouble(-1.0, 1.0)
                column.add(i to v)
                dense[i, j] = v
            }
            columns.add(column)
        }
        return SparseMatrix.ofColumns(rows, cols, columns) to dense
    }

    @Test
    fun `the sparse product agrees with the independent oracle`() {
        val rng = Random(20260919)
        for (shape in listOf(Triple(1, 1, 1), Triple(5, 4, 3), Triple(9, 2, 7))) {
            val (left, _) = sparseAndDense(shape.first, shape.second, rng)
            val (right, _) = sparseAndDense(shape.second, shape.third, rng)

            val product = left * right
            val expected = ReferenceSparseBlas.gemm(1.0, left, false, right, false)

            assertEquals(expected.rows, product.rows, "shape=$shape rows")
            assertEquals(expected.cols, product.cols, "shape=$shape cols")
            for (i in 0 until shape.first) {
                for (j in 0 until shape.third) {
                    assertClose(expected[i, j], product[i, j], "shape=$shape entry $i $j")
                }
            }
        }
    }

    @Test
    fun `the direct dense destination agrees with the independent oracle for every transpose`() {
        val rng = Random(20260923)
        val n = 4
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val (a, _) = sparseAndDense(5, 3, rng)
                val m = if (transposeA) 3 else 5
                val k = if (transposeA) 5 else 3
                val (b, _) = sparseAndDense(if (transposeB) n else k, if (transposeB) k else n, rng)
                val expected = randomMatrix(m, n, rng)
                val actual = expected.copyOf()

                ReferenceSparseBlas.gemm(0.5, a, transposeA, b, transposeB, -0.25, expected)
                koblas.gemm(0.5, a, transposeA, b, transposeB, -0.25, actual, scratch = Scratch())

                assertClose(expected, actual, "sparse-sparse dense transposeA=$transposeA transposeB=$transposeB")
            }
        }
    }

    @Test
    fun `a sparse result product supports scaling transpose and structure`() {
        val a = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 1.0), listOf(1 to 2.0), listOf(0 to 0.0)))
        val b = SparseMatrix.ofColumns(2, 3, listOf(listOf(0 to 4.0), listOf(1 to 5.0), listOf(0 to -4.0, 1 to 6.0)))

        val actual = koblas.gemm(-2.0, a, true, b, false)
        val zero = koblas.gemm(-0.0, a, true, b, false)

        assertEquals(3, actual.rows)
        assertEquals(3, actual.cols)
        assertContentEquals(intArrayOf(0, 2, 3, 6), actual.copyColumnPointers())
        assertContentEquals(intArrayOf(0, 2, 1, 0, 1, 2), actual.copyRowIndices())
        assertContentEquals(doubleArrayOf(-8.0, -0.0, -20.0, 8.0, -24.0, 0.0), actual.values)
        assertContentEquals(actual.copyColumnPointers(), zero.copyColumnPointers())
        assertContentEquals(actual.copyRowIndices(), zero.copyRowIndices())
        assertTrue(zero.values.all { it == 0.0 })
    }

    // A discovered column is either sorted or swept back into ascending order, whichever is cheaper for it,
    // so both fixtures below assert which branch they took as well as the result.
    @Test
    fun `a discovered column is ascending whether it was sorted or swept`() {
        val rng = Random(20261020)
        val cases = listOf(
            Triple(96, 0.02, false),
            Triple(24, 0.9, true),
        )
        for ((order, density, expectSweep) in cases) {
            val (left, denseLeft) = sparseAndDense(order, order, rng, density = density)
            val (right, denseRight) = sparseAndDense(order, order, rng, density = density)
            val product = left * right
            val context = "order=$order density=$density"

            val swept = (0 until product.cols).count {
                sweepsTouchedRows(product.colPointers[it + 1] - product.colPointers[it], order)
            }
            assertEquals(
                expectSweep,
                swept > product.cols / 2,
                "$context put $swept of ${product.cols} columns through the sweep",
            )
            for (j in 0 until product.cols) {
                var previous = -1
                product.forEachInColumn(j) { i, _ ->
                    assertTrue(i > previous, "$context column $j has $i after $previous")
                    previous = i
                }
            }
            val expected = DenseMatrix.zero(order, order)
            koblas.gemm(1.0, denseLeft, false, denseRight, false, 0.0, expected)
            val actual = DenseMatrix.zero(order, order)
            for (j in 0 until product.cols) product.forEachInColumn(j) { i, v -> actual[i, j] = v }
            assertClose(expected.values, actual.values, "$context product")
        }
    }

    // A stored zero is a position the patterns meet at, so it selects and scatters like any other entry.
    @Test
    fun `the sparse product keeps a stored zero of either operand`() {
        val one = SparseMatrix.ofColumns(1, 1, listOf(listOf(0 to 1.0)))
        val zero = SparseMatrix.ofColumns(1, 1, listOf(listOf(0 to 0.0)))
        assertEquals(1, zero.nnz, "the operand itself has to store the zero for this to say anything")

        assertEquals(1, (one * zero).nnz, "a stored zero in the second operand was dropped")
        assertEquals(1, (zero * one).nnz, "a stored zero in the first operand was dropped")
    }

    @Test
    fun `a later reached column keeps stored zeros in the product pattern`() {
        for ((leftValue, rightValue) in listOf(0.0 to 2.0, 1.0 to 0.0)) {
            val a = SparseMatrix.wrap(3, 2, intArrayOf(0, 0, 1), intArrayOf(2), doubleArrayOf(leftValue))
            val b = SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 2), intArrayOf(0, 1), doubleArrayOf(1.0, rightValue))

            val product = a * b

            assertEquals(3, product.rows)
            assertEquals(2, product.cols)
            assertContentEquals(intArrayOf(0, 0, 1), product.copyColumnPointers())
            assertContentEquals(intArrayOf(2), product.copyRowIndices())
            assertContentEquals(doubleArrayOf(0.0), product.values)
        }
    }

    @Test
    fun `the sparse product keeps an entry the arithmetic cancels to zero`() {
        val a = SparseMatrix.ofColumns(1, 2, listOf(listOf(0 to 1.0), listOf(0 to 1.0)))
        val b = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to 1.0, 1 to -1.0)))

        val product = a * b

        assertEquals(1, product.nnz, "the cancelled entry was dropped")
        assertEquals(0.0, product[0, 0], 0.0)
    }

    @Test
    fun `the sparse product assigns its first contribution directly`() {
        val negativeZero = SparseMatrix.ofColumns(1, 1, listOf(listOf(0 to -0.0)))
        val one = SparseMatrix.ofColumns(1, 1, listOf(listOf(0 to 1.0)))

        val product = negativeZero * one

        assertEquals((-0.0).toBits(), product.values.single().toBits())
    }

    @Test
    fun `disjoint products own fresh empty arrays`() {
        val a = SparseMatrix.wrap(3, 2, intArrayOf(0, 1, 1), intArrayOf(0), doubleArrayOf(1.0))
        val b = SparseMatrix.wrap(2, 2, intArrayOf(0, 0, 1), intArrayOf(1), doubleArrayOf(2.0))

        val first = a * b
        val second = a * b

        assertNotSame(a.colPointers, first.colPointers)
        assertNotSame(b.colPointers, first.colPointers)
        assertNotSame(first.colPointers, second.colPointers)
        assertNotSame(first.rowIndices, second.rowIndices)
        assertNotSame(first.values, second.values)
        assertContentEquals(intArrayOf(0, 0, 0), first.copyColumnPointers())
    }

    // The shortcut that avoids row-sized scratch must not swallow a product with structure to find.
    @Test
    fun `the empty result shortcut does not swallow a product that has structure`() {
        val a = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to 1.0)))
        val b = SparseMatrix.ofColumns(1, 2, listOf(emptyList(), listOf(0 to 2.0)))

        val product = a * b

        assertEquals(2, product.rows)
        assertEquals(2, product.cols)
        assertEquals(1, product.nnz)
        assertEquals(2.0, product[0, 1])
    }

    @Test
    fun `sparse syrk provides dense and selected sparse results`() {
        val a = SparseMatrix.ofColumns(3, 2, listOf(listOf(0 to 1.0, 2 to 2.0), listOf(1 to 3.0, 2 to -1.0)))
        for (transpose in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                val n = if (transpose) a.cols else a.rows
                val dense = DenseMatrix.wrap(n, n, DoubleArray(n * n) { Double.NaN })

                val sparse = koblas.syrk(a, transpose, lower)
                koblas.syrk(1.0, a, transpose, 0.0, dense, lower, scratch = Scratch())

                for (j in 0 until n) {
                    for (i in 0 until n) {
                        if (if (lower) i >= j else i <= j) {
                            assertEquals(sparse[i, j], dense[i, j])
                        } else {
                            assertTrue(dense[i, j].isNaN())
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `sparse syrk forms products of stored entries only`() {
        val exceptional = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to Double.POSITIVE_INFINITY, 1 to 0.0)))
        val implicit = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to Double.POSITIVE_INFINITY)))
        val implicitDense = DenseMatrix.zero(2)

        val stored = koblas.syrk(exceptional)
        koblas.syrk(1.0, implicit, false, 0.0, implicitDense)

        assertTrue(stored[1, 0].isNaN(), "a stored zero must form zero times infinity")
        assertEquals(0.0, implicitDense[1, 0], "implicit sparse zero must not form zero times infinity")
    }

    @Test
    fun `sparse syrk follows diagonal stored adjacency in both orientations`() {
        val n = 1024
        val diagonal = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to (j % 7 + 1.0)) })

        for (transpose in booleanArrayOf(false, true)) {
            val result = koblas.syrk(diagonal, transpose)
            assertEquals(n, result.nnz)
            assertContentEquals(IntArray(n) { it }, result.copyRowIndices())
            for (j in 0 until n) assertEquals((j % 7 + 1.0) * (j % 7 + 1.0), result.values[j])
        }
    }

    @Test
    fun `sparse syrk handles hypersparse rectangular adjacency`() {
        val columns = List(193) { j ->
            when (j) {
                2 -> listOf(0 to 2.0, 256 to -3.0)
                191 -> listOf(128 to 4.0)
                else -> emptyList()
            }
        }
        val a = SparseMatrix.ofColumns(257, 193, columns)

        for (lower in booleanArrayOf(false, true)) {
            val ordinary = koblas.syrk(a, transpose = false, lower)
            assertEquals(4, ordinary.nnz)
            assertEquals(4.0, ordinary[0, 0])
            assertEquals(16.0, ordinary[128, 128])
            assertEquals(9.0, ordinary[256, 256])
            assertEquals(-6.0, if (lower) ordinary[256, 0] else ordinary[0, 256])

            val transposed = koblas.syrk(a, transpose = true, lower)
            assertEquals(2, transposed.nnz)
            assertEquals(13.0, transposed[2, 2])
            assertEquals(16.0, transposed[191, 191])
        }
    }

    @Test
    fun `sparse syrk agrees with the independent sparse oracle`() {
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
        for (lower in booleanArrayOf(false, true)) {
            val expected = DenseMatrix.wrap(4, 4, DoubleArray(16) { 0.25 * it })
            val actual = expected.copyOf()
            ReferenceSparseBlas.syrk(1.25, a, false, -0.5, expected, lower)

            koblas.syrk(1.25, a, false, -0.5, actual, lower)

            assertContentEquals(expected.values, actual.values, "lower=$lower")
        }
        assertEquals(ReferenceSparseBlas.syrk(a), koblas.syrk(a), "fresh sparse syrk")
    }

    @Test
    fun `dense syrk reuses dirty scratch across different patterns and triangle flags`() {
        val patterns = listOf(
            SparseMatrix.ofColumns(
                5,
                4,
                listOf(
                    listOf(0 to 2.0, 2 to -1.0),
                    listOf(1 to 3.0, 4 to 0.0),
                    emptyList(),
                    listOf(0 to -2.0, 3 to 4.0),
                ),
            ),
            SparseMatrix.ofColumns(
                5,
                4,
                listOf(
                    listOf(1 to -1.0, 3 to 2.0),
                    emptyList(),
                    listOf(0 to 0.0, 2 to 3.0, 4 to -2.0),
                    listOf(2 to 1.0),
                ),
            ),
        )
        for (engine in listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.simd).distinct()) {
            val workspace = Scratch()
            // Epoch one collides with the first column unless marks are cleared; nonzero row counts
            // expose an adjacency builder that assumes its borrowed pointers are fresh.
            for (transpose in booleanArrayOf(false, true)) {
                for (a in patterns) {
                    val order = if (transpose) a.cols else a.rows
                    withSymmetricRankScratch(workspace.arrays, order, a.rows, a.nnz) {
                            sums,
                            marks,
                            touched,
                            pointers,
                            columns,
                            positions,
                            cursor,
                        ->
                        sums.fill(Double.NaN)
                        for (buffer in listOf(marks, touched, pointers, columns, positions, cursor)) buffer.fill(1)
                    }
                }
            }
            repeat(2) { pass ->
                for (transpose in booleanArrayOf(false, true)) {
                    for (lower in booleanArrayOf(true, false)) {
                        for ((pattern, a) in patterns.withIndex()) {
                            val order = if (transpose) a.cols else a.rows
                            val initial = DenseMatrix.wrap(order, order, DoubleArray(order * order) { -1000.0 - it })
                            for (j in 0 until order) {
                                for (i in 0 until order) {
                                    if (if (lower) i >= j else i <= j) initial[i, j] = (i - j) * 0.25
                                }
                            }
                            val expected = initial.copyOf()
                            val actual = initial.copyOf()
                            ReferenceSparseBlas.syrk(0.875, a, transpose, -0.25, expected, lower)

                            engine.syrk(0.875, a, transpose, -0.25, actual, lower, scratch = workspace)

                            val context = "${engine.name} pass=$pass pattern=$pattern transpose=$transpose lower=$lower"
                            assertClose(expected, actual, context)
                            for (j in 0 until order) {
                                for (i in 0 until order) {
                                    if (if (lower) i < j else i > j) assertEquals(initial[i, j], actual[i, j], context)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
