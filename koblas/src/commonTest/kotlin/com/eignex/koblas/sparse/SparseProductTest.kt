package com.eignex.koblas.sparse

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DimensionMismatch
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.UnsafeKoblasApi
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.copyOf
import com.eignex.koblas.dense.PanelWork
import com.eignex.koblas.koblas
import com.eignex.koblas.partialPanelWidth
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import com.eignex.koblas.sparse.internal.sweepsTouchedRows
import com.eignex.koblas.times
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
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
    fun `every transpose combination agrees with the dense product`() {
        val rng = Random(20260826)
        val rightHandSides = partialPanelWidth(PanelWork.SparseRightHandSides)
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val (sparse, dense) = sparseAndDense(5, 3, rng)
                val m = if (transposeA) 3 else 5
                val k = if (transposeA) 5 else 3
                val b = if (transposeB) randomMatrix(rightHandSides, k, rng) else randomMatrix(k, rightHandSides, rng)
                val c = randomMatrix(m, rightHandSides, rng)

                val fromDense = c.copyOf()
                koblas.gemm(0.75, dense, transposeA, b, transposeB, -0.5, fromDense)
                val fromSparse = c.copyOf()
                koblas.gemm(0.75, sparse, transposeA, b, transposeB, -0.5, fromSparse, workspace = Workspace())

                assertClose(fromDense, fromSparse, "transposeA=$transposeA transposeB=$transposeB")
            }
        }
    }

    @Test
    fun `every transpose combination from the right agrees with the dense product`() {
        val rng = Random(20260917)
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val (sparse, dense) = sparseAndDense(5, 3, rng)
                val k = if (transposeA) 3 else 5
                val n = if (transposeA) 5 else 3
                val b = if (transposeB) randomMatrix(k, 4, rng) else randomMatrix(4, k, rng)
                val c = randomMatrix(4, n, rng)

                val fromDense = c.copyOf()
                koblas.gemm(0.75, b, transposeB, dense, transposeA, -0.5, fromDense)
                val fromSparse = c.copyOf()
                koblas.gemm(
                    0.75, sparse, transposeA, b, transposeB, -0.5, fromSparse,
                    right = true, workspace = Workspace(),
                )

                assertClose(fromDense, fromSparse, "right transposeA=$transposeA transposeB=$transposeB")
            }
        }
    }

    @Test
    fun `one right-hand side agrees with gemv`() {
        val rng = Random(20260827)
        val (sparse, _) = sparseAndDense(6, 4, rng)
        val x = randomVector(4, rng)
        val c = DenseMatrix.zero(6, 1)

        koblas.gemm(1.0, sparse, false, DenseMatrix.wrap(4, 1, x.copyOf()), false, 0.0, c)

        assertClose(koblas.gemv(sparse, x), c.values, "gemm over one column is gemv")
    }

    @Test
    fun `gemv forms zero products for stored entries only`() {
        val a = SparseMatrix.ofColumns(2, 1, listOf(listOf(0 to Double.POSITIVE_INFINITY)))
        val y = DoubleArray(2)

        koblas.gemv(1.0, a, doubleArrayOf(0.0), 0.0, y)

        assertTrue(y[0].isNaN(), "the stored infinity produced ${y[0]}")
        assertEquals(0.0, y[1], "the missing entry was not structural zero")
    }

    @Test
    fun `left sparse gemm forms zero products for stored entries`() {
        val a = SparseMatrix.ofColumns(1, 1, listOf(listOf(0 to Double.POSITIVE_INFINITY)))
        val c = DenseMatrix.zero(1, 1)

        koblas.gemm(1.0, a, false, DenseMatrix.zero(1, 1), false, 0.0, c)

        assertTrue(c[0, 0].isNaN())
    }

    @Test
    fun `right sparse gemm forms zero products for stored entries`() {
        val a = SparseMatrix.ofColumns(1, 1, listOf(listOf(0 to 0.0)))
        val b = DenseMatrix.wrap(1, 1, doubleArrayOf(Double.POSITIVE_INFINITY))
        val c = DenseMatrix.zero(1, 1)

        koblas.gemm(1.0, a, false, b, false, 0.0, c, right = true)

        assertTrue(c[0, 0].isNaN())
    }

    @Test
    fun `a beta of zero overwrites a destination it never reads`() {
        val rng = Random(20260828)
        val (sparse, _) = sparseAndDense(4, 3, rng)
        val b = randomMatrix(3, 2, rng)
        val c = DenseMatrix.wrap(4, 2, DoubleArray(8) { Double.NaN })

        koblas.gemm(1.0, sparse, false, b, false, 0.0, c)

        assertTrue(c.values.all { it.isFinite() }, "a NaN survived beta = 0")
    }

    @Test
    fun `an alpha of zero leaves the destination scaled by beta alone`() {
        val rng = Random(20260829)
        val (sparse, _) = sparseAndDense(4, 3, rng)
        val b = randomMatrix(3, 2, rng)
        val c = randomMatrix(4, 2, rng)
        val expected = DenseMatrix.wrap(4, 2, DoubleArray(8) { c.values[it] * 2.0 })

        val actual = c.copyOf()
        koblas.gemm(0.0, sparse, false, b, false, 2.0, actual)

        assertClose(expected, actual, "alpha = 0")
    }

    @Test
    fun `the product of a matrix and an identity is the matrix`() {
        val rng = Random(20260830)
        val (sparse, dense) = sparseAndDense(5, 4, rng)

        assertClose(dense, sparse * DenseMatrix.diagonal(4), "A times I")
    }

    @Test
    fun `the convenience overload takes its shape from the operands`() {
        val rng = Random(20260831)
        val (sparse, _) = sparseAndDense(5, 4, rng)

        val c = sparse * randomMatrix(4, 3, rng)

        assertEquals(5, c.rows)
        assertEquals(3, c.cols)
    }

    @Test
    fun `a dense operand that does not meet the sparse one from the right is rejected`() {
        val rng = Random(20260918)
        val (sparse, _) = sparseAndDense(5, 4, rng)

        assertFailsWith<DimensionMismatch> {
            koblas.gemm(
                1.0,
                sparse,
                false,
                randomMatrix(3, 3, rng),
                false,
                0.0,
                DenseMatrix.zero(3, 4),
                right = true,
            )
        }
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
                koblas.gemm(0.5, a, transposeA, b, transposeB, -0.25, actual, Workspace())

                assertClose(expected, actual, "sparse-sparse dense transposeA=$transposeA transposeB=$transposeB")
            }
        }
    }

    @Test
    fun `the sparse product keeps its rows ascending within a column`() {
        val rng = Random(20260920)
        val (left, _) = sparseAndDense(12, 9, rng, density = 0.5)
        val (right, _) = sparseAndDense(9, 7, rng, density = 0.5)

        val product = left * right

        for (j in 0 until product.cols) {
            var previous = -1
            product.forEachInColumn(j) { i, _ ->
                assertTrue(i > previous, "column $j has $i after $previous")
                previous = i
            }
        }
    }

    /**
     * A discovered column comes out ascending whichever way it was put in order.
     *
     * The rows of a column arrive in the order the contributing columns held them, and the scheduling either
     * sorts what it collected or sweeps the range it came from, whichever is cheaper for that column. The
     * two are different code over the same marks, so both are exercised here: a wide sparse product whose
     * columns touch a fraction of their range, and a dense-ish one whose columns touch most of it. The
     * predicate itself is asserted at both fixtures so this cannot quietly become one branch twice.
     */
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

    @Test
    fun `a sparse matrix times a sparse identity is the matrix`() {
        val rng = Random(20260921)
        val (a, _) = sparseAndDense(6, 5, rng)
        val identity = SparseMatrix.ofColumns(5, 5, List(5) { j -> listOf(j to 1.0) })

        assertEquals(a, a * identity, "A times I should give A back")
    }

    /**
     * The pattern of the product is what the patterns of the operands meet at. A stored zero is a position
     * either of them holds, so it selects and scatters like any other entry rather than being skipped for
     * the value it carries.
     */
    @Test
    fun `the sparse product keeps a stored zero of either operand`() {
        val one = SparseMatrix.ofColumns(1, 1, listOf(listOf(0 to 1.0)))
        val zero = SparseMatrix.ofColumns(1, 1, listOf(listOf(0 to 0.0)))
        assertEquals(1, zero.nnz, "the operand itself has to store the zero for this to say anything")

        assertEquals(1, (one * zero).nnz, "a stored zero in the second operand was dropped")
        assertEquals(1, (zero * one).nnz, "a stored zero in the first operand was dropped")
    }

    @Test
    fun `the sparse product keeps an entry the arithmetic cancels to zero`() {
        // The single entry of the product is 1 * 1 + 1 * -1, a position the patterns meet at all the same.
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
    fun `a sparse operand that does not meet the first is rejected`() {
        val rng = Random(20260922)
        val (a, _) = sparseAndDense(5, 4, rng)
        val (b, _) = sparseAndDense(3, 4, rng)

        assertFailsWith<DimensionMismatch> { a * b }
    }

    @Test
    fun `a destination of the wrong shape is rejected`() {
        val rng = Random(20260901)
        val (sparse, _) = sparseAndDense(5, 4, rng)
        val b = randomMatrix(4, 3, rng)

        assertFailsWith<DimensionMismatch> {
            koblas.gemm(1.0, sparse, false, b, false, 0.0, DenseMatrix.zero(5, 2))
        }
    }

    @Test
    fun `a second operand that does not meet the first is rejected`() {
        val rng = Random(20260902)
        val (sparse, _) = sparseAndDense(5, 4, rng)

        assertFailsWith<DimensionMismatch> { sparse * randomMatrix(3, 3, rng) }
    }
}
