package com.eignex.koblas.internal.portable.sparse

import com.eignex.koblas.*
import com.eignex.koblas.assertClose
import com.eignex.koblas.copyOf
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.engineWithKernels
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.operands.*
import com.eignex.koblas.internal.portable.PortableEngine
import com.eignex.koblas.partialPanelWidth
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.sparse.*
import com.eignex.koblas.sparse.ReferenceSparseBlas
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

// Whether a call stages is the backend's own answer, so a runtime with no vector backend would never stage
// and leave the staged layout untested. [AdjacentPreferringPanels] asks for adjacent right-hand sides and
// keeps the portable arithmetic, which puts the copy, the addressing and the writeback under test everywhere.
class SparseDenseProductTest {
    private val staging = engineWithKernels(panels = AdjacentPreferringPanels)
    private val plain = engineWithKernels(panels = PortableEngine)

    private fun randomSparse(rows: Int, cols: Int, rng: Random, density: Double = 0.35): SparseMatrix =
        SparseMatrix.ofColumns(
            rows,
            cols,
            List(cols) { (0 until rows).filter { rng.nextDouble() < density }.map { it to rng.nextDouble(-1.0, 1.0) } },
        )

    // One more right-hand side than the selected backend groups, so every call ends on a partial panel.
    @Test
    fun `every transpose combination on either side agrees with the sparse oracle`() {
        val rng = Random(20260826)
        val sides = partialPanelWidth(PanelWork.SparseRightHandSides)
        for (right in booleanArrayOf(false, true)) {
            for (transposeA in booleanArrayOf(false, true)) {
                for (transposeB in booleanArrayOf(false, true)) {
                    val a = randomSparse(5, 3, rng)
                    val rows = if (transposeA) 3 else 5
                    val inner = if (transposeA) 5 else 3
                    // From the left op(B) is inner by sides; from the right it is sides by the rows of op(A).
                    val (bRows, bCols) = if (right) sides to rows else inner to sides
                    val b = if (transposeB) randomMatrix(bCols, bRows, rng) else randomMatrix(bRows, bCols, rng)
                    val c = if (right) randomMatrix(sides, inner, rng) else randomMatrix(rows, sides, rng)
                    val expected = c.copyOf()
                    ReferenceSparseBlas.gemm(0.75, a, transposeA, b, transposeB, -0.5, expected, right)

                    val actual = c.copyOf()
                    koblas.gemm(0.75, a, transposeA, b, transposeB, -0.5, actual, right = right, scratch = Scratch())

                    assertClose(expected, actual, "right=$right transposeA=$transposeA transposeB=$transposeB")
                }
            }
        }
    }

    @Test
    fun `transposed products leave empty columns at their scaled value`() {
        val order = 8
        val a = SparseMatrix.ofColumns(
            order,
            order,
            List(order) { column ->
                when (column) {
                    0 -> emptyList()
                    1 -> listOf(1 to 0.0)
                    else -> List(order) { it to 1.0 }
                }
            },
        )
        for (sides in intArrayOf(1, 3, 8)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val b = if (transposeB) {
                    DenseMatrix.wrap(sides, order, DoubleArray(order * sides) { 2.0 })
                } else {
                    DenseMatrix.wrap(order, sides, DoubleArray(order * sides) { 2.0 })
                }
                for (alpha in doubleArrayOf(1.0, Double.POSITIVE_INFINITY)) {
                    for (beta in doubleArrayOf(0.0, 1.0, 0.5)) {
                        val initial = DoubleArray(order * sides) { if (beta == 0.0) Double.NaN else -0.0 }
                        val expected = DenseMatrix.wrap(order, sides, initial.copyOf())
                        ReferenceSparseBlas.gemm(alpha, a, true, b, transposeB, beta, expected)

                        for ((name, engine) in sparsePanelEngines()) {
                            val actual = DenseMatrix.wrap(order, sides, initial.copyOf())
                            engine.gemm(alpha, a, true, b, transposeB, beta, actual, scratch = Scratch())

                            assertContentEquals(
                                expected.values,
                                actual.values,
                                "$name sides=$sides transposeB=$transposeB alpha=$alpha beta=$beta",
                            )
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `a staged product agrees with the same product in the caller layout`() {
        val rng = Random(20261001)
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val m = ORDER
                val k = DEPTH
                val n = SIDES
                val a = denseEnough(if (transposeA) k else m, if (transposeA) m else k, rng)
                val b = if (transposeB) randomMatrix(n, k, rng) else randomMatrix(k, n, rng)
                val c0 = randomMatrix(m, n, rng)
                val context = "transposeA=$transposeA transposeB=$transposeB"

                val expected = c0.copyOf()
                ReferenceSparseBlas.gemm(ALPHA, a, transposeA, b, transposeB, BETA, expected)
                val stagedResult = c0.copyOf()
                staging.gemm(
                    ALPHA, a, transposeA, b, transposeB, BETA, stagedResult, right = false, scratch = Scratch(),
                )
                val plainResult = c0.copyOf()
                plain.gemm(
                    ALPHA, a, transposeA, b, transposeB, BETA, plainResult, right = false, scratch = Scratch(),
                )

                // A transposed operand against a transposed dense block already has its right-hand sides
                // adjacent, so that one pair takes the adjacent body with no copy at all.
                assertEquals(
                    !(transposeA && transposeB),
                    rhsStaged(productRhsPlan(staging, m, n, k, a.nnz, transposeA, transposeB)),
                    "$context staged the wrong way round",
                )
                assertClose(expected.values, stagedResult.values, "staged product $context")
                assertClose(plainResult.values, stagedResult.values, "staged against unstaged $context")
            }
        }
    }

    @Test
    fun `a staged symmetric product agrees with the same product in the caller layout`() {
        val rng = Random(20261002)
        for (lower in booleanArrayOf(true, false)) {
            val order = ORDER
            // A thinner operand runs unstaged and would leave this comparing two unstaged calls.
            val a = filled(order, rng)
            val b = randomMatrix(order, SIDES, rng)
            val c0 = randomMatrix(order, SIDES, rng)
            val expected = c0.copyOf()
            ReferenceSparseBlas.symm(ALPHA, a, b, BETA, expected, lower)

            assertTrue(
                rhsStaged(symmetricRhsPlan(staging, order, SIDES, a.nnz)),
                "the symmetric fixture did not stage at lower=$lower",
            )
            val staged = c0.copyOf()
            staging.symm(ALPHA, a, b, BETA, staged, lower, right = false, scratch = Scratch())
            val unstaged = c0.copyOf()
            plain.symm(ALPHA, a, b, BETA, unstaged, lower, right = false, scratch = Scratch())

            assertClose(expected.values, staged.values, "staged symmetric product lower=$lower")
            assertClose(unstaged.values, staged.values, "staged against unstaged lower=$lower")
        }
    }

    // Staging writes a block back, which is a second place an alias could be read after being overwritten.
    @Test
    fun `a staged call reads the operand it was given when that operand is the destination`() {
        val rng = Random(20261006)
        val order = ORDER
        val a = filled(order, rng)
        val shared = randomMatrix(order, SIDES, rng)

        val expectedProduct = shared.copyOf()
        plain.gemm(ALPHA, a, false, shared.copyOf(), false, BETA, expectedProduct, right = false, scratch = Scratch())
        val aliasedProduct = shared.copyOf()
        staging.gemm(ALPHA, a, false, aliasedProduct, false, BETA, aliasedProduct, right = false, scratch = Scratch())
        val expectedSymmetric = shared.copyOf()
        plain.symm(ALPHA, a, shared.copyOf(), BETA, expectedSymmetric, lower = true, right = false, scratch = Scratch())
        val aliasedSymmetric = shared.copyOf()
        staging.symm(
            ALPHA,
            a,
            aliasedSymmetric,
            BETA,
            aliasedSymmetric,
            lower = true,
            right = false,
            scratch = Scratch(),
        )

        assertClose(expectedProduct.values, aliasedProduct.values, "a staged product against its own source")
        assertClose(
            expectedSymmetric.values,
            aliasedSymmetric.values,
            "a staged symmetric product against its own source",
        )
    }

    // A negative zero distinguishes writing the entry back from writing a fresh accumulator; every finite
    // check in this file would pass either way.
    @Test
    fun `a staged destination returns an entry the product never reached`() {
        val rng = Random(20261004)
        val m = ORDER
        val k = DEPTH
        val a = withEmptyRow(denseEnough(m, k, rng), row = EMPTY_ROW)
        val b = randomMatrix(k, SIDES, rng)
        val c = randomMatrix(m, SIDES, rng)
        for (side in 0 until SIDES) c.values[EMPTY_ROW + side * m] = -0.0

        staging.gemm(ALPHA, a, false, b, false, 1.0, c, right = false, scratch = Scratch())

        for (side in 0 until SIDES) {
            val at = EMPTY_ROW + side * m
            assertEquals(-0.0, c.values[at], "the untouched entry at $at came back as ${c.values[at]}")
            assertTrue(1.0 / c.values[at] < 0.0, "the untouched entry at $at lost its sign")
        }
    }

    /** Repeated staged calls over one shape keep asking for the same lengths, so the loans are reused. */
    @Test
    fun `a staged call reuses the buffers a previous one returned`() {
        val rng = Random(20261005)
        val a = denseEnough(ORDER, DEPTH, rng)
        val b = randomMatrix(DEPTH, SIDES, rng)
        val c = randomMatrix(ORDER, SIDES, rng)
        val workspace = Scratch()

        staging.gemm(ALPHA, a, false, b, false, BETA, c, right = false, scratch = workspace)
        val afterFirst = workspace.idleLengths()
        repeat(4) { staging.gemm(ALPHA, a, false, b, false, BETA, c, right = false, scratch = workspace) }

        assertTrue(afterFirst > 0, "a staged call borrowed nothing at all")
        assertEquals(afterFirst, workspace.idleLengths(), "a repeated staged call asked for new lengths")
    }

    // The opposite triangle is NaN, so reading it from either side would show in the product.
    @Test
    fun `a symmetric product reads only the selected triangle`() {
        val b = DenseMatrix.wrap(3, 2, doubleArrayOf(2.0, -3.0, 5.0, 11.0, 13.0, 17.0))
        val rightInput = DenseMatrix.wrap(2, 3, doubleArrayOf(2.0, 11.0, -3.0, 13.0, 5.0, 17.0))
        for (lower in booleanArrayOf(false, true)) {
            val a = nanOutsideTriangle(lower)
            val left = DenseMatrix.zero(3, 2)
            val right = DenseMatrix.zero(2, 3)

            koblas.symm(1.0, a, b, 0.0, left, lower)
            koblas.symm(1.0, a, rightInput, 0.0, right, lower, right = true)

            assertContentEquals(doubleArrayOf(27.0, 14.0, 28.0), left.values.copyOfRange(0, 3), "lower=$lower")
            for (j in 0 until 3) for (i in 0 until 2) assertEquals(left[j, i], right[i, j], "lower=$lower")
        }
    }

    // Adding a zero for the absent mirrored half would turn the infinity into a NaN.
    @Test
    fun `a diagonal only symmetric operand mirrors nothing`() {
        val a = SparseMatrix.ofColumns(1, 1, listOf(listOf(0 to 2.0)))
        for ((name, engine) in sparsePanelEngines()) {
            for (sides in intArrayOf(1, 3, 8)) {
                val b = DenseMatrix.wrap(1, sides, DoubleArray(sides) { 3.0 })
                val c = DenseMatrix.wrap(1, sides, DoubleArray(sides))

                engine.symm(Double.POSITIVE_INFINITY, a, b, 0.0, c, lower = true, right = false)

                for (side in 0 until sides) {
                    assertEquals(
                        Double.POSITIVE_INFINITY,
                        c.values[side],
                        "$name at $sides sides: an absent mirrored half was evaluated",
                    )
                }
            }
        }
    }

    // The only stored entry is on the diagonal of the second column, so the first row is neither scattered
    // into nor mirrored back into; the second row is touched, which is why only the first is asserted about.
    @Test
    fun `a symmetric row no entry reaches keeps its sign`() {
        val a = SparseMatrix.ofColumns(2, 2, listOf(emptyList(), listOf(1 to 3.0)))
        for ((name, engine) in sparsePanelEngines()) {
            for (sides in intArrayOf(1, 3)) {
                val b = DenseMatrix.wrap(2, sides, DoubleArray(2 * sides))
                val c = DenseMatrix.wrap(2, sides, DoubleArray(2 * sides) { -0.0 })

                engine.symm(1.0, a, b, 1.0, c, lower = true, right = false)

                for (side in 0 until sides) {
                    val at = side * 2
                    assertEquals(-0.0, c.values[at], "$name at $sides sides: entry $at changed")
                    assertTrue(1.0 / c.values[at] < 0.0, "$name at $sides sides: entry $at lost its sign")
                }
            }
        }
    }

    // Dense enough to pass the copy-cost rule, but the widest panel over this depth needs more than one array.
    @Test
    fun `a panel no array can hold is not staged`() {
        val wide = engineWithKernels(panels = WidePanels)
        val depth = Int.MAX_VALUE / 4

        val plan = productRhsPlan(
            wide,
            1,
            WidePanels.GROUP,
            depth,
            Int.MAX_VALUE,
            transposeSparse = true,
            transposeDense = false,
        )

        assertFalse(rhsStaged(plan))
    }

    // A stored infinity against a dense zero and a stored zero against a dense infinity are both NaN.
    @Test
    fun `a sparse dense product forms the product of every stored entry`() {
        for (right in booleanArrayOf(false, true)) {
            for (storedInfinity in booleanArrayOf(false, true)) {
                val a = SparseMatrix.ofColumns(
                    1,
                    1,
                    listOf(listOf(0 to if (storedInfinity) Double.POSITIVE_INFINITY else 0.0)),
                )
                val b = DenseMatrix.wrap(1, 1, doubleArrayOf(if (storedInfinity) 0.0 else Double.POSITIVE_INFINITY))
                val c = DenseMatrix.zero(1, 1)

                koblas.gemm(1.0, a, false, b, false, 0.0, c, right = right)

                assertTrue(c[0, 0].isNaN(), "right=$right storedInfinity=$storedInfinity produced ${c[0, 0]}")
            }
        }
    }

    @Test
    fun `an alpha of zero leaves the destination scaled by beta alone`() {
        val rng = Random(20260829)
        val sparse = randomSparse(4, 3, rng)
        val b = randomMatrix(3, 2, rng)
        val c = randomMatrix(4, 2, rng)
        val expected = DenseMatrix.wrap(4, 2, DoubleArray(8) { c.values[it] * 2.0 })

        val actual = c.copyOf()
        koblas.gemm(0.0, sparse, false, b, false, 2.0, actual)

        assertClose(expected, actual, "alpha = 0")
    }

    /** A symmetric operand whose stored opposite triangle is NaN, so a read of it shows in any product. */
    private fun nanOutsideTriangle(lower: Boolean): SparseMatrix = if (lower) {
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

    /** Stored entries enough that staging pays for itself under the shipped crossover. */
    private fun denseEnough(rows: Int, cols: Int, rng: Random): SparseMatrix {
        val columns = List(cols) { List(rows) { i -> i to rng.nextDouble(-1.0, 1.0) } }
        return SparseMatrix.ofColumns(rows, cols, columns)
    }

    private fun withEmptyRow(a: SparseMatrix, row: Int): SparseMatrix {
        val columns = List(a.cols) { j ->
            buildList { a.forEachInColumn(j) { i, value -> if (i != row) add(i to value) } }
        }
        return SparseMatrix.ofColumns(a.rows, a.cols, columns)
    }

    /** Every position stored, which is what a symmetric product needs before its three copies pay for it. */
    private fun filled(order: Int, rng: Random): SparseMatrix = SparseMatrix.ofColumns(
        order,
        order,
        List(order) { (0 until order).map { i -> i to rng.nextDouble(-1.0, 1.0) } },
    )

    private companion object {
        const val ALPHA = 0.875
        const val BETA = -0.25

        /** Extents the staging rule admits; a smaller fixture would silently compare two unstaged calls. */
        const val ORDER = 24
        const val DEPTH = 16

        /** A row of the operand with nothing stored, so the destination there is never reached. */
        const val EMPTY_ROW = 4

        /** Wider than the grouping, so a call cuts more than one group and leaves a short last one. */
        const val SIDES = 5
    }
}

/** [AdjacentPreferringPanels] at the widest group the staging rule admits, wider than any copy-cost ratio. */
private object WidePanels : EngineKernels by AdjacentPreferringPanels {
    const val GROUP = 64

    override fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): Int =
        if (columns <= 0) 1 else minOf(GROUP, columns)
}
