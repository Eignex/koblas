package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.EngineImplementation
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.copyOf
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.engineWithKernels
import com.eignex.koblas.internal.dense.PanelWork
import com.eignex.koblas.internal.dense.PortablePanelKernels
import com.eignex.koblas.internal.operands.staged
import com.eignex.koblas.internal.sparse.SparsePanelKernels
import com.eignex.koblas.internal.sparse.productRhsPlan
import com.eignex.koblas.internal.sparse.rhsStaged
import com.eignex.koblas.internal.sparse.symmetricRhsPlan
import com.eignex.koblas.internal.sparse.triangularRhsPlan
import com.eignex.koblas.koblas
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// Whether a call stages is the backend's own answer, so a runtime with no vector backend would never stage
// and leave this untested. [AdjacentPreferringPanels] asks for adjacent right-hand sides and keeps the
// portable arithmetic, which puts the copy, the addressing and the writeback under test everywhere.
class SparseRhsPanelTest {
    private val stagingPanels = SparsePanelKernels(engineWithKernels(panels = AdjacentPreferringPanels()))
    private val staging = engineWith(AdjacentPreferringPanels())
    private val plain = engineWith(PortablePanelKernels)

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

                        for ((name, engine) in engines()) {
                            val actual = DenseMatrix.wrap(order, sides, initial.copyOf())
                            engine.gemm(alpha, a, true, b, transposeB, beta, actual, workspace = Workspace())

                            assertGemmAgreesWithReference(
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

    private fun assertGemmAgreesWithReference(expected: DoubleArray, actual: DoubleArray, context: String) {
        assertContentEquals(expected, actual, context)
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
                val b = if (transposeB) randomDense(n, k, rng) else randomDense(k, n, rng)
                val c0 = randomDense(m, n, rng)
                val context = "transposeA=$transposeA transposeB=$transposeB"

                val expected = DenseMatrix.wrap(m, n, c0.values.copyOf())
                ReferenceSparseBlas.gemm(ALPHA, a, transposeA, b, transposeB, BETA, expected)
                val stagedResult = run(a, transposeA, b, transposeB, c0, staging)
                val plainResult = run(a, transposeA, b, transposeB, c0, plain)

                // A transposed operand against a transposed dense block already has its right-hand sides
                // adjacent, so that one pair takes the adjacent body with no copy at all.
                assertEquals(
                    !(transposeA && transposeB),
                    rhsStaged(productRhsPlan(stagingPanels, m, n, k, a.nnz, transposeA, transposeB)),
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
            val b = randomDense(order, SIDES, rng)
            val c0 = randomDense(order, SIDES, rng)
            val expected = DenseMatrix.wrap(order, SIDES, c0.values.copyOf())
            ReferenceSparseBlas.symm(ALPHA, a, b, BETA, expected, lower)

            assertTrue(
                rhsStaged(symmetricRhsPlan(stagingPanels, order, SIDES, a.nnz)),
                "the symmetric fixture did not stage at lower=$lower",
            )
            val staged = DenseMatrix.wrap(order, SIDES, c0.values.copyOf())
            staging.symm(ALPHA, a, b, BETA, staged, lower, right = false, workspace = Workspace())
            val unstaged = DenseMatrix.wrap(order, SIDES, c0.values.copyOf())
            plain.symm(ALPHA, a, b, BETA, unstaged, lower, right = false, workspace = Workspace())

            assertClose(expected.values, staged.values, "staged symmetric product lower=$lower")
            assertClose(unstaged.values, staged.values, "staged against unstaged lower=$lower")
        }
    }

    @Test
    fun `a staged triangular block agrees with the same call in the caller layout`() {
        val rng = Random(20261003)
        for (solve in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (unit in booleanArrayOf(false, true)) {
                    for (lower in booleanArrayOf(true, false)) {
                        val order = ORDER
                        // A thinner triangle would run unstaged.
                        val t = wholeTriangle(order, lower, rng)
                        val source = randomDense(order, SIDES, rng)
                        val context = "solve=$solve transpose=$transpose unit=$unit lower=$lower"
                        assertTrue(
                            rhsStaged(triangularRhsPlan(stagingPanels, order, SIDES, t.nnz, transpose)),
                            "the triangular fixture did not stage at $context",
                        )

                        val staged = DenseMatrix.wrap(order, SIDES, source.values.copyOf())
                        val unstaged = DenseMatrix.wrap(order, SIDES, source.values.copyOf())
                        if (solve) {
                            staging.trsm(t, staged, lower, transpose, unit, false, ALPHA, Workspace())
                            plain.trsm(t, unstaged, lower, transpose, unit, false, ALPHA, Workspace())
                        } else {
                            staging.trmm(t, staged, lower, transpose, unit, false, ALPHA, Workspace())
                            plain.trmm(t, unstaged, lower, transpose, unit, false, ALPHA, Workspace())
                        }
                        assertClose(unstaged.values, staged.values, "staged triangular block $context")
                    }
                }
            }
        }
    }

    // Staging writes a block back, which is a second place an alias could be read after being overwritten.
    @Test
    fun `a staged call reads the operand it was given when that operand is the destination`() {
        val rng = Random(20261006)
        val order = ORDER
        val a = filled(order, rng)
        val shared = randomDense(order, SIDES, rng)

        val expectedProduct = DenseMatrix.wrap(order, SIDES, shared.values.copyOf())
        plain.gemm(
            ALPHA, a, false, DenseMatrix.wrap(order, SIDES, shared.values.copyOf()), false,
            BETA, expectedProduct, right = false, workspace = Workspace(),
        )
        val aliasedProduct = DenseMatrix.wrap(order, SIDES, shared.values.copyOf())
        staging.gemm(
            ALPHA, a, false, aliasedProduct, false, BETA, aliasedProduct, right = false, workspace = Workspace(),
        )
        assertClose(expectedProduct.values, aliasedProduct.values, "a staged product against its own source")

        val expectedSymmetric = DenseMatrix.wrap(order, SIDES, shared.values.copyOf())
        plain.symm(
            ALPHA,
            a,
            DenseMatrix.wrap(order, SIDES, shared.values.copyOf()),
            BETA,
            expectedSymmetric,
            lower = true,
            right = false,
            workspace = Workspace(),
        )
        val aliasedSymmetric = DenseMatrix.wrap(order, SIDES, shared.values.copyOf())
        staging.symm(
            ALPHA,
            a,
            aliasedSymmetric,
            BETA,
            aliasedSymmetric,
            lower = true,
            right = false,
            workspace = Workspace(),
        )
        assertClose(
            expectedSymmetric.values,
            aliasedSymmetric.values,
            "a staged symmetric product against its own source",
        )
    }

    // The order is odd so a whole lower triangle holds a multiple of it, which is what lets one array be
    // both the triangle's coefficients and the block's elements.
    @Test
    fun `a staged triangular call reads the triangle it was given when the block is its values`() {
        val order = 17
        val entries = order * (order + 1) / 2
        val sides = entries / order
        val pointers = IntArray(order + 1)
        val rows = IntArray(entries)
        var at = 0
        for (j in 0 until order) {
            for (i in j until order) {
                rows[at] = i
                at++
            }
            pointers[j + 1] = at
        }
        val block = DoubleArray(entries) { 0.25 + (it % 7) * 0.125 }

        val independent = SparseMatrix.wrap(order, order, pointers, rows, block.copyOf())
        val expected = DenseMatrix.wrap(order, sides, block.copyOf())
        plain.trsm(independent, expected, true, false, false, false, ALPHA, Workspace())

        val sharedTriangle = SparseMatrix.wrap(order, order, pointers, rows, block.copyOf())
        val aliased = DenseMatrix.wrap(order, sides, sharedTriangle.values)
        assertTrue(
            rhsStaged(triangularRhsPlan(stagingPanels, order, sides, sharedTriangle.nnz, transpose = false)),
            "the aliased triangular fixture did not stage",
        )
        staging.trsm(sharedTriangle, aliased, true, false, false, false, ALPHA, Workspace())

        assertClose(expected.values, aliased.values, "a staged solve against its own triangle")
    }

    // A negative zero distinguishes writing the entry back from writing a fresh accumulator; every finite
    // check in this file would pass either way.
    @Test
    fun `a staged destination returns an entry the product never reached`() {
        val rng = Random(20261004)
        val m = ORDER
        val k = DEPTH
        val a = withEmptyRow(denseEnough(m, k, rng), row = EMPTY_ROW)
        val b = randomDense(k, SIDES, rng)
        val c = randomDense(m, SIDES, rng)
        for (side in 0 until SIDES) c.values[EMPTY_ROW + side * m] = -0.0

        staging.gemm(ALPHA, a, false, b, false, 1.0, c, right = false, workspace = Workspace())

        for (side in 0 until SIDES) {
            val at = EMPTY_ROW + side * m
            assertEquals(-0.0, c.values[at], "the untouched entry at $at came back as ${c.values[at]}")
            assertTrue(1.0 / c.values[at] < 0.0, "the untouched entry at $at lost its sign")
        }
    }

    // One extent is zero and the other is a million, which separates returning early from stepping through
    // a million empty panels.
    @Test
    fun `an empty destination stages nothing and borrows nothing`() {
        val workspace = Workspace()
        val wide = 1 shl 20
        val emptySparse = SparseMatrix.wrap(0, 0, IntArray(1), IntArray(0), DoubleArray(0))
        val emptyDense = DenseMatrix.wrap(0, wide, DoubleArray(0))

        staging.symm(ALPHA, emptySparse, emptyDense, BETA, emptyDense, true, right = false, workspace = workspace)
        staging.gemm(ALPHA, emptySparse, false, emptyDense, false, BETA, emptyDense, false, workspace)
        staging.trsm(emptySparse, emptyDense, true, false, false, false, ALPHA, workspace)
        staging.trmm(emptySparse, emptyDense, true, false, false, false, ALPHA, workspace)

        assertEquals(0, workspace.idleLengths(), "an empty call took a loan from the workspace")
    }

    /** Repeated staged calls over one shape keep asking for the same lengths, so the loans are reused. */
    @Test
    fun `a staged call reuses the buffers a previous one returned`() {
        val rng = Random(20261005)
        val m = ORDER
        val k = DEPTH
        val a = denseEnough(m, k, rng)
        val b = randomDense(k, SIDES, rng)
        val c = randomDense(m, SIDES, rng)
        val workspace = Workspace()

        staging.gemm(ALPHA, a, false, b, false, BETA, c, right = false, workspace = workspace)
        val afterFirst = workspace.idleLengths()
        repeat(4) { staging.gemm(ALPHA, a, false, b, false, BETA, c, right = false, workspace = workspace) }

        assertTrue(afterFirst > 0, "a staged call borrowed nothing at all")
        assertEquals(afterFirst, workspace.idleLengths(), "a repeated staged call asked for new lengths")
    }

    // Adding a zero for the absent mirrored half would turn the infinity into a NaN.
    @Test
    fun `a diagonal only symmetric operand mirrors nothing`() {
        val a = SparseMatrix.ofColumns(1, 1, listOf(listOf(0 to 2.0)))
        for (engine in engines()) {
            for (sides in intArrayOf(1, 3, 8)) {
                val b = DenseMatrix.wrap(1, sides, DoubleArray(sides) { 3.0 })
                val c = DenseMatrix.wrap(1, sides, DoubleArray(sides))
                engine.second.symm(
                    Double.POSITIVE_INFINITY,
                    a,
                    b,
                    0.0,
                    c,
                    lower = true,
                    right = false,
                    workspace = null,
                )
                for (side in 0 until sides) {
                    assertEquals(
                        Double.POSITIVE_INFINITY,
                        c.values[side],
                        "${engine.first} at $sides sides: an absent mirrored half was evaluated",
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
        for (engine in engines()) {
            for (sides in intArrayOf(1, 3)) {
                val b = DenseMatrix.wrap(2, sides, DoubleArray(2 * sides))
                val c = DenseMatrix.wrap(2, sides, DoubleArray(2 * sides) { -0.0 })
                engine.second.symm(1.0, a, b, 1.0, c, lower = true, right = false, workspace = null)
                for (side in 0 until sides) {
                    val at = side * 2
                    assertEquals(-0.0, c.values[at], "${engine.first} at $sides sides: entry $at changed")
                    assertTrue(
                        1.0 / c.values[at] < 0.0,
                        "${engine.first} at $sides sides: entry $at lost its sign",
                    )
                }
            }
        }
    }

    @Test
    fun `a zero right hand side keeps its skip in every layout`() {
        val triangle = SparseMatrix.ofColumns(
            2,
            2,
            listOf(
                listOf(0 to 1.0, 1 to Double.POSITIVE_INFINITY),
                listOf(1 to 1.0),
            ),
        )
        val sides = 4
        // Live right-hand sides exercise the mixed mask; only zero right-hand sides have a no-read result.
        val source = DoubleArray(2 * sides)
        source[0] = 1.0
        source[4] = 1.0

        for ((name, engine) in engines()) {
            val b = DenseMatrix.wrap(2, sides, source.copyOf())
            engine.trsm(triangle, b, lower = true, transpose = false, unitDiag = false, right = false, alpha = 1.0)

            assertEquals(0.0, b[1, 1], "$name: a zero right-hand side was not skipped")
            assertEquals(0.0, b[1, 3], "$name: a zero right-hand side was not skipped")
        }
    }

    @Test
    fun `a zero right hand side keeps its skip in a triangular multiply`() {
        val triangle = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 1.0, 1 to Double.POSITIVE_INFINITY), listOf(1 to 1.0)),
        )
        val sides = 4
        val source = DoubleArray(2 * sides)
        source[0] = 1.0
        source[4] = 1.0

        for ((name, engine) in engines()) {
            val b = DenseMatrix.wrap(2, sides, source.copyOf())
            engine.trmm(triangle, b, lower = true, transpose = false, unitDiag = false, right = false, alpha = 1.0)

            assertEquals(0.0, b[1, 1], "$name: a zero right-hand side was not skipped")
        }
    }

    // One right-hand side uses a leaf that a wider panel cannot reach, so its zero skip is checked separately.
    @Test
    fun `a zero right hand side keeps its skip at one right hand side`() {
        val solveTriangle = SparseMatrix.ofColumns(
            2,
            2,
            listOf(
                listOf(0 to 1.0, 1 to Double.POSITIVE_INFINITY),
                listOf(1 to 1.0),
            ),
        )
        val multiplyTriangle = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 1.0, 1 to Double.POSITIVE_INFINITY), listOf(1 to 1.0)),
        )

        for ((name, engine) in engines()) {
            val dead = DenseMatrix.wrap(2, 1, doubleArrayOf(0.0, 0.0))
            engine.trsm(solveTriangle, dead, true, transpose = false, unitDiag = false, right = false)
            assertEquals(0.0, dead[1, 0], "$name: a zero right-hand side was not skipped by the solve")

            val quiet = DenseMatrix.wrap(2, 1, doubleArrayOf(0.0, 0.0))
            engine.trmm(multiplyTriangle, quiet, true, transpose = false, unitDiag = false, right = false)
            assertEquals(0.0, quiet[1, 0], "$name: a zero right-hand side was not skipped by the multiply")
        }
    }

    // The two-by-two fixtures above are too small to stage, so the same rule is checked here in the staged
    // layout, with one group holding a dead right-hand side and another all live.
    @Test
    fun `a zero right hand side keeps its skip inside a staged panel`() {
        val order = 16
        val sides = 4
        val columns = List(order) { j ->
            (j until order).map { i ->
                i to when {
                    i == j -> 1.0
                    i == 1 && j == 0 -> Double.POSITIVE_INFINITY
                    else -> 1.0
                }
            }
        }
        val triangle = SparseMatrix.ofColumns(order, order, columns)
        val source = DoubleArray(order * sides)
        for (side in intArrayOf(0, 2, 3)) for (row in 0 until order) source[row + side * order] = 1.0
        val b = DenseMatrix.wrap(order, sides, source.copyOf())

        assertTrue(
            rhsStaged(triangularRhsPlan(stagingPanels, order, sides, triangle.nnz, transpose = false)),
            "the fixture did not stage",
        )

        staging.trsm(triangle, b, lower = true, transpose = false, unitDiag = false, right = false, alpha = 1.0)

        for (row in 0 until order) {
            assertEquals(0.0, b[row, 1], "a zero right-hand side was written at row $row")
        }
    }

    @Test
    fun `triangular panels compute live right hand sides across layouts`() {
        for (order in intArrayOf(2, 16)) {
            val columns = List(order) { j -> (j until order).map { i -> i to if (i == j) 2.0 else 0.25 } }
            val triangle = SparseMatrix.ofColumns(order, order, columns)
            val dense = DenseMatrix.wrap(order, order, DoubleArray(order * order))
            for (j in columns.indices) for ((i, value) in columns[j]) dense.values[i + j * order] = value
            for (sides in intArrayOf(1, 4)) {
                for (mixed in booleanArrayOf(false, true)) {
                    val source = DoubleArray(order * sides) { if (mixed && it / order == 1) 0.0 else 1.0 }
                    for (solve in booleanArrayOf(false, true)) {
                        val expected = DenseMatrix.wrap(order, sides, source.copyOf())
                        if (solve) {
                            ReferenceBlas.trsm(dense, expected, lower = true)
                        } else {
                            ReferenceBlas.trmm(dense, expected, lower = true)
                        }

                        for ((name, engine) in engines()) {
                            val actual = DenseMatrix.wrap(order, sides, source.copyOf())
                            if (solve) {
                                engine.trsm(triangle, actual, lower = true, workspace = Workspace())
                            } else {
                                engine.trmm(triangle, actual, lower = true, workspace = Workspace())
                            }

                            assertTriangularAgreesWithReference(
                                expected.values,
                                actual.values,
                                "$name order=$order sides=$sides mixed=$mixed solve=$solve",
                            )
                        }
                    }
                }
            }
        }
    }

    private fun assertTriangularAgreesWithReference(expected: DoubleArray, actual: DoubleArray, context: String) {
        assertClose(expected, actual, context)
    }

    /** Every sparse engine this runtime has, so a rule is checked where it is implemented and where selected. */
    private fun engines(): List<Pair<String, KoblasEngine>> = buildList {
        add("strided" to plain)
        add("staged" to staging)
        add("selected" to koblas)
        add("scalar" to BuiltinEngines.scalar)
        BuiltinEngines.simd?.let { add("simd" to it) }
    }

    private fun run(
        a: SparseMatrix,
        transposeA: Boolean,
        b: DenseMatrix,
        transposeB: Boolean,
        c0: DenseMatrix,
        engine: KoblasEngine,
    ): DenseMatrix {
        val c = c0.copyOf()
        engine.gemm(ALPHA, a, transposeA, b, transposeB, BETA, c, right = false, workspace = Workspace())
        return c
    }

    private fun randomDense(rows: Int, cols: Int, rng: Random): DenseMatrix =
        DenseMatrix.wrap(rows, cols, DoubleArray(rows * cols) { rng.nextDouble(-1.0, 1.0) })

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

    /** Every position of the selected triangle, with a diagonal that dominates the column it heads. */
    private fun wholeTriangle(order: Int, lower: Boolean, rng: Random): SparseMatrix = SparseMatrix.ofColumns(
        order,
        order,
        List(order) { j ->
            val rows = if (lower) j until order else 0..j
            rows.map { i -> i to if (i == j) 2.0 * order else rng.nextDouble(-1.0, 1.0) }
        },
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

/** An engine's sparse algorithms over a chosen panel backend, which is what selects the staged schedule. */
internal fun engineWith(panels: KoblasEngine): KoblasEngine = engineWithKernels(panels = panels)

/**
 * A backend that asks for adjacent right-hand sides and computes with the portable bodies, so the staged
 * schedule runs on a platform with no vector bodies at all.
 *
 * [group] is deliberately not a lane count and not a power of two, so a call cuts groups the shipped
 * backends never would.
 */
internal open class AdjacentPreferringPanels(
    private val group: Int = 3,
    private val minimumRows: Int = 1,
    private val entryThreshold: Int? = null,
    private val delegate: KoblasEngine = PortablePanelKernels,
) : KoblasEngine() {
    override val implementations = setOf(EngineImplementation.Panels)

    override fun multiDot(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        beta: Double,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ): Unit = delegate.multiDot(alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, beta, y, yOffset, yStride)

    override fun columnUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
    ): Unit = delegate.columnUpdate(alpha, a, aOffset, lda, x, xOffset, xStride, rows, columns, y, yOffset, yStride)

    override fun coupledUpdateDot(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        rows: Int,
        columns: Int,
        y: DoubleArray,
        yOffset: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        sums: DoubleArray,
        sumOffset: Int,
        xStride: Int,
        yStride: Int,
        coefficientStride: Int,
        sumStride: Int,
    ): Unit = delegate.coupledUpdateDot(
        alpha,
        a,
        aOffset,
        lda,
        x,
        xOffset,
        rows,
        columns,
        y,
        yOffset,
        coefficients,
        coefficientOffset,
        sums,
        sumOffset,
        xStride,
        yStride,
        coefficientStride,
        sumStride,
    )

    override fun rankUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        rows: Int,
        columns: Int,
        coefficients: DoubleArray,
        coefficientOffset: Int,
        coefficientStride: Int,
    ): Unit = delegate.rankUpdate(
        alpha,
        a,
        aOffset,
        lda,
        x,
        xOffset,
        xStride,
        rows,
        columns,
        coefficients,
        coefficientOffset,
        coefficientStride,
    )

    override fun rankTwoUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        lda: Int,
        x: DoubleArray,
        xOffset: Int,
        xStride: Int,
        y: DoubleArray,
        yOffset: Int,
        yStride: Int,
        rows: Int,
        columns: Int,
        u: DoubleArray,
        uOffset: Int,
        uStride: Int,
        v: DoubleArray,
        vOffset: Int,
        vStride: Int,
    ): Unit = delegate.rankTwoUpdate(
        alpha,
        a,
        aOffset,
        lda,
        x,
        xOffset,
        xStride,
        y,
        yOffset,
        yStride,
        rows,
        columns,
        u,
        uOffset,
        uStride,
        v,
        vOffset,
        vStride,
    )

    override fun indexedColumnUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        y: DoubleArray,
        yOffset: Int,
    ): Unit = delegate.indexedColumnUpdate(
        alpha,
        a,
        aOffset,
        rowStride,
        indexStride,
        indices,
        values,
        fromIndex,
        columns,
        rows,
        y,
        yOffset,
    )

    override fun indexedRankUpdate(
        alpha: Double,
        a: DoubleArray,
        aOffset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        x: DoubleArray,
        xOffset: Int,
    ): Unit = delegate.indexedRankUpdate(
        alpha,
        a,
        aOffset,
        rowStride,
        indexStride,
        indices,
        values,
        fromIndex,
        columns,
        rows,
        x,
        xOffset,
    )

    override fun indexedCoupledUpdate(
        alpha: Double,
        a: DoubleArray,
        b: DoubleArray,
        offset: Int,
        rowStride: Int,
        indexStride: Int,
        indices: IntArray,
        values: DoubleArray,
        fromIndex: Int,
        columns: Int,
        rows: Int,
        pivot: Int,
        sums: DoubleArray,
        sumOffset: Int,
        excluded: Int,
        destinationOffset: Int,
        destinationRowStride: Int,
        destinationIndexStride: Int,
    ): Unit = delegate.indexedCoupledUpdate(
        alpha,
        a,
        b,
        offset,
        rowStride,
        indexStride,
        indices,
        values,
        fromIndex,
        columns,
        rows,
        pivot,
        sums,
        sumOffset,
        excluded,
        destinationOffset,
        destinationRowStride,
        destinationIndexStride,
    )

    /** The portable bodies this backend computes with, which a subclass recording its runs delegates to. */
    protected val bodies: KoblasEngine get() = delegate

    override val name: String get() = "adjacent-preferring"

    override fun executionGroup(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): Int = when {
        !sparse(work) || !contiguous -> delegate.executionGroup(work, rows, columns, contiguous)
        columns <= 0 -> 1
        else -> minOf(group, columns)
    }

    override fun prefersContiguous(work: PanelWork, rows: Int, columns: Int): Boolean =
        sparse(work) && rows >= minimumRows

    override fun implementationFor(work: PanelWork, rows: Int, columns: Int, contiguous: Boolean): String = when {
        !sparse(work) || !contiguous || rows < minimumRows ->
            delegate.implementationFor(work, rows, columns, contiguous)

        entryThreshold != null && columns < entryThreshold -> "$name-short"

        else -> name
    }

    /** Both sparse panel shapes, since what this stands in for is a backend that vectorises either. */
    private fun sparse(work: PanelWork): Boolean =
        work == PanelWork.SparseRightHandSides || work == PanelWork.SparseRightHandSideReduction
}
