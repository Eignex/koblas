package com.eignex.koblas.internal.scheduling.sparse

import com.eignex.koblas.*
import com.eignex.koblas.dense.ReferenceBlas
import com.eignex.koblas.internal.engine.panelProvider
import com.eignex.koblas.internal.kernels.PanelWork
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import com.eignex.koblas.internal.operations.*
import kotlin.random.Random
import kotlin.test.*

class TriangularSchedulingTest {
    private val staging = engineWithKernels(panels = AdjacentPreferringPanels)
    private val plain = engineWithKernels(panels = ScalarEngine)

    private fun triangle(n: Int, lower: Boolean, rng: Random, density: Double = 0.4): SparseMatrix {
        val columns = ArrayList<List<Pair<Int, Double>>>(n)
        for (j in 0 until n) {
            val column = ArrayList<Pair<Int, Double>>()
            for (i in 0 until n) {
                val inTriangle = if (lower) i >= j else i <= j
                if (!inTriangle) continue
                val v = when {
                    i == j -> 2.0 + rng.nextDouble()
                    rng.nextDouble() < density -> rng.nextDouble(-1.0, 1.0)
                    else -> 0.0
                }
                if (v != 0.0) column.add(i to v)
            }
            columns.add(column)
        }
        return SparseMatrix.ofColumns(n, n, columns)
    }

    @Test
    fun `matrix solves agree with the scalar reference across triangle flags`() {
        assertTriangularAgreesWithReference(solve = true)
    }

    @Test
    fun `matrix multiplies agree with the scalar reference across triangle flags`() {
        assertTriangularAgreesWithReference(solve = false)
    }

    private fun assertTriangularAgreesWithReference(solve: Boolean) {
        val rng = Random(20261013)
        val order = 32
        for (engine in listOfNotNull(KoblasEngine.scalar, KoblasEngine.simd).distinct()) {
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    val work = if (transpose) PanelWork.SparseRightHandSideReduction else PanelWork.SparseRightHandSides
                    val group = engine.panelProvider.executionGroup(
                        work,
                        order,
                        1024,
                        contiguous = engine == KoblasEngine.simd,
                    )
                    val sides = group + 1
                    for (unitDiag in booleanArrayOf(false, true)) {
                        // Small off-diagonal entries keep the implicit unit triangle well conditioned too.
                        val dense = wellConditioned(order, rng)
                        val sparse = SparseMatrix.ofColumns(
                            order,
                            order,
                            List(order) { j ->
                                val rows = if (lower) j until order else 0..j
                                rows.map { i ->
                                    if (i != j) dense[i, j] /= order
                                    i to if (unitDiag && i == j) Double.NaN else dense[i, j]
                                }
                            },
                        )
                        for (right in booleanArrayOf(false, true)) {
                            val input = if (right) randomMatrix(sides, order, rng) else randomMatrix(order, sides, rng)
                            val expected = input.copyOf()
                            val actual = input.copyOf()
                            if (solve) {
                                ReferenceBlas.trsm(dense, expected, lower, transpose, unitDiag, right, alpha = -0.75)
                                engine.trsm(
                                    sparse,
                                    actual,
                                    lower,
                                    transpose,
                                    unitDiag,
                                    right,
                                    -0.75,
                                    scratch =
                                    Scratch(),
                                )
                            } else {
                                ReferenceBlas.trmm(dense, expected, lower, transpose, unitDiag, right, alpha = -0.75)
                                engine.trmm(
                                    sparse,
                                    actual,
                                    lower,
                                    transpose,
                                    unitDiag,
                                    right,
                                    -0.75,
                                    scratch =
                                    Scratch(),
                                )
                            }
                            val context = "${engine.name} solve=$solve lower=$lower transpose=$transpose " +
                                "unit=$unitDiag right=$right"
                            assertClose(expected, actual, context)
                        }
                    }
                }
            }
        }
    }

    @Test
    fun `an alpha of zero empties the destination without reading the triangle`() {
        val n = 3
        // A NaN diagonal, which a multiply or a solve would spread through the answer.
        val poisoned = SparseMatrix.ofColumns(n, n, List(n) { j -> listOf(j to Double.NaN) })
        for (solve in booleanArrayOf(false, true)) {
            val b = DenseMatrix.wrap(n, 2, DoubleArray(n * 2) { 1.0 })

            if (solve) {
                koblas.trsm(poisoned, b, lower = true, alpha = 0.0)
            } else {
                koblas.trmm(poisoned, b, lower = true, alpha = 0.0)
            }

            assertContentEquals(DoubleArray(n * 2), b.values, "solve=$solve")
        }
    }

    @Test
    fun `alpha scales the right hand side before the solve`() {
        val rng = Random(20260823)
        val n = 5
        val sparse = triangle(n, lower = true, rng)
        val b = randomMatrix(n, 2, rng)

        val scaled = DenseMatrix.wrap(n, 2, DoubleArray(b.values.size) { 2.5 * b.values[it] })
        koblas.trsm(sparse, scaled, lower = true)
        val withAlpha = DenseMatrix.wrap(n, 2, b.values.copyOf())
        koblas.trsm(sparse, withAlpha, lower = true, alpha = 2.5)

        assertClose(scaled, withAlpha, "alpha must scale B", tolerance = 1e-9)
    }

    // Whether a call stages is the backend's own answer, so these run over [AdjacentPreferringPanels], which
    // asks for adjacent right-hand sides on every platform.
    @Test
    fun `a staged triangular block agrees with the same call in the caller layout`() {
        val rng = Random(20261003)
        for (solve in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (unit in booleanArrayOf(false, true)) {
                    for (lower in booleanArrayOf(true, false)) {
                        // A thinner triangle would run unstaged.
                        val t = wholeTriangle(ORDER, lower, rng)
                        val source = randomMatrix(ORDER, SIDES, rng)
                        val context = "solve=$solve transpose=$transpose unit=$unit lower=$lower"
                        assertTrue(
                            triangularRhsPlan(staging, ORDER, SIDES, t.nnz, transpose).staged,
                            "the triangular fixture did not stage at $context",
                        )

                        val staged = source.copyOf()
                        val unstaged = source.copyOf()
                        if (solve) {
                            staging.trsm(t, staged, lower, transpose, unit, false, ALPHA, scratch = Scratch())
                            plain.trsm(t, unstaged, lower, transpose, unit, false, ALPHA, scratch = Scratch())
                        } else {
                            staging.trmm(t, staged, lower, transpose, unit, false, ALPHA, scratch = Scratch())
                            plain.trmm(t, unstaged, lower, transpose, unit, false, ALPHA, scratch = Scratch())
                        }

                        assertClose(unstaged.values, staged.values, "staged triangular block $context")
                    }
                }
            }
        }
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
        plain.trsm(independent, expected, true, false, false, false, ALPHA, scratch = Scratch())

        val sharedTriangle = SparseMatrix.wrap(order, order, pointers, rows, block.copyOf())
        val aliased = DenseMatrix.wrap(order, sides, sharedTriangle.values)
        assertTrue(
            triangularRhsPlan(staging, order, sides, sharedTriangle.nnz, transpose = false).staged,
            "the aliased triangular fixture did not stage",
        )
        staging.trsm(sharedTriangle, aliased, true, false, false, false, ALPHA, scratch = Scratch())

        assertClose(expected.values, aliased.values, "a staged solve against its own triangle")
    }

    // One right-hand side uses a leaf that a wider panel cannot reach, and four put live and dead ones in one
    // mask. Only a zero right-hand side has a no-read result, so that is the one asserted about.
    @Test
    fun `a zero right hand side keeps its skip in every layout`() {
        val triangle = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 1.0, 1 to Double.POSITIVE_INFINITY), listOf(1 to 1.0)),
        )
        for (solve in booleanArrayOf(true, false)) {
            for (sides in intArrayOf(1, 4)) {
                val source = DoubleArray(2 * sides)
                for (live in 1 until sides step 2) source[2 * live] = 1.0

                for ((name, engine) in sparsePanelEngines()) {
                    val b = DenseMatrix.wrap(2, sides, source.copyOf())

                    if (solve) {
                        engine.trsm(triangle, b, lower = true)
                    } else {
                        engine.trmm(triangle, b, lower = true)
                    }

                    for (dead in 0 until sides step 2) {
                        assertEquals(0.0, b[1, dead], "$name solve=$solve sides=$sides: side $dead was not skipped")
                    }
                }
            }
        }
    }

    // The two-by-two fixture above is too small to stage, so the same rule is checked here in the staged
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
            triangularRhsPlan(staging, order, sides, triangle.nnz, transpose = false).staged,
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

                        for ((name, engine) in sparsePanelEngines()) {
                            val actual = DenseMatrix.wrap(order, sides, source.copyOf())
                            if (solve) {
                                engine.trsm(triangle, actual, lower = true, scratch = Scratch())
                            } else {
                                engine.trmm(triangle, actual, lower = true, scratch = Scratch())
                            }

                            assertClose(
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

        /** Extents the staging rule admits; a smaller fixture would silently compare two unstaged calls. */
        const val ORDER = 24

        /** Wider than the grouping, so a call cuts more than one group and leaves a short last one. */
        const val SIDES = 5
    }
}
