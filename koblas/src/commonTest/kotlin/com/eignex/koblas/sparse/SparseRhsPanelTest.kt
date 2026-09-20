package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.SparseMatrix
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.copyOf
import com.eignex.koblas.dense.DensePanelKernels
import com.eignex.koblas.dense.PanelWork
import com.eignex.koblas.dense.PortablePanelKernels
import com.eignex.koblas.dense.ScalarVectorKernels
import com.eignex.koblas.koblas
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The staged orientation, on every platform.
 *
 * Whether a call stages its right-hand sides is the backend's answer about its own bodies, so on a runtime
 * with no vector backend nothing would ever stage and this code would go untested. The backend below says
 * it prefers adjacent right-hand sides and delegates the arithmetic to the portable bodies, which makes the
 * staged schedule run everywhere while leaving the arithmetic the portable one: what is under test here is
 * the copy, the addressing and the writeback, not a vector body.
 *
 * Every case is checked against the same call in the caller's own layout and against [ReferenceSparseBlas],
 * which is a traversal over stored coordinates and shares no code with either.
 */
class SparseRhsPanelTest {
    private val staging = engineWith(AdjacentPreferringPanels())
    private val plain = engineWith(PortablePanelKernels)

    @Test
    fun `a staged product agrees with the same product in the caller's layout`() {
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

                val components = panelComponents(a, transposeA, transposeB, m, n, k)
                assertTrue(
                    components.any { it.startsWith("adjacent-preferring") },
                    "$context reached no adjacent panel: $components",
                )
                // A transposed operand against a transposed dense block already has its right-hand sides
                // adjacent, so that one pair takes the adjacent body with no copy at all.
                assertEquals(
                    !(transposeA && transposeB),
                    components.any { it.startsWith(SPARSE_STAGING) },
                    "$context staged the wrong way round: $components",
                )
                assertClose(expected.values, stagedResult.values, "staged product $context")
                assertClose(plainResult.values, stagedResult.values, "staged against unstaged $context")
            }
        }
    }

    @Test
    fun `a staged symmetric product agrees with the same product in the caller's layout`() {
        val rng = Random(20261002)
        for (lower in booleanArrayOf(true, false)) {
            val order = ORDER
            // A symmetric product copies three passes per right-hand side, so its operand has to hold that
            // much more before the copy pays for itself; a thinner one runs in the caller's layout and would
            // leave this test comparing two unstaged calls.
            val a = filled(order, rng)
            val b = randomDense(order, SIDES, rng)
            val c0 = randomDense(order, SIDES, rng)
            val expected = DenseMatrix.wrap(order, SIDES, c0.values.copyOf())
            ReferenceSparseBlas.symm(ALPHA, a, b, BETA, expected, lower)

            assertTrue(
                stages(SparseMatrixOperation.SymmLeft, a, order * SIDES, order, SIDES, false, lower),
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
    fun `a staged triangular block agrees with the same call in the caller's layout`() {
        val rng = Random(20261003)
        for (solve in booleanArrayOf(true, false)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (unit in booleanArrayOf(false, true)) {
                    for (lower in booleanArrayOf(true, false)) {
                        val order = ORDER
                        // Every position of the selected triangle, because the staging rule asks for two
                        // passes per right-hand side and a thinner triangle would run unstaged.
                        val t = wholeTriangle(order, lower, rng)
                        val source = randomDense(order, SIDES, rng)
                        val context = "solve=$solve transpose=$transpose unit=$unit lower=$lower"
                        val operation = if (solve) {
                            SparseMatrixOperation.TrsmLeft
                        } else {
                            SparseMatrixOperation.TrmmLeft
                        }
                        assertTrue(
                            stages(operation, t, order * SIDES, order, SIDES, transpose, lower),
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

    /**
     * The staged paths against an operand that shares the destination's buffer.
     *
     * Staging copies a block into adjacent order and writes it back, which is a second place an alias could
     * be read after it was overwritten. The public calls stage an aliased operand before any of that, and
     * the check is that the answer is the one an unaliased call gives.
     */
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

    /**
     * A staged triangular block whose triangle is stored in the block's own buffer.
     *
     * The order is odd so that a whole lower triangle holds a multiple of it, which is what lets one array
     * be both the triangle's coefficients and the block's elements; that many entries is also what the
     * staging rule asks for at this order, so the call really does stage.
     */
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
            stages(SparseMatrixOperation.TrsmLeft, sharedTriangle, entries, order, sides, false, true),
            "the aliased triangular fixture did not stage",
        )
        staging.trsm(sharedTriangle, aliased, true, false, false, false, ALPHA, Workspace())

        assertClose(expected.values, aliased.values, "a staged solve against its own triangle")
    }

    /**
     * A staged destination is copied in and written back, so an entry the product never reached has to come
     * back as it went out rather than as the sum of nothing.
     *
     * A negative zero is what distinguishes the two: writing back a fresh accumulator would leave a positive
     * one there, and every finite check in this file would still pass.
     */
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

    /**
     * An operand with no elements to write is validated and then left alone.
     *
     * One extent of a dense block may be zero while the other is far larger than anything a test can walk,
     * which is the case that separates returning early from planning a group traversal over nothing: the
     * workspace is asked for nothing at all, and the call returns rather than stepping through a million
     * empty panels.
     */
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

    /**
     * Where [alpha] multiplies, which is a question about categories rather than about rounding.
     *
     * The scattered half of a product forms `value · (alpha · B)` and the gathered half sums the stored
     * products and scales once. With an alpha too small to be recovered and operands large enough to
     * overflow, the two orders reach different answers, and this pins the one the implementation states.
     */
    @Test
    fun `alpha multiplies where the product contract says it does`() {
        val tiny = 1e-300
        val large = 1e300
        val a = SparseMatrix.ofColumns(1, 1, listOf(listOf(0 to large)))
        val b = DenseMatrix.wrap(1, 1, doubleArrayOf(large))
        val scattered = DenseMatrix.wrap(1, 1, doubleArrayOf(0.0))
        plain.gemm(tiny, a, false, b, false, 0.0, scattered, right = false, workspace = null)
        assertEquals(large, scattered.values[0], "the scattered half did not form value · (alpha · B)")

        val gathered = DenseMatrix.wrap(1, 1, doubleArrayOf(0.0))
        plain.gemm(tiny, a, true, b, false, 0.0, gathered, right = false, workspace = null)
        assertEquals(
            Double.POSITIVE_INFINITY,
            gathered.values[0],
            "the gathered half did not sum the stored products before scaling",
        )
    }

    /**
     * The symmetric product gives each stored entry one multiplier and spends it on both halves.
     *
     * A stored entry too large to multiply by an operand, against a multiplier too small to be recovered
     * afterwards, is where the three orders differ: scaling the entry first keeps both halves finite, and
     * scaling either operand first, or the sum, does not. The two halves also have to agree with each
     * other, which is what makes this one rule rather than two.
     */
    @Test
    fun `a symmetric product gives every stored entry one multiplier`() {
        val tiny = 1e-300
        val large = 1e300
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0, 1 to large), listOf(1 to 1.0)))
        val b = DenseMatrix.wrap(2, 1, doubleArrayOf(large, large))
        val c = DenseMatrix.wrap(2, 1, doubleArrayOf(0.0, 0.0))

        plain.symm(tiny, a, b, 0.0, c, lower = true, right = false, workspace = null)

        // (alpha · value) is one, so the scattered half and the mirrored half both stay finite.
        assertEquals(large, c.values[1], "the scattered half did not scale the stored entry first")
        assertEquals(large, c.values[0], "the mirrored half did not scale the stored entry first")
    }

    /**
     * A symmetric operand storing only its diagonal mirrors nothing, and the destination says so.
     *
     * The half that gathers rows back into a column has no row to gather here, and an implementation that
     * added a zero for it anyway would turn an infinity into a NaN and a negative zero into a positive one.
     * Checked through the public entry point at one right-hand side and at several, because the written-out
     * column and the coupled panel are two implementations of the same rule.
     */
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

    /**
     * A row no stored entry reaches keeps what the destination multiplier left there, sign and all.
     *
     * The operand below stores one entry, on the diagonal of its second column, so the first row takes part
     * in no product at all: not as a row something scatters into, and not as a column something mirrors
     * back into. A traversal that added a zero for the mirrored half it does not have would turn the
     * negative zero there into a positive one. The second row is touched, and a stored diagonal against a
     * zero right-hand side does form its product, which is why only the first is asserted about.
     */
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

    /** The same rule where the panel runs rather than the written-out column, which must agree with it. */
    @Test
    fun `a symmetric panel multiplies as its written out column does`() {
        val tiny = 1e-300
        val large = 1e300
        val sides = 4
        val a = SparseMatrix.ofColumns(2, 2, listOf(listOf(0 to 1.0, 1 to large), listOf(1 to 1.0)))
        val b = DenseMatrix.wrap(2, sides, DoubleArray(2 * sides) { large })
        val wide = DenseMatrix.wrap(2, sides, DoubleArray(2 * sides))

        plain.symm(tiny, a, b, 0.0, wide, lower = true, right = false, workspace = Workspace())

        for (side in 0 until sides) {
            assertEquals(large, wide.values[1 + side * 2], "the scattered half at side $side")
            assertEquals(large, wide.values[side * 2], "the mirrored half at side $side")
        }
    }

    /**
     * The triangular skip rule, in every layout and on every engine this runtime has.
     *
     * A right-hand side that is exactly zero on entry contributes no update, and liveness is read from the
     * raw value before the division, so a pivot that underflows to zero still forms its stored products.
     * Neither of those is the dense substitution's rule and neither may be lost to a panel that evaluates
     * what the written-out loop skipped, so the fixture is run through the strided schedule, the staged one
     * and whatever the platform selected.
     *
     * The first group below is all live and reaches the panel; the second holds a zero and keeps the
     * written-out loop. Both have to agree with the same hand-computed answer.
     */
    @Test
    fun `a zero right hand side keeps its skip in every layout`() {
        val triangle = SparseMatrix.ofColumns(
            2,
            2,
            listOf(
                listOf(0 to Double.POSITIVE_INFINITY, 1 to Double.POSITIVE_INFINITY),
                listOf(1 to 1.0),
            ),
        )
        val sides = 4
        // Right-hand sides 0 and 2 underflow their pivot; 1 and 3 are zero on entry and are skipped.
        val source = DoubleArray(2 * sides)
        source[0] = Double.MIN_VALUE
        source[4] = Double.MIN_VALUE

        for ((name, engine) in engines()) {
            val b = DenseMatrix.wrap(2, sides, source.copyOf())
            engine.trsm(triangle, b, lower = true, transpose = false, unitDiag = false, right = false, alpha = 1.0)

            assertTrue(b[1, 0].isNaN(), "$name: an underflowed live pivot did not form its stored product")
            assertTrue(b[1, 2].isNaN(), "$name: an underflowed live pivot did not form its stored product")
            assertEquals(0.0, b[1, 1], "$name: a zero right-hand side was not skipped")
            assertEquals(0.0, b[1, 3], "$name: a zero right-hand side was not skipped")
        }
    }

    /** The same rule for the multiply, whose panel is the same one with the other sign. */
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

            assertEquals(Double.POSITIVE_INFINITY, b[1, 0], "$name: a live right-hand side lost its update")
            assertEquals(0.0, b[1, 1], "$name: a zero right-hand side was not skipped")
        }
    }

    /**
     * The same two rules for a call with a single right-hand side, which no panel sees at all.
     *
     * One right-hand side is written out by the traversal rather than handed to a panel, so the skip and
     * the raw-value liveness live in a leaf of their own and have to be checked there: a block of four
     * cannot reach it, and a leaf that read the divided pivot instead of the raw one would pass every test
     * above.
     */
    @Test
    fun `a zero right hand side keeps its skip at one right hand side`() {
        val solveTriangle = SparseMatrix.ofColumns(
            2,
            2,
            listOf(
                listOf(0 to Double.POSITIVE_INFINITY, 1 to Double.POSITIVE_INFINITY),
                listOf(1 to 1.0),
            ),
        )
        val multiplyTriangle = SparseMatrix.ofColumns(
            2,
            2,
            listOf(listOf(0 to 1.0, 1 to Double.POSITIVE_INFINITY), listOf(1 to 1.0)),
        )

        for ((name, engine) in engines()) {
            val underflowed = DenseMatrix.wrap(2, 1, doubleArrayOf(Double.MIN_VALUE, 0.0))
            engine.trsm(solveTriangle, underflowed, true, transpose = false, unitDiag = false, right = false)
            assertTrue(underflowed[1, 0].isNaN(), "$name: an underflowed live pivot skipped its stored product")

            val dead = DenseMatrix.wrap(2, 1, doubleArrayOf(0.0, 0.0))
            engine.trsm(solveTriangle, dead, true, transpose = false, unitDiag = false, right = false)
            assertEquals(0.0, dead[1, 0], "$name: a zero right-hand side was not skipped by the solve")
            assertTrue(1.0 / dead[1, 0] > 0.0, "$name: a zero right-hand side lost its sign")

            val live = DenseMatrix.wrap(2, 1, doubleArrayOf(1.0, 0.0))
            engine.trmm(multiplyTriangle, live, true, transpose = false, unitDiag = false, right = false)
            assertEquals(Double.POSITIVE_INFINITY, live[1, 0], "$name: a live right-hand side lost its update")

            val quiet = DenseMatrix.wrap(2, 1, doubleArrayOf(0.0, 0.0))
            engine.trmm(multiplyTriangle, quiet, true, transpose = false, unitDiag = false, right = false)
            assertEquals(0.0, quiet[1, 0], "$name: a zero right-hand side was not skipped by the multiply")
        }
    }

    /**
     * The same rule where the block really is staged, with one group holding a dead right-hand side and
     * another all live.
     *
     * The two-by-two fixtures above are too small for the staging rule to pay for itself, so they check the
     * semantics in the caller's layout on every engine and this checks them in the staged one. A right-hand
     * side of zeros has no pivot to spread, so an infinite coefficient in the triangle never reaches it and
     * every entry of it stays exactly zero; a live one does reach it and does not.
     */
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

        val route = staging.routeOf(
            SparseMatrixOperation.TrsmLeft,
            SparseCall(triangle, 1.0, destinationElements = b.values.size, rightHandSides = sides, lower = true),
        )
        assertTrue(route.components.any { it.startsWith(SPARSE_STAGING) }, "the fixture did not stage: $route")

        staging.trsm(triangle, b, lower = true, transpose = false, unitDiag = false, right = false, alpha = 1.0)

        for (row in 0 until order) {
            assertEquals(0.0, b[row, 1], "a zero right-hand side was written at row $row")
            assertTrue(1.0 / b[row, 1] > 0.0, "a zero right-hand side lost its sign at row $row")
        }
        assertTrue(b[1, 0].isInfinite(), "a live right-hand side did not form its stored product")
        assertTrue(b[1, 3].isInfinite(), "an all-live group did not form its stored product")
    }

    /** Every sparse engine this runtime has, so a rule is checked where it is implemented and where selected. */
    private fun engines(): List<Pair<String, SparseBlas>> = buildList {
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
        engine: SparseAlgorithms,
    ): DenseMatrix {
        val c = c0.copyOf()
        engine.gemm(ALPHA, a, transposeA, b, transposeB, BETA, c, right = false, workspace = Workspace())
        return c
    }

    private fun panelComponents(
        a: SparseMatrix,
        transposeA: Boolean,
        transposeB: Boolean,
        m: Int,
        n: Int,
        k: Int,
    ): List<String> = staging.routeOf(
        SparseMatrixOperation.GemmDense,
        SparseCall(
            a,
            ALPHA,
            BETA,
            destinationElements = m * n,
            depth = k,
            rightHandSides = n,
            transposeSparse = transposeA,
            transposeDense = transposeB,
        ),
    ).components

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

    /** Whether a call of this shape stages its right-hand sides, asked of the engine that will run it. */
    @Suppress("LongParameterList") // the operation and the five call facts the staging rule reads
    private fun stages(
        operation: SparseMatrixOperation,
        a: SparseMatrix,
        destinationElements: Int,
        depth: Int,
        sides: Int,
        transpose: Boolean,
        lower: Boolean,
    ): Boolean = staging.routeOf(
        operation,
        SparseCall(
            a,
            ALPHA,
            BETA,
            destinationElements = destinationElements,
            depth = depth,
            rightHandSides = sides,
            transposeSparse = transpose,
            lower = lower,
        ),
    ).components.any { it.startsWith(SPARSE_STAGING) }

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

    private fun triangle(order: Int, rng: Random): SparseMatrix {
        val columns = List(order) { j ->
            buildList {
                for (i in j until order) {
                    val value = if (i == j) 2.0 + abs(rng.nextDouble()) else rng.nextDouble(-1.0, 1.0)
                    if (i == j || (i + j) % 3 != 0) add(i to value)
                }
            }
        }
        return SparseMatrix.ofColumns(order, order, columns)
    }

    private companion object {
        const val ALPHA = 0.875
        const val BETA = -0.25

        /**
         * Extents the staging rule admits, which is what a staged test has to be built on.
         *
         * The rule asks for four stored entries per dense element a staged panel copies, and a destination
         * is copied twice, so a product of this order needs most of its positions stored before the copy is
         * worth making. A smaller fixture would silently compare two unstaged calls.
         */
        const val ORDER = 24
        const val DEPTH = 16

        /** A row of the operand with nothing stored, so the destination there is never reached. */
        const val EMPTY_ROW = 4

        /** Wider than the grouping below, so a call cuts more than one group and leaves a short last one. */
        const val SIDES = 5
    }
}

/** An engine's sparse algorithms over a chosen panel backend, which is what selects the staged schedule. */
internal fun engineWith(panels: DensePanelKernels): SparseAlgorithms = SparseAlgorithms(
    ScalarVectorKernels,
    ScalarIndexedSparseKernels,
    SparsePanelKernels(ScalarVectorKernels, panels),
)

/**
 * A backend that asks for adjacent right-hand sides and computes with the portable bodies.
 *
 * Not a vector backend and not pretending to be one: what it changes is the answer to the two questions the
 * sparse scheduling asks before it decides to copy, which is how the staged schedule is exercised on a
 * platform that has no vector bodies at all. [group] is deliberately not a lane count and not a power of
 * two, so a call cuts groups the shipped backends never would.
 */
internal open class AdjacentPreferringPanels(
    private val group: Int = 3,
    private val minimumRows: Int = 1,
    private val entryThreshold: Int? = null,
    private val delegate: DensePanelKernels = PortablePanelKernels,
) : DensePanelKernels by delegate {
    /** The portable bodies this backend computes with, which a subclass recording its runs delegates to. */
    protected val bodies: DensePanelKernels get() = delegate

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
