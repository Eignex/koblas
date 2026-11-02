package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngine
import com.eignex.koblas.MAX_IDLE_LENGTHS
import com.eignex.koblas.PortableEngine
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.copyOf
import com.eignex.koblas.engineWithKernels
import com.eignex.koblas.internal.dense.PortablePanelKernels
import com.eignex.koblas.internal.dense.PortableProductKernels
import com.eignex.koblas.internal.dense.PortableTriangularKernels
import com.eignex.koblas.internal.dense.SYMMETRIC_BLOCK
import com.eignex.koblas.internal.dense.TRIANGULAR_DIAGONAL_BLOCK
import com.eignex.koblas.poisonedTriangle
import com.eignex.koblas.randomMatrix
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

// The sixteen flag combinations between them decide which entry is a coefficient, which direction the
// substitution runs and which operand of the product between blocks is the triangle, so the sweep is the
// whole of it. The orders cross the diagonal block, below which a call is one substitution and no product.
class TriangularSchedulingTest {
    private val rng = Random(20261103)

    @Test
    fun `trsm solves every transpose and diagonal mode from the left with an upper triangle`() = withDenseBlas { blas ->
        assertSolves(blas, right = false, lower = false)
    }

    @Test
    fun `trsm solves every transpose and diagonal mode from the left with a lower triangle`() = withDenseBlas { blas ->
        assertSolves(blas, right = false, lower = true)
    }

    @Test
    fun `trsm solves every transpose and diagonal mode from the right with an upper triangle`() =
        withDenseBlas { blas ->
            assertSolves(blas, right = true, lower = false)
        }

    @Test
    fun `trsm solves every transpose and diagonal mode from the right with a lower triangle`() = withDenseBlas { blas ->
        assertSolves(blas, right = true, lower = true)
    }

    @Test
    fun `trmm multiplies every uplo transpose and diagonal mode from the left`() = withDenseBlas { blas ->
        assertMultiplies(blas, right = false)
    }

    @Test
    fun `trmm multiplies every uplo transpose and diagonal mode from the right`() = withDenseBlas { blas ->
        assertMultiplies(blas, right = true)
    }

    @Test
    fun `trsm agrees with the oracle over three blocks from the left with an upper triangle`() = withDenseBlas { blas ->
        assertSolves(blas, right = false, lower = false, shapes = threeBlockShapes())
    }

    @Test
    fun `trsm agrees with the oracle over three blocks from the left with a lower triangle`() = withDenseBlas { blas ->
        assertSolves(blas, right = false, lower = true, shapes = threeBlockShapes())
    }

    @Test
    fun `trsm agrees with the oracle over three blocks from the right with an upper triangle`() =
        withDenseBlas { blas ->
            assertSolves(blas, right = true, lower = false, shapes = threeBlockShapes())
        }

    @Test
    fun `trsm agrees with the oracle over three blocks from the right with a lower triangle`() = withDenseBlas { blas ->
        assertSolves(blas, right = true, lower = true, shapes = threeBlockShapes())
    }

    @Test
    fun `trmm agrees with the oracle over three blocks from the left with an upper triangle`() = withDenseBlas { blas ->
        assertMultiplies(blas, right = false, lowerModes = booleanArrayOf(false), shapes = threeBlockShapes())
    }

    @Test
    fun `trmm agrees with the oracle over three blocks from the left with a lower triangle`() = withDenseBlas { blas ->
        assertMultiplies(blas, right = false, lowerModes = booleanArrayOf(true), shapes = threeBlockShapes())
    }

    @Test
    fun `trmm agrees with the oracle over three blocks from the right with an upper triangle`() =
        withDenseBlas { blas ->
            assertMultiplies(blas, right = true, lowerModes = booleanArrayOf(false), shapes = threeBlockShapes())
        }

    @Test
    fun `trmm agrees with the oracle over three blocks from the right with a lower triangle`() = withDenseBlas { blas ->
        assertMultiplies(blas, right = true, lowerModes = booleanArrayOf(true), shapes = threeBlockShapes())
    }

    /** Two full diagonal blocks and a tail, with a partial backend-derived right-hand-side group. */
    private fun threeBlockShapes(): List<Pair<Int, Int>> = listOf(2 * TRIANGULAR_DIAGONAL_BLOCK + 5 to VECTOR_SIDES + 3)

    private fun assertSolves(
        blas: KoblasEngine,
        right: Boolean,
        lower: Boolean,
        shapes: List<Pair<Int, Int>> = triangularShapes(),
    ) = forEachCase(booleanArrayOf(lower), shapes) { order, sides, _, transpose, unitDiag ->
        val (triangle, _) = poisonedTriangle(rng, order, lower, unitDiag)
        // Keep the larger unit-diagonal fixtures well conditioned so error in constructing the right-hand
        // sides cannot dominate the scheduling comparison.
        if (order > 2 * TRIANGULAR_DIAGONAL_BLOCK) {
            for (column in 0 until order) {
                for (row in 0 until order) {
                    if (row != column) triangle[row, column] /= order
                }
            }
        }
        val solution = randomMatrix(if (right) sides else order, if (right) order else sides, rng)
        // The right-hand sides come from the reference multiply, so the check does not depend on the
        // solve it is checking.
        val b = solution.copyOf()
        ReferenceBlas.trmm(triangle, b, lower, transpose, unitDiag, right, 1.0 / ALPHA)

        blas.trsm(triangle, b, lower, transpose, unitDiag, right, ALPHA)

        assertClose(
            solution.values,
            b.values,
            "trsm ${describe(order, sides, lower, transpose, unitDiag, right)}",
            SOLVE_TOLERANCE,
        )
    }

    private fun assertMultiplies(
        blas: KoblasEngine,
        right: Boolean,
        lowerModes: BooleanArray = booleanArrayOf(false, true),
        shapes: List<Pair<Int, Int>> = triangularShapes(),
    ) = forEachCase(lowerModes, shapes) { order, sides, lower, transpose, unitDiag ->
        val (triangle, _) = poisonedTriangle(rng, order, lower, unitDiag)
        val start = randomMatrix(if (right) sides else order, if (right) order else sides, rng)
        val expected = start.copyOf()
        ReferenceBlas.trmm(triangle, expected, lower, transpose, unitDiag, right, ALPHA)
        val actual = start.copyOf()

        blas.trmm(triangle, actual, lower, transpose, unitDiag, right, ALPHA)

        assertClose(
            expected.values,
            actual.values,
            "trmm ${describe(order, sides, lower, transpose, unitDiag, right)}",
            TOLERANCE,
        )
    }

    /** A zero multiplier writes zeros over the whole block and reads no coefficient at all. */
    @Test
    fun `a zero alpha fills the block without reading the triangle`() = withDenseBlas { blas ->
        val order = TRIANGULAR_DIAGONAL_BLOCK + 3
        val sides = 5
        val triangle = DenseMatrix.wrap(order, order, DoubleArray(order * order) { Double.NaN })
        for (solve in booleanArrayOf(false, true)) {
            val b = randomMatrix(order, sides, rng)

            if (solve) {
                blas.trsm(triangle, b, lower = true, alpha = 0.0)
            } else {
                blas.trmm(triangle, b, lower = true, alpha = 0.0)
            }

            assertTrue(b.values.all { it == 0.0 }, "a zero alpha left something behind, solve=$solve")
        }
    }

    // Scaling by alpha writes the block before the substitution starts, so a call that scaled first and
    // staged afterwards would feed the substitution coefficients the caller never gave it.
    @Test
    fun `a triangle sharing its own block is staged before the block is scaled`() = withDenseBlas { blas ->
        val order = TRIANGULAR_DIAGONAL_BLOCK + 3
        val workspace = Workspace()
        for (solve in booleanArrayOf(false, true)) {
            // A finite matrix throughout, because the operand it shares is the block as well as the
            // triangle: the poison a separate triangle carries above its diagonal would be data here.
            val triangle = dominantDiagonal(order)
            val expected = triangle.copyOf()
            if (solve) {
                ReferenceBlas.trsm(triangle.copyOf(), expected, lower = true, alpha = ALPHA)
            } else {
                ReferenceBlas.trmm(triangle.copyOf(), expected, lower = true, alpha = ALPHA)
            }
            val aliased = triangle.copyOf()

            if (solve) {
                blas.trsm(aliased, aliased, lower = true, alpha = ALPHA, workspace = workspace)
            } else {
                blas.trmm(aliased, aliased, lower = true, alpha = ALPHA, workspace = workspace)
            }

            assertClose(expected.values, aliased.values, "aliased triangle solve=$solve", SOLVE_TOLERANCE)
        }
    }

    // Twenty four shrinking windows exceed the workspace retention limit and expose scratch sized by call order.
    @Test
    fun `portable trmm retains bounded scratch over twenty four blocks`() {
        val blas = testDenseBlas
        longThinScratch(blas, 24 * TRIANGULAR_DIAGONAL_BLOCK, solve = false)
    }

    @Test
    fun `portable trsm retains bounded scratch over twenty four blocks`() {
        val blas = testDenseBlas
        longThinScratch(blas, 24 * TRIANGULAR_DIAGONAL_BLOCK, solve = true)
    }

    @Test
    fun `candidate trmm retains bounded scratch over twenty four blocks`() {
        val blas = BuiltinEngines.simd ?: return
        longThinScratch(blas, 24 * TRIANGULAR_DIAGONAL_BLOCK, solve = false)
    }

    @Test
    fun `candidate trsm retains bounded scratch over twenty four blocks`() {
        val blas = BuiltinEngines.simd ?: return
        longThinScratch(blas, 24 * TRIANGULAR_DIAGONAL_BLOCK, solve = true)
    }

    private fun longThinScratch(blas: KoblasEngine, order: Int, solve: Boolean) {
        val workspace = Workspace()
        val triangle = dominantDiagonal(order)
        val b = randomMatrix(order, 1, rng)

        if (solve) {
            blas.trsm(triangle, b, lower = true, workspace = workspace)
        } else {
            blas.trmm(triangle, b, lower = true, workspace = workspace)
        }
        val afterOne = workspace.idleLengths()
        repeat(3) {
            if (solve) {
                blas.trsm(triangle, b, lower = true, workspace = workspace)
            } else {
                blas.trmm(triangle, b, lower = true, workspace = workspace)
            }
        }

        assertTrue(afterOne > 0, "the product updates did not retain their scratch")
        assertTrue(
            afterOne <= MAX_IDLE_LENGTHS,
            "order $order solve=$solve left $afterOne lengths behind, past what a workspace retains",
        )
        assertEquals(
            afterOne,
            workspace.idleLengths(),
            "order $order solve=$solve asked for a new length on a repeated call",
        )
    }

    /** The same for a symmetric product, whose strips beside the diagonal shrink the same way. */
    @Test
    fun `a long thin portable symmetric product retains bounded scratch`() {
        assertSymmetricScratchIsBounded(testDenseBlas)
    }

    @Test
    fun `a long thin candidate symmetric product retains bounded scratch`() {
        val blas = BuiltinEngines.simd ?: return
        assertSymmetricScratchIsBounded(blas)
    }

    private fun assertSymmetricScratchIsBounded(blas: KoblasEngine) {
        val order = 16 * SYMMETRIC_BLOCK
        val workspace = Workspace()
        val a = dominantDiagonal(order)
        val b = randomMatrix(order, 1, rng)
        val c = randomMatrix(order, 1, rng)

        blas.symm(ALPHA, a, b, -0.25, c, lower = true, workspace = workspace)
        val afterOne = workspace.idleLengths()
        repeat(3) { blas.symm(ALPHA, a, b, -0.25, c, lower = true, workspace = workspace) }

        assertTrue(afterOne > 0, "a symmetric product over ${order / SYMMETRIC_BLOCK} blocks borrowed nothing")
        assertTrue(afterOne <= MAX_IDLE_LENGTHS, "symm left $afterOne lengths behind")
        assertEquals(afterOne, workspace.idleLengths(), "symm asked for a new length on a repeated call")
    }

    // The scheduling does clamp what it is given, but a recommendation wider than the call has right-hand
    // sides answers about work that is not there, and the next caller need not clamp.
    @Test
    fun `a substitution backend recommends a group inside the sides it was asked about`() =
        withBackends { _, _, triangles ->
            for (sides in intArrayOf(0, 1, 2, 3, 7, 64, 1024)) {
                for (order in intArrayOf(1, TRIANGULAR_DIAGONAL_BLOCK)) {
                    val group = triangles.rightHandSideGroup(order, sides)

                    assertTrue(group >= 1, "${triangles.name} recommended $group for $sides sides")
                    if (sides > 0) {
                        assertTrue(
                            group <= sides,
                            "${triangles.name} recommended $group for $sides sides at order $order",
                        )
                    }
                }
            }
        }

    // Under both recorders at once, so each half of the claim is checked against what the schedule handed
    // over: the diagonal blocks substituted and the product windows run between them.
    @Test
    fun `trmm substitutes diagonal blocks covering the order from the left`() {
        assertTriangularBlocks(right = false, solve = false)
    }

    @Test
    fun `trmm substitutes diagonal blocks covering the order from the right`() {
        assertTriangularBlocks(right = true, solve = false)
    }

    @Test
    fun `trsm substitutes diagonal blocks covering the order from the left`() {
        assertTriangularBlocks(right = false, solve = true)
    }

    @Test
    fun `trsm substitutes diagonal blocks covering the order from the right`() {
        assertTriangularBlocks(right = true, solve = true)
    }

    private fun assertTriangularBlocks(right: Boolean, solve: Boolean) = withBackends { panels, products, triangles ->
        for (lower in booleanArrayOf(false, true)) {
            for (transpose in booleanArrayOf(false, true)) {
                assertTriangularBlocksAt(panels, products, triangles, lower, transpose, right, solve)
            }
        }
    }

    @Suppress("LongParameterList") // the backends under test plus the three flags the schedule depends on
    private fun assertTriangularBlocksAt(
        panels: KoblasEngine,
        products: KoblasEngine,
        triangles: KoblasEngine,
        lower: Boolean,
        transpose: Boolean,
        right: Boolean,
        solve: Boolean,
    ) {
        val order = 2 * TRIANGULAR_DIAGONAL_BLOCK + 5
        val sides = 2 * VECTOR_SIDES + 1
        val recordedTriangles = RecordingTriangles(triangles)
        val recordedProducts = RecordingProducts(products)
        val blas = engineWithKernels(PortableEngine, panels, recordedProducts, recordedTriangles)
        val (triangle, _) = poisonedTriangle(rng, order, lower, unitDiag = false)
        val b = randomMatrix(if (right) sides else order, if (right) order else sides, rng)
        val context = "${if (solve) "trsm" else "trmm"} " +
            describe(order, sides, lower, transpose, unitDiag = false, right = right) + " on ${triangles.name}"

        if (solve) {
            blas.trsm(triangle, b, lower, transpose, right = right, alpha = ALPHA)
        } else {
            blas.trmm(triangle, b, lower, transpose, right = right, alpha = ALPHA)
        }

        assertDiagonalBlocks(recordedTriangles, order, lower, transpose, right, solve, context)
    }

    @Suppress("LongParameterList") // the recording, the claim being checked and the flags it is checked at
    private fun assertDiagonalBlocks(
        recorded: RecordingTriangles,
        order: Int,
        lower: Boolean,
        transpose: Boolean,
        right: Boolean,
        solve: Boolean,
        context: String,
    ) {
        val flip = transpose != right
        assertTrue(recorded.blocks.isNotEmpty(), "$context: no diagonal block was substituted")
        assertEquals(order, coveredOrder(recorded), "$context: the diagonal blocks did not cover the order")
        for (block in recorded.blocks) {
            assertEquals(flip, block.transposed, "$context: the substitution read the wrong corner")
            assertEquals(lower != flip, block.lower, "$context: the substitution ran the wrong way")
            assertEquals(solve, block.solve, "$context: the wrong substitution ran")
        }
    }

    /** The steps of the order the recorded diagonal blocks covered between them, which must be all of them. */
    private fun coveredOrder(recorded: RecordingTriangles): Int {
        val covered = HashSet<Int>()
        for (block in recorded.blocks) for (step in block.start until block.start + block.size) covered.add(step)
        return covered.size
    }

    /** Every transpose and diagonal mode for the selected triangles and shapes. */
    private inline fun forEachCase(
        lowerModes: BooleanArray,
        shapes: List<Pair<Int, Int>>,
        body: (order: Int, sides: Int, lower: Boolean, transpose: Boolean, unitDiag: Boolean) -> Unit,
    ) {
        for ((order, sides) in shapes) {
            for (lower in lowerModes) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        body(order, sides, lower, transpose, unitDiag)
                    }
                }
            }
        }
    }

    /** One single block, one full block and one short tail, with narrow and grouped right-hand sides. */
    private fun triangularShapes(): List<Pair<Int, Int>> =
        intArrayOf(1, TRIANGULAR_DIAGONAL_BLOCK, TRIANGULAR_DIAGONAL_BLOCK + 5).flatMap { order ->
            listOf(order to 1, order to VECTOR_SIDES + 3)
        }

    @Suppress("LongParameterList") // the shape and the four flags, which is what identifies a case
    private fun describe(
        order: Int,
        sides: Int,
        lower: Boolean,
        transpose: Boolean,
        unitDiag: Boolean,
        right: Boolean,
    ): String = "order=$order sides=$sides lower=$lower transpose=$transpose unit=$unitDiag right=$right"

    /** A square whose diagonal dominates, so a solve over it is well conditioned and nothing in it is NaN. */
    private fun dominantDiagonal(order: Int): DenseMatrix {
        val a = randomMatrix(order, order, rng)
        for (i in 0 until order) a[i, i] = 3.0 + i % 2
        return a
    }

    private companion object {
        const val ALPHA = 0.875
        const val TOLERANCE = 1e-9

        /** A solve over several blocks accumulates the substitution's error, so its bound is looser. */
        const val SOLVE_TOLERANCE = 1e-7
    }
}

/**
 * Right-hand sides enough to fill whatever the substitution backend groups them in, read from the backend
 * because a count that fills one machine's lane block leaves another's empty.
 */
internal val VECTOR_SIDES: Int = BuiltinEngines.simd
    ?.rightHandSideGroup(TRIANGULAR_DIAGONAL_BLOCK, UNGROUPED_SIDES)
    ?: 4

/** Wider than any grouping a backend recommends, so asking with it returns the recommendation itself. */
private const val UNGROUPED_SIDES = 1024

/** Runs [body] against every composition of panel, product and substitution backends this platform has. */
internal fun withBackends(body: (KoblasEngine, KoblasEngine, KoblasEngine) -> Unit) {
    body(PortablePanelKernels, PortableProductKernels, PortableTriangularKernels)
    BuiltinEngines.simd?.let { body(it, it, it) }
}
