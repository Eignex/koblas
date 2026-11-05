package com.eignex.koblas.internal.scheduling.dense

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.kernels.EngineKernels
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import com.eignex.koblas.internal.operations.triangle
import com.eignex.koblas.internal.storage.MAX_IDLE_LENGTHS
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
    fun `trsm solves every flag combination inside and across diagonal blocks`() = withEngines { blas ->
        forEachCase(triangularShapes()) { order, sides, flags -> assertSolves(blas, order, sides, flags) }
    }

    @Test
    fun `trmm multiplies every flag combination inside and across diagonal blocks`() = withEngines { blas ->
        forEachCase(triangularShapes()) { order, sides, flags -> assertMultiplies(blas, order, sides, flags) }
    }

    // Two full diagonal blocks and a tail, with a partial right-hand-side group. The diagonal mode alternates
    // with the transpose, since it decides only what the substitution reads and the shorter shapes sweep it.
    @Test
    fun `trsm agrees with the oracle over three diagonal blocks`() = withEngines { blas ->
        forEachThreeBlockCase { order, sides, flags -> assertSolves(blas, order, sides, flags) }
    }

    @Test
    fun `trmm agrees with the oracle over three diagonal blocks`() = withEngines { blas ->
        forEachThreeBlockCase { order, sides, flags -> assertMultiplies(blas, order, sides, flags) }
    }

    private fun assertSolves(blas: KoblasEngine, order: Int, sides: Int, flags: Flags) {
        val (triangle, _) = poisonedTriangle(rng, order, flags.lower, flags.unitDiag)
        // Keep the larger unit-diagonal fixtures well conditioned so error in constructing the right-hand
        // sides cannot dominate the scheduling comparison.
        if (order > 2 * TRIANGULAR_DIAGONAL_BLOCK) {
            for (column in 0 until order) {
                for (row in 0 until order) {
                    if (row != column) triangle[row, column] /= order
                }
            }
        }
        val solution = randomMatrix(if (flags.right) sides else order, if (flags.right) order else sides, rng)
        // The right-hand sides come from the reference multiply, so the check does not depend on the
        // solve it is checking.
        val b = solution.copyOf()
        ReferenceBlas.trmm(triangle, b, flags.lower, flags.transpose, flags.unitDiag, flags.right, 1.0 / ALPHA)

        blas.trsm(triangle, b, flags.lower, flags.transpose, flags.unitDiag, flags.right, ALPHA)

        assertClose(solution.values, b.values, "trsm order=$order sides=$sides $flags", SOLVE_TOLERANCE)
    }

    private fun assertMultiplies(blas: KoblasEngine, order: Int, sides: Int, flags: Flags) {
        val (triangle, _) = poisonedTriangle(rng, order, flags.lower, flags.unitDiag)
        val start = randomMatrix(if (flags.right) sides else order, if (flags.right) order else sides, rng)
        val expected = start.copyOf()
        ReferenceBlas.trmm(triangle, expected, flags.lower, flags.transpose, flags.unitDiag, flags.right, ALPHA)
        val actual = start.copyOf()

        blas.trmm(triangle, actual, flags.lower, flags.transpose, flags.unitDiag, flags.right, ALPHA)

        assertClose(expected.values, actual.values, "trmm order=$order sides=$sides $flags", TOLERANCE)
    }

    /** A zero multiplier writes zeros over the whole block and reads no coefficient at all. */
    @Test
    fun `a zero alpha fills the block without reading the triangle`() = withEngines { blas ->
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
    fun `a triangle sharing its own block is staged before the block is scaled`() = withEngines { blas ->
        val order = TRIANGULAR_DIAGONAL_BLOCK + 3
        val workspace = Scratch()
        for (solve in booleanArrayOf(false, true)) {
            // A finite matrix throughout, because the operand it shares is the block as well as the
            // triangle: the poison a separate triangle carries above its diagonal would be data here.
            val triangle = randomMatrix(order, order, rng)
            for (i in 0 until order) triangle[i, i] = 3.0 + i % 2
            val expected = triangle.copyOf()
            if (solve) {
                ReferenceBlas.trsm(triangle.copyOf(), expected, lower = true, alpha = ALPHA)
            } else {
                ReferenceBlas.trmm(triangle.copyOf(), expected, lower = true, alpha = ALPHA)
            }
            val aliased = triangle.copyOf()

            if (solve) {
                blas.trsm(aliased, aliased, lower = true, alpha = ALPHA, scratch = workspace)
            } else {
                blas.trmm(aliased, aliased, lower = true, alpha = ALPHA, scratch = workspace)
            }

            assertClose(expected.values, aliased.values, "aliased triangle solve=$solve", SOLVE_TOLERANCE)
        }
    }

    // More shrinking windows than a workspace retains lengths, so scratch sized by call order rather than by
    // the schedule's own blocks would be evicted and borrowed again.
    @Test
    fun `a long thin triangular call retains bounded scratch`() = withEngines { blas ->
        val order = (MAX_IDLE_LENGTHS + 2) * TRIANGULAR_DIAGONAL_BLOCK
        val triangle = DenseMatrix.wrap(order, order, DoubleArray(order * order) { (it % 7 - 3) / (8.0 * order) })
        for (i in 0 until order) triangle[i, i] = 3.0 + i % 2
        for (solve in booleanArrayOf(false, true)) {
            val workspace = Scratch()
            val b = DenseMatrix.wrap(order, 1, DoubleArray(order) { (it % 5 - 2) / 4.0 })
            val call = {
                if (solve) {
                    blas.trsm(triangle, b, lower = true, scratch = workspace)
                } else {
                    blas.trmm(triangle, b, lower = true, scratch = workspace)
                }
            }

            call()
            val afterOne = workspace.idleLengths()
            repeat(3) { call() }

            val context = "${blas.name} order $order solve=$solve"
            assertTrue(afterOne > 0, "$context: the product updates did not retain their scratch")
            assertTrue(afterOne <= MAX_IDLE_LENGTHS, "$context left $afterOne lengths behind")
            assertEquals(afterOne, workspace.idleLengths(), "$context asked for a new length on a repeated call")
        }
    }

    // Under a recorder, so the claim is checked against the diagonal blocks the schedule handed over.
    @Test
    fun `triangular calls substitute diagonal blocks covering the order`() =
        withBackends { panels, products, triangles ->
            for (solve in booleanArrayOf(false, true)) {
                for (right in booleanArrayOf(false, true)) {
                    for (lower in booleanArrayOf(false, true)) {
                        for (transpose in booleanArrayOf(false, true)) {
                            val flags = Flags(lower, transpose, unitDiag = false, right = right)
                            assertTriangularBlocks(panels, products, triangles, flags, solve)
                        }
                    }
                }
            }
        }

    private fun assertTriangularBlocks(
        panels: EngineKernels,
        products: EngineKernels,
        triangles: EngineKernels,
        flags: Flags,
        solve: Boolean,
    ) {
        val order = 2 * TRIANGULAR_DIAGONAL_BLOCK + 5
        val sides = 2 * VECTOR_SIDES + 1
        val recorded = RecordingTriangles(triangles)
        val blas = engineWithKernels(ScalarEngine, panels, products, recorded)
        val (triangle, _) = poisonedTriangle(rng, order, flags.lower, unitDiag = false)
        val b = randomMatrix(if (flags.right) sides else order, if (flags.right) order else sides, rng)
        val context = "${if (solve) "trsm" else "trmm"} order=$order sides=$sides $flags on ${triangles.name}"

        if (solve) {
            blas.trsm(triangle, b, flags.lower, flags.transpose, right = flags.right, alpha = ALPHA)
        } else {
            blas.trmm(triangle, b, flags.lower, flags.transpose, right = flags.right, alpha = ALPHA)
        }

        val flip = flags.transpose != flags.right
        assertTrue(recorded.blocks.isNotEmpty(), "$context: no diagonal block was substituted")
        assertEquals(order, coveredOrder(recorded), "$context: the diagonal blocks did not cover the order")
        for (block in recorded.blocks) {
            assertEquals(flip, block.transposed, "$context: the substitution read the wrong corner")
            assertEquals(flags.lower != flip, block.lower, "$context: the substitution ran the wrong way")
            assertEquals(solve, block.solve, "$context: the wrong substitution ran")
        }
    }

    /** The steps of the order the recorded diagonal blocks covered between them, which must be all of them. */
    private fun coveredOrder(recorded: RecordingTriangles): Int {
        val covered = HashSet<Int>()
        for (block in recorded.blocks) for (step in block.start until block.start + block.size) covered.add(step)
        return covered.size
    }

    /** The four flags of a triangular call, which with its shape identify a case. */
    private data class Flags(val lower: Boolean, val transpose: Boolean, val unitDiag: Boolean, val right: Boolean)

    /** Every side, triangle, transpose and diagonal mode at each of [shapes]. */
    private inline fun forEachCase(shapes: List<Pair<Int, Int>>, body: (order: Int, sides: Int, flags: Flags) -> Unit) {
        for ((order, sides) in shapes) {
            for (right in booleanArrayOf(false, true)) {
                for (lower in booleanArrayOf(false, true)) {
                    for (transpose in booleanArrayOf(false, true)) {
                        for (unitDiag in booleanArrayOf(false, true)) {
                            body(order, sides, Flags(lower, transpose, unitDiag, right))
                        }
                    }
                }
            }
        }
    }

    /** Every side, triangle and transpose over three diagonal blocks, with both diagonal modes among them. */
    private inline fun forEachThreeBlockCase(body: (order: Int, sides: Int, flags: Flags) -> Unit) {
        for (right in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                for (transpose in booleanArrayOf(false, true)) {
                    val flags = Flags(lower, transpose, unitDiag = lower == transpose, right = right)
                    body(2 * TRIANGULAR_DIAGONAL_BLOCK + 5, VECTOR_SIDES + 1, flags)
                }
            }
        }
    }

    /**
     * A single entry with grouped sides, one full block, and one block past it with a short tail, which is
     * the first order whose product between blocks runs.
     */
    private fun triangularShapes(): List<Pair<Int, Int>> = listOf(
        1 to VECTOR_SIDES + 3,
        TRIANGULAR_DIAGONAL_BLOCK to 1,
        TRIANGULAR_DIAGONAL_BLOCK + 5 to 1,
    )

    private companion object {
        const val ALPHA = 0.875
        const val TOLERANCE = 1e-9

        /** A solve over several blocks accumulates the substitution's error, so its bound is looser. */
        const val SOLVE_TOLERANCE = 1e-7
    }
}
