package com.eignex.koblas.dense

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.KoblasEngineApi
import com.eignex.koblas.MAX_IDLE_LENGTHS
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.poisonedTriangle
import com.eignex.koblas.randomMatrix
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The triangular matrix routines over the blocks they are scheduled with.
 *
 * All sixteen flag combinations on each of the two routines run through one algorithm, so the sweep below is
 * the whole of it rather than a sample: a side, a triangle, a transpose and a diagonal mode between them
 * decide which entry of the triangle is a coefficient, which direction the substitution runs in and which
 * operand of the product between blocks is the triangle, and getting any of those wrong is a wrong answer
 * and not a slow one.
 *
 * The orders cross the diagonal block deliberately. Below it a call is one substitution and no product at
 * all, which is a different path from the one a real solve takes, and a suite that only checked small
 * triangles would be checking the easy half.
 */
class TriangularBlockTest {
    private val rng = Random(20261103)

    @Test
    fun `trsm solves every uplo transpose and diagonal mode from the left`() = withDenseBlas { blas ->
        assertSolves(blas, right = false)
    }

    @Test
    fun `trsm solves every uplo transpose and diagonal mode from the right`() = withDenseBlas { blas ->
        assertSolves(blas, right = true)
    }

    @Test
    fun `trmm multiplies every uplo transpose and diagonal mode from the left`() = withDenseBlas { blas ->
        assertMultiplies(blas, right = false)
    }

    @Test
    fun `trmm multiplies every uplo transpose and diagonal mode from the right`() = withDenseBlas { blas ->
        assertMultiplies(blas, right = true)
    }

    private fun assertSolves(blas: DenseBlas, right: Boolean) =
        forEachCase { order, sides, lower, transpose, unitDiag ->
            val (triangle, _) = poisonedTriangle(rng, order, lower, unitDiag)
            val solution = randomMatrix(if (right) sides else order, if (right) order else sides, rng)
            // The right-hand sides come from the reference multiply, so the check does not depend on the
            // solve it is checking.
            val b = copyOf(solution)
            ReferenceBlas.trmm(triangle, b, lower, transpose, unitDiag, right)

            blas.trsm(triangle, b, lower, transpose, unitDiag, right)

            assertClose(
                solution.values,
                b.values,
                "trsm ${describe(order, sides, lower, transpose, unitDiag, right)}",
                SOLVE_TOLERANCE,
            )
        }

    private fun assertMultiplies(blas: DenseBlas, right: Boolean) =
        forEachCase { order, sides, lower, transpose, unitDiag ->
            val (triangle, _) = poisonedTriangle(rng, order, lower, unitDiag)
            val start = randomMatrix(if (right) sides else order, if (right) order else sides, rng)
            val expected = copyOf(start)
            ReferenceBlas.trmm(triangle, expected, lower, transpose, unitDiag, right, ALPHA)
            val actual = copyOf(start)

            blas.trmm(triangle, actual, lower, transpose, unitDiag, right, ALPHA)

            assertClose(
                expected.values,
                actual.values,
                "trmm ${describe(order, sides, lower, transpose, unitDiag, right)}",
                TOLERANCE,
            )
        }

    /**
     * The diagonal is divided by, and dividing is not multiplying by a reciprocal.
     *
     * A subnormal diagonal is where the two part company: its reciprocal overflows, so a substitution that
     * multiplied by one would answer with an infinity where the division has a finite answer to give. The
     * comparison is exact rather than close, because there is nothing approximate about the difference.
     *
     * The right-hand sides are several lane blocks wide, so the vector body is the one being asked.
     */
    @Test
    fun `a solve divides by a subnormal diagonal rather than multiplying by its reciprocal`() = withDenseBlas { blas ->
        val sides = VECTOR_SIDES + 1
        val subnormal = Double.MIN_VALUE
        for (right in booleanArrayOf(false, true)) {
            // The off-diagonal coefficient is zero, so the second step contributes nothing to the first
            // and the quotient is the whole of the answer.
            val triangle = DenseMatrix.wrap(2, 2, doubleArrayOf(subnormal, 0.0, 0.0, 1.0))
            val b = DenseMatrix.wrap(if (right) sides else 2, if (right) 2 else sides, DoubleArray(2 * sides))
            val step = if (right) sides else 1
            for (side in 0 until sides) {
                val at = if (right) side else side * 2
                b.values[at] = 1e-300
                b.values[at + step] = 0.5
            }

            blas.trsm(triangle, b, lower = true, right = right)

            for (side in 0 until sides) {
                val at = if (right) side else side * 2
                assertEquals(
                    1e-300 / subnormal,
                    b.values[at],
                    "side $side of a subnormal diagonal right=$right",
                )
                assertTrue(b.values[at].isFinite(), "side $side lost the finite quotient right=$right")
                assertEquals(0.5, b.values[at + step], "the second step changed, side $side right=$right")
            }
        }
    }

    /**
     * A singular diagonal produces what the division produces and is not detected.
     *
     * The BLAS solves carry no status and report nothing, so a zero pivot is an infinity or a NaN in the
     * result, and the caller who needs the distinction tests the diagonal first. Both cases are here
     * because they are different: a nonzero numerator over zero is an infinity and a zero one is a NaN.
     */
    @Test
    fun `a zero pivot yields the infinity or the NaN the division yields`() = withDenseBlas { blas ->
        val sides = VECTOR_SIDES + 1
        val triangle = DenseMatrix.wrap(1, 1, doubleArrayOf(0.0))
        val b = DenseMatrix.wrap(1, sides, DoubleArray(sides) { if (it == 0) 0.0 else 1.0 })
        // One right-hand side is zero and the rest are not, so both sides of the singular case run in the
        // same call and neither can be mistaken for the other.

        blas.trsm(triangle, b, lower = true)

        assertTrue(b.values[0].isNaN(), "a zero over a zero pivot was not a NaN")
        for (side in 1 until sides) {
            assertEquals(Double.POSITIVE_INFINITY, b.values[side], "side $side over a zero pivot")
        }
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

    /**
     * A triangle that shares the block it works on is read as the caller supplied it.
     *
     * The scaling by alpha writes the block before the substitution starts, so a call that scaled first and
     * staged afterwards would feed the substitution coefficients the caller never gave it.
     */
    @Test
    fun `a triangle sharing its own block is staged before the block is scaled`() = withDenseBlas { blas ->
        val order = TRIANGULAR_DIAGONAL_BLOCK + 3
        val workspace = Workspace()
        for (solve in booleanArrayOf(false, true)) {
            // A finite matrix throughout, because the operand it shares is the block as well as the
            // triangle: the poison a separate triangle carries above its diagonal would be data here.
            val triangle = dominantDiagonal(order)
            val expected = copyOf(triangle)
            if (solve) {
                ReferenceBlas.trsm(copyOf(triangle), expected, lower = true, alpha = ALPHA)
            } else {
                ReferenceBlas.trmm(copyOf(triangle), expected, lower = true, alpha = ALPHA)
            }
            val aliased = copyOf(triangle)

            if (solve) {
                blas.trsm(aliased, aliased, lower = true, alpha = ALPHA, workspace = workspace)
            } else {
                blas.trmm(aliased, aliased, lower = true, alpha = ALPHA, workspace = workspace)
            }

            assertClose(expected.values, aliased.values, "aliased triangle solve=$solve", SOLVE_TOLERANCE)
        }
    }

    /**
     * A long thin call reuses its scratch instead of asking for a new length at every block.
     *
     * The windows of a triangular schedule shrink: a left solve of this order over one right-hand side
     * accumulates a destination column for each of sixteen diagonal blocks, and those columns are all
     * different lengths. A workspace lends by exact length and keeps a bounded number of them, so asking at
     * the exact extent would cycle every one of them out of the pool and allocate again on every warmed
     * call, which neither a wide call nor a short one would show.
     */
    @Test
    fun `a long thin solve keeps its scratch inside what a workspace retains`() = withDenseBlas { blas ->
        // Three orders an octave apart, because the claim is that what the schedule borrows is bounded by
        // its own blocks rather than by the call: a bound that only held at one size would be a property of
        // that size. Each is past the eight distinct lengths a workspace retains, counted in blocks.
        for (blocks in intArrayOf(9, 17, 24)) longThinScratch(blas, blocks * TRIANGULAR_DIAGONAL_BLOCK)
    }

    private fun longThinScratch(blas: DenseBlas, order: Int) {
        val workspace = Workspace()
        for (solve in booleanArrayOf(false, true)) {
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

            assertTrue(afterOne > 0, "a call over ${order / TRIANGULAR_DIAGONAL_BLOCK} blocks borrowed nothing")
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
    }

    /** The same for a symmetric product, whose strips beside the diagonal shrink the same way. */
    @Test
    fun `a long thin symmetric product keeps its scratch inside what a workspace retains`() = withDenseBlas { blas ->
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

    /**
     * A substitution backend recommends a grouping inside the work it was asked about.
     *
     * The contract is the backend's to keep rather than the caller's to impose: the shared scheduling does
     * clamp what it is given, but a backend whose recommendation is wider than the call has right-hand
     * sides has answered a question about work that is not there, and the next caller need not clamp.
     */
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

    /**
     * The route names the substitutions and the products the call really ran.
     *
     * Under both recorders at once, so each half of the claim is checked against what the schedule handed
     * over: which diagonal blocks were substituted, with which flags and strides, and which product windows
     * ran between them.
     */
    @Test
    fun `a triangular route names the diagonal blocks and products the call really cut`() = withBackends {
            panels,
            products,
            triangles,
        ->
        for (right in booleanArrayOf(false, true)) {
            for (lower in booleanArrayOf(false, true)) {
                for (transpose in booleanArrayOf(false, true)) {
                    assertTriangularRoute(panels, products, triangles, lower, transpose, right, solve = true)
                    assertTriangularRoute(panels, products, triangles, lower, transpose, right, solve = false)
                }
            }
        }
    }

    @Suppress("LongParameterList") // the backends under test plus the three flags the route depends on
    private fun assertTriangularRoute(
        panels: DensePanelKernels,
        products: DenseProductKernels,
        triangles: DenseTriangularKernels,
        lower: Boolean,
        transpose: Boolean,
        right: Boolean,
        solve: Boolean,
    ) {
        val order = 2 * TRIANGULAR_DIAGONAL_BLOCK + 5
        val sides = 2 * VECTOR_SIDES + 1
        val recordedTriangles = RecordingTriangles(triangles)
        val recordedProducts = RecordingProducts(products)
        val blas = PortableDenseBlas(ScalarVectorKernels, panels, recordedProducts, recordedTriangles)
        val (triangle, _) = poisonedTriangle(rng, order, lower, unitDiag = false)
        val b = randomMatrix(if (right) sides else order, if (right) order else sides, rng)
        val operation = if (solve) DenseMatrixOperation.Trsm else DenseMatrixOperation.Trmm
        val context = "${if (solve) "trsm" else "trmm"} " +
            describe(order, sides, lower, transpose, unitDiag = false, right = right) + " on ${triangles.name}"

        if (solve) {
            blas.trsm(triangle, b, lower, transpose, right = right, alpha = ALPHA)
        } else {
            blas.trmm(triangle, b, lower, transpose, right = right, alpha = ALPHA)
        }

        val route = blas.routeOf(
            operation,
            DenseCall(
                b.rows,
                b.cols,
                ALPHA,
                depth = order,
                lower = lower,
                transposeA = transpose,
                right = right,
            ),
        )
        assertDiagonalBlocks(recordedTriangles, triangles, route, order, lower, transpose, right, solve, context)
        val tiles = route.components.filter { it.endsWith("/product-block") }.map { it.substringBefore('/') }
        val reached = recordedProducts.blocks
            .flatMap { products.implementationsFor(it.rows, it.columns, it.depth) }
            .distinct()
        assertEquals(reached, tiles, "$context: the route named $tiles for its products")
    }

    @Suppress("LongParameterList") // the recording, the claim being checked and the flags it is checked at
    private fun assertDiagonalBlocks(
        recorded: RecordingTriangles,
        triangles: DenseTriangularKernels,
        route: DenseMatrixRoute,
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
        val bodies = recorded.blocks
            .flatMap { triangles.implementationsFor(it.size, it.sides, it.rhsStride == 1) }
            .distinct()
        val entry = if (solve) "diagonal-solve" else "diagonal-multiply"
        val named = route.components.filter { it.endsWith("/$entry") }.map { it.substringBefore('/') }
        assertEquals(bodies, named, "$context: the route named $named for its substitutions")
        val callerStride = if (right) 1 else order
        val gathered = recorded.blocks.any { it.rhsStride == 1 && callerStride != 1 }
        assertEquals(
            gathered,
            route.components.any { it == "$TRIANGULAR_GATHER/rhs-block" },
            "$context: a gather ran=$gathered and the route said otherwise in ${route.components}",
        )
    }

    /** The steps of the order the recorded diagonal blocks covered between them, which must be all of them. */
    private fun coveredOrder(recorded: RecordingTriangles): Int {
        val covered = HashSet<Int>()
        for (block in recorded.blocks) for (step in block.start until block.start + block.size) covered.add(step)
        return covered.size
    }

    /**
     * Every flag combination on one side, at three orders and two side counts.
     *
     * The side is the caller's, not a loop here: each side is a complete sweep of the other flags, and a
     * case that ran both would do twice the work before it could say which of the two failed.
     */
    private inline fun forEachCase(
        body: (order: Int, sides: Int, lower: Boolean, transpose: Boolean, unitDiag: Boolean) -> Unit,
    ) {
        // One order inside a single diagonal block, one that fills it exactly, and one that leaves a short
        // last block; one side count below a lane block and one that leaves a tail above it.
        for (order in intArrayOf(1, TRIANGULAR_DIAGONAL_BLOCK, TRIANGULAR_DIAGONAL_BLOCK + 5)) {
            for (sides in intArrayOf(1, VECTOR_SIDES + 3)) {
                for (lower in booleanArrayOf(false, true)) {
                    for (transpose in booleanArrayOf(false, true)) {
                        for (unitDiag in booleanArrayOf(false, true)) {
                            body(order, sides, lower, transpose, unitDiag)
                        }
                    }
                }
            }
        }
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

    private fun copyOf(a: DenseMatrix) = DenseMatrix.wrap(a.rows, a.cols, a.values.copyOf())

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
 * Right-hand sides enough to fill whatever the substitution backend groups them in.
 *
 * Read from the backend rather than written down, for the reason the product fixtures are: a count that
 * fills one machine's lane block leaves another's empty, and a vector body nothing reaches is one nothing
 * checks. Where there is no vector backend the number only has to be more than one.
 */
@OptIn(KoblasEngineApi::class)
internal val VECTOR_SIDES: Int = BuiltinEngines.simd
    ?.triangularKernels
    ?.rightHandSideGroup(TRIANGULAR_DIAGONAL_BLOCK, UNGROUPED_SIDES)
    ?: 4

/** Wider than any grouping a backend recommends, so asking with it returns the recommendation itself. */
private const val UNGROUPED_SIDES = 1024

/** Runs [body] against every composition of panel, product and substitution backends this platform has. */
@OptIn(KoblasEngineApi::class)
internal fun withBackends(body: (DensePanelKernels, DenseProductKernels, DenseTriangularKernels) -> Unit) {
    body(PortablePanelKernels, PortableProductKernels, PortableTriangularKernels)
    BuiltinEngines.simd?.let { body(it.panelKernels, it.productKernels, it.triangularKernels) }
}
