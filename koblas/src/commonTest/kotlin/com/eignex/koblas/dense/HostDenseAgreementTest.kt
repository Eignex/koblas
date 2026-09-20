package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.DenseVector
import com.eignex.koblas.Workspace
import com.eignex.koblas.assertClose
import com.eignex.koblas.poisonedSymmetric
import com.eignex.koblas.poisonedTriangle
import com.eignex.koblas.randomMatrix
import com.eignex.koblas.randomVector
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.openBlas
import com.eignex.koblas.wellConditioned
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * What a composed default computes, against the textbook definition rather than against itself.
 *
 * These need a library, so they say so and stop where none is installed. What they are for is the half the
 * recording binding cannot reach: whether the operands, the leading dimensions, the increments and the
 * triangle flags that [HostDenseBlas] hands across actually describe the call it was given. A wrong `uplo` or
 * a transposed operand comes back as wrong numbers here and as nothing at all in a test that only counts
 * calls.
 *
 * The tolerance is loose on purpose. The arithmetic is the library's, which is free to accumulate in another
 * order, so agreement to the last bit is not what is being asked; a flag translated the wrong way misses by
 * far more than any accumulation order does.
 *
 * `minimumWork = 0` keeps the fixtures small enough to be read. Whether the size rule fires is settled
 * against the policy's own number in [HostDenseBlasTest], not here.
 */
class HostDenseAgreementTest {
    private fun composed(host: Blas): HostDenseBlas =
        HostDenseBlas(PortableDenseBlas(ScalarVectorKernels, PortablePanelKernels), host, minimumWork = 0)

    /** The composition and the binding under it, or a line saying what this host could not check. */
    private inline fun withHost(what: String, block: (HostDenseBlas, Blas) -> Unit) {
        val host = openBlas()
        if (host == null) {
            println("SKIPPED: no CBLAS library installed; $what was not verified on this host")
            return
        }
        block(composed(host), host)
    }

    /**
     * Every routine these cases compute through actually reaches the library, or the coverage has lapsed.
     *
     * The composition here admits any size, so what can send one of these calls back to the portable
     * schedule is a compatibility rule or a library that does not export the entry point. Both are correct
     * outcomes and a silent loss of coverage at the same time, so they are told apart rather than left to be
     * noticed: an operation the installed binding does not export is named and skipped, and `syr2k`, which
     * is meant to stay whatever the library exports, is asserted to stay.
     *
     * `gemmt` is the one this distinction is really for. Accelerate does not export `cblas_dgemmt`, so on
     * macOS that row is portable and the composed `gemmt` path is genuinely unexercised there.
     */
    @Test
    fun `every routine these cases cover reaches the library except the one that may not`() =
        withHost("the routing behind these cases") { blas, host ->
            val order = ORDER
            val admitted = listOf(
                DenseMatrixOperation.Gemv to DenseCall(ROWS, COLUMNS, alpha = ALPHA, beta = BETA),
                DenseMatrixOperation.Symv to DenseCall(order, order, alpha = ALPHA, beta = BETA),
                DenseMatrixOperation.Ger to DenseCall(ROWS, COLUMNS, alpha = ALPHA),
                DenseMatrixOperation.Syr to DenseCall(order, order, alpha = ALPHA),
                DenseMatrixOperation.Syr2 to DenseCall(order, order, alpha = ALPHA),
                DenseMatrixOperation.Trsv to DenseCall(order, order),
                DenseMatrixOperation.Trmv to DenseCall(order, order),
                DenseMatrixOperation.Symm to DenseCall(order, SIDES, alpha = ALPHA, beta = BETA, depth = order),
                DenseMatrixOperation.Trsm to DenseCall(order, SIDES, alpha = ALPHA, depth = order),
                DenseMatrixOperation.Trmm to DenseCall(order, SIDES, alpha = ALPHA, depth = order),
                DenseMatrixOperation.Gemm to
                    DenseCall(ROWS, COLUMNS, alpha = PLACED_ALPHA, beta = BETA, depth = DEPTH),
                DenseMatrixOperation.Gemmt to
                    DenseCall(order, order, alpha = PLACED_ALPHA, beta = BETA, depth = DEPTH),
                DenseMatrixOperation.Syrk to
                    DenseCall(order, order, alpha = PLACED_ALPHA, beta = BETA, depth = DEPTH),
            )

            for ((operation, call) in admitted) {
                val entry = requireNotNull(HostDensePolicy.entryPointFor(operation))
                if (entry !in host.directlyImplemented) {
                    println(
                        "SKIPPED: ${host.vendor.vendorName} does not export ${entry.entryPoint}; " +
                            "the composed $operation path was not exercised on this host",
                    )
                    continue
                }
                val route = blas.routeOf(operation, call)
                assertEquals(HOST_SCHEDULING, route.scheduling, "$operation no longer reaches the library")
                assertTrue(route.host != null, "$operation named no vendor call")
            }
            val pair = blas.routeOf(
                DenseMatrixOperation.Syr2k,
                DenseCall(order, order, alpha = PLACED_ALPHA, beta = BETA, depth = DEPTH),
            )
            assertEquals(DENSE_SCHEDULING, pair.scheduling, "a rank-2k update reached the library")
        }

    @Test
    fun `the matrix vector routines agree with the definition on both orientations`() =
        withHost("the matrix-vector routines") { blas, _ ->
            val rng = Random(11)
            val a = randomMatrix(ROWS, COLUMNS, rng)
            for (transpose in booleanArrayOf(false, true)) {
                val x = randomVector(if (transpose) ROWS else COLUMNS, rng)
                val y = randomVector(if (transpose) COLUMNS else ROWS, rng)
                val expected = y.copyOf()
                ReferenceBlas.gemv(ALPHA, a, x, BETA, expected, transpose)

                val actual = y.copyOf()
                blas.gemv(ALPHA, a, x, BETA, actual, transpose)

                assertClose(expected, actual, "gemv transpose=$transpose", TOLERANCE)
            }
        }

    @Test
    fun `the symmetric routines read only the triangle they were given`() = withHost(
        "the symmetric routines",
    ) { blas, _ ->
        val rng = Random(12)
        for (lower in booleanArrayOf(true, false)) {
            val (full, poisoned) = poisonedSymmetric(rng, ORDER, lower)
            val x = randomVector(ORDER, rng)
            val y = randomVector(ORDER, rng)
            val b = randomMatrix(ORDER, SIDES, rng)
            val expectedVector = y.copyOf()
            ReferenceBlas.symv(ALPHA, full, x, BETA, expectedVector, lower)
            val expectedMatrix = randomMatrix(ORDER, SIDES, Random(13))
            val actualMatrix = DenseMatrix.wrap(ORDER, SIDES, expectedMatrix.values.copyOf())
            ReferenceBlas.symm(ALPHA, full, b, BETA, expectedMatrix, lower)

            val actualVector = y.copyOf()
            blas.symv(ALPHA, poisoned, x, BETA, actualVector, lower)
            blas.symm(ALPHA, poisoned, b, BETA, actualMatrix, lower)

            assertClose(expectedVector, actualVector, "symv lower=$lower", TOLERANCE)
            assertClose(expectedMatrix, actualMatrix, "symm lower=$lower", TOLERANCE)
        }
    }

    @Test
    fun `a symmetric product from the right is the other side and not the same call`() =
        withHost("the right-side symmetric product") { blas, _ ->
            val rng = Random(14)
            val (full, poisoned) = poisonedSymmetric(rng, ORDER, lower = true)
            val b = randomMatrix(SIDES, ORDER, rng)
            val expected = randomMatrix(SIDES, ORDER, Random(15))
            val actual = DenseMatrix.wrap(SIDES, ORDER, expected.values.copyOf())
            ReferenceBlas.symm(ALPHA, full, b, BETA, expected, lower = true, right = true)

            blas.symm(ALPHA, poisoned, b, BETA, actual, lower = true, right = true)

            assertClose(expected, actual, "symm right", TOLERANCE)
        }

    @Test
    fun `the rank updates write only the selected triangle`() = withHost("the rank updates") { blas, _ ->
        val rng = Random(16)
        for (lower in booleanArrayOf(true, false)) {
            val x = DenseVector.wrap(randomVector(ORDER, rng))
            val y = DenseVector.wrap(randomVector(ORDER, rng))
            val base = randomMatrix(ORDER, ORDER, Random(17))
            val expectedSyr = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
            val expectedSyr2 = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
            ReferenceBlas.syr(ALPHA, x, expectedSyr, lower)
            ReferenceBlas.syr2(ALPHA, x, y, expectedSyr2, lower)

            val actualSyr = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
            val actualSyr2 = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
            blas.syr(ALPHA, x, actualSyr, lower)
            blas.syr2(ALPHA, x, y, actualSyr2, lower)

            assertClose(expectedSyr, actualSyr, "syr lower=$lower", TOLERANCE)
            assertClose(expectedSyr2, actualSyr2, "syr2 lower=$lower", TOLERANCE)
        }
    }

    @Test
    fun `a general rank update covers the whole destination`() = withHost("ger") { blas, _ ->
        val rng = Random(18)
        val x = randomVector(ROWS, rng)
        val y = randomVector(COLUMNS, rng)
        val base = randomMatrix(ROWS, COLUMNS, rng)
        val expected = DenseMatrix.wrap(ROWS, COLUMNS, base.values.copyOf())
        ReferenceBlas.ger(ALPHA, x, y, expected)

        val actual = DenseMatrix.wrap(ROWS, COLUMNS, base.values.copyOf())
        blas.ger(ALPHA, x, y, actual)

        assertClose(expected, actual, "ger", TOLERANCE)
    }

    /**
     * Every triangular traversal, with an implicit diagonal that is a NaN where the flag says it is implied.
     *
     * The poisoned fixture is what makes the unit-diagonal flag observable: a library that read the stored
     * diagonal instead of taking it as one would come back all NaN rather than slightly wrong.
     */
    @Test
    fun `the triangular vector routines cover all eight traversals`() =
        withHost("the triangular vector routines") { blas, _ ->
            val rng = Random(19)
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (unitDiag in booleanArrayOf(false, true)) {
                        val (poisoned, explicit) = poisonedTriangle(rng, ORDER, lower, unitDiag)
                        val b = randomVector(ORDER, rng)
                        val where = "lower=$lower transpose=$transpose unit=$unitDiag"

                        val expectedSolve = b.copyOf()
                        ReferenceBlas.trsv(explicit, expectedSolve, lower, transpose, unitDiag)
                        val actualSolve = b.copyOf()
                        blas.trsv(poisoned, actualSolve, lower, transpose, unitDiag)

                        val expectedMultiply = b.copyOf()
                        ReferenceBlas.trmv(explicit, expectedMultiply, lower, transpose, unitDiag)
                        val actualMultiply = b.copyOf()
                        blas.trmv(poisoned, actualMultiply, lower, transpose, unitDiag)

                        assertClose(expectedSolve, actualSolve, "trsv $where", TOLERANCE)
                        assertClose(expectedMultiply, actualMultiply, "trmv $where", TOLERANCE)
                    }
                }
            }
        }

    @Test
    fun `the triangular matrix routines cover both sides and both traversals`() =
        withHost("the triangular matrix routines") { blas, _ ->
            val rng = Random(20)
            for (lower in booleanArrayOf(true, false)) {
                for (transpose in booleanArrayOf(false, true)) {
                    for (right in booleanArrayOf(false, true)) {
                        val (poisoned, explicit) = poisonedTriangle(rng, ORDER, lower, unitDiag = false)
                        val rows = if (right) SIDES else ORDER
                        val columns = if (right) ORDER else SIDES
                        val base = randomMatrix(rows, columns, rng)
                        val where = "lower=$lower transpose=$transpose right=$right"

                        val expectedSolve = DenseMatrix.wrap(rows, columns, base.values.copyOf())
                        ReferenceBlas.trsm(explicit, expectedSolve, lower, transpose, false, right, ALPHA)
                        val actualSolve = DenseMatrix.wrap(rows, columns, base.values.copyOf())
                        blas.trsm(poisoned, actualSolve, lower, transpose, false, right, ALPHA)

                        val expectedMultiply = DenseMatrix.wrap(rows, columns, base.values.copyOf())
                        ReferenceBlas.trmm(explicit, expectedMultiply, lower, transpose, false, right, ALPHA)
                        val actualMultiply = DenseMatrix.wrap(rows, columns, base.values.copyOf())
                        blas.trmm(poisoned, actualMultiply, lower, transpose, false, right, ALPHA)

                        assertClose(expectedSolve, actualSolve, "trsm $where", TOLERANCE)
                        assertClose(expectedMultiply, actualMultiply, "trmm $where", TOLERANCE)
                    }
                }
            }
        }

    @Test
    fun `a product agrees on every combination of transposes`() = withHost("the product") { blas, _ ->
        val rng = Random(21)
        for (transposeA in booleanArrayOf(false, true)) {
            for (transposeB in booleanArrayOf(false, true)) {
                val a = if (transposeA) randomMatrix(DEPTH, ROWS, rng) else randomMatrix(ROWS, DEPTH, rng)
                val b = if (transposeB) randomMatrix(COLUMNS, DEPTH, rng) else randomMatrix(DEPTH, COLUMNS, rng)
                val base = randomMatrix(ROWS, COLUMNS, rng)
                val expected = DenseMatrix.wrap(ROWS, COLUMNS, base.values.copyOf())
                ReferenceBlas.gemm(PLACED_ALPHA, a, transposeA, b, transposeB, BETA, expected)

                val actual = DenseMatrix.wrap(ROWS, COLUMNS, base.values.copyOf())
                blas.gemm(PLACED_ALPHA, a, transposeA, b, transposeB, BETA, actual)

                assertClose(expected, actual, "gemm transA=$transposeA transB=$transposeB", TOLERANCE)
            }
        }
    }

    /**
     * `gemmt` and `syrk` go across; `syr2k` is here for its numbers and not as host coverage.
     *
     * The policy never hands a rank-2k update to a library, because this library documents it as two
     * separately accumulated products and a library's own need not be. So its rows below check the portable
     * schedule the composition falls back to, and [HostDenseContractTest] is where that fallback is asserted.
     */
    @Test
    fun `the triangle selected products write one triangle and leave the other alone`() =
        withHost("the triangle-selected products") { blas, _ ->
            val rng = Random(22)
            for (lower in booleanArrayOf(true, false)) {
                val a = randomMatrix(ORDER, DEPTH, rng)
                val b = randomMatrix(DEPTH, ORDER, rng)
                val base = randomMatrix(ORDER, ORDER, rng)
                val where = "lower=$lower"

                val expectedGemmt = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
                ReferenceBlas.gemmt(PLACED_ALPHA, a, false, b, false, BETA, expectedGemmt, lower)
                val actualGemmt = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
                blas.gemmt(PLACED_ALPHA, a, false, b, false, BETA, actualGemmt, lower)

                val expectedSyrk = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
                ReferenceBlas.syrk(PLACED_ALPHA, a, false, BETA, expectedSyrk, lower)
                val actualSyrk = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
                blas.syrk(PLACED_ALPHA, a, false, BETA, actualSyrk, lower)

                val other = randomMatrix(ORDER, DEPTH, rng)
                val expectedSyr2k = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
                ReferenceBlas.syr2k(ALPHA, a, other, false, BETA, expectedSyr2k, lower)
                val actualSyr2k = DenseMatrix.wrap(ORDER, ORDER, base.values.copyOf())
                blas.syr2k(ALPHA, a, other, false, BETA, actualSyr2k, lower)

                assertClose(expectedGemmt, actualGemmt, "gemmt $where", TOLERANCE)
                assertClose(expectedSyrk, actualSyrk, "syrk $where", TOLERANCE)
                assertClose(expectedSyr2k, actualSyr2k, "syr2k $where", TOLERANCE)
            }
        }

    /**
     * An operand that is the destination's own buffer, which a whole-call binding cannot be handed.
     *
     * The staged copy is what makes these answerable at all, so the check is that the answer is the one the
     * definition gives when the two are separate matrices, not merely that nothing crashed.
     */
    @Test
    fun `an operand sharing the destination gives the answer two separate operands would`() =
        withHost("the staged aliases") { blas, _ ->
            val rng = Random(23)
            val workspace = Workspace()
            val square = wellConditioned(ORDER, rng)
            val separate = DenseMatrix.wrap(ORDER, ORDER, square.values.copyOf())

            val expectedProduct = DenseMatrix.wrap(ORDER, ORDER, square.values.copyOf())
            ReferenceBlas.gemm(PLACED_ALPHA, separate, false, separate, false, BETA, expectedProduct)
            val product = DenseMatrix.wrap(ORDER, ORDER, square.values.copyOf())
            blas.gemm(PLACED_ALPHA, product, false, product, false, BETA, product, workspace)

            val expectedSolve = DenseMatrix.wrap(ORDER, ORDER, square.values.copyOf())
            ReferenceBlas.trsm(separate, expectedSolve, lower = true, alpha = ALPHA)
            val solve = DenseMatrix.wrap(ORDER, ORDER, square.values.copyOf())
            blas.trsm(solve, solve, lower = true, alpha = ALPHA, workspace = workspace)

            val vector = square.values
            val expectedGemv = DoubleArray(ORDER)
            ReferenceBlas.gemv(ALPHA, separate, vector.copyOf(ORDER), 0.0, expectedGemv)

            assertClose(expectedProduct, product, "gemm over one buffer", TOLERANCE)
            assertClose(expectedSolve, solve, "trsm over one buffer", TOLERANCE)
            assertTrue(expectedGemv.all { it.isFinite() }, "the alias fixture produced no finite oracle")
        }

    @Test
    fun `a zero multiplier scales the destination and reads no operand`() =
        withHost("the zero-multiplier no-read rule") { blas, _ ->
            val poisoned = DenseMatrix.wrap(ORDER, ORDER, DoubleArray(ORDER * ORDER) { Double.NaN })
            val destination = DenseMatrix.wrap(ORDER, ORDER, DoubleArray(ORDER * ORDER) { 2.0 + it })
            val expected = DoubleArray(ORDER * ORDER) { BETA * (2.0 + it) }

            blas.gemm(0.0, poisoned, false, poisoned, false, BETA, destination)

            assertClose(expected, destination.values, "gemm with a zero multiplier", TOLERANCE)
        }

    @Test
    fun `a zero destination multiplier overwrites whatever stood there`() =
        withHost("the zero-beta no-read rule") { blas, _ ->
            val rng = Random(24)
            val a = randomMatrix(ROWS, DEPTH, rng)
            val b = randomMatrix(DEPTH, COLUMNS, rng)
            val destination = DenseMatrix.wrap(ROWS, COLUMNS, DoubleArray(ROWS * COLUMNS) { Double.NaN })
            val expected = DenseMatrix.zero(ROWS, COLUMNS)
            ReferenceBlas.gemm(PLACED_ALPHA, a, false, b, false, 0.0, expected)

            blas.gemm(PLACED_ALPHA, a, false, b, false, 0.0, destination)

            assertClose(expected, destination, "gemm over a poisoned destination", TOLERANCE)
        }

    @Test
    fun `a bad shape is refused before anything is written`() = withHost("shape validation") { blas, _ ->
        val a = DenseMatrix.zero(ROWS, DEPTH)
        val destination = DenseMatrix.wrap(ROWS, COLUMNS, DoubleArray(ROWS * COLUMNS) { 3.0 })

        val failure = runCatching {
            blas.gemm(
                1.0,
                a,
                false,
                DenseMatrix.zero(DEPTH + 1, COLUMNS),
                false,
                0.0,
                destination,
            )
        }

        assertTrue(failure.isFailure, "a mismatched product was accepted")
        assertEquals(List(ROWS * COLUMNS) { 3.0 }, destination.values.toList(), "a refused call wrote anyway")
    }

    private companion object {
        const val ROWS = 7
        const val COLUMNS = 5
        const val DEPTH = 6
        const val ORDER = 6
        const val SIDES = 4
        const val ALPHA = 0.75

        /**
         * The multiplier `gemm`, `gemmt` and `syrk` carry here, which has to be one.
         *
         * Those three document their result as a multiplier applied to an accumulated sum, and the policy
         * hands them to a library only where there is no multiplier to place. A scaled fixture would run
         * these cases on the portable schedule instead and quietly stop covering the binding, which the
         * route assertion below is the second guard against.
         */
        const val PLACED_ALPHA = 1.0
        const val BETA = -0.5

        /** Loose enough for another accumulation order, far tighter than any mistranslated flag. */
        const val TOLERANCE = 1e-9
    }
}
