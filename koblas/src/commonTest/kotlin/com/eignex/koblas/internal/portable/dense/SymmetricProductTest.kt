package com.eignex.koblas.internal.portable.dense

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.engine.*
import com.eignex.koblas.workspace.*
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SymmetricProductTest {
    private val rng = Random(20261102)

    @Test
    fun `symm matches gemm on the full matrix and reads only the selected triangle`() = withEngines { blas ->
        // The last order is past one symmetric block, so a zero beta also has to overwrite the strips beside
        // the diagonal blocks rather than scale them.
        for (n in intArrayOf(1, 5, 12, SYMMETRIC_BLOCK + 3)) {
            for (lower in booleanArrayOf(true, false)) {
                val p = 4
                for (alpha in doubleArrayOf(0.0, 1.0, 0.75)) {
                    for (beta in doubleArrayOf(0.0, 1.0, -0.5)) {
                        val (full, poisoned) = poisonedSymmetric(rng, n, lower)
                        val b = randomMatrix(n, p, rng)
                        val c0 = DoubleArray(n * p) { if (beta == 0.0) Double.NaN else rng.nextDouble(-1.0, 1.0) }
                        val expected = DenseMatrix(n, p, c0.copyOf())
                        ReferenceBlas.gemm(alpha, full, false, b, false, beta, expected)
                        val actual = DenseMatrix(n, p, c0.copyOf())

                        blas.symm(alpha, poisoned, b, beta, actual, lower)

                        assertClose(expected, actual, "symm n=$n a=$alpha b=$beta lower=$lower", TOLERANCE)
                    }
                }
            }
        }
    }

    @Test
    fun `symm agrees with the oracle on both sides and reads only the stored triangle`() = withEngines { blas ->
        // One block and a tail, one short order, and a single entry, since the blocks are the order's rather
        // than the destination's.
        for ((order, sides) in listOf(SYMMETRIC_BLOCK + 3 to 7, 5 to 2, 1 to 3)) {
            for (lower in booleanArrayOf(false, true)) {
                for (right in booleanArrayOf(false, true)) {
                    assertSymmetricProductAgreesWithReference(blas, order, sides, lower, right)
                }
            }
        }
    }

    @Test
    fun `symm agrees with the oracle over three blocks on either side with either triangle`() = withEngines { blas ->
        val order = 2 * SYMMETRIC_BLOCK + 3
        for (lower in booleanArrayOf(false, true)) {
            for (right in booleanArrayOf(false, true)) {
                assertSymmetricProductAgreesWithReference(blas, order, VECTOR_SIDES + 3, lower, right)
            }
        }
    }

    private fun assertSymmetricProductAgreesWithReference(
        blas: KoblasEngine,
        order: Int,
        sides: Int,
        lower: Boolean,
        right: Boolean,
    ) {
        val (full, poisoned) = poisonedSymmetric(rng, order, lower)
        val rows = if (right) sides else order
        val columns = if (right) order else sides
        val b = randomMatrix(rows, columns, rng)
        val start = randomMatrix(rows, columns, rng)
        val expected = start.copyOf()
        val actual = start.copyOf()
        // The oracle takes the full symmetric matrix, so its answer does not depend on which
        // half this call was given.
        ReferenceBlas.symm(0.875, full, b, -0.25, expected, lower = true, right = right)

        blas.symm(0.875, poisoned, b, -0.25, actual, lower, right)

        assertClose(
            expected.values,
            actual.values,
            "symm order=$order sides=$sides lower=$lower right=$right",
            TOLERANCE,
        )
    }

    /** A symmetric product whose dense operand is its destination reads what the caller supplied. */
    @Test
    fun `symm over its own destination agrees with the oracle`() = withEngines { blas ->
        val order = SYMMETRIC_BLOCK + 3
        val (full, poisoned) = poisonedSymmetric(rng, order, lower = true)
        val start = randomMatrix(order, order, rng)
        val expected = start.copyOf()
        ReferenceBlas.symm(0.875, full, start.copyOf(), -0.25, expected, lower = true)
        val aliased = start.copyOf()

        blas.symm(0.875, poisoned, aliased, -0.25, aliased, lower = true)

        assertClose(expected.values, aliased.values, "symm into its own dense operand", TOLERANCE)
    }

    // The strips beside the diagonal shrink with every block, so more blocks than a workspace retains lengths
    // expose scratch sized by the strip rather than by the block.
    @Test
    fun `a long thin symmetric product retains bounded scratch`() = withEngines { blas ->
        val order = (MAX_IDLE_LENGTHS + 2) * SYMMETRIC_BLOCK
        val workspace = Workspace()
        val a = DenseMatrix.wrap(order, order, DoubleArray(order * order) { (it % 7 - 3) / (8.0 * order) })
        for (i in 0 until order) a[i, i] = 3.0 + i % 2
        val b = DenseMatrix.wrap(order, 1, DoubleArray(order) { (it % 5 - 2) / 4.0 })
        val c = DenseMatrix.wrap(order, 1, DoubleArray(order) { (it % 3 - 1) / 2.0 })

        blas.withWorkspace(workspace).symm(0.875, a, b, -0.25, c, lower = true)
        val afterOne = workspace.idleLengths()
        repeat(3) { blas.withWorkspace(workspace).symm(0.875, a, b, -0.25, c, lower = true) }

        val context = "${blas.name} over ${order / SYMMETRIC_BLOCK} blocks"
        assertTrue(afterOne > 0, "$context borrowed nothing")
        assertTrue(afterOne <= MAX_IDLE_LENGTHS, "$context left $afterOne lengths behind")
        assertEquals(afterOne, workspace.idleLengths(), "$context asked for a new length on a repeated call")
    }

    private companion object {
        const val TOLERANCE = 1e-9
    }
}
