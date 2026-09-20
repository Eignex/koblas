package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.BlasOperation
import com.eignex.koblas.vendor.openBlas
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The two places [DenseBlas] states a result a legal `dgemm` or `dsyr2k` need not produce, so the two places
 * an installed library has to be held back.
 *
 * Each case runs twice: against a binding that records and computes nothing, where a call sent across leaves
 * the destination untouched, and against whatever library is installed, which is what would notice a guard
 * holding only because the recorder is not a real library. Every fixture is past
 * [HostDensePolicy.MINIMUM_WORK], so the size rule is not what keeps these calls at home.
 */
class HostDenseContractTest {
    private fun composed(host: Blas): HostDenseBlas =
        HostDenseBlas(PortableDenseBlas(ScalarVectorKernels, PortablePanelKernels), host)

    // Reference BLAS scales a coefficient as it goes in places, where an infinite multiplier against a
    // stored zero is a NaN; here the two answers are a matrix of infinities and a matrix of NaNs.
    @Test
    fun `an infinite multiplier keeps the placement this library documents`() {
        val order = PRODUCT_ORDER
        val a = DenseMatrix.wrap(order, order, DoubleArray(order * order) { if (it / order == 0) 1.0 else 0.0 })
        val b = DenseMatrix.wrap(order, order, DoubleArray(order * order) { if (it % order == 0) 1.0 else 0.0 })
        val recorder = RecordingBlas()

        // The fixture is only interesting if the product really is one and the other terms really are zeros.
        assertTrue(order.toLong() * order * order >= HostDensePolicy.MINIMUM_WORK, "fixture below the policy")
        assertEquals(0.0, a.values[1 + order], "the fixture has no zero entry to scale")

        for (host in hosts(recorder)) {
            val c = DenseMatrix.zero(order, order)
            val blas = composed(host)

            val route = blas.routeOf(
                DenseMatrixOperation.Gemm,
                DenseCall(order, order, alpha = Double.POSITIVE_INFINITY, beta = 0.0, depth = order),
            )
            blas.gemm(Double.POSITIVE_INFINITY, a, false, b, false, 0.0, c)

            assertEquals(DENSE_SCHEDULING, route.scheduling, "an infinite multiplier was sent to a library")
            assertEquals(null, route.host)
            assertTrue(c.values.none { it.isNaN() }, "a zero entry produced a NaN against an infinite multiplier")
            assertTrue(c.values.all { it == Double.POSITIVE_INFINITY }, "the documented result is an infinity")
        }
        assertEquals(emptyList(), recorder.calls.map { it.operation }, "the library was called")
    }

    // Each separate sum overflows, one to an infinity and the other to its negative, so composing them is a
    // NaN where an interleaved traversal would cancel them to zero. The sums are computed by direct loops
    // rather than through [ReferenceBlas], whose `syr2k` is the interleaved form.
    @Test
    fun `a rank two-k update is never handed to a library because it is two separate products`() {
        val rows = PAIR_ROWS
        val depth = PAIR_DEPTH
        val a = DenseMatrix.wrap(rows, depth, DoubleArray(rows * depth) { if (it % 2 == 0) 1.0 else -1.0 })
        val b = DenseMatrix.wrap(rows, depth, DoubleArray(rows * depth) { Double.MAX_VALUE / 2 })
        val recorder = RecordingBlas()

        var first = 0.0
        var second = 0.0
        var fused = 0.0
        for (t in 0 until depth) {
            first += a.values[1 + t * rows] * b.values[t * rows]
            second += b.values[1 + t * rows] * a.values[t * rows]
            fused += a.values[1 + t * rows] * b.values[t * rows] + b.values[1 + t * rows] * a.values[t * rows]
        }
        // The fixture has to be on the edge rather than near it, or the two formulations agree and prove
        // nothing: each separate sum leaves the finite range in the opposite direction, and the pair cancels.
        assertEquals(Double.NEGATIVE_INFINITY, first, "the first product did not overflow")
        assertEquals(Double.POSITIVE_INFINITY, second, "the second product did not overflow")
        assertEquals(0.0, fused, "the interleaved formulation does not differ on this fixture")
        assertTrue(rows.toLong() * rows * depth >= HostDensePolicy.MINIMUM_WORK, "fixture below the policy")

        for (host in hosts(recorder)) {
            val c = DenseMatrix.zero(rows, rows)
            val blas = composed(host)

            val route = blas.routeOf(DenseMatrixOperation.Syr2k, DenseCall(rows, rows, depth = depth))
            blas.syr2k(1.0, a, b, transpose = false, beta = 0.0, c = c, lower = true)

            assertEquals(DENSE_SCHEDULING, route.scheduling, "a rank-2k update was sent to a library")
            assertEquals(null, route.host)
            assertTrue(c[1, 0].isNaN(), "the two products were not composed separately; the entry is ${c[1, 0]}")
        }
        assertEquals(emptyList(), recorder.calls.map { it.operation }, "the library was called")
    }

    // The multiplier and both operands are finite, so neither a finiteness test nor the repartitioning
    // [DenseBlas.gemm] allows would hold this call back; only where the multiplier lands separates the two,
    // and [PreScalingBlas] is a legal formulation that places it on an operand.
    @Test
    fun `a scaled product is kept here because a library may scale an operand instead`() {
        val depth = SINGLE_DEPTH
        val order = SCALED_ORDER
        for (case in SCALED_CASES) {
            val a = DenseMatrix.wrap(order, depth, DoubleArray(order * depth) { case.left })
            val b = DenseMatrix.wrap(depth, order, DoubleArray(depth * order) { case.right })
            val fake = PreScalingBlas()
            val blas = composed(fake)

            // What the two formulations give, computed here so the fixture is shown to tell them apart.
            val owned = (case.left * case.right) * case.alpha
            val preScaled = case.left * (case.alpha * case.right)
            val direct = DenseMatrix.zero(order, order)
            // A second instance, so that what [fake] recorded is only what the composition asked of it.
            PreScalingBlas().gemm(case.alpha, a, false, b, false, 0.0, direct)

            val route = blas.routeOf(
                DenseMatrixOperation.Gemm,
                DenseCall(order, order, alpha = case.alpha, beta = 0.0, depth = depth),
            )
            val c = DenseMatrix.zero(order, order)
            blas.gemm(case.alpha, a, false, b, false, 0.0, c)

            assertTrue(order.toLong() * order * depth >= HostDensePolicy.MINIMUM_WORK, "fixture below the policy")
            assertTrue(
                owned.toRawBits() != preScaled.toRawBits(),
                "the fixture does not tell the two placements apart: both give $owned",
            )
            assertEquals(preScaled.toRawBits(), direct.values[0].toRawBits(), "the fake does not pre-scale")
            assertEquals(DENSE_SCHEDULING, route.scheduling, "a scaled product was sent to a library")
            assertEquals(null, route.host)
            assertEquals(emptyList(), fake.recorded(), "the library was called for a scaled product")
            assertTrue(
                c.values.all { it.toRawBits() == owned.toRawBits() },
                "alpha ${case.alpha} placed the multiplier on an operand: got ${c.values[0]}, owned is $owned",
            )
        }
    }

    // Without this the cases above would pass on a policy that simply never reached a library.
    @Test
    fun `a unit multiplier on the same product does reach the library`() {
        val order = PRODUCT_ORDER
        val a = DenseMatrix.wrap(order, order, DoubleArray(order * order) { 1.0 + it % 3 })
        val recorder = RecordingBlas()
        val blas = composed(recorder)

        val route = blas.routeOf(DenseMatrixOperation.Gemm, DenseCall(order, order, depth = order))
        blas.gemm(1.0, a, false, a, false, 0.0, DenseMatrix.zero(order, order))

        assertEquals(HOST_SCHEDULING, route.scheduling)
        assertEquals(listOf(BlasOperation.Gemm), recorder.calls.map { it.operation })
    }

    /** The recorder always, and whatever library this host has beside it. */
    private fun hosts(recorder: RecordingBlas): List<Blas> {
        val installed = openBlas()
        if (installed == null) {
            println("SKIPPED: no CBLAS library installed; the guards were checked against the recorder only")
        }
        return listOfNotNull(recorder, installed)
    }

    /**
     * One of the two finite fixtures that separate the placements, with the answer each one gives.
     *
     * @property alpha the multiplier, finite and not one.
     * @property left every entry of the left operand.
     * @property right every entry of the right operand.
     */
    private class ScaledCase(val alpha: Double, val left: Double, val right: Double)

    /**
     * A binding whose `gemm` scales the right operand before multiplying, which BLAS permits, so the check
     * holds on a host whose installed library happens to scale last. Only `gemm` computes.
     */
    private class PreScalingBlas(private val recorder: RecordingBlas = RecordingBlas()) : Blas by recorder {
        /** Every call the composition made, which for a correctly held-back product is none. */
        fun recorded(): List<BlasOperation> = recorder.calls.map { it.operation }

        override fun gemm(
            alpha: Double,
            a: DenseMatrix,
            transposeA: Boolean,
            b: DenseMatrix,
            transposeB: Boolean,
            beta: Double,
            c: DenseMatrix,
        ) {
            recorder.gemm(alpha, a, transposeA, b, transposeB, beta, c)
            val depth = if (transposeA) a.rows else a.cols
            for (column in 0 until c.cols) {
                for (row in 0 until c.rows) {
                    var sum = 0.0
                    for (step in 0 until depth) {
                        val left = if (transposeA) a.values[step + row * a.rows] else a.values[row + step * a.rows]
                        val right = if (transposeB) {
                            b.values[column + step * b.rows]
                        } else {
                            b.values[
                                step +
                                    column * b.rows,
                            ]
                        }
                        sum += left * (alpha * right)
                    }
                    val previous = if (beta == 0.0) 0.0 else beta * c.values[row + column * c.rows]
                    c.values[row + column * c.rows] = sum + previous
                }
            }
        }
    }

    private companion object {
        /** Past the policy's size as a cubic product, so only the multiplier decides. */
        const val PRODUCT_ORDER = 16

        /** One step, so no partition of a shared dimension can account for a difference. */
        const val SINGLE_DEPTH = 1

        /** With [SINGLE_DEPTH], past the policy's size, so only the multiplier holds the call back. */
        const val SCALED_ORDER = 32

        /**
         * Two finite fixtures, one overflowing where the multiplier lands on the operand and one where it
         * does not. Together they rule out a finiteness test standing in for the unit-multiplier rule.
         */
        val SCALED_CASES = listOf(
            ScaledCase(Double.MAX_VALUE, left = 0.0, right = 2.0),
            ScaledCase(0.5, left = Double.MAX_VALUE, right = 2.0),
        )

        /** Two rows, so one off-diagonal entry carries the whole question. */
        const val PAIR_ROWS = 2

        /** Long enough for each separate sum to leave the finite range, and past the policy's size. */
        const val PAIR_DEPTH = 512
    }
}
