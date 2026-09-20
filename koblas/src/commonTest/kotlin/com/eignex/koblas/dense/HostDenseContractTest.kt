package com.eignex.koblas.dense

import com.eignex.koblas.DenseMatrix
import com.eignex.koblas.vendor.Blas
import com.eignex.koblas.vendor.BlasOperation
import com.eignex.koblas.vendor.openBlas
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * The guarantees [DenseBlas] states, on an engine that has a library to hand.
 *
 * A built-in call means the same thing on every platform. A library is part of the host rather than something
 * the caller asked for, so an ordinary call cannot start behaving differently because one is installed; where
 * a routine's documented result and a library's freedom can be told apart, the portable schedule has to be
 * chosen, and chosen before anything is written. These are the two places this library states a result that a
 * legal `dgemm` or `dsyr2k` need not produce, so they are the two places the policy has to hold back.
 *
 * Each case is run twice over. Once against a binding that records and computes nothing, where a call sent
 * across leaves the destination untouched and is caught twice, by the recorder and by the number; and once
 * against whatever library is installed, where a call sent across would come back with that library's answer.
 * Both are needed: the first fails on this host with no library at all, and the second is the one that would
 * notice a guard that holds only because the recorder is not a real library.
 *
 * Every fixture here is past [HostDensePolicy.MINIMUM_WORK], so the size rule is not what keeps these calls
 * at home. The policy runs at its own number rather than a forced one for the same reason.
 */
class HostDenseContractTest {
    private fun composed(host: Blas): HostDenseBlas =
        HostDenseBlas(PortableDenseBlas(ScalarVectorKernels, PortablePanelKernels), host)

    /**
     * `alpha` scales an accumulated sum, so a zero entry of an operand cannot produce a NaN on its own.
     *
     * Reference BLAS scales a coefficient as it goes in places, and an infinite multiplier against a stored
     * zero is a NaN there. The product below is one everywhere it is defined, with every other term a zero
     * times a zero, so the two answers are a matrix of infinities and a matrix of NaNs.
     */
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

    /**
     * A rank-2k update is the two products it is defined as, composed rather than fused.
     *
     * Finite operands are enough to tell that apart, which is why no multiplier check would do here. Each
     * separate sum below overflows, one to an infinity and the other to its negative, so composing them is a
     * NaN; a traversal that added each pair of terms before accumulating would cancel them and return zero.
     * The two sums are computed here by direct loops rather than through [ReferenceBlas], whose `syr2k` is
     * the interleaved form and would give the other answer.
     */
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

    /**
     * The same routine with a finite multiplier does go across, so the guard above is the multiplier's.
     *
     * Without this the first case would pass on a policy that simply never reached a library.
     */
    @Test
    fun `a finite multiplier on the same product does reach the library`() {
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

    private companion object {
        /** Past the policy's size as a cubic product, so only the multiplier decides. */
        const val PRODUCT_ORDER = 16

        /** Two rows, so one off-diagonal entry carries the whole question. */
        const val PAIR_ROWS = 2

        /** Long enough for each separate sum to leave the finite range, and past the policy's size. */
        const val PAIR_DEPTH = 512
    }
}
