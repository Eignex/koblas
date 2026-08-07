package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.randomVector
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The compiled-in sparse kernels against the portable interface defaults they may replace.
 *
 * Runs on every target, which is the point: on all but the JVM the two are the same code and this is a cheap
 * tautology, while on the JVM it is the gathered `usdot` being held to the loop's answer. The nonzero counts
 * sweep across the JVM's threshold (128) and across the vector width either side of it, because the shapes
 * that break a vectorized kernel are the tail below one register and a count that is not a whole number of
 * registers.
 */
class PlatformSparseVectorKernelsTest {

    /** A sparse vector of exactly [nnz] entries over [size] positions, one per evenly spaced block. */
    private fun sparse(size: Int, nnz: Int, rng: Random): SparseVector {
        val stride = size / nnz
        val indices = IntArray(nnz) { k -> k * stride + rng.nextInt(stride) }
        return SparseVector.of(size, indices, DoubleArray(nnz) { rng.nextDouble(-1.0, 1.0) })
    }

    @Test
    fun `dot against a dense operand matches the portable loop at every count`() {
        val counts = (1..17) + listOf(31, 32, 33, 127, 128, 129, 255, 512, 1000)
        for (nnz in counts) {
            val rng = Random(nnz * 7919)
            val size = nnz * 8
            val x = sparse(size, nnz, rng)
            val y = randomVector(size, rng)
            val expected = ReferenceSparseLinearAlgebra.dot(x, y)
            val actual = PlatformSparseVectorKernels.dot(x, y)
            // Not exact: a vectorized reduction sums lanes in a different order, so the two associate
            // differently. The bound scales with the count because that reordering is what accumulates.
            assertTrue(
                abs(actual - expected) <= 1e-13 * nnz * (1.0 + abs(expected)),
                "nnz=$nnz: $actual vs $expected",
            )
        }
    }

    @Test
    fun `dot rejects mismatched sizes on both paths`() {
        val rng = Random(4)
        for (nnz in intArrayOf(4, 200)) { // one below the JVM threshold, one above it
            val x = sparse(nnz * 8, nnz, rng)
            val wrong = DoubleArray(x.size + 1)
            val failed = try {
                PlatformSparseVectorKernels.dot(x, wrong)
                false
            } catch (_: IllegalArgumentException) {
                true
            }
            assertTrue(failed, "nnz=$nnz should have rejected a dense operand of the wrong length")
        }
    }

    /** The other three routines are the interface defaults on every target; this pins that they still are. */
    @Test
    fun `the routines with no vector form are the portable ones`() {
        val rng = Random(11)
        val x = sparse(4096, 512, rng)
        val dense = randomVector(4096, rng)

        val expectedAxpy = dense.copyOf().also { ReferenceSparseLinearAlgebra.axpy(it, 2.5, x) }
        val actualAxpy = dense.copyOf().also { PlatformSparseVectorKernels.axpy(it, 2.5, x) }
        assertTrue(expectedAxpy.contentEquals(actualAxpy), "axpy diverged from the portable loop")

        val expectedScatter = dense.copyOf().also { ReferenceSparseLinearAlgebra.scatter(x, it) }
        val actualScatter = dense.copyOf().also { PlatformSparseVectorKernels.scatter(x, it) }
        assertTrue(expectedScatter.contentEquals(actualScatter), "scatter diverged from the portable loop")

        val other = sparse(4096, 400, Random(12))
        assertTrue(
            PlatformSparseVectorKernels.dot(x, other) == ReferenceSparseLinearAlgebra.dot(x, other),
            "the sparse-against-sparse merge must stay the portable one",
        )
    }
}
