package com.eignex.koblas.sparse

import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.randomVector
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class PlatformSparseKernelsTest {

    private fun sparse(size: Int, nnz: Int, rng: Random): F64SparseVector {
        val stride = size / nnz
        val indices = IntArray(nnz) { k -> k * stride + rng.nextInt(stride) }
        return F64SparseVector.of(size, indices, DoubleArray(nnz) { rng.nextDouble(-1.0, 1.0) })
    }

    @Test
    fun `dot against a dense operand matches the portable loop at every count`() {
        val counts = (1..17) + listOf(31, 32, 33, 127, 128, 129, 255, 512, 1000)
        for (nnz in counts) {
            val rng = Random(nnz * 7919)
            val size = nnz * 8
            val x = sparse(size, nnz, rng)
            val y = randomVector(size, rng)
            val expected = F64ReferenceSparseLinearAlgebra.dot(x, y)
            val actual = F64PlatformSparseKernels.dot(x, y)
            // A vectorized reduction sums lanes in a different order, so the bound scales with the count.
            assertTrue(
                abs(actual - expected) <= 1e-13 * nnz * (1.0 + abs(expected)),
                "nnz=$nnz: $actual vs $expected",
            )
        }
    }

    @Test
    fun `the reductions match the portable loop at every count`() {
        val counts = (1..17) + listOf(31, 32, 33, 127, 128, 129, 255, 512, 1000)
        for (nnz in counts) {
            val rng = Random(nnz * 6151)
            val x = sparse(nnz * 8, nnz, rng)
            // Both reduce over the stored values, so a vectorized kernel sums lanes in a different order and
            // the bound scales with the count, as it does for dot above.
            val expectedAsum = F64ReferenceSparseLinearAlgebra.asum(x)
            val actualAsum = F64PlatformSparseKernels.asum(x)
            assertTrue(
                abs(actualAsum - expectedAsum) <= 1e-13 * nnz * (1.0 + abs(expectedAsum)),
                "asum nnz=$nnz: $actualAsum vs $expectedAsum",
            )
            val expectedNrm2 = F64ReferenceSparseLinearAlgebra.nrm2(x)
            val actualNrm2 = F64PlatformSparseKernels.nrm2(x)
            assertTrue(
                abs(actualNrm2 - expectedNrm2) <= 1e-13 * nnz * (1.0 + abs(expectedNrm2)),
                "nrm2 nnz=$nnz: $actualNrm2 vs $expectedNrm2",
            )
        }
    }

    /** The rescaling path both reductions share, which a plain sum of squares would answer with 0 or Inf. */
    @Test
    fun `the reductions survive values that would overflow a plain sum of squares`() {
        for (scale in doubleArrayOf(1e200, 1e-200)) {
            val rng = Random(20260825)
            val nnz = 64
            val values = DoubleArray(nnz) { rng.nextDouble(0.5, 1.0) * scale }
            val x = F64SparseVector.of(nnz * 4, IntArray(nnz) { it * 4 }, values)
            val expectedNrm2 = F64ReferenceSparseLinearAlgebra.nrm2(x)
            val actualNrm2 = F64PlatformSparseKernels.nrm2(x)
            assertTrue(actualNrm2.isFinite() && actualNrm2 > 0.0, "nrm2 at scale $scale is $actualNrm2")
            assertTrue(
                abs(actualNrm2 - expectedNrm2) <= 1e-13 * nnz * expectedNrm2,
                "nrm2 scale=$scale: $actualNrm2 vs $expectedNrm2",
            )
            val expectedAsum = F64ReferenceSparseLinearAlgebra.asum(x)
            val actualAsum = F64PlatformSparseKernels.asum(x)
            assertTrue(
                abs(actualAsum - expectedAsum) <= 1e-13 * nnz * expectedAsum,
                "asum scale=$scale: $actualAsum vs $expectedAsum",
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
                F64PlatformSparseKernels.dot(x, wrong)
                false
            } catch (_: IllegalArgumentException) {
                true
            }
            assertTrue(failed, "nnz=$nnz should have rejected a dense operand of the wrong length")
        }
    }

    @Test
    fun `the routines with no vector form are the portable ones`() {
        val rng = Random(11)
        val x = sparse(4096, 512, rng)
        val dense = randomVector(4096, rng)

        val expectedAxpy = dense.copyOf().also { F64ReferenceSparseLinearAlgebra.axpy(it, 2.5, x) }
        val actualAxpy = dense.copyOf().also { F64PlatformSparseKernels.axpy(it, 2.5, x) }
        assertTrue(expectedAxpy.contentEquals(actualAxpy), "axpy diverged from the portable loop")

        val expectedScatter = dense.copyOf().also { F64ReferenceSparseLinearAlgebra.scatter(x, it) }
        val actualScatter = dense.copyOf().also { F64PlatformSparseKernels.scatter(x, it) }
        assertTrue(expectedScatter.contentEquals(actualScatter), "scatter diverged from the portable loop")

        val other = sparse(4096, 400, Random(12))
        assertTrue(
            F64PlatformSparseKernels.dot(x, other) == F64ReferenceSparseLinearAlgebra.dot(x, other),
            "the sparse-against-sparse merge must stay the portable one",
        )
    }
}
