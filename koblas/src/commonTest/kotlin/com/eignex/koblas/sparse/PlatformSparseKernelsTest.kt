package com.eignex.koblas.sparse

import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseVector
import com.eignex.koblas.randomVector
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
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
    fun `dot against a sparse operand matches the portable merge`() {
        val rng = Random(20260830)
        for (nnz in listOf(1, 7, 32, 129, 512)) {
            val size = nnz * 8
            val x = sparse(size, nnz, rng)
            val y = sparse(size, nnz, rng)
            val expected = F64ReferenceSparseLinearAlgebra.dot(x, y)
            val actual = F64PlatformSparseKernels.dot(x, y)
            assertTrue(
                abs(actual - expected) <= 1e-13 * nnz * (1.0 + abs(expected)),
                "nnz=$nnz: $actual vs $expected",
            )
        }
    }

    @Test
    fun `axpy matches the portable sparse update`() {
        val rng = Random(20260831)
        for (nnz in listOf(1, 7, 32, 129, 512)) {
            val x = sparse(nnz * 8, nnz, rng)
            val expected = randomVector(x.size, rng)
            val actual = expected.copyOf()
            F64ReferenceSparseLinearAlgebra.axpy(expected, -0.75, x)
            F64PlatformSparseKernels.axpy(actual, -0.75, x)
            assertClose(expected, actual, "nnz=$nnz", tolerance = 1e-15)
        }
    }

    @Test
    fun `scatter matches the portable sparse write`() {
        val rng = Random(20260901)
        for (nnz in listOf(1, 7, 32, 129, 512)) {
            val x = sparse(nnz * 8, nnz, rng)
            val expected = randomVector(x.size, rng)
            val actual = expected.copyOf()
            F64ReferenceSparseLinearAlgebra.scatter(x, expected)
            F64PlatformSparseKernels.scatter(x, actual)
            assertContentEquals(expected, actual, "nnz=$nnz")
        }
    }

    @Test
    fun `gather matches the portable sparse read`() {
        val rng = Random(20260902)
        for (nnz in listOf(1, 7, 32, 129, 512)) {
            val pattern = sparse(nnz * 8, nnz, rng)
            val expected = F64SparseVector.of(pattern.size, pattern.indices, pattern.values)
            val actual = F64SparseVector.of(pattern.size, pattern.indices, pattern.values)
            val from = randomVector(pattern.size, rng)
            F64ReferenceSparseLinearAlgebra.gather(expected, from)
            F64PlatformSparseKernels.gather(actual, from)
            assertContentEquals(expected.values, actual.values, "nnz=$nnz")
        }
    }

    @Test
    fun `gather zero matches the portable sparse move`() {
        val rng = Random(20260903)
        for (nnz in listOf(1, 7, 32, 129, 512)) {
            val pattern = sparse(nnz * 8, nnz, rng)
            val expected = F64SparseVector.of(pattern.size, pattern.indices, pattern.values)
            val actual = F64SparseVector.of(pattern.size, pattern.indices, pattern.values)
            val expectedFrom = randomVector(pattern.size, rng)
            val actualFrom = expectedFrom.copyOf()
            F64ReferenceSparseLinearAlgebra.gatherZero(expected, expectedFrom)
            F64PlatformSparseKernels.gatherZero(actual, actualFrom)
            assertContentEquals(expected.values, actual.values, "values nnz=$nnz")
            assertContentEquals(expectedFrom, actualFrom, "source nnz=$nnz")
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
}
