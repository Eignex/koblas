package com.eignex.koblas.sparse

import com.eignex.koblas.BuiltinEngines
import com.eignex.koblas.SparseVector
import com.eignex.koblas.assertClose
import com.eignex.koblas.koblas
import com.eignex.koblas.randomVector
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class PlatformSparseKernelsTest {

    private val engines
        get() = listOfNotNull(BuiltinEngines.scalar, BuiltinEngines.c, BuiltinEngines.simd).distinct()

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
            val expected = ReferenceSparseBlas.dot(x, y)
            val actual = koblas.sparseKernels.dot(x, y)
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
            val expected = ReferenceSparseBlas.dot(x, y)
            val actual = koblas.sparseKernels.dot(x, y)
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
            ReferenceSparseBlas.axpy(expected, -0.75, x)
            koblas.sparseKernels.axpy(actual, -0.75, x)
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
            ReferenceSparseBlas.scatter(x, expected)
            koblas.sparseKernels.scatter(x, actual)
            assertContentEquals(expected, actual, "nnz=$nnz")
        }
    }

    @Test
    fun `gather matches the portable sparse read`() {
        val rng = Random(20260902)
        for (nnz in listOf(1, 7, 32, 129, 512)) {
            val pattern = sparse(nnz * 8, nnz, rng)
            val expected = SparseVector.of(pattern.size, pattern.indices, pattern.values)
            val actual = SparseVector.of(pattern.size, pattern.indices, pattern.values)
            val from = randomVector(pattern.size, rng)
            ReferenceSparseBlas.gather(expected, from)
            koblas.sparseKernels.gather(actual, from)
            assertContentEquals(expected.values, actual.values, "nnz=$nnz")
        }
    }

    @Test
    fun `gather zero matches the portable sparse move`() {
        val rng = Random(20260903)
        for (nnz in listOf(1, 7, 32, 129, 512)) {
            val pattern = sparse(nnz * 8, nnz, rng)
            val expected = SparseVector.of(pattern.size, pattern.indices, pattern.values)
            val actual = SparseVector.of(pattern.size, pattern.indices, pattern.values)
            val expectedFrom = randomVector(pattern.size, rng)
            val actualFrom = expectedFrom.copyOf()
            ReferenceSparseBlas.gatherZero(expected, expectedFrom)
            koblas.sparseKernels.gatherZero(actual, actualFrom)
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
            val expectedAsum = ReferenceSparseBlas.asum(x)
            val actualAsum = koblas.sparseKernels.asum(x)
            assertTrue(
                abs(actualAsum - expectedAsum) <= 1e-13 * nnz * (1.0 + abs(expectedAsum)),
                "asum nnz=$nnz: $actualAsum vs $expectedAsum",
            )
            val expectedNrm2 = ReferenceSparseBlas.nrm2(x)
            val actualNrm2 = koblas.sparseKernels.nrm2(x)
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
            val x = SparseVector.of(nnz * 4, IntArray(nnz) { it * 4 }, values)
            val expectedNrm2 = ReferenceSparseBlas.nrm2(x)
            val actualNrm2 = koblas.sparseKernels.nrm2(x)
            assertTrue(actualNrm2.isFinite() && actualNrm2 > 0.0, "nrm2 at scale $scale is $actualNrm2")
            assertTrue(
                abs(actualNrm2 - expectedNrm2) <= 1e-13 * nnz * expectedNrm2,
                "nrm2 scale=$scale: $actualNrm2 vs $expectedNrm2",
            )
            val expectedAsum = ReferenceSparseBlas.asum(x)
            val actualAsum = koblas.sparseKernels.asum(x)
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
                koblas.sparseKernels.dot(x, wrong)
                false
            } catch (_: IllegalArgumentException) {
                true
            }
            assertTrue(failed, "nnz=$nnz should have rejected a dense operand of the wrong length")
        }
    }

    @Test
    fun `raw indexed operations honor independent offsets`() {
        val indices = intArrayOf(99, 1, 4, 6, 99)
        val values = doubleArrayOf(99.0, 99.0, 2.0, -3.0, 4.0, 99.0)
        val dense = doubleArrayOf(10.0, 5.0, 20.0, 30.0, 7.0, 50.0, -2.0)
        for (engine in engines) {
            val kernels = engine.sparseKernels
            assertEquals(-19.0, kernels.dot(indices, 1, values, 2, 3, dense), engine.name)

            val updated = dense.copyOf()
            kernels.axpy(updated, 0.5, indices, 1, values, 2, 3)
            assertContentEquals(doubleArrayOf(10.0, 6.0, 20.0, 30.0, 5.5, 50.0, 0.0), updated, engine.name)

            val scattered = DoubleArray(dense.size) { 8.0 }
            kernels.scatter(indices, 1, values, 2, 3, scattered)
            assertContentEquals(doubleArrayOf(8.0, 2.0, 8.0, 8.0, -3.0, 8.0, 4.0), scattered, engine.name)
        }
    }

    @Test
    fun `raw reductions permit repeated unsorted support`() {
        val indices = intArrayOf(2, 0, 2)
        val values = doubleArrayOf(3.0, 4.0, 5.0)
        val dense = doubleArrayOf(2.0, 7.0, -1.0)
        for (engine in engines) {
            assertEquals(0.0, engine.sparseKernels.dot(indices, 0, values, 0, 3, dense), engine.name)
            assertEquals(kotlin.math.sqrt(6.0), engine.sparseKernels.nrm2(indices, 0, 3, dense), engine.name)
        }
    }

    @Test
    fun `raw empty slices at array ends are legal`() {
        for (engine in engines) {
            val kernels = engine.sparseKernels
            val destination = doubleArrayOf(3.0)
            assertEquals(0.0, kernels.dot(intArrayOf(9), 1, doubleArrayOf(9.0), 1, 0, DoubleArray(0)))
            kernels.axpy(destination, 2.0, intArrayOf(9), 1, doubleArrayOf(9.0), 1, 0)
            kernels.scatter(intArrayOf(9), 1, doubleArrayOf(9.0), 1, 0, destination)
            assertEquals(0.0, kernels.nrm2(intArrayOf(9), 1, 0, DoubleArray(0)))
            assertContentEquals(doubleArrayOf(3.0), destination, engine.name)
        }
    }

    @Test
    fun `raw mutations validate before changing the destination`() {
        for (engine in engines) {
            val kernels = engine.sparseKernels
            for (indices in listOf(intArrayOf(0, 3), intArrayOf(1, 1), intArrayOf(2, 0))) {
                val destination = doubleArrayOf(5.0, 6.0, 7.0)
                assertFailsWith<IllegalArgumentException>(engine.name) {
                    kernels.axpy(destination, 1.0, indices, 0, doubleArrayOf(2.0, 3.0), 0, 2)
                }
                assertContentEquals(doubleArrayOf(5.0, 6.0, 7.0), destination, engine.name)
            }
            val destination = doubleArrayOf(5.0, 6.0)
            assertFailsWith<IllegalArgumentException>(engine.name) {
                kernels.scatter(intArrayOf(0, 1), 0, destination, 0, 2, destination)
            }
            assertContentEquals(doubleArrayOf(5.0, 6.0), destination, engine.name)
        }
    }

    @Test
    fun `raw operations reject invalid windows and dimensions`() {
        for (engine in engines) {
            val kernels = engine.sparseKernels
            assertFailsWith<IllegalArgumentException>(engine.name) {
                kernels.dot(intArrayOf(0), 1, doubleArrayOf(1.0), 0, 1, doubleArrayOf(2.0))
            }
            assertFailsWith<IllegalArgumentException>(engine.name) {
                kernels.dot(intArrayOf(0), 0, doubleArrayOf(1.0), 0, -1, doubleArrayOf(2.0))
            }
            assertFailsWith<IllegalArgumentException>(engine.name) {
                kernels.nrm2(intArrayOf(1), 0, 1, doubleArrayOf(2.0))
            }
            val destination = doubleArrayOf(7.0)
            assertFailsWith<IllegalArgumentException>(engine.name) {
                kernels.scatter(intArrayOf(0), 0, doubleArrayOf(1.0), 1, 1, destination)
            }
            assertContentEquals(doubleArrayOf(7.0), destination, engine.name)
        }
    }

    @Test
    fun `raw axpy with zero alpha does not evaluate values`() {
        for (engine in engines) {
            val destination = doubleArrayOf(Double.NaN, -0.0)
            engine.sparseKernels.axpy(
                destination,
                -0.0,
                intArrayOf(0, 1),
                0,
                doubleArrayOf(Double.POSITIVE_INFINITY, Double.NaN),
                0,
                2,
            )
            assertTrue(destination[0].isNaN(), engine.name)
            assertEquals((-0.0).toBits(), destination[1].toBits(), engine.name)
        }
    }

    @Test
    fun `raw scatter preserves explicit zeros and exceptional values`() {
        val source = doubleArrayOf(-0.0, Double.NaN, Double.NEGATIVE_INFINITY)
        for (engine in engines) {
            val destination = DoubleArray(3)
            engine.sparseKernels.scatter(intArrayOf(0, 1, 2), 0, source, 0, 3, destination)
            assertEquals((-0.0).toBits(), destination[0].toBits(), engine.name)
            assertTrue(destination[1].isNaN(), engine.name)
            assertEquals(Double.NEGATIVE_INFINITY, destination[2], engine.name)
        }
    }

    @Test
    fun `indexed norm rescales and propagates nonfinite values`() {
        val support = intArrayOf(1, 3)
        for (engine in engines) {
            val kernels = engine.sparseKernels
            assertEquals(
                5e200,
                kernels.nrm2(support, 0, 2, doubleArrayOf(0.0, 3e200, 0.0, 4e200)),
                1e188,
                engine.name,
            )
            assertEquals(
                5e-200,
                kernels.nrm2(support, 0, 2, doubleArrayOf(0.0, 3e-200, 0.0, 4e-200)),
                1e-212,
                engine.name,
            )
            assertTrue(
                kernels.nrm2(support, 0, 2, doubleArrayOf(0.0, Double.NaN, 0.0, 1.0)).isNaN(),
                engine.name,
            )
            assertEquals(
                Double.POSITIVE_INFINITY,
                kernels.nrm2(
                    support,
                    0,
                    2,
                    doubleArrayOf(0.0, Double.NEGATIVE_INFINITY, 0.0, 1.0),
                ),
                engine.name,
            )
        }
    }

    @Test
    fun `container operations agree with raw indexed operations`() {
        val x = SparseVector.of(6, intArrayOf(1, 4), doubleArrayOf(2.0, -3.0))
        val dense = doubleArrayOf(1.0, 5.0, 2.0, 3.0, 7.0, 4.0)
        for (engine in engines) {
            val kernels = engine.sparseKernels
            assertEquals(
                kernels.dot(x, dense),
                kernels.dot(x.indices, 0, x.values, 0, x.values.size, dense),
                engine.name,
            )
            val containerUpdate = dense.copyOf()
            val rawUpdate = dense.copyOf()
            kernels.axpy(containerUpdate, 0.25, x)
            kernels.axpy(rawUpdate, 0.25, x.indices, 0, x.values, 0, x.values.size)
            assertContentEquals(containerUpdate, rawUpdate, engine.name)
            val containerScatter = DoubleArray(6)
            val rawScatter = DoubleArray(6)
            kernels.scatter(x, containerScatter)
            kernels.scatter(x.indices, 0, x.values, 0, x.values.size, rawScatter)
            assertContentEquals(containerScatter, rawScatter, engine.name)
            assertEquals(
                kernels.nrm2(x),
                kernels.nrm2(x.indices, 0, x.indices.size, containerScatter),
                1e-15,
                engine.name,
            )
        }
    }

    @Test
    fun `platform raw leaves agree with scalar across long offset slices`() {
        val count = 4096
        val dimension = count * 2 + 3
        val indices = IntArray(count + 4) { -1 }
        val values = DoubleArray(count + 6) { Double.NaN }
        for (k in 0 until count) {
            indices[2 + k] = 2 * k + 1
            values[3 + k] = (k % 17 - 8) * 0.125
        }
        val dense = DoubleArray(dimension) { (it % 23 - 11) * 0.0625 }
        val scalar = BuiltinEngines.scalar.sparseKernels
        val expectedDot = scalar.dot(indices, 2, values, 3, count, dense)
        val expectedNorm = scalar.nrm2(indices, 2, count, dense)
        for (engine in engines) {
            val kernels = engine.sparseKernels
            val actualDot = kernels.dot(indices, 2, values, 3, count, dense)
            assertTrue(
                abs(actualDot - expectedDot) <= 1e-12 * (1.0 + abs(expectedDot)),
                engine.name,
            )
            assertEquals(expectedNorm, kernels.nrm2(indices, 2, count, dense), 1e-12, engine.name)

            val expectedAxpy = dense.copyOf()
            val actualAxpy = dense.copyOf()
            scalar.axpy(expectedAxpy, -0.75, indices, 2, values, 3, count)
            kernels.axpy(actualAxpy, -0.75, indices, 2, values, 3, count)
            assertClose(expectedAxpy, actualAxpy, engine.name, tolerance = 1e-15)

            val expectedScatter = dense.copyOf()
            val actualScatter = dense.copyOf()
            scalar.scatter(indices, 2, values, 3, count, expectedScatter)
            kernels.scatter(indices, 2, values, 3, count, actualScatter)
            assertContentEquals(expectedScatter, actualScatter, engine.name)
        }
    }
}
