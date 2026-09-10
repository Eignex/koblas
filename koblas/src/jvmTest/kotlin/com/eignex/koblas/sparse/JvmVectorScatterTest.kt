package com.eignex.koblas.sparse

import com.eignex.koblas.SparseVector
import com.eignex.koblas.assertClose
import com.eignex.koblas.dense.simdAvailable
import org.junit.Assume
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class JvmVectorScatterTest {

    @Test
    fun `the property takes precedence over the environment variable`() {
        assertEquals(
            JvmVectorScatterMode.OFF,
            JvmVectorScatterMode.configured("off", "on"),
        )
    }

    @Test
    fun `the scatter mode accepts its documented values`() {
        assertEquals(JvmVectorScatterMode.AUTO, JvmVectorScatterMode.configured(null, null))
        assertEquals(JvmVectorScatterMode.AUTO, JvmVectorScatterMode.configured(" AUTO ", null))
        assertEquals(JvmVectorScatterMode.ON, JvmVectorScatterMode.configured(null, "on"))
        assertEquals(JvmVectorScatterMode.OFF, JvmVectorScatterMode.configured("off", null))
    }

    @Test
    fun `an unknown scatter mode fails clearly`() {
        assertFailsWith<IllegalStateException> {
            JvmVectorScatterMode.configured("sometimes", null)
        }
    }

    @Test
    fun `on enables indexed stores despite automatic eligibility`() {
        val enabled = jvmVectorScatterEnabled(
            JvmVectorScatterMode.ON,
            vectorApiAvailable = true,
            autoScatterEligible = false,
        )

        assertEquals(true, enabled)
    }

    @Test
    fun `off retains scalar stores despite automatic eligibility`() {
        val enabled = jvmVectorScatterEnabled(
            JvmVectorScatterMode.OFF,
            vectorApiAvailable = true,
            autoScatterEligible = true,
        )

        assertEquals(false, enabled)
    }

    @Test
    fun `on requires the Vector API module`() {
        assertFailsWith<IllegalStateException> {
            jvmVectorScatterEnabled(
                JvmVectorScatterMode.ON,
                vectorApiAvailable = false,
                autoScatterEligible = true,
            )
        }
    }

    @Test
    fun `indexed scatter agrees with the portable implementation`() {
        forEachPattern { x, dense ->
            val expected = dense.copyOf()
            ReferenceSparseBlas.scatter(x, expected)

            val actual = dense.copyOf()
            SparseSimd.scatter(x.indices, 0, x.values, 0, x.values.size, actual)

            assertClose(expected, actual, "scatter nnz=${x.values.size}")
        }
    }

    @Test
    fun `indexed gather agrees with the portable implementation`() {
        forEachPattern { x, dense ->
            val expected = SparseVector.wrap(x.size, x.indices.copyOf(), x.values.copyOf())
            ReferenceSparseBlas.gather(expected, dense.copyOf())

            val actual = SparseVector.wrap(x.size, x.indices.copyOf(), x.values.copyOf())
            SparseSimd.gather(actual.indices, 0, actual.values, 0, actual.values.size, dense.copyOf())

            assertClose(expected.values, actual.values, "gather nnz=${x.values.size}")
        }
    }

    @Test
    fun `indexed gather zero agrees with the portable implementation`() {
        forEachPattern { x, dense ->
            val expectedX = SparseVector.wrap(x.size, x.indices.copyOf(), x.values.copyOf())
            val expectedDense = dense.copyOf()
            ReferenceSparseBlas.gatherZero(expectedX, expectedDense)

            val actualX = SparseVector.wrap(x.size, x.indices.copyOf(), x.values.copyOf())
            val actualDense = dense.copyOf()
            SparseSimd.gatherZero(actualX.indices, 0, actualX.values, 0, actualX.values.size, actualDense)

            assertClose(expectedX.values, actualX.values, "gatherZero values nnz=${x.values.size}")
            assertClose(expectedDense, actualDense, "gatherZero dense nnz=${x.values.size}")
        }
    }

    @Test
    fun `indexed axpy agrees with the portable implementation`() {
        forEachPattern { x, dense ->
            val expected = dense.copyOf()
            ReferenceSparseBlas.axpy(expected, -0.75, x)

            val actual = dense.copyOf()
            SparseSimd.axpy(x.indices, 0, x.values, 0, x.values.size, actual, -0.75)

            assertClose(expected, actual, "axpy nnz=${x.values.size}")
        }
    }

    private fun forEachPattern(block: (SparseVector, DoubleArray) -> Unit) {
        Assume.assumeTrue("the Vector API module is unavailable", simdAvailable)
        for (nnz in intArrayOf(1, 2, 3, 4, 5, 7, 8, 9)) {
            val random = Random(nnz)
            val size = 2 * nnz + 1
            val x = SparseVector.wrap(
                size,
                IntArray(nnz) { 2 * it + 1 },
                DoubleArray(nnz) { random.nextDouble(-1.0, 1.0) },
            )
            block(x, DoubleArray(size) { random.nextDouble(-1.0, 1.0) })
        }
    }
}
