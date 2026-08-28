package com.eignex.koblas.sparse

import com.eignex.koblas.assertClose
import com.eignex.koblas.core.F64SparseVector
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
        val scatter = JvmVectorScatter.resolve(
            JvmVectorScatterMode.ON,
            vectorApiAvailable = true,
            autoScatterEligible = false,
        )

        assertEquals("indexed-store", scatter.path)
    }

    @Test
    fun `off retains scalar stores despite automatic eligibility`() {
        val scatter = JvmVectorScatter.resolve(
            JvmVectorScatterMode.OFF,
            vectorApiAvailable = true,
            autoScatterEligible = true,
        )

        assertEquals("scalar", scatter.path)
    }

    @Test
    fun `on requires the Vector API module`() {
        assertFailsWith<IllegalStateException> {
            JvmVectorScatter.resolve(
                JvmVectorScatterMode.ON,
                vectorApiAvailable = false,
                autoScatterEligible = true,
            )
        }
    }

    @Test
    fun `scatter metadata reports the resolved mode and path`() {
        val scatter = JvmVectorScatter.resolve(
            JvmVectorScatterMode.ON,
            vectorApiAvailable = true,
            autoScatterEligible = false,
        )

        assertEquals(
            mapOf(
                "jvm.vector.scatter.mode" to "on",
                "jvm.vector.scatter.path" to "indexed-store",
            ),
            scatter.metadata.options,
        )
    }

    @Test
    fun `indexed scatter agrees with the portable implementation`() {
        forEachPattern { x, dense ->
            val expected = dense.copyOf()
            F64ReferenceSparseLinearAlgebra.scatter(x, expected)

            val actual = dense.copyOf()
            SparseSimd.scatter(x.indices, x.values, actual)

            assertClose(expected, actual, "scatter nnz=${x.values.size}")
        }
    }

    @Test
    fun `indexed gather agrees with the portable implementation`() {
        forEachPattern { x, dense ->
            val expected = F64SparseVector.wrap(x.size, x.indices.copyOf(), x.values.copyOf())
            F64ReferenceSparseLinearAlgebra.gather(expected, dense.copyOf())

            val actual = F64SparseVector.wrap(x.size, x.indices.copyOf(), x.values.copyOf())
            SparseSimd.gather(actual.indices, actual.values, dense.copyOf())

            assertClose(expected.values, actual.values, "gather nnz=${x.values.size}")
        }
    }

    @Test
    fun `indexed gather zero agrees with the portable implementation`() {
        forEachPattern { x, dense ->
            val expectedX = F64SparseVector.wrap(x.size, x.indices.copyOf(), x.values.copyOf())
            val expectedDense = dense.copyOf()
            F64ReferenceSparseLinearAlgebra.gatherZero(expectedX, expectedDense)

            val actualX = F64SparseVector.wrap(x.size, x.indices.copyOf(), x.values.copyOf())
            val actualDense = dense.copyOf()
            SparseSimd.gatherZero(actualX.indices, actualX.values, actualDense)

            assertClose(expectedX.values, actualX.values, "gatherZero values nnz=${x.values.size}")
            assertClose(expectedDense, actualDense, "gatherZero dense nnz=${x.values.size}")
        }
    }

    @Test
    fun `indexed axpy agrees with the portable implementation`() {
        forEachPattern { x, dense ->
            val expected = dense.copyOf()
            F64ReferenceSparseLinearAlgebra.axpy(expected, -0.75, x)

            val actual = dense.copyOf()
            SparseSimd.axpy(x.indices, x.values, actual, -0.75)

            assertClose(expected, actual, "axpy nnz=${x.values.size}")
        }
    }

    private fun forEachPattern(block: (F64SparseVector, DoubleArray) -> Unit) {
        Assume.assumeTrue("the Vector API module is unavailable", simdAvailable)
        for (nnz in intArrayOf(1, 2, 3, 4, 5, 7, 8, 9)) {
            val random = Random(nnz)
            val size = 2 * nnz + 1
            val x = F64SparseVector.wrap(
                size,
                IntArray(nnz) { 2 * it + 1 },
                DoubleArray(nnz) { random.nextDouble(-1.0, 1.0) },
            )
            block(x, DoubleArray(size) { random.nextDouble(-1.0, 1.0) })
        }
    }
}
