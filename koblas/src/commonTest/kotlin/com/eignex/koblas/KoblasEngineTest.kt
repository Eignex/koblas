package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertSame

class KoblasEngineTest {
    @Test
    fun `portable factory selects owned scalar kernels without a vendor`() {
        val engine = BuiltinEngines.scalar

        assertNull(engine.vendor)
        assertSame(PortableEngine, engine)
        assertEquals("scalar", engine.name)
        assertEquals(BuiltinEngines.scalar.name, engine.name)
    }

    @Test
    fun `portable selection leaves the platform default intact`() {
        val default = koblas

        BuiltinEngines.scalar
        assertSame(default, koblas)
    }

    @Test
    fun `sparse convenience operations execute the engine selected kernels`() {
        val x = SparseVector.of(4, intArrayOf(1, 3), doubleArrayOf(2.0, 4.0))
        val calls = mutableListOf<String>()
        val dense = object : KoblasEngine() {
            override val implementations = setOf(EngineImplementation.Nrm2)

            override val name = "selected-dense"
            override fun nrm2(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                assertSame(x.values, v)
                calls += name
                return 0.0
            }
        }
        val indexed = object : KoblasEngine() {
            override val implementations = setOf(EngineImplementation.IndexedDot)
            override val name = "selected-indexed"
            override fun dotDense(
                indices: IntArray,
                indexOffset: Int,
                values: DoubleArray,
                valueOffset: Int,
                count: Int,
                dense: DoubleArray,
            ): Double {
                assertSame(x.indices, indices)
                assertSame(x.values, values)
                calls += name
                return 0.0
            }
        }
        val engine = engineWithKernels(vectors = dense, indexed = indexed)

        engine.nrm2(x)
        engine.dot(x, DoubleArray(x.size))

        assertEquals(listOf(dense.name, indexed.name), calls)
    }
}
