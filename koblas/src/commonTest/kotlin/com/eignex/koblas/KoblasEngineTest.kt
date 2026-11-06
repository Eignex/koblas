package com.eignex.koblas

import com.eignex.koblas.engine.*
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.EngineKernels
import kotlin.test.*

class KoblasEngineTest {
    @Test
    fun `typed vector calls use the chosen engine primitives`() {
        val engine = object : EngineKernels {
            override val name = "recording"
            override fun sum(v: DoubleArray, vOff: Int, len: Int, vStride: Int): Double {
                assertEquals(3, len)
                assertEquals(5, vOff)
                assertEquals(-2, vStride)
                return 42.0
            }
        }
        val view = StridedVector(DoubleArray(7), 5, 3, -2)

        assertEquals(42.0, engine.sum(view))
    }

    @Test
    fun `typed reductions agree with scalar window references`() {
        for (engine in listOf(BuiltinEngines.scalar, koblas)) {
            for (stride in intArrayOf(-2, 1, 2)) {
                val source = vectorView(stride)
                val other = vectorView(-stride)
                val scalar = BuiltinEngines.scalar

                assertReductionAgreesWithReference(
                    scalar.dot(source.values, source.offset, other.values, other.offset, 3, stride, -stride),
                    engine.dot(source, other),
                )
                assertReductionAgreesWithReference(
                    scalar.nrm2(source.values, source.offset, 3, stride),
                    engine.nrm2(source),
                )
                assertReductionAgreesWithReference(
                    scalar.asum(source.values, source.offset, 3, stride),
                    engine.asum(source),
                )
                assertReductionAgreesWithReference(
                    scalar.sum(source.values, source.offset, 3, stride),
                    engine.sum(source),
                )
                assertEquals(scalar.iamax(source.values, source.offset, 3, stride), engine.iamax(source))
            }
        }
    }

    @Test
    fun `typed vector mutations preserve padding and agree with scalar windows`() {
        for (operation in listOf("scale", "axpy", "copy", "rot", "swap")) {
            for (stride in intArrayOf(-2, 1, 2)) {
                val x = vectorView(stride)
                val y = vectorView(-stride)
                val expectedX = x.values.copyOf()
                val expectedY = y.values.copyOf()
                val scalar = BuiltinEngines.scalar
                when (operation) {
                    "scale" -> scalar.scale(expectedX, x.offset, -2.0, 3, stride)
                    "axpy" -> scalar.axpy(expectedY, y.offset, 0.75, expectedX, x.offset, 3, -stride, stride)
                    "copy" -> scalar.copy(expectedX, x.offset, expectedY, y.offset, 3, stride, -stride)
                    "rot" -> scalar.rot(expectedX, x.offset, expectedY, y.offset, 3, 0.6, 0.8, stride, -stride)
                    "swap" -> scalar.swap(expectedX, x.offset, expectedY, y.offset, 3, stride, -stride)
                }

                when (operation) {
                    "scale" -> koblas.scale(x, -2.0)
                    "axpy" -> koblas.axpy(0.75, x, y)
                    "copy" -> koblas.copy(x, y)
                    "rot" -> koblas.rot(x, y, 0.6, 0.8)
                    "swap" -> koblas.swap(x, y)
                }

                assertVectorAgreesWithReference(expectedX, x.values)
                assertVectorAgreesWithReference(expectedY, y.values)
            }
        }
    }

    @Test
    fun `typed matrix vector calls preserve views across dense and sparse storage`() {
        val dense = DenseMatrix.ofRows(
            arrayOf(
                doubleArrayOf(2.0, 0.0, 0.0),
                doubleArrayOf(1.0, 3.0, 0.0),
                doubleArrayOf(-1.0, 2.0, 4.0),
            ),
        )
        val sparse = SparseMatrix.ofColumns(
            3,
            3,
            List(3) { j ->
                (j until 3).map { i -> i to dense[i, j] }
            },
        )
        for (sparseStorage in booleanArrayOf(false, true)) {
            for (transpose in booleanArrayOf(false, true)) {
                for (stride in intArrayOf(-2, 1, 2)) {
                    for (operation in listOf("gemv", "symv", "trsv", "trmv")) {
                        val x = vectorView(stride)
                        val y = vectorView(-stride)
                        val expectedX = x.values.copyOf()
                        val expectedY = y.values.copyOf()
                        val scalar = BuiltinEngines.scalar
                        when (operation) {
                            "gemv" -> scalar.gemv(
                                0.75, dense, expectedX, x.offset, stride, -0.5, expectedY, y.offset, -stride, transpose,
                            )

                            "symv" -> scalar.symv(
                                0.75, dense, expectedX, x.offset, stride, -0.5, expectedY, y.offset, -stride, true,
                            )

                            "trsv" -> scalar.trsv(dense, expectedX, x.offset, stride, true, transpose)

                            "trmv" -> scalar.trmv(dense, expectedX, x.offset, stride, true, transpose)
                        }

                        if (sparseStorage) {
                            when (operation) {
                                "gemv" -> koblas.gemv(0.75, sparse, x, -0.5, y, transpose)
                                "symv" -> koblas.symv(0.75, sparse, x, -0.5, y)
                                "trsv" -> koblas.trsv(sparse, x, true, transpose)
                                "trmv" -> koblas.trmv(sparse, x, true, transpose)
                            }
                        } else {
                            when (operation) {
                                "gemv" -> koblas.gemv(0.75, dense, x, -0.5, y, transpose)
                                "symv" -> koblas.symv(0.75, dense, x, -0.5, y)
                                "trsv" -> koblas.trsv(dense, x, true, transpose)
                                "trmv" -> koblas.trmv(dense, x, true, transpose)
                            }
                        }

                        assertVectorAgreesWithReference(expectedX, x.values)
                        assertVectorAgreesWithReference(expectedY, y.values)
                    }
                }
            }
        }
    }

    @Test
    fun `typed rank updates preserve matrix view padding`() {
        for (operation in listOf("ger", "syr", "syr2")) {
            val backing = DenseMatrix.ofRows(Array(5) { i -> DoubleArray(5) { j -> (i + j) * 0.25 } })
            val region = backing.view(1, 1, 3, 3)
            val expected = DenseMatrix.wrap(5, 5, backing.values.copyOf())
            val expectedRegion = expected.view(1, 1, 3, 3)
            val x = vectorView(-2)
            val y = vectorView(2)
            when (operation) {
                "ger" -> BuiltinEngines.scalar.ger(
                    0.75,
                    x.values,
                    x.offset,
                    x.stride,
                    y.values,
                    y.offset,
                    y.stride,
                    expectedRegion,
                )

                "syr" -> BuiltinEngines.scalar.syr(0.75, x.values, x.offset, x.stride, expectedRegion, false)

                "syr2" -> BuiltinEngines.scalar.syr2(
                    0.75, x.values, x.offset, x.stride, y.values, y.offset, y.stride, expectedRegion, false,
                )
            }

            when (operation) {
                "ger" -> koblas.ger(0.75, x, y, region)
                "syr" -> koblas.syr(0.75, x, region, false)
                "syr2" -> koblas.syr2(0.75, x, y, region, false)
            }

            assertVectorAgreesWithReference(expected.values, backing.values)
        }
    }

    @Test
    fun `typed matrix vector calls validate logical sizes before writing`() {
        val a = DenseMatrix.diagonal(3)
        val short = StridedVector(DoubleArray(10) { 7.0 }, 0, 2)
        val full = DenseVector.zero(3)
        val expected = short.values.copyOf()

        assertFailsWith<DimensionMismatch> { koblas.gemv(0.0, a, full, 0.0, short) }
        assertFailsWith<DimensionMismatch> { koblas.symv(0.0, a, short, 0.0, full) }
        assertFailsWith<DimensionMismatch> { koblas.ger(0.0, full, short, a) }
        assertFailsWith<DimensionMismatch> { koblas.syr(0.0, short, a) }
        assertFailsWith<DimensionMismatch> { koblas.syr2(0.0, full, short, a) }
        assertFailsWith<DimensionMismatch> { koblas.trsv(a, short, true) }
        assertFailsWith<DimensionMismatch> { koblas.trmv(a, short, true) }
        assertContentEquals(expected, short.values)
    }

    @Test
    fun `typed matrix vector calls stage overlapping views`() {
        val a = DenseMatrix.ofRows(
            arrayOf(
                doubleArrayOf(2.0, 1.0, 0.0),
                doubleArrayOf(0.0, 3.0, -1.0),
                doubleArrayOf(1.0, 0.0, 4.0),
            ),
        )
        val sparse = SparseMatrix.ofColumns(
            3,
            3,
            List(3) { j ->
                (0 until 3).map { i -> i to a[i, j] }
            },
        )
        for (sparseStorage in booleanArrayOf(false, true)) {
            val backing = doubleArrayOf(7.0, 1.0, 2.0, 3.0, 7.0)
            val x = StridedVector(backing, 1, 3)
            val y = StridedVector(backing, 3, 3, -1)
            val expected = backing.copyOf()
            BuiltinEngines.scalar.gemv(0.75, a, x.toDoubleArray(), 0, 1, -0.5, expected, 3, -1)

            if (sparseStorage) {
                koblas.gemv(0.75, sparse, x, -0.5, y)
            } else {
                koblas.gemv(0.75, a, x, -0.5, y)
            }

            assertVectorAgreesWithReference(expected, backing)
        }
    }

    @Test
    fun `typed matrix vector zero multipliers skip poisoned inputs`() {
        val dense = DenseMatrix.wrap(2, 2, DoubleArray(4) { Double.NaN })
        val sparse = SparseMatrix.wrap(2, 2, intArrayOf(0, 1, 2), intArrayOf(0, 1), DoubleArray(2) { Double.NaN })
        val x = StridedVector(DoubleArray(5) { Double.NaN }, 3, 2, -2)
        for (sparseStorage in booleanArrayOf(false, true)) {
            val y = StridedVector(DoubleArray(5) { Double.NaN }, 1, 2, 2)

            if (sparseStorage) {
                koblas.gemv(0.0, sparse, x, 0.0, y)
            } else {
                koblas.gemv(0.0, dense, x, 0.0, y)
            }

            assertContentEquals(doubleArrayOf(0.0, 0.0), y.toDoubleArray())
            assertTrue(y.values[0].isNaN())
            assertTrue(y.values[2].isNaN())
            assertTrue(y.values[4].isNaN())
        }
    }

    @Test
    fun `typed sparse gather and scatter preserve vector view padding`() {
        for (stride in intArrayOf(-2, 1, 2)) {
            val source = vectorView(stride)
            val sparse = SparseVector.wrap(3, intArrayOf(0, 2), DoubleArray(2))
            val expectedSource = source.values.copyOf()
            val expectedSparse = DoubleArray(2)
            for (k in sparse.indices.indices) {
                val position = source.offset + sparse.indices[k] * stride
                expectedSparse[k] = expectedSource[position]
                expectedSource[position] = 0.0
            }

            koblas.gather(source, sparse)
            assertContentEquals(expectedSparse, sparse.values)
            koblas.gatherZero(source, sparse)

            assertContentEquals(expectedSparse, sparse.values)
            assertContentEquals(expectedSource, source.values)
            val destination = vectorView(-stride)
            val expected = destination.values.copyOf()
            for (k in sparse.indices.indices) {
                expected[destination.offset + sparse.indices[k] * destination.stride] = sparse.values[k]
            }

            koblas.scatter(sparse, destination)

            assertContentEquals(expected, destination.values)
        }
    }

    private fun vectorView(stride: Int): DenseVector {
        val offset = if (stride > 0) 1 else 1 - 2 * stride
        return StridedVector(DoubleArray(8) { -7.0 }, offset, 3, stride).also { vector ->
            for (i in 0 until 3) vector[i] = i * 0.75 - 1.0
        }
    }

    private fun assertVectorAgreesWithReference(expected: DoubleArray, actual: DoubleArray) {
        assertClose(expected, actual, "typed engine vector")
    }

    private fun assertReductionAgreesWithReference(expected: Double, actual: Double) {
        assertClose(expected, actual, "typed engine reduction")
    }

    private val partial = object : EngineKernels {
        override val name = "partial"
        override val implementations = setOf(EngineImplementation.Dot)
        override fun dot(
            a: DoubleArray,
            aOff: Int,
            b: DoubleArray,
            bOff: Int,
            len: Int,
            aStride: Int,
            bStride: Int,
        ): Double {
            var sum = 0.0
            for (i in 0 until len) sum += a[aOff + i * aStride] * b[bOff + i * bStride]
            return sum
        }
    }

    @Test
    fun `a partial provider never inherits portable arithmetic`() {
        assertFailsWith<UnsupportedOperationException> { partial.sum(doubleArrayOf(1.0), 0, 1) }
        assertFailsWith<UnsupportedOperationException> {
            partial.dot(
                intArrayOf(0),
                0,
                doubleArrayOf(1.0),
                0,
                1,
                doubleArrayOf(1.0),
            )
        }
        assertFailsWith<UnsupportedOperationException> { partial.tileRows }
        assertNull(partial.implementationFor(EngineImplementation.Sum, 1))
    }

    @Test
    fun `conditional product geometry is rejected before packing`() {
        val profile = EngineProfile(
            "test",
            listOf(EngineRule("portable", setOf(EngineImplementation.Products), rows = 32..Int.MAX_VALUE)),
        )

        assertFailsWith<IllegalArgumentException> { combine(profile, BuiltinEngines.scalar) }
    }
}
