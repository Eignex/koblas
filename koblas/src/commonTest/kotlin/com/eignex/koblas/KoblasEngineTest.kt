package com.eignex.koblas

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.engine.*
import com.eignex.koblas.internal.host.environmentVariableOrNull
import com.eignex.koblas.internal.kernels.*
import com.eignex.koblas.internal.kernels.EngineKernels
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import kotlin.test.*
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

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
        val view = DenseVector.wrap(DoubleArray(7), 5, 3, -2)

        assertEquals(42.0, engine.sum(view))
    }

    @Test
    fun `typed reductions agree with scalar window references`() {
        for (engine in listOf(KoblasEngine.scalar, koblas)) {
            for (stride in intArrayOf(-2, 1, 2)) {
                val source = vectorView(stride)
                val other = vectorView(-stride)
                val scalar = KoblasEngine.scalar

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
                val scalar = KoblasEngine.scalar
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
                        val scalar = KoblasEngine.scalar
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
                "ger" -> KoblasEngine.scalar.ger(
                    0.75,
                    x.values,
                    x.offset,
                    x.stride,
                    y.values,
                    y.offset,
                    y.stride,
                    expectedRegion,
                )

                "syr" -> KoblasEngine.scalar.syr(0.75, x.values, x.offset, x.stride, expectedRegion, false)

                "syr2" -> KoblasEngine.scalar.syr2(
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
        val short = DenseVector.wrap(DoubleArray(10) { 7.0 }, 0, 2)
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
            val x = DenseVector.wrap(backing, 1, 3)
            val y = DenseVector.wrap(backing, 3, 3, -1)
            val expected = backing.copyOf()
            KoblasEngine.scalar.gemv(0.75, a, x.toDoubleArray(), 0, 1, -0.5, expected, 3, -1)

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
        val x = DenseVector.wrap(DoubleArray(5) { Double.NaN }, 3, 2, -2)
        for (sparseStorage in booleanArrayOf(false, true)) {
            val y = DenseVector.wrap(DoubleArray(5) { Double.NaN }, 1, 2, 2)

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
        return DenseVector.wrap(DoubleArray(8) { -7.0 }, offset, 3, stride).also { vector ->
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
    fun `exact built in engines do not resolve a host library`() {
        assertEquals(null, KoblasEngine.scalar.vendor)
        assertEquals(null, KoblasEngine.simd?.vendor)
    }

    @Test
    fun `exact SIMD uses supplied kernels from their intrinsic minimum length`() {
        val simd = KoblasEngine.simd ?: return
        val leaf = simd.providers.first { it.id == "jvm-simd" && EngineImplementation.Iamax in it.implementations }
            as EngineKernels

        assertEquals(
            leaf.name,
            simd.implementationFor(EngineImplementation.Iamax, leaf.minimumLength(EngineImplementation.Iamax)),
        )
    }

    // Kotlin/Native can still compose whole host calls over the owned matrix backends, which is checked
    // separately; the desktop JVM default with the Vector API is checked by JvmBackendTest.
    @Test
    fun `without the Vector API the owned matrix backends are portable`() {
        if (KoblasEngine.simd != null) return

        assertEquals(ScalarEngine.name, koblas.panelProvider.implementationFor(PanelWork.RankUpdate, WIDE, 4))
    }

    /** Below any arm's threshold the call overhead outweighs the arithmetic, so the portable loop runs. */
    @Test
    fun `a single entry call stays with the portable kernels`() {
        assertEquals(ScalarEngine.name, koblas.implementationFor(EngineImplementation.Dot, 1))
    }

    @Test
    fun `explicitly required vendor libraries must pass the loader checks`() {
        val requested = environmentVariableOrNull("KOBLAS_REQUIRE_VENDORS") ?: return
        for (name in requested.split(',')) {
            val vendor = assertNotNull(
                Vendor.entries.singleOrNull { it.name.equals(name.trim(), ignoreCase = true) },
                "Unknown required vendor: $name",
            )

            assertNotNull(
                KoblasEngine.vendor(vendor),
                "Required ${vendor.vendorName} failed to load or pass ABI and thread checks",
            )
        }
    }

    @Test
    fun `exact vendor providers do not supply portable only operations`() {
        val engine = installed() ?: return skipped("exact vendor separation")

        assertEquals(setOf(engine.id), engine.providers.map { it.id }.toSet())
        assertTrue(EngineImplementation.Sum !in engine.implementations)
        assertFailsWith<UnsupportedOperationException> { engine.sum(doubleArrayOf(1.0), 0, 1) }
        assertEquals(null, engine.implementationFor(EngineImplementation.Sum, 1))
    }

    @Test
    fun `two instances of one vendor are independent of each other`() {
        val first = installed() ?: return skipped("independent instances")
        val second = assertNotNull(
            KoblasEngine.vendor(requireNotNull(first.vendor)),
            "the same vendor failed to open twice",
        )

        assertTrue(first !== second, "opening a vendor twice returned one shared instance")
        assertEquals(first.vendor, second.vendor)
        assertEquals(first.libraryPath, second.libraryPath)
        assertEquals(first.implementations, second.implementations)

        // Using one must not disturb the other: both compute the same answer from the same operands.
        val x = DenseVector.wrap(doubleArrayOf(1.0, 2.0, 3.0, 4.0))
        assertEquals(first.dot(x.values, 0, x.values, 0, x.size), second.dot(x.values, 0, x.values, 0, x.size))
    }

    private fun installed(): KoblasEngine? = Vendor.entries.firstNotNullOfOrNull { KoblasEngine.vendor(it) }

    private fun skipped(what: String) {
        println("SKIPPED: no CBLAS library installed; $what was not verified on this host")
    }

    private fun provider(id: String, operations: Set<EngineImplementation>): KoblasEngine = object : EngineKernels {
        override val name = id
        override val implementations = operations
    }

    @Test
    fun `the default level one kernels agree with the scalar loops`() = assertLevel1KernelsAgreeWithReference(koblas)

    @Test
    fun `the default scaling agrees with the reference across windows`() = assertScaleAgreesWithReference(koblas)

    @Test
    fun `the default reductions agree with the scalar loops`() = assertReductionsAgreeWithReference(koblas)

    @Test
    fun `the default norm agrees with the scalar oracle at extreme magnitudes`() =
        assertNormExtremesAgreeWithReference(koblas)

    @Test
    fun `the default norm agrees with the scalar oracle across offsets and lengths`() =
        assertNormWindowsAgreeWithReference(koblas)

    @Test
    fun `the default iamax honours its contract`() = assertIamaxHonoursItsContract(koblas)

    @Test
    fun `the default swap agrees with the scalar loop`() = assertSwapAgreesWithReference(koblas)

    @Test
    fun `the default rot kernel agrees with the portable one`() = assertRotKernelAgreesWithReference(koblas)

    // The KoblasEngine.axpy contract: a zero alpha returns without evaluating the products, so infinity
    // times zero never reaches the destination.
    @Test
    fun `a zero alpha axpy evaluates no product`() {
        val x = DoubleArray(64) { Double.POSITIVE_INFINITY }
        val y = DoubleArray(64)

        koblas.axpy(y, 0, 0.0, x, 0, 64)

        assertContentEquals(DoubleArray(64), y, "zero axpy evaluated infinity times zero")
    }

    /**
     * A caller walking a matrix reaches its last row or column with an empty tail, so every routine here is
     * called with a zero length. Each must read nothing and return the identity.
     */
    @Test
    fun `every default kernel accepts a zero length run`() {
        val v = doubleArrayOf(1.0, 2.0, 3.0)

        val dot = koblas.dot(v, 3, v, 3, 0)
        val nrm2 = koblas.nrm2(v, 3, 0)
        val asum = koblas.asum(v, 3, 0)
        val sum = koblas.sum(v, 3, 0)
        val iamax = koblas.iamax(v, 3, 0)
        koblas.axpy(v, 3, 2.0, v, 3, 0)
        koblas.scale(v, 3, 2.0, 0)
        koblas.swap(v, 3, v, 3, 0)

        assertEquals(0.0, dot, "dot over nothing")
        assertEquals(0.0, nrm2, "nrm2 over nothing")
        assertEquals(0.0, asum, "asum over nothing")
        assertEquals(0.0, sum, "sum over nothing")
        assertEquals(-1, iamax, "iamax over nothing")
        assertEquals(listOf(1.0, 2.0, 3.0), v.toList(), "a zero-length write touched the vector")
    }

    /** Equal runs are staged before the vendor call, without changing arithmetic providers. */
    @Test
    fun `an aliased rotation stays on the selected vendor`() {
        val arm = koblas.implementationFor(EngineImplementation.Rot, LONG_COLUMN)
        if (koblas.providers.none { it.vendor != null && it.name == arm }) {
            println("SKIPPED: no measured host profile routes rotation to a binding on this host")
            return
        }
        val values = DoubleArray(LONG_COLUMN) { (it - 17) / 13.0 }
        val expected = values.copyOf()
        KoblasEngine.scalar.rot(expected, 0, expected, 0, LONG_COLUMN, 0.75, -0.25)

        koblas.rot(values, 0, values, 0, LONG_COLUMN, 0.75, -0.25)

        assertEquals(arm, koblas.implementationFor(EngineImplementation.Rot, LONG_COLUMN))
        assertClose(expected, values, "aliased vendor rotation")
    }

    private companion object {
        /** Wider than any arm's crossover, so the accelerated one is reached if there is one. */
        const val WIDE = 1 shl 16

        /** Past the Native Level 1 crossover, so a contiguous rotation could reach the library. */
        const val LONG_COLUMN = 256
    }
}
