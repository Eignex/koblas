package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.kernels.KernelProvider
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertSame

class EngineSelectionTest {
    @Test
    fun `absent and automatic settings use platform selection`() {
        for (setting in listOf(null, "auto", " AUTO ")) {
            val engine = engineFromSetting(setting, automatic = { ScalarEngine })

            assertSame(KoblasEngine.scalar, engine)
        }
    }

    @Test
    fun `scalar setting does not resolve accelerated providers`() {
        val engine = engineFromSetting(
            " SCALAR ",
            automatic = { error("Unexpected automatic selection") },
            simd = { error("Unexpected SIMD selection") },
            openVendor = { error("Unexpected vendor selection") },
        )

        assertSame(KoblasEngine.scalar, engine)
    }

    @Test
    fun `SIMD setting uses the exact SIMD engine`() {
        val supplied = engineWithKernels()

        val engine = engineFromSetting("simd", simd = { supplied })

        assertSame(supplied, engine)
    }

    @Test
    fun `unavailable explicit engines fail without automatic selection`() {
        for (setting in listOf("simd") + Vendor.entries.map { it.vendorName }) {
            var automaticCalls = 0
            assertFailsWith<IllegalStateException>(setting) {
                engineFromSetting(
                    setting,
                    automatic = {
                        automaticCalls++
                        ScalarEngine
                    },
                    simd = { null },
                    openVendor = { null },
                )
            }
            assertEquals(0, automaticCalls, setting)
        }
    }

    @Test
    fun `unknown settings fail without opening providers`() {
        for (setting in listOf("", " ", "unknown")) {
            assertFailsWith<IllegalArgumentException>(setting) {
                engineFromSetting(
                    setting,
                    automatic = { error("Unexpected automatic selection") },
                    simd = { error("Unexpected SIMD selection") },
                    openVendor = { error("Unexpected vendor selection") },
                )
            }
        }
    }

    @Test
    fun `named selections share whole call routing and portable product coverage`() {
        var hostCalls = 0
        val host = object : KernelProvider {
            override val name = "test host"
            override val vendor = Vendor.OpenBlas
            override val implementations = setOf(EngineImplementation.Gemv)

            override fun gemv(
                alpha: Double,
                a: DenseMatrix,
                x: DenseVector,
                beta: Double,
                y: DenseVector,
                transpose: Boolean,
                scratch: Scratch?,
            ) {
                hostCalls++
                KoblasEngine.scalar.gemv(alpha, a, x, beta, y, transpose, scratch)
            }
        }
        val launch = engineFromSetting("openblas", openVendor = { host })
        val programmatic = requireNotNull(configuredVendorEngine(Vendor.OpenBlas) { host })
        val a = DenseMatrix.wrap(2, 2, doubleArrayOf(1.0, -2.0, 3.0, 4.0))
        val x = DenseVector.wrap(doubleArrayOf(2.0, -1.0))
        val expectedVector = DenseVector.zero(2)
        val expectedProduct = DenseMatrix.zero(2, 2)
        KoblasEngine.scalar.gemv(1.5, a, x, 0.0, expectedVector)
        KoblasEngine.scalar.gemm(1.5, a, false, a, false, 0.0, expectedProduct)

        for (engine in listOf(launch, programmatic)) {
            val vector = DenseVector.zero(2)
            val product = DenseMatrix.zero(2, 2)

            engine.gemv(1.5, a, x, 0.0, vector)
            engine.gemm(1.5, a, false, a, false, 0.0, product)

            assertSame(host, engine.wholeCallFor(EngineImplementation.Gemv, 2, 2))
            assertNull(engine.wholeCallFor(EngineImplementation.Gemm, 2, 2, 2))
            assertProductAgreesWithReference(expectedVector.values, vector.values)
            assertProductAgreesWithReference(expectedProduct.values, product.values)
        }
        assertEquals(2, hostCalls)
        assertFailsWith<UnsupportedOperationException> { host.sum(x) }
        assertFailsWith<UnsupportedOperationException> {
            host.gemm(1.5, a, false, a, false, 0.0, DenseMatrix.zero(2, 2))
        }
    }

    @Test
    fun `vendor settings retain portable coverage for missing operations`() {
        for (vendor in Vendor.entries) {
            val provider = object : KernelProvider {
                override val name = "test ${vendor.vendorName}"
                override val vendor = vendor
                override val implementations = setOf(EngineImplementation.Copy)

                override fun copy(
                    a: DoubleArray,
                    aOff: Int,
                    b: DoubleArray,
                    bOff: Int,
                    len: Int,
                    aStride: Int,
                    bStride: Int,
                ) = ScalarEngine.copy(a, aOff, b, bOff, len, aStride, bStride)
            }
            val opened = mutableListOf<Vendor>()

            val engine = engineFromSetting(
                vendor.vendorName,
                automatic = { error("Unexpected automatic selection") },
                simd = { error("Unexpected SIMD selection") },
                openVendor = {
                    opened += it
                    provider
                },
            )

            val programmatic = configuredVendorEngine(vendor) { provider }!!

            assertEquals(listOf(vendor), opened)
            for (operation in EngineImplementation.entries.filter { it.vector }) {
                for (length in listOf(1, 16, 4096)) {
                    for (contiguous in listOf(false, true)) {
                        assertEquals(
                            engine.implementationFor(operation, length, contiguous),
                            programmatic.implementationFor(operation, length, contiguous),
                        )
                    }
                }
            }
            assertLevel1KernelsAgreeWithReference(programmatic)
            assertEquals(provider.name, engine.implementationFor(EngineImplementation.Copy, 16))
            assertEquals(ScalarEngine.name, engine.implementationFor(EngineImplementation.Sum, 16))
            assertLevel1KernelsAgreeWithReference(engine)
        }
    }
    private fun assertProductAgreesWithReference(expected: DoubleArray, actual: DoubleArray) =
        assertClose(expected, actual, "named engine product")
}
