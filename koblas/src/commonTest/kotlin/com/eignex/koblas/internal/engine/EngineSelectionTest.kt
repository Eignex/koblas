package com.eignex.koblas.internal.engine

import com.eignex.koblas.*
import com.eignex.koblas.dense.*
import com.eignex.koblas.internal.kernels.KernelProvider
import com.eignex.koblas.internal.kernels.scalar.ScalarEngine
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
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

            assertEquals(listOf(vendor), opened)
            assertEquals(provider.name, engine.implementationFor(EngineImplementation.Copy, 16))
            assertEquals(ScalarEngine.name, engine.implementationFor(EngineImplementation.Sum, 16))
            assertLevel1KernelsAgreeWithReference(engine)
        }
    }
}
