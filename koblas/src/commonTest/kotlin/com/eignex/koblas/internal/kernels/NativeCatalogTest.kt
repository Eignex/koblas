package com.eignex.koblas.internal.kernels

import kotlin.test.*

class NativeCatalogTest {
    private val host = NativeHost(1, 1, 0xfffu, 0xfffu, 0xfffu, 1, 6, 154, 0, 1)
    private val context = NativeContext(0, 0, null, null)
    private val geometry = NativeGeometry(0, 1, 1, 1, 1, 16, 16, 1, 0x30001, 0x30001, 0, 64, 0)
    private val fp64 = NativeKernel(
        9001, 100, 100, 0, 0u, NativeTypes(1, 1, 1, 1, 0), geometry,
        NativeState(0, 0, 0), 2, 3, 1, 1, 1,
    )

    @Test
    fun `ACE without double accumulation cannot serve double arithmetic`() {
        val ace = fp64.copy(requiredFeatures = 1uL shl 10, types = NativeTypes(2, 2, 2, 2, 0))

        assertEquals(8, nativeEligibility(ace, host, context))
    }

    @Test
    fun `AVX ten does not imply ACE availability`() {
        val avxTen = fp64.copy(requiredFeatures = 1uL shl 9)
        val avxHost = host.copy(hardware = 1uL shl 9, usable = 1uL shl 9)

        assertEquals(0, nativeEligibility(avxTen, avxHost, context))
        assertEquals(5, nativeEligibility(avxTen.copy(requiredFeatures = 1uL shl 10), avxHost, context))
    }

    @Test
    fun `build CPU and OS unavailability remain distinguishable`() {
        val kernel = fp64.copy(requiredFeatures = 4u)

        assertEquals(9, nativeEligibility(kernel, host.copy(built = 0u), context))
        assertEquals(5, nativeEligibility(kernel, host.copy(hardware = 0u), context))
        assertEquals(6, nativeEligibility(kernel, host.copy(usable = 0u), context))
    }

    @Test
    fun `permission and thread preparation have distinct lifetimes`() {
        val kernel = fp64.copy(state = NativeState(4, 4, 4))

        assertEquals(7, nativeEligibility(kernel, host, context))
        assertEquals(8, nativeEligibility(kernel, host, context.copy(readyProcess = 4)))
        assertEquals(0, nativeEligibility(kernel, host, context.copy(readyProcess = 4, readyThread = 4)))
        assertEquals(
            6,
            nativeEligibility(kernel.copy(reason = 6), host, context.copy(readyProcess = 4, readyThread = 4)),
        )
    }

    @Test
    fun `a future double tile preserves geometry without streaming vector length`() {
        val kernel = fp64.copy(requiredFeatures = 1uL shl 10, state = NativeState(0, 0, 4))

        assertEquals(0, nativeEligibility(kernel, host, context))
        assertEquals(16, kernel.geometry.rows)
        assertEquals(0, kernel.geometry.registerBits)
        assertEquals(4, kernel.state.call)
        assertNull(context.streamingVectorBytes)
    }

    @Test
    fun `unknown feature bits survive record decoding`() {
        val words = IntArray(64)
        words[0] = 1
        words[1] = 256
        words[13] = Int.MIN_VALUE
        words[23] = Int.MIN_VALUE
        words[52] = 64

        assertEquals(1uL shl 63, NativeRecords.host(words).hardware)
        assertEquals(1uL shl 63, NativeRecords.kernel(words).requiredFeatures)
        assertNull(NativeRecords.context(words).ordinaryVectorBytes)
        words[54] = 1
        assertEquals(64, NativeRecords.context(words).ordinaryVectorBytes)
    }

    @Test
    fun `record decoding rejects incompatible prefixes`() {
        val words = IntArray(64)
        words[0] = 2
        words[1] = 256

        assertFailsWith<IllegalArgumentException> { NativeRecords.host(words) }
        words[0] = 1
        words[1] = 252
        assertFailsWith<IllegalArgumentException> { NativeRecords.host(words) }
        words[1] = 256
        words[3] = 2
        assertFailsWith<IllegalArgumentException> { NativeRecords.host(words) }
    }

    @Test
    fun `real descriptors round trip through exact queries`() {
        NativeCatalog.kernels.forEach { kernel ->
            val result = NativeRecords.kernel(checkNotNull(NativeProbe.query(2, kernelId = kernel.id)))

            assertEquals(kernel, result)
            assertEquals(kernel.operation, kernel.id / 16)
            assertEquals(kernel.variant, kernel.id % 16)
            assertTrue(kernel.types.isDouble)
        }
    }
}
