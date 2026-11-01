package com.eignex.koblas.internal.jvm25

import com.eignex.koblas.internal.jvm25.simd.simdAvailable
import com.eignex.koblas.internal.jvm25.simd.simdBitSizeFor
import org.junit.Assume
import kotlin.test.Test
import kotlin.test.assertEquals

class SimdSpeciesTest {
    @Test
    fun `a 512-bit x86 unit narrows to 256 bits`() {
        Assume.assumeTrue("the Vector API module is unavailable", simdAvailable)
        for (arch in listOf("amd64", "x86_64", "x64")) {
            assertEquals(256, simdBitSizeFor(arch, 512), arch)
        }
    }

    @Test
    fun `every other unit keeps its preferred width`() {
        Assume.assumeTrue("the Vector API module is unavailable", simdAvailable)
        val cases = listOf("amd64" to 256, "amd64" to 128, "aarch64" to 512, "aarch64" to 256, "aarch64" to 128)
        for ((arch, bits) in cases) {
            assertEquals(bits, simdBitSizeFor(arch, bits), "$arch $bits")
        }
    }
}
