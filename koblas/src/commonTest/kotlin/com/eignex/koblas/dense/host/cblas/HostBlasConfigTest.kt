package com.eignex.koblas.dense.host.cblas

import kotlin.test.*

class HostBlasConfigTest {
    @Test
    fun `a default LP64 build is accepted`() {
        assertTrue(isLp64OpenBlas("OpenBLAS 0.3.21 DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64"))
        assertTrue(isLp64OpenBlas("OpenBLAS 0.3.28 DYNAMIC_ARCH NO_AFFINITY Zen MAX_THREADS=128"))
        assertTrue(isLp64OpenBlas("OpenBLAS 0.3.21 USE64BITINT_OFF DYNAMIC_ARCH Haswell"))
    }

    @Test
    fun `an ILP64 build is rejected under either spelling`() {
        assertFalse(
            isLp64OpenBlas("OpenBLAS 0.3.21 USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64"),
        )
        assertFalse(isLp64OpenBlas("OpenBLAS 0.3.28 INTERFACE64 DYNAMIC_ARCH Zen MAX_THREADS=128"))
    }

    @Test
    fun `a library without OpenBLAS ABI metadata is rejected`() {
        assertFalse(isLp64OpenBlas(""))
        assertFalse(isLp64OpenBlas("unknown CBLAS implementation"))
    }
}
