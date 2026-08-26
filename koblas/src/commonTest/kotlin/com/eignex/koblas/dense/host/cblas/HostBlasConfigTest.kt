package com.eignex.koblas.dense.host.cblas

import kotlin.test.*

class HostBlasConfigTest {

    @Test
    fun `a default LP64 build is accepted`() {
        // Config strings in the shape Debian's libopenblas0 and a Homebrew build report.
        assertFalse(isIlp64OpenBlas("OpenBLAS 0.3.21 DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64"))
        assertFalse(isIlp64OpenBlas("OpenBLAS 0.3.28 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Zen MAX_THREADS=128"))
        assertFalse(isIlp64OpenBlas(""), "a build with no config string cannot be judged, so it passes")
        assertFalse(isIlp64OpenBlas("OpenBLAS 0.3.21 USE64BITINT_OFF DYNAMIC_ARCH Haswell"))
    }

    @Test
    fun `an ILP64 build is rejected under either spelling`() {
        assertTrue(isIlp64OpenBlas("OpenBLAS 0.3.21 USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64"))
        assertTrue(isIlp64OpenBlas("OpenBLAS 0.3.28 INTERFACE64 DYNAMIC_ARCH Zen MAX_THREADS=128"))
    }

    /**
     * The config string decides on its own when it names the width, so a build that reports one is judged
     * without the cost, and the risk, of factorizing through a library koblas has not accepted yet.
     */
    @Test
    fun `a build that names its width in its config string is judged without probing`() {
        var probes = 0

        assertTrue(
            isIlp64Build("OpenBLAS 0.3.28 INTERFACE64 DYNAMIC_ARCH Zen") {
                probes++
                null
            },
        )

        assertEquals(0, probes, "the config string already said so")
    }

    /**
     * A vendor is free to report no config string, and an ILP64 build of one exports the same unsuffixed
     * symbols as LP64, so the width of what it writes is the only evidence left.
     */
    @Test
    fun `a build with no config string is judged by the pivots it writes`() {
        assertTrue(isIlp64Build("") { intArrayOf(2, 0, 2) }, "64-bit pivots went unnoticed")
        assertFalse(isIlp64Build("") { intArrayOf(2, 2, 0) }, "32-bit pivots were turned away")
    }

    @Test
    fun `a build that answers neither question is taken as LP64`() {
        assertFalse(isIlp64Build("") { null }, "a library with no LAPACKE to ask should keep working")
        assertFalse(isIlp64Build("OpenBLAS 0.3.21 DYNAMIC_ARCH Haswell") { null })
    }

    @Test
    fun `pivots written 64 bits wide are judged ILP64`() {
        assertTrue(isIlp64PivotWidth(intArrayOf(2, 0, 2)))
    }

    @Test
    fun `pivots written 32 bits wide are judged LP64`() {
        assertFalse(isIlp64PivotWidth(intArrayOf(2, 2, 0)))
    }

    @Test
    fun `a pivot buffer that matches neither width is judged LP64`() {
        assertFalse(isIlp64PivotWidth(intArrayOf(0, 0, 0)), "a routine that wrote nothing cannot be judged")
        assertFalse(isIlp64PivotWidth(intArrayOf(1, 0, 2)), "a first pivot the probe did not ask for")
        assertFalse(isIlp64PivotWidth(intArrayOf(2, 0)), "too few words to tell the widths apart")
    }
}
