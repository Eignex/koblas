package com.eignex.koblas.dense

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** The one part of the ILP64 check that can be tested without an ILP64 OpenBLAS installed. */
class OpenBlasConfigTest {

    @Test
    fun `a default LP64 build is accepted`() {
        // Config strings in the shape Debian's libopenblas0 and a Homebrew build report.
        assertFalse(isIlp64OpenBlas("OpenBLAS 0.3.21 DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64"))
        assertFalse(isIlp64OpenBlas("OpenBLAS 0.3.28 NO_LAPACKE DYNAMIC_ARCH NO_AFFINITY Zen MAX_THREADS=128"))
        assertFalse(isIlp64OpenBlas(""), "a build with no config string cannot be judged, so it passes")
        // A token that merely starts with the marker is not the marker; matching substrings would reject
        // this build for saying that the interface it does not have is off.
        assertFalse(isIlp64OpenBlas("OpenBLAS 0.3.21 USE64BITINT_OFF DYNAMIC_ARCH Haswell"))
    }

    @Test
    fun `an ILP64 build is rejected under either spelling`() {
        assertTrue(isIlp64OpenBlas("OpenBLAS 0.3.21 USE64BITINT DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64"))
        assertTrue(isIlp64OpenBlas("OpenBLAS 0.3.28 INTERFACE64 DYNAMIC_ARCH Zen MAX_THREADS=128"))
    }
}
