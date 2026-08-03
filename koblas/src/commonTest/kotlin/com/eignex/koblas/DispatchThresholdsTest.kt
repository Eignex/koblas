package com.eignex.koblas

import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * The dispatch thresholds themselves, which are ordinary values and worth the same scrutiny as anything
 * else that decides which code path runs.
 *
 * The routing they gate is verified by the conformance suites: a gated routine has to give the same answer
 * on either side of its threshold, which those suites already check by sweeping sizes that straddle these
 * values. What is checked here is that the numbers are coherent and that the override plumbing exists,
 * since a threshold silently resolving to zero or to nonsense would route everything the wrong way without
 * any test noticing.
 */
class DispatchThresholdsTest {

    @Test
    fun `thresholds are non-negative and ordered by how much work amortizes a call`() {
        val t = dispatchThresholds
        assertTrue(t.level2 >= 0 && t.level3 >= 0 && t.lapack >= 0, "negative threshold: $t")
        // Level 2 does O(n^2) work over O(n^2) data, so it can never amortize a call earlier than level 3
        // does; on a platform where the portable kernels win outright it is Int.MAX_VALUE.
        assertTrue(t.level2 >= t.level3, "level 2 (${t.level2}) should not dispatch before level 3 (${t.level3})")
    }

    @Test
    fun `the platform defaults are the ones this target was measured with`() {
        val defaults = platformDispatchThresholds
        // Off the JVM the portable kernels are scalar loops and lose from the smallest size measured; on
        // the JVM they are SIMD and win level 2 at every size, so the two platforms differ by design.
        // The signal is which kernels are in force, not which platform: a JVM launched without the
        // incubator vector module runs the same scalar loops the native targets do, and inherits their
        // thresholds rather than the SIMD-derived ones.
        if (mathBackend.startsWith("simd")) {
            assertTrue(defaults.level2 == Int.MAX_VALUE, "SIMD level 2 should stay portable, got ${defaults.level2}")
            assertTrue(defaults.level3 in 1..1024, "SIMD level 3 threshold looks wrong: ${defaults.level3}")
            assertTrue(defaults.lapack in 1..1024, "SIMD lapack threshold looks wrong: ${defaults.lapack}")
        } else {
            assertTrue(defaults.level3 == 0, "scalar kernels should dispatch level 3 from the start")
            assertTrue(defaults.level2 == 0, "scalar kernels should dispatch level 2 from the start")
        }
    }
}
