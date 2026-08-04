package com.eignex.koblas

import com.eignex.koblas.dense.PlatformVectorKernels
import com.eignex.koblas.dense.denseKernels

/**
 * Short human-readable identifier for the vector kernels the current process resolved: `"scalar"` on any
 * non-JVM target (or a JVM started without `--add-modules=jdk.incubator.vector`), `"simd(4 lanes)"` on a
 * JVM with AVX2, `"simd(8 lanes)"` with AVX-512, and a `"+openblas"` suffix when a host backend is
 * registered for long runs. Print at startup to verify your runtime picked up what you expected.
 *
 * Common rather than per-target now that [PlatformVectorKernels] names itself: it is just
 * `denseKernels.name`. The two `actual`s it replaces had drifted — the JVM's was an eagerly computed `val`
 * that never mentioned a registered host backend, while the other reported one.
 */
public val mathBackend: String get() = denseKernels.name
